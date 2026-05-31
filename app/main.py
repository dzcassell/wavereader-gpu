"""FastAPI app: REST API + static single-page UI."""
import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, db, gpu, logging_setup, transcribe, worker

log = logging.getLogger("wavereader.api")


class RetranscribeReq(BaseModel):
    model: str | None = None
    engine: str | None = None
    preprocess: bool | None = None
    vad: bool | None = None


class SettingsReq(BaseModel):
    recursive: bool | None = None

_GPU_STATUS = "not loaded"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _GPU_STATUS
    logging_setup.setup()
    log.info("starting wavereader-gpu: engine=%s model=%s scan_dir=%s",
             config.WHISPER_ENGINE, config.WHISPER_MODEL, config.SCAN_DIR)
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    db.init()
    stuck = db.requeue_stuck()
    if stuck:
        log.info("re-queued %d recording(s) left in 'processing' by a previous run", stuck)
    # Load the model eagerly so the first transcription isn't cold, and so a
    # broken GPU setup surfaces at startup instead of silently mid-job.
    try:
        _GPU_STATUS = await asyncio.to_thread(transcribe.load)
        print(f"[startup] model loaded: {_GPU_STATUS}")
    except Exception as e:
        _GPU_STATUS = f"MODEL LOAD FAILED: {e}"
        print(f"[startup] {_GPU_STATUS}")
    asyncio.create_task(worker.scanner_loop())
    asyncio.create_task(worker.worker_loop())
    yield


app = FastAPI(title="wavereader-gpu", lifespan=lifespan)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]")


@app.get("/api/health")
async def health():
    return {"status": _GPU_STATUS, "engine": config.WHISPER_ENGINE, "model": config.WHISPER_MODEL}


@app.get("/api/gpu")
async def gpu_status():
    return await asyncio.to_thread(gpu.gpu_info)


@app.get("/api/models")
async def models():
    return {
        "models": transcribe.AVAILABLE_MODELS,
        "engines": transcribe.AVAILABLE_ENGINES,
        "default_model": config.WHISPER_MODEL,
        "default_engine": config.WHISPER_ENGINE,
        "loaded": transcribe.loaded_models(),
    }


@app.post("/api/models/free")
async def free_models():
    freed = await asyncio.to_thread(transcribe.free_models)
    log.info("free models requested via API: freed %d", len(freed))
    return {"freed": freed, "loaded": transcribe.loaded_models()}


@app.get("/api/stats")
async def stats():
    return db.counts()


def _recursive_enabled() -> bool:
    return db.get_setting("recursive", "1" if config.SCAN_RECURSIVE_DEFAULT else "0") == "1"


@app.get("/api/settings")
async def get_settings():
    return {
        "scan_dir": config.SCAN_DIR,
        "recursive": _recursive_enabled(),
        "default_preprocess": config.PREPROCESS,
        "default_vad": config.VAD,
    }


@app.post("/api/settings")
async def update_settings(req: SettingsReq):
    if req.recursive is not None:
        db.set_setting("recursive", "1" if req.recursive else "0")
        log.info("setting changed: recursive=%s", req.recursive)
    return {"scan_dir": config.SCAN_DIR, "recursive": _recursive_enabled()}


@app.get("/api/logs")
async def logs(limit: int = 300, level: str | None = None):
    limit = max(1, min(limit, config.LOG_BUFFER_LINES))
    return {"lines": logging_setup.recent(limit, level)}


@app.get("/api/recordings")
async def recordings(q: str | None = None):
    return db.list_recordings(q)


@app.get("/api/recordings/{rec_id}")
async def recording(rec_id: int):
    rec = db.get_recording(rec_id)
    if not rec:
        raise HTTPException(404, "not found")
    return rec


@app.get("/api/recordings/{rec_id}/download")
async def download(rec_id: int):
    rec = db.get_recording(rec_id)
    if not rec:
        raise HTTPException(404, "not found")
    if not os.path.exists(rec["path"]):
        raise HTTPException(410, "file no longer on disk")
    return FileResponse(rec["path"], filename=rec["filename"],
                        media_type="application/octet-stream")


def _srt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _to_srt(segments: list) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_srt_ts(seg['start'])} --> {_srt_ts(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _to_vtt(segments: list) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _srt_ts(seg["start"]).replace(",", ".")  # WebVTT uses '.' for ms
        end = _srt_ts(seg["end"]).replace(",", ".")
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


@app.get("/api/recordings/{rec_id}/export")
async def export(rec_id: int, fmt: str = "txt"):
    rec = db.get_recording(rec_id)
    if not rec:
        raise HTTPException(404, "not found")
    if rec["status"] != "done":
        raise HTTPException(409, f"not transcribed (status={rec['status']})")
    stem = os.path.splitext(rec["filename"])[0]
    if fmt == "txt":
        return PlainTextResponse(rec.get("text") or "", headers={
            "Content-Disposition": f'attachment; filename="{stem}.txt"'})
    if fmt == "srt":
        return Response(_to_srt(rec["segments"]), media_type="application/x-subrip", headers={
            "Content-Disposition": f'attachment; filename="{stem}.srt"'})
    if fmt == "vtt":
        return Response(_to_vtt(rec["segments"]), media_type="text/vtt", headers={
            "Content-Disposition": f'attachment; filename="{stem}.vtt"'})
    raise HTTPException(400, "fmt must be txt, srt, or vtt")


def _deletable(path: str) -> bool:
    """Only allow unlinking files that live under the scan or upload directories."""
    try:
        real = os.path.realpath(path)
        roots = [os.path.realpath(config.SCAN_DIR), os.path.realpath(config.UPLOAD_DIR)]
        return any(os.path.commonpath([real, r]) == r for r in roots)
    except (ValueError, OSError):
        return False


@app.delete("/api/recordings/{rec_id}")
async def delete_recording(rec_id: int):
    rec = db.get_recording(rec_id)
    if not rec:
        raise HTTPException(404, "not found")
    path = rec["path"]
    deleted_file = False
    if os.path.exists(path):
        if not _deletable(path):
            log.warning("refusing to delete out-of-bounds path id=%s %s", rec_id, path)
            raise HTTPException(403, "file is outside the scan/upload directories")
        try:
            os.remove(path)
            deleted_file = True
        except OSError as e:
            log.error("failed to unlink id=%s %s: %s", rec_id, path, e)
            raise HTTPException(500, f"could not delete file: {e}")
    db.delete_recording(rec_id)
    log.info("deleted id=%s %s (file_removed=%s)", rec_id, rec["filename"], deleted_file)
    return {"deleted": True, "file_removed": deleted_file}


@app.post("/api/recordings/{rec_id}/retranscribe")
async def retranscribe(rec_id: int, req: RetranscribeReq | None = None):
    rec = db.get_recording(rec_id)
    if not rec:
        raise HTTPException(404, "not found")
    model = req.model if req else None
    engine = req.engine if req else None
    preprocess = req.preprocess if req else None
    vad = req.vad if req else None
    if model and model not in transcribe.AVAILABLE_MODELS:
        raise HTTPException(400, f"unknown model {model}")
    if engine and engine not in transcribe.AVAILABLE_ENGINES:
        raise HTTPException(400, f"unknown engine {engine}")
    db.requeue(rec_id, model=model, engine=engine, preprocess=preprocess, vad=vad)
    log.info("re-queued id=%s (model=%s engine=%s preprocess=%s vad=%s)",
             rec_id, model or config.WHISPER_MODEL, engine or config.WHISPER_ENGINE,
             preprocess, vad)
    return {"status": "queued", "model": model or config.WHISPER_MODEL,
            "engine": engine or config.WHISPER_ENGINE}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in config.AUDIO_EXTS:
        raise HTTPException(400, f"unsupported type {ext}")
    safe = _SAFE_NAME.sub("_", os.path.basename(file.filename or "upload")) or "upload"
    dest = os.path.join(config.UPLOAD_DIR, safe)
    # Avoid clobbering an existing upload of the same name.
    if os.path.exists(dest):
        base, e = os.path.splitext(safe)
        dest = os.path.join(config.UPLOAD_DIR, f"{base}_{int(time.time())}{e}")
    size = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
            size += len(chunk)
    st = os.stat(dest)
    rec_id = db.add_recording(os.path.basename(dest), "upload", dest, size, st.st_mtime)
    if rec_id is None:
        raise HTTPException(409, "already ingested")
    log.info("uploaded id=%s %s (%d bytes)", rec_id, os.path.basename(dest), size)
    return {"id": rec_id, "status": "queued"}


# Serve the UI at "/". Mounted last so it doesn't shadow /api routes.
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
