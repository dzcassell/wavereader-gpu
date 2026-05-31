"""FastAPI app: REST API + static single-page UI."""
import asyncio
import os
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, db, gpu, transcribe, worker


class RetranscribeReq(BaseModel):
    model: str | None = None
    engine: str | None = None

_GPU_STATUS = "not loaded"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _GPU_STATUS
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    db.init()
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
    return {"freed": freed, "loaded": transcribe.loaded_models()}


@app.get("/api/stats")
async def stats():
    return db.counts()


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


@app.post("/api/recordings/{rec_id}/retranscribe")
async def retranscribe(rec_id: int, req: RetranscribeReq | None = None):
    rec = db.get_recording(rec_id)
    if not rec:
        raise HTTPException(404, "not found")
    model = req.model if req else None
    engine = req.engine if req else None
    if model and model not in transcribe.AVAILABLE_MODELS:
        raise HTTPException(400, f"unknown model {model}")
    if engine and engine not in transcribe.AVAILABLE_ENGINES:
        raise HTTPException(400, f"unknown engine {engine}")
    db.requeue(rec_id, model=model, engine=engine)
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
    return {"id": rec_id, "status": "queued"}


# Serve the UI at "/". Mounted last so it doesn't shadow /api routes.
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
