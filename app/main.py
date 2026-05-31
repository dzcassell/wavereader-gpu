"""FastAPI app: REST API + static single-page UI."""
import asyncio
import logging
import os
import re
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask

from . import config, db, entities, gpu, logging_setup, transcribe, voiceprint, worker

log = logging.getLogger("wavereader.api")

_AUDIO_MIME = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg",
               ".m4a": "audio/mp4", ".ogg": "audio/ogg"}


class RetranscribeReq(BaseModel):
    model: str | None = None
    engine: str | None = None
    preprocess: bool | None = None
    vad: bool | None = None


class SettingsReq(BaseModel):
    recursive: bool | None = None


class ClipReq(BaseModel):
    ranges: list[tuple[float, float]]


class WatchReq(BaseModel):
    terms: list[str]


class SpeakerReq(BaseModel):
    name: str


class TagReq(BaseModel):
    segments: list[int]
    speaker_id: int | None = None
    name: str | None = None


class UntagReq(BaseModel):
    segments: list[int]

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
async def recordings(q: str | None = None, alerts_only: bool = False):
    return db.list_recordings(q, alerts_only)


@app.get("/api/entities")
async def entities_index(type: str | None = None, limit: int = 500):
    """Aggregate 'heard' index: distinct callsigns/Q-codes/frequencies with counts
    and the recordings/timestamps where each was heard."""
    agg: dict = {}
    for rid, fname, ents in db.iter_entities():
        for e in ents:
            if type and type != "all" and e["type"] != type:
                continue
            d = agg.setdefault((e["type"], e["value"]),
                               {"type": e["type"], "value": e["value"], "count": 0, "occurrences": []})
            d["count"] += 1
            if len(d["occurrences"]) < 100:
                d["occurrences"].append({"id": rid, "filename": fname, "start": e["start"]})
    items = sorted(agg.values(), key=lambda x: (-x["count"], x["value"]))[:limit]
    return {"entities": items}


@app.get("/api/search")
async def search(q: str, limit: int = 200):
    if not q.strip():
        return {"hits": []}
    hits = await asyncio.to_thread(db.search_segments, q.strip(), limit)
    return {"hits": hits}


@app.get("/api/watch")
async def get_watch():
    raw = db.get_setting("watch_terms", "") or ""
    return {"terms": [t.strip() for t in raw.splitlines() if t.strip()]}


@app.post("/api/watch")
async def set_watch(req: WatchReq):
    terms = [t.strip() for t in req.terms if t.strip()]
    db.set_setting("watch_terms", "\n".join(terms))
    log.info("watch terms updated: %d term(s)", len(terms))
    return {"terms": terms}


@app.post("/api/reindex")
async def reindex():
    """Recompute entities + alerts for every transcribed recording (e.g. after
    changing watch terms, or to backfill transcripts made before this feature)."""
    terms = (await get_watch())["terms"]

    def run():
        n = 0
        for rid, _fname, segs, text in db.iter_done_segments():
            ents = entities.extract(segs)
            db.set_entities(rid, ents, entities.match_terms(text, ents, terms))
            n += 1
        return n

    count = await asyncio.to_thread(run)
    log.info("reindex complete: %d recordings", count)
    return {"reindexed": count}


# --- Speakers / voice tagging (Phase 1: manual enrollment) ---

@app.get("/api/speakers")
async def speakers():
    return {"speakers": db.list_speakers()}


@app.post("/api/speakers")
async def add_speaker(req: SpeakerReq):
    if not req.name.strip():
        raise HTTPException(400, "name required")
    sid = db.create_speaker(req.name)
    return {"id": sid, "name": req.name.strip()}


@app.delete("/api/speakers/{speaker_id}")
async def remove_speaker(speaker_id: int):
    db.delete_speaker(speaker_id)
    log.info("deleted speaker id=%s", speaker_id)
    return {"deleted": True}


@app.post("/api/recordings/{rec_id}/tag")
async def tag(rec_id: int, req: TagReq):
    rec = db.get_recording(rec_id)
    if not rec:
        raise HTTPException(404, "not found")
    segs = rec.get("segments") or []
    if not segs:
        raise HTTPException(400, "recording has no segments")
    if req.speaker_id:
        sid = req.speaker_id
        name = db.speaker_name(sid)
        if not name:
            raise HTTPException(404, "speaker not found")
    elif req.name and req.name.strip():
        sid = db.create_speaker(req.name)
        name = req.name.strip()
    else:
        raise HTTPException(400, "speaker_id or name required")

    path = rec["path"]
    can_embed = os.path.exists(path)

    def work():
        enrolled, short = 0, 0
        for i in req.segments:
            if i < 0 or i >= len(segs):
                continue
            s = segs[i]
            start, end = s.get("start", 0) or 0, s.get("end", 0) or 0
            db.set_segment_tag(rec_id, i, sid, "manual", None)
            if can_embed and (end - start) >= config.MIN_ENROLL_SEC:
                try:
                    emb = voiceprint.embed(path, start, end)
                    db.add_voiceprint(sid, rec_id, i, start, end, emb.tobytes(), int(emb.shape[0]))
                    enrolled += 1
                except Exception as e:
                    log.warning("embed failed rec=%s seg=%s: %s", rec_id, i, e)
            elif end - start < config.MIN_ENROLL_SEC:
                short += 1
        if enrolled:
            _build_profile(sid)   # keep the speaker's profile current
        return enrolled, short

    enrolled, short = await asyncio.to_thread(work)
    log.info("tagged rec=%s speaker=%s segs=%d enrolled=%d (skipped %d too-short)",
             rec_id, name, len(req.segments), enrolled, short)
    return {"speaker_id": sid, "name": name, "tagged": len(req.segments),
            "enrolled": enrolled, "skipped_short": short}


@app.post("/api/recordings/{rec_id}/untag")
async def untag(rec_id: int, req: UntagReq):
    for i in req.segments:
        db.remove_segment_tag(rec_id, i)
        db.remove_voiceprint(rec_id, i)
    return {"untagged": len(req.segments)}


# --- Speaker profiles + auto-identification (Phase 2) ---

def _build_profile(speaker_id: int) -> int:
    """Recompute a speaker's profile (centroid) from its voiceprints. Returns count."""
    blobs = db.voiceprint_blobs(speaker_id)
    if not blobs:
        return 0
    c = voiceprint.centroid([voiceprint.blob_to_np(b) for b in blobs])
    db.set_profile(speaker_id, c.tobytes(), int(c.shape[0]), len(blobs))
    return len(blobs)


def _load_profiles() -> list[dict]:
    return [{"id": p["id"], "name": p["name"], "emb": voiceprint.blob_to_np(p["emb"])}
            for p in db.get_profiles()]


def _identify_recording(rec_id: int, threshold: float, overwrite: bool = True,
                        profiles: list = None) -> int:
    rec = db.get_recording(rec_id)
    if not rec or not os.path.exists(rec["path"]):
        return 0
    segs = rec.get("segments") or []
    if profiles is None:
        profiles = _load_profiles()
    if not profiles:
        return 0
    tags = db.get_segment_tags(rec_id)
    n = 0
    for i, s in enumerate(segs):
        ex = tags.get(i)
        if ex and ex["source"] == "manual":
            continue                          # never override a human tag
        if ex and ex["source"] == "auto" and not overwrite:
            continue
        start, end = s.get("start", 0) or 0, s.get("end", 0) or 0
        if end - start < config.MIN_ENROLL_SEC:
            continue
        try:
            emb = voiceprint.embed(rec["path"], start, end)
        except Exception as e:
            log.warning("identify embed failed rec=%s seg=%s: %s", rec_id, i, e)
            continue
        sid, score = voiceprint.identify(emb, profiles)
        if sid is not None and score >= threshold:
            db.set_segment_tag(rec_id, i, sid, "auto", round(score, 3))
            n += 1
        elif ex and ex["source"] == "auto":
            db.remove_segment_tag(rec_id, i)   # was auto, no longer confident
    return n


@app.post("/api/speakers/{speaker_id}/profile")
async def build_profile(speaker_id: int):
    n = await asyncio.to_thread(_build_profile, speaker_id)
    return {"speaker_id": speaker_id, "voiceprints": n}


@app.post("/api/profiles")
async def build_all_profiles():
    def run():
        return sum(1 for sp in db.list_speakers() if _build_profile(sp["id"]))
    built = await asyncio.to_thread(run)
    log.info("built %d speaker profile(s)", built)
    return {"profiles_built": built}


@app.post("/api/recordings/{rec_id}/identify")
async def identify_one(rec_id: int, threshold: float | None = None):
    th = config.SPK_THRESHOLD if threshold is None else threshold
    n = await asyncio.to_thread(_identify_recording, rec_id, th, True, None)
    log.info("identify rec=%s: %d auto-tag(s) at threshold %.2f", rec_id, n, th)
    return {"identified": n, "threshold": th}


@app.post("/api/identify")
async def identify_all(threshold: float | None = None, only_new: bool = True):
    """Scan transcribed files and auto-tag recognized voices. Runs in the background
    (can be long); the Tags column fills in live as it progresses."""
    th = config.SPK_THRESHOLD if threshold is None else threshold
    if not db.get_profiles():
        raise HTTPException(400, "no speaker profiles yet — tag some voices first")

    async def run():
        def work():
            profiles = _load_profiles()
            files, tags = 0, 0
            for rid, _f, _segs, _t in db.iter_done_segments():
                if only_new and any(t["source"] == "auto"
                                    for t in db.get_segment_tags(rid).values()):
                    continue
                tags += _identify_recording(rid, th, True, profiles)
                files += 1
                if files % 25 == 0:
                    log.info("identify-all progress: %d files, %d auto-tags so far", files, tags)
            return files, tags
        files, tags = await asyncio.to_thread(work)
        log.info("identify-all complete: %d files scanned, %d auto-tags at threshold %.2f",
                 files, tags, th)

    asyncio.create_task(run())
    return {"started": True, "threshold": th, "only_new": only_new}


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


@app.get("/api/recordings/{rec_id}/audio")
async def audio(rec_id: int):
    """Serve the audio inline (no attachment) for the in-page player. FileResponse
    handles HTTP Range requests, so seeking works."""
    rec = db.get_recording(rec_id)
    if not rec:
        raise HTTPException(404, "not found")
    if not os.path.exists(rec["path"]):
        raise HTTPException(410, "file no longer on disk")
    ext = os.path.splitext(rec["filename"])[1].lower()
    return FileResponse(rec["path"], media_type=_AUDIO_MIME.get(ext, "application/octet-stream"))


def _build_clip(src: str, ranges: list[tuple[float, float]], out: str) -> None:
    """Stitch the given time ranges of src into one wav via ffmpeg atrim+concat."""
    trims, labels = [], []
    for i, (s, e) in enumerate(ranges):
        trims.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]")
        labels.append(f"[a{i}]")
    fc = ";".join(trims) + ";" + "".join(labels) + f"concat=n={len(ranges)}:v=0:a=1[out]"
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", src, "-filter_complex", fc, "-map", "[out]", out]
    subprocess.run(cmd, capture_output=True, check=True, timeout=900)


@app.post("/api/recordings/{rec_id}/clip")
async def clip(rec_id: int, req: ClipReq):
    """Export a single .wav containing only the selected transcript ranges."""
    rec = db.get_recording(rec_id)
    if not rec:
        raise HTTPException(404, "not found")
    if not os.path.exists(rec["path"]):
        raise HTTPException(410, "file no longer on disk")
    ranges = [(max(0.0, float(s)), float(e)) for s, e in req.ranges if float(e) > float(s)]
    if not ranges:
        raise HTTPException(400, "no valid ranges")
    if len(ranges) > 1000:
        raise HTTPException(400, "too many ranges (max 1000)")
    ranges.sort()
    stem = os.path.splitext(rec["filename"])[0]
    fd, out = tempfile.mkstemp(suffix=".wav", prefix="wr_clip_")
    os.close(fd)
    try:
        await asyncio.to_thread(_build_clip, rec["path"], ranges, out)
    except Exception as e:
        try:
            os.remove(out)
        except OSError:
            pass
        log.error("clip failed id=%s: %s", rec_id, e)
        raise HTTPException(500, "clip extraction failed")
    log.info("clip id=%s: %d range(s) -> %s_clip.wav", rec_id, len(ranges), stem)
    return FileResponse(out, filename=f"{stem}_clip.wav", media_type="audio/wav",
                        background=BackgroundTask(os.remove, out))


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
