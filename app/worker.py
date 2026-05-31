"""Background loops: a directory scanner and a single-GPU transcription worker.

Both run as asyncio tasks. The heavy transcribe() call is pushed to a thread so it
never blocks the event loop / HTTP server. Only one file transcribes at a time
because the GPU is the bottleneck.
"""
import asyncio
import os
import time
import traceback

from . import config, db, transcribe

_log = print  # swap for logging if desired


async def scanner_loop() -> None:
    """Periodically ingest new, stable files from the read-only scan directory."""
    while True:
        try:
            _scan_once()
        except Exception:
            _log("[scanner] error:\n" + traceback.format_exc())
        await asyncio.sleep(config.SCAN_INTERVAL)


def _scan_once() -> None:
    if not os.path.isdir(config.SCAN_DIR):
        return
    now = time.time()
    for entry in os.scandir(config.SCAN_DIR):
        if not entry.is_file():
            continue
        if os.path.splitext(entry.name)[1].lower() not in config.AUDIO_EXTS:
            continue
        st = entry.stat()
        # Skip files still being copied (mtime too recent).
        if now - st.st_mtime < config.STABLE_SECONDS:
            continue
        rec_id = db.add_recording(entry.name, "scan", entry.path, st.st_size, st.st_mtime)
        if rec_id:
            _log(f"[scanner] queued {entry.name} (id={rec_id})")


async def worker_loop() -> None:
    """Pull pending recordings one at a time and transcribe them."""
    while True:
        row = db.next_pending()
        if row is None:
            await asyncio.sleep(2)
            continue
        rec_id = row["id"]
        path = row["path"]
        _log(f"[worker] transcribing id={rec_id} {row['filename']}")
        db.set_status(rec_id, "processing")
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            result = await asyncio.to_thread(
                transcribe.transcribe, path, row["req_model"], row["req_engine"])
            db.save_transcript(
                rec_id, result["text"], result["segments"], result["duration"],
                result["language"], result["model"], result["engine"],
            )
            _log(f"[worker] done id={rec_id} ({len(result['segments'])} segments)")
        except Exception as e:
            _log(f"[worker] FAILED id={rec_id}: {e}\n" + traceback.format_exc())
            db.set_status(rec_id, "error", error=str(e), completed_at=time.time())
