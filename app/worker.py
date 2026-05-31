"""Background loops: a directory scanner and a single-GPU transcription worker.

Both run as asyncio tasks. The heavy transcribe() call is pushed to a thread so it
never blocks the event loop / HTTP server. Only one file transcribes at a time
because the GPU is the bottleneck.
"""
import asyncio
import logging
import os
import time
import traceback

from . import config, db, transcribe

log = logging.getLogger("wavereader.worker")
scanlog = logging.getLogger("wavereader.scanner")


def _recursive_enabled() -> bool:
    return db.get_setting("recursive", "1" if config.SCAN_RECURSIVE_DEFAULT else "0") == "1"


async def scanner_loop() -> None:
    """Periodically ingest new, stable files from the scan directory."""
    scanlog.info("scanner started: dir=%s interval=%ss stable=%ss",
                 config.SCAN_DIR, config.SCAN_INTERVAL, config.STABLE_SECONDS)
    while True:
        try:
            _scan_once()
        except Exception:
            scanlog.error("scan error:\n%s", traceback.format_exc())
        await asyncio.sleep(config.SCAN_INTERVAL)


def _iter_audio_files(root: str, recursive: bool):
    """Yield (name, fullpath, stat) for audio files under root."""
    if recursive:
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if os.path.splitext(name)[1].lower() in config.AUDIO_EXTS:
                    full = os.path.join(dirpath, name)
                    try:
                        yield name, full, os.stat(full)
                    except OSError as e:
                        scanlog.debug("stat failed %s: %s", full, e)
    else:
        for entry in os.scandir(root):
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in config.AUDIO_EXTS:
                yield entry.name, entry.path, entry.stat()


def _scan_once() -> None:
    if not os.path.isdir(config.SCAN_DIR):
        scanlog.warning("scan dir does not exist: %s", config.SCAN_DIR)
        return
    recursive = _recursive_enabled()
    now = time.time()
    seen = queued = skipped_unstable = 0
    for name, path, st in _iter_audio_files(config.SCAN_DIR, recursive):
        seen += 1
        age = now - st.st_mtime
        if age < config.STABLE_SECONDS:  # still being copied
            skipped_unstable += 1
            scanlog.debug("skip (unstable, age=%.1fs): %s", age, path)
            continue
        rec_id = db.add_recording(name, "scan", path, st.st_size, st.st_mtime)
        if rec_id:
            queued += 1
            scanlog.info("queued id=%s %s (%d bytes)", rec_id, path, st.st_size)
    scanlog.debug("scan complete: recursive=%s seen=%d queued=%d unstable=%d",
                  recursive, seen, queued, skipped_unstable)


async def worker_loop() -> None:
    """Pull pending recordings one at a time and transcribe them."""
    log.info("worker started")
    while True:
        row = db.next_pending()
        if row is None:
            await asyncio.sleep(2)
            continue
        rec_id = row["id"]
        path = row["path"]
        model = row["req_model"] or config.WHISPER_MODEL
        engine = row["req_engine"] or config.WHISPER_ENGINE
        log.info("transcribing id=%s %s (engine=%s model=%s)", rec_id, row["filename"], engine, model)
        db.set_status(rec_id, "processing")
        t0 = time.monotonic()
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            result = await asyncio.to_thread(
                transcribe.transcribe, path, row["req_model"], row["req_engine"])
            elapsed = time.monotonic() - t0
            dur = result["duration"] or 0
            rtf = (dur / elapsed) if elapsed > 0 else 0
            db.save_transcript(
                rec_id, result["text"], result["segments"], result["duration"],
                result["language"], result["model"], result["engine"],
            )
            log.info("done id=%s: %d segments, audio=%.1fs in %.1fs (%.1fx realtime)",
                     rec_id, len(result["segments"]), dur, elapsed, rtf)
        except Exception as e:
            log.error("FAILED id=%s after %.1fs: %s\n%s",
                      rec_id, time.monotonic() - t0, e, traceback.format_exc())
            db.set_status(rec_id, "error", error=str(e), completed_at=time.time())
