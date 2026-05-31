"""Runtime configuration, all overridable via environment variables."""
import os

# --- Paths (inside the container) ---
SCAN_DIR = os.getenv("SCAN_DIR", "/data/incoming")      # read-only bind mount of host transfer dir
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/data/uploads")   # files uploaded via the UI
DB_PATH = os.getenv("DB_PATH", "/data/wavereader.db")

# --- Transcription engine ---
# "faster_whisper" (default, fastest) or "transformers" (PyTorch fallback, guaranteed Blackwell/sm_120 support)
WHISPER_ENGINE = os.getenv("WHISPER_ENGINE", "faster_whisper")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("DEVICE", "cuda")
# float16 is ideal on the 5070 Ti. Use "int8_float16" to cut VRAM, "float32" for CPU.
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
LANGUAGE = os.getenv("LANGUAGE", "en")
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "5"))
VAD = os.getenv("VAD", "true").lower() in ("1", "true", "yes")

# Bias the model toward amateur/CB radio vocabulary. Whisper uses this as a soft hint.
INITIAL_PROMPT = os.getenv(
    "INITIAL_PROMPT",
    "Amateur radio voice contact. Callsigns, phonetic alphabet (alpha bravo charlie), "
    "CQ, QSL, QSO, QTH, QRZ, roger, copy, over, break, 73, signal report.",
)

# --- Scanner behavior ---
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "30"))    # seconds between directory scans
STABLE_SECONDS = int(os.getenv("STABLE_SECONDS", "15"))  # file must be untouched this long before ingest
# Default for the UI "scan recursively" toggle (persisted in the DB once set).
SCAN_RECURSIVE_DEFAULT = os.getenv("SCAN_RECURSIVE", "true").lower() in ("1", "true", "yes")
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()       # DEBUG|INFO|WARNING|ERROR
LOG_DIR = os.getenv("LOG_DIR", "/data/logs")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024)))  # 5 MB per file
LOG_BACKUPS = int(os.getenv("LOG_BACKUPS", "5"))         # rotated files kept
LOG_BUFFER_LINES = int(os.getenv("LOG_BUFFER_LINES", "2000"))  # in-memory ring for the UI panel
