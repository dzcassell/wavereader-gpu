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

# --- Audio recovery / pre-cleaning ---
# When on, the audio is run through an ffmpeg filter chain before transcription
# to pull speech out of weak/noisy signals. Default off so clean FM audio isn't
# altered; toggle per-file in the UI, or flip the default here.
PREPROCESS = os.getenv("PREPROCESS", "false").lower() in ("1", "true", "yes")
# Tuned for comms voice: band-limit to the speech band, FFT denoise, then bring
# up quiet passages. Override the whole chain via env if you want.
PREPROCESS_FILTERS = os.getenv(
    "PREPROCESS_FILTERS",
    "highpass=f=250,lowpass=f=3000,afftdn=nf=-20,dynaudnorm=f=150:g=15",
)

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

# Optional: POST {id, file, matched} here when a transcript hits a watch term.
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# --- Speaker identification (voiceprints) ---
# SpeechBrain ECAPA-TDNN embedding model; downloaded to the model-cache volume.
SPK_MODEL = os.getenv("SPK_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
# A segment must be at least this long (seconds) to enroll a usable voiceprint.
MIN_ENROLL_SEC = float(os.getenv("MIN_ENROLL_SEC", "1.5"))
# Cosine-similarity threshold for auto-identifying a speaker (ECAPA voiceprints).
# Higher = stricter (fewer, more-confident matches). ~0.40-0.55 is a sane range.
SPK_THRESHOLD = float(os.getenv("SPK_THRESHOLD", "0.45"))
# Pre-clean the audio of each voiceprint clip (band-limit + denoise) before ECAPA.
# Must match between enrollment and identification — run "Rebuild voiceprints" after
# changing these so existing prints are recomputed the same way.
SPK_PREPROCESS = os.getenv("SPK_PREPROCESS", "true").lower() in ("1", "true", "yes")
# Band-pass only by default: length-invariant, so whole-file-decode and per-segment
# clipping yield identical embeddings (and it's much cheaper than FFT denoise).
SPK_PREPROCESS_FILTERS = os.getenv("SPK_PREPROCESS_FILTERS", "highpass=f=250,lowpass=f=3000")
# kNN matching: average the top-K most-similar enrolled prints per speaker.
SPK_TOPK = int(os.getenv("SPK_TOPK", "3"))
# A segment must be at least this long to *auto-identify* (stricter than enrollment).
SPK_ID_MIN_SEC = float(os.getenv("SPK_ID_MIN_SEC", "2.0"))
# Require best - second-best speaker score to exceed this (0 = off). Cuts confusions.
SPK_MIN_MARGIN = float(os.getenv("SPK_MIN_MARGIN", "0.0"))
# Skip enrolling a voiceprint from a clip whose Whisper avg_logprob is below this
# (garbage audio -> garbage print). Set very low to effectively disable.
SPK_ENROLL_MIN_LOGPROB = float(os.getenv("SPK_ENROLL_MIN_LOGPROB", "-1.5"))
# How many embeddings to run through the GPU per batch.
SPK_BATCH = int(os.getenv("SPK_BATCH", "16"))
# If a file has at least this many segments to embed, decode it once and slice in
# memory instead of clipping each segment separately.
SPK_DECODE_ALL_MIN = int(os.getenv("SPK_DECODE_ALL_MIN", "4"))

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()       # DEBUG|INFO|WARNING|ERROR
LOG_DIR = os.getenv("LOG_DIR", "/data/logs")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024)))  # 5 MB per file
LOG_BACKUPS = int(os.getenv("LOG_BACKUPS", "5"))         # rotated files kept
LOG_BUFFER_LINES = int(os.getenv("LOG_BUFFER_LINES", "2000"))  # in-memory ring for the UI panel
