"""Speaker voiceprints via SpeechBrain ECAPA-TDNN.

embed(path, start, end) clips a segment to 16k mono and returns a normalized
192-dim embedding. Cosine similarity between embeddings measures voice similarity.
The model is loaded lazily on first use so the app still starts if speaker-ID
isn't being used.
"""
import logging
import os
import subprocess
import tempfile
import threading

import numpy as np

from . import config

log = logging.getLogger("wavereader.voiceprint")

_model = None
_lock = threading.Lock()


def _load():
    global _model
    with _lock:
        if _model is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:  # older speechbrain layout
                from speechbrain.pretrained import EncoderClassifier
            savedir = os.path.join(os.getenv("HF_HOME", "/data/model-cache"), "ecapa")
            log.info("loading speaker model %s on %s", config.SPK_MODEL, config.DEVICE)
            _model = EncoderClassifier.from_hparams(
                source=config.SPK_MODEL, savedir=savedir,
                run_opts={"device": config.DEVICE},
            )
            log.info("speaker model loaded")
    return _model


def _clip(path: str, start: float, end: float) -> str:
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="wr_emb_")
    os.close(fd)
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", path,
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
           "-ac", "1", "-ar", "16000", tmp]
    subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    return tmp


def embed(path: str, start: float, end: float) -> np.ndarray:
    """Return a normalized float32 voiceprint for the given time range."""
    import torch  # noqa
    import torchaudio
    tmp = _clip(path, start, end)
    try:
        signal, _sr = torchaudio.load(tmp)
        emb = _load().encode_batch(signal).squeeze().detach().cpu().numpy().astype("float32")
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else emb
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def blob_to_np(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two already-normalized vectors."""
    return float(np.dot(a, b))


def centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    """Normalized mean of voiceprints — a speaker's profile vector."""
    m = np.vstack(embeddings).mean(axis=0)
    norm = float(np.linalg.norm(m))
    return (m / norm if norm > 0 else m).astype("float32")


def identify(emb: np.ndarray, profiles: list[dict]) -> tuple:
    """Return (best_speaker_id, score) over profiles [{id, emb}], or (None, 0.0)."""
    best_id, best = None, -1.0
    for p in profiles:
        sc = cosine(emb, p["emb"])
        if sc > best:
            best, best_id = sc, p["id"]
    return best_id, max(0.0, best)
