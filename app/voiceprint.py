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
import time

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
            log.info("loading speaker model %s on %s (savedir=%s)",
                     config.SPK_MODEL, config.DEVICE, savedir)
            t0 = time.monotonic()
            try:
                _model = EncoderClassifier.from_hparams(
                    source=config.SPK_MODEL, savedir=savedir,
                    run_opts={"device": config.DEVICE},
                )
            except Exception as e:
                log.error("speaker model load FAILED: %s", e)
                raise
            log.info("speaker model loaded in %.1fs", time.monotonic() - t0)
    return _model


def _clip(path: str, start: float, end: float) -> str:
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="wr_emb_")
    os.close(fd)
    af = (["-af", config.SPK_PREPROCESS_FILTERS]
          if config.SPK_PREPROCESS and config.SPK_PREPROCESS_FILTERS else [])
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", path,
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
           "-ac", "1", "-ar", "16000", *af, tmp]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        os.remove(tmp)
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise RuntimeError(f"ffmpeg clip failed (rc={proc.returncode}): {' | '.join(tail)}")
    return tmp


def embed(path: str, start: float, end: float) -> np.ndarray:
    """Return a normalized float32 voiceprint for the given time range."""
    import soundfile as sf
    import torch
    log.debug("embed: %s [%.2f-%.2f] dur=%.2fs preprocess=%s",
              os.path.basename(path), start, end, end - start, config.SPK_PREPROCESS)
    tmp = _clip(path, start, end)
    try:
        # Read with soundfile (libsndfile) — avoids torchaudio's TorchCodec backend.
        sig, _sr = sf.read(tmp, dtype="float32")
        if sig.ndim > 1:                       # safety; clips are forced mono
            sig = sig.mean(axis=1)
        signal = torch.from_numpy(np.ascontiguousarray(sig)).unsqueeze(0)  # [1, T]
        emb = _load().encode_batch(signal).squeeze().detach().cpu().numpy().astype("float32")
        norm = float(np.linalg.norm(emb))
        log.debug("embed: ok, dim=%d norm=%.3f", emb.shape[0], norm)
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


def identify(emb: np.ndarray, speakers: list[dict], topk: int = 3) -> list[dict]:
    """Rank speakers for one voiceprint using kNN: the mean of each speaker's top-K
    most-similar enrolled prints. speakers = [{id, name, embs(np [N,dim])}].
    Returns [{id, name, score}] sorted best-first (empty if no speakers)."""
    ranked = []
    for sp in speakers:
        embs = sp["embs"]
        if embs is None or len(embs) == 0:
            continue
        sims = embs @ emb                       # cosine (all normalized)
        k = min(topk, sims.shape[0])
        score = float(np.sort(sims)[-k:].mean())
        ranked.append({"id": sp["id"], "name": sp["name"], "score": round(score, 3)})
    ranked.sort(key=lambda x: -x["score"])
    return ranked
