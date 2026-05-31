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


def _patch_torchaudio():
    """SpeechBrain 1.0.x calls torchaudio's legacy backend API, which the very new
    torchaudio (cu128/Blackwell build) removed. We load audio via soundfile anyway,
    so re-add these as harmless shims to keep SpeechBrain importing/running."""
    try:
        import torchaudio
    except Exception:
        return
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None


def _load():
    global _model
    with _lock:
        if _model is None:
            _patch_torchaudio()
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
    dur = max(0.0, end - start)
    # Input seeking (-ss before -i) jumps straight to the segment instead of
    # decoding the whole file up to it — orders of magnitude faster on long files.
    cmd = ["ffmpeg", "-nostdin", "-y", "-ss", f"{start:.3f}", "-i", path,
           "-t", f"{dur:.3f}", "-ac", "1", "-ar", "16000", *af, tmp]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        os.remove(tmp)
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise RuntimeError(f"ffmpeg clip failed (rc={proc.returncode}): {' | '.join(tail)}")
    return tmp


def _clip_array(path: str, start: float, end: float):
    """One segment as a 16k-mono float32 array via a per-segment ffmpeg clip."""
    import soundfile as sf
    if end - start < 0.3:
        return None
    tmp = _clip(path, start, end)
    try:
        sig, _sr = sf.read(tmp, dtype="float32")
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        return np.ascontiguousarray(sig)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _decode_full(path: str) -> np.ndarray:
    """Decode the whole file once to 16k mono float32 (with the clean-audio chain)."""
    af = (["-af", config.SPK_PREPROCESS_FILTERS]
          if config.SPK_PREPROCESS and config.SPK_PREPROCESS_FILTERS else [])
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
           "-ac", "1", "-ar", "16000", *af, "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, timeout=1800)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise RuntimeError(f"ffmpeg decode failed (rc={proc.returncode}): {' | '.join(tail)}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _embed_clips(clips: list) -> list:
    """Batch a list of (possibly None) 16k-mono arrays through ECAPA on the GPU.
    Returns normalized embeddings aligned to input (None where the clip was None)."""
    import torch
    results = [None] * len(clips)
    items = [(k, c) for k, c in enumerate(clips) if c is not None and c.shape[0] > 0]
    if not items:
        return results
    model = _load()
    batch = max(1, config.SPK_BATCH)
    for b in range(0, len(items), batch):
        chunk = items[b:b + batch]
        maxlen = max(c.shape[0] for _, c in chunk)
        wavs = torch.zeros(len(chunk), maxlen, dtype=torch.float32)
        lens = torch.ones(len(chunk), dtype=torch.float32)
        for j, (_k, c) in enumerate(chunk):
            wavs[j, :c.shape[0]] = torch.from_numpy(c)
            lens[j] = c.shape[0] / maxlen
        with torch.no_grad():
            out = model.encode_batch(wavs, lens)        # [B, 1, D]
        embs = out.squeeze(1).detach().cpu().numpy().astype("float32")
        for j, (k, _c) in enumerate(chunk):
            v = embs[j]
            nrm = float(np.linalg.norm(v))
            results[k] = v / nrm if nrm > 0 else v
    log.debug("embedded %d clip(s) on %s", len(items), config.DEVICE)
    return results


def embed_segments(path: str, ranges: list) -> list:
    """Embed many segments of one file. Decodes the whole file once when there are
    enough segments to make that worthwhile, else clips each individually; either way
    the embeddings are computed in GPU batches. Returns a list aligned to `ranges`."""
    if not ranges:
        return []
    if len(ranges) >= config.SPK_DECODE_ALL_MIN:
        audio = _decode_full(path)
        sr = 16000
        clips = []
        for s, e in ranges:
            i0, i1 = max(0, int(s * sr)), int(e * sr)
            c = audio[i0:i1]
            clips.append(c if c.shape[0] >= int(0.3 * sr) else None)
    else:
        clips = [_clip_array(path, s, e) for s, e in ranges]
    return _embed_clips(clips)


def embed(path: str, start: float, end: float) -> np.ndarray:
    """Return a normalized float32 voiceprint for one time range."""
    out = embed_segments(path, [(start, end)])
    if not out or out[0] is None:
        raise RuntimeError("clip too short to embed")
    return out[0]


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
