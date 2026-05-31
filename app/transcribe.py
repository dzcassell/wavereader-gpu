"""Transcription engines. Two backends behind one interface.

faster_whisper  -> CTranslate2, fastest, lowest VRAM. Preferred.
transformers    -> PyTorch + HF Whisper. Heavier, but ships CUDA 12.8 / sm_120
                   kernels so it is the reliable fallback on Blackwell (RTX 50xx).

A segment is {"start": float, "end": float, "text": str}.

Models are loaded on demand and cached by (engine, model) so a recording can be
re-transcribed with a different model without disturbing the default. Each cached
model holds VRAM until the process restarts.
"""
import threading
from typing import Optional

from . import config

# Models the UI offers for re-transcription.
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"]
AVAILABLE_ENGINES = ["faster_whisper", "transformers"]

_cache: dict[tuple[str, str], object] = {}
_cache_lock = threading.Lock()


def _gpu_summary() -> str:
    try:
        import torch  # noqa
        if torch.cuda.is_available():
            return f"cuda: {torch.cuda.get_device_name(0)}"
        return "cuda: NOT available (torch sees no GPU)"
    except Exception as e:  # torch may not be installed for the faster_whisper-only path
        return f"cuda: unknown ({e})"


def _build_faster_whisper(model: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model, device=config.DEVICE, compute_type=config.COMPUTE_TYPE)


def _build_transformers(model: str):
    import torch
    from transformers import pipeline
    dtype = torch.float16 if config.DEVICE == "cuda" else torch.float32
    model_id = model if "/" in model else f"openai/whisper-{model}"
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=dtype,
        device=0 if config.DEVICE == "cuda" else -1,
        return_timestamps=True,
        chunk_length_s=30,
    )


def _get_model(engine: str, model: str):
    key = (engine, model)
    with _cache_lock:
        if key not in _cache:
            if engine == "faster_whisper":
                _cache[key] = _build_faster_whisper(model)
            elif engine == "transformers":
                _cache[key] = _build_transformers(model)
            else:
                raise ValueError(f"Unknown WHISPER_ENGINE: {engine}")
        return _cache[key]


def load() -> str:
    """Eagerly load the default model at startup. Returns a status string."""
    _get_model(config.WHISPER_ENGINE, config.WHISPER_MODEL)
    return f"engine={config.WHISPER_ENGINE} model={config.WHISPER_MODEL} {_gpu_summary()}"


def loaded_models() -> list[str]:
    with _cache_lock:
        return [f"{e}:{m}" for (e, m) in _cache.keys()]


def free_models() -> list[str]:
    """Drop all cached models and release VRAM. A model currently mid-transcription
    stays alive via the worker's local reference, so this is safe to call any time;
    the next job reloads whatever it needs."""
    with _cache_lock:
        freed = [f"{e}:{m}" for (e, m) in _cache.keys()]
        _cache.clear()
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return freed


def _transcribe_faster_whisper(mdl, path: str):
    vad_params = {"min_silence_duration_ms": 500} if config.VAD else None
    segments_iter, info = mdl.transcribe(
        path,
        language=config.LANGUAGE or None,
        beam_size=config.BEAM_SIZE,
        vad_filter=config.VAD,
        vad_parameters=vad_params,
        initial_prompt=config.INITIAL_PROMPT,
    )
    segments = []
    for s in segments_iter:  # generator: streaming decode
        segments.append({"start": round(s.start, 2), "end": round(s.end, 2),
                         "text": s.text.strip()})
    return segments, info.duration, info.language


def _transcribe_transformers(mdl, path: str):
    out = mdl(path, generate_kwargs={"language": config.LANGUAGE} if config.LANGUAGE else {})
    segments = []
    duration = 0.0
    for ch in out.get("chunks", []):
        ts = ch.get("timestamp") or (None, None)
        start = ts[0] if ts[0] is not None else 0.0
        end = ts[1] if ts[1] is not None else start
        duration = max(duration, end or 0.0)
        segments.append({"start": round(start, 2), "end": round(end, 2),
                         "text": ch["text"].strip()})
    if not segments:  # no chunking -> single block
        segments = [{"start": 0.0, "end": 0.0, "text": out.get("text", "").strip()}]
    return segments, duration, config.LANGUAGE


def transcribe(path: str, model: Optional[str] = None, engine: Optional[str] = None) -> dict:
    """Transcribe one file, optionally overriding model/engine for this job.

    Returns {segments, text, duration, language, model, engine}.
    """
    engine = engine or config.WHISPER_ENGINE
    model = model or config.WHISPER_MODEL
    mdl = _get_model(engine, model)
    if engine == "faster_whisper":
        segments, duration, language = _transcribe_faster_whisper(mdl, path)
    else:
        segments, duration, language = _transcribe_transformers(mdl, path)
    text = "\n".join(s["text"] for s in segments if s["text"])
    return {
        "segments": segments,
        "text": text,
        "duration": duration,
        "language": language,
        "model": model,
        "engine": engine,
    }
