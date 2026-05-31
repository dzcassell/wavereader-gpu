# wavereader-gpu

Web app that transcribes amateur/CB radio voice recordings (e.g. `.wav` files
pulled off an Icom IC-7610 SD card) using the host's **RTX 5070 Ti** GPU and
OpenAI Whisper `large-v3`. The UI lists each file with a caret to expand its
timestamped transcript and a button to download the original audio.

## How it works

- **FastAPI** backend serves a single-page UI and a small REST API.
- A **scanner** watches a read-only bind-mounted host directory and auto-queues
  new, fully-copied files. You can also **drag/drop upload** through the UI.
- A single background **worker** transcribes one file at a time (the GPU is the
  bottleneck) with **Silero VAD** enabled to skip dead air, and a ham-radio
  `initial_prompt` to bias callsigns, Q-codes, and the phonetic alphabet.
- State (jobs + transcripts) lives in **SQLite**; model weights and uploads
  persist in a Docker volume.

## Quick start (on the Debian 13 GPU host)

Prereqs: Docker, the **NVIDIA Container Toolkit**, and a recent driver
(CUDA 12.8-class, required for the Blackwell RTX 50xx). Verify the GPU is
visible to containers:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

Then:

```bash
cd /opt/wavereader-gpu        # or wherever you cloned it
cp .env.example .env
# edit .env -> set RECORDINGS_DIR to your IC-7610 transfer directory
mkdir -p "$(grep RECORDINGS_DIR .env | cut -d= -f2)"

docker compose up -d --build
docker compose logs -f        # watch for "model loaded: ... cuda: NVIDIA GeForce RTX 5070 Ti"
```

Open <http://HOST_IP:8080>. The header chip shows the live engine/GPU status.

> First start downloads the `large-v3` weights (~3 GB) into the `wavereader-data`
> volume; subsequent starts are fast.

## Blackwell (RTX 50xx) note

The container ships **two engines**. Default is `faster_whisper` (fastest, lowest
VRAM). If it errors at the first transcription with something like
`no kernel image is available for execution on the device`, CTranslate2's
prebuilt kernels don't yet cover sm_120 on your build — flip to the PyTorch
backend, which is installed from the CUDA 12.8 wheel index and does include
sm_120:

```bash
# in .env
WHISPER_ENGINE=transformers
```
```bash
docker compose up -d
```

The worker also surfaces a hard failure as a job error (and the startup log shows
the GPU name) rather than silently falling back to CPU.

## Tuning

| Env | Default | Notes |
|-----|---------|-------|
| `WHISPER_MODEL` | `large-v3` | `medium` is faster / lower VRAM |
| `COMPUTE_TYPE` | `float16` | `int8_float16` cuts VRAM (faster_whisper only) |
| `LANGUAGE` | `en` | set empty to auto-detect |
| `BEAM_SIZE` | `5` | lower = faster, slightly less accurate |
| `VAD` | `true` | trims silence/dead air before decoding |
| `INITIAL_PROMPT` | ham vocab | override to bias other vocabularies |
| `SCAN_INTERVAL` | `30` | seconds between directory scans |
| `STABLE_SECONDS` | `15` | file must be untouched this long before ingest |

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/recordings` | list all recordings + status (`?q=` full-text search over filenames + transcripts) |
| GET | `/api/recordings/{id}` | one recording with transcript segments |
| GET | `/api/recordings/{id}/download` | download original audio |
| GET | `/api/recordings/{id}/export?fmt=txt\|srt\|vtt` | download transcript as plain text, SRT, or WebVTT |
| POST | `/api/recordings/{id}/retranscribe` | re-queue a file; JSON body `{"model": "...", "engine": "..."}` optionally overrides model/engine for that run |
| POST | `/api/upload` | upload an audio file (multipart `file`) |
| GET | `/api/health` | engine load status |
| GET | `/api/gpu` | live GPU telemetry (name, utilization, VRAM, temp) via `nvidia-smi` |
| GET | `/api/models` | available models/engines, configured defaults, and currently loaded models |
| POST | `/api/models/free` | drop all cached models and release VRAM |
| GET | `/api/stats` | backlog tally (total / pending / processing / done / error) |

### Re-transcribing with a different model

Expand any file and use the **Re-transcribe with `<model>` `<engine>`** control at the
bottom of the transcript. Each distinct (engine, model) pair is loaded on demand
and cached in VRAM for the life of the container, so the first run of a new model
pays a load cost and subsequent ones are fast. The list of selectable models is
`tiny … large-v3-turbo`; both engines are selectable. This is also how you recover
a file that errored — just re-queue it.

The header has a **Free models** button that drops every cached model and releases
VRAM in one click (the GPU chip beside it shows the memory drop). The backlog
**progress bar** under the search box shows how much of the on-disk queue has been
transcribed, plus what is still queued or errored.

## Caveats for radio audio

Whisper is trained on broadcast/clean speech. Weak-signal SSB, heavy QSB/QRM,
and rapid callsign exchanges will produce errors — especially callsigns and
serial numbers. The VAD + initial prompt help, but treat transcripts as a
searchable first pass, not a log-quality record. Digital modes (FT8/PSK/RTTY)
and CW are **not** decoded — this handles voice (SSB/AM/FM) only.
