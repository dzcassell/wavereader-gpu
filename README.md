# wavereader-gpu

Web app that transcribes amateur/CB radio voice recordings (e.g. `.wav` files
pulled off an Icom IC-7610 SD card) using the host's **RTX 5070 Ti** GPU and
OpenAI Whisper `large-v3`. The UI lists each file with a caret to expand its
timestamped transcript and a button to download the original audio.

## How it works

- **FastAPI** backend serves a single-page UI and a small REST API.
- A **scanner** watches a bind-mounted host directory (optionally recursively)
  and auto-queues new, fully-copied files. You can also **drag/drop upload**
  through the UI.
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
# .env defaults RECORDINGS_DIR to /data1/recordings/incoming -- edit if yours differs

docker compose up -d --build
docker compose logs -f        # watch for "model loaded: ... cuda: NVIDIA GeForce RTX 5070 Ti"
```

Open <http://HOST_IP:8080>. The header chip shows the live engine/GPU status.
(If 8080 is taken by another container, set `WEB_PORT` in `.env` to a free port.)

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
| `VAD` | `true` | trims silence/dead air before decoding; turn off to recover quiet/marginal speech |
| `PREPROCESS` | `false` | pre-clean weak audio (band-limit + denoise + normalize) before transcribing |
| `PREPROCESS_FILTERS` | comms voice chain | ffmpeg `-af` chain used when `PREPROCESS` is on |
| `INITIAL_PROMPT` | ham vocab | override to bias other vocabularies |
| `SCAN_INTERVAL` | `30` | seconds between directory scans |
| `STABLE_SECONDS` | `15` | file must be untouched this long before ingest |
| `SCAN_RECURSIVE` | `true` | default for the UI "Scan recursively" toggle |
| `LOG_LEVEL` | `INFO` | `DEBUG` for verbose troubleshooting |
| `LOG_MAX_BYTES` / `LOG_BACKUPS` | `5 MB` / `5` | rotating file log in `/data/logs` |
| `LOG_BUFFER_LINES` | `2000` | lines kept in memory for the in-UI log panel |

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/recordings` | list all recordings + status (`?q=` full-text search over filenames + transcripts) |
| GET | `/api/recordings/{id}` | one recording with transcript segments |
| GET | `/api/recordings/{id}/download` | download original audio |
| GET | `/api/recordings/{id}/audio` | stream audio inline (Range-enabled) for the in-page player |
| POST | `/api/recordings/{id}/clip` | export one .wav of selected ranges; body `{"ranges": [[start,end], …]}` |
| GET | `/api/recordings/{id}/export?fmt=txt\|srt\|vtt` | download transcript as plain text, SRT, or WebVTT |
| POST | `/api/recordings/{id}/retranscribe` | re-queue a file; JSON body `{"model": "...", "engine": "..."}` optionally overrides model/engine for that run |
| DELETE | `/api/recordings/{id}` | **delete the file from disk** and remove the record (guarded to the scan/upload dirs) |
| POST | `/api/upload` | upload an audio file (multipart `file`) |
| GET | `/api/health` | engine load status |
| GET | `/api/gpu` | live GPU telemetry (name, utilization, VRAM, temp) via `nvidia-smi` |
| GET | `/api/models` | available models/engines, configured defaults, and currently loaded models |
| POST | `/api/models/free` | drop all cached models and release VRAM |
| GET | `/api/stats` | backlog tally (total / pending / processing / done / error) |
| GET | `/api/settings` | current scan dir + recursive flag |
| POST | `/api/settings` | update settings, e.g. `{"recursive": true}` |
| GET | `/api/logs?limit=&level=` | recent log lines from the in-memory ring buffer |

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

## Playback & clipping

Expand a finished transcript and you get an inline **audio player**:
- **Click any timestamp** to jump there; the current segment highlights as it plays.
- **Check segments** (or **Select all**) to build a selection — a bar shows the count
  and total duration, with:
  - **▶ Play selection** — auditions just the chosen ranges back-to-back.
  - **Export .wav** — server stitches the selected ranges (ffmpeg `atrim`+`concat`)
    into a single clean `<name>_clip.wav` and downloads it.
  - **Copy** / **.txt** — the selected text.

So the workflow is: listen → check the phrases worth keeping → export an edited-down
.wav of just that audio.

## Recovering weak audio

Goal: pull as many words as possible out of marginal recordings.

- **Confidence flags** — each segment carries Whisper's `avg_logprob` and
  `no_speech_prob`. Shaky segments are highlighted amber in the transcript (hover
  for the numbers), and a note tallies how many are low-confidence. That tells you
  exactly which files are worth a recovery pass.
- **Re-transcribe with recovery options** — the per-file control has two toggles:
  - **Clean audio** — runs the recording through an ffmpeg chain
    (`PREPROCESS_FILTERS`: speech-band bandpass + FFT denoise + dynamic normalize)
    before transcribing. Best lever for weak/noisy SSB.
  - **VAD** — on by default to trim dead air, but it can clip quiet speech; **turn
    it off** to let Whisper attempt the marginal passages.
  Combine with a larger model (e.g. `large-v3`) for the hardest files.
- Set `PREPROCESS=true` in `.env` to clean *everything* by default, or leave it off
  and clean only the files that need it from the UI.

## Logging

Everything logs to **container stdout** (`docker compose logs -f wavereader`) and a
**rotating file** at `/data/logs/wavereader.log` (persisted in the `wavereader-data`
volume). Scans, queueing, model loads, per-file transcription timing
(incl. realtime factor), deletes, and errors with stack traces are all recorded.
Set `LOG_LEVEL=DEBUG` for per-file scan decisions.

The UI has a collapsible **Logs** panel at the bottom (with a level filter) that
tails the last `LOG_BUFFER_LINES` records live — handy for troubleshooting without
shelling into the host.

## Recursive scanning

The **Scan recursively** checkbox (top of the page, default on) controls whether
the scanner walks subdirectories of the watched folder. It's persisted in the DB
and applied on the next scan cycle — no restart needed. The watched path itself is
fixed at container start by `RECORDINGS_DIR` (Docker can only see mounted paths).

## Deleting files (destructive)

The trash-can icon on each row **permanently deletes the file from the host disk**
and removes its record, after a confirm dialog. This is why the recordings mount is
read-**write** in `docker-compose.yml`. Deletion is guarded to paths under the
scan/upload directories. If you'd rather never touch originals, re-add `:ro` to the
mount — delete will then 403 on scanned files (uploads still delete).

## Caveats for radio audio

Whisper is trained on broadcast/clean speech. Weak-signal SSB, heavy QSB/QRM,
and rapid callsign exchanges will produce errors — especially callsigns and
serial numbers. The VAD + initial prompt help, but treat transcripts as a
searchable first pass, not a log-quality record. Digital modes (FT8/PSK/RTTY)
and CW are **not** decoded — this handles voice (SSB/AM/FM) only.
