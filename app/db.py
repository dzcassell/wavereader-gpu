"""Thin SQLite layer. One connection guarded by a lock; the workload is light."""
import json
import sqlite3
import threading
import time
from typing import Any, Optional

from . import config

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT NOT NULL,
    source        TEXT NOT NULL,            -- 'scan' or 'upload'
    path          TEXT NOT NULL,            -- absolute path inside container
    size          INTEGER,
    mtime         REAL,
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending|processing|done|error
    duration      REAL,
    language      TEXT,
    model         TEXT,
    engine        TEXT,
    text          TEXT,                     -- full plain transcript
    segments_json TEXT,                     -- [{start,end,text}, ...]
    error         TEXT,
    req_model     TEXT,                     -- per-job model override (NULL = config default)
    req_engine    TEXT,                     -- per-job engine override (NULL = config default)
    created_at    REAL NOT NULL,
    completed_at  REAL,
    UNIQUE(path, size, mtime)
);
CREATE INDEX IF NOT EXISTS idx_status ON recordings(status);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS speakers (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS voiceprints (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id   INTEGER NOT NULL,
    recording_id INTEGER NOT NULL,
    seg_index    INTEGER NOT NULL,
    start        REAL,
    end          REAL,
    dim          INTEGER NOT NULL,
    emb          BLOB NOT NULL,
    created_at   REAL NOT NULL,
    UNIQUE(recording_id, seg_index)
);

CREATE TABLE IF NOT EXISTS segment_tags (
    recording_id INTEGER NOT NULL,
    seg_index    INTEGER NOT NULL,
    speaker_id   INTEGER NOT NULL,
    source       TEXT NOT NULL DEFAULT 'manual',  -- manual | auto
    confidence   REAL,
    PRIMARY KEY (recording_id, seg_index)
);

CREATE TABLE IF NOT EXISTS speaker_profiles (
    speaker_id INTEGER PRIMARY KEY,
    dim        INTEGER NOT NULL,
    emb        BLOB NOT NULL,
    n          INTEGER NOT NULL,
    updated_at REAL NOT NULL
);
"""

# Columns added after initial release; ALTER on existing databases.
_MIGRATIONS = [
    ("req_model", "ALTER TABLE recordings ADD COLUMN req_model TEXT"),
    ("req_engine", "ALTER TABLE recordings ADD COLUMN req_engine TEXT"),
    ("req_preprocess", "ALTER TABLE recordings ADD COLUMN req_preprocess INTEGER"),
    ("req_vad", "ALTER TABLE recordings ADD COLUMN req_vad INTEGER"),
    ("entities_json", "ALTER TABLE recordings ADD COLUMN entities_json TEXT"),
    ("alerts", "ALTER TABLE recordings ADD COLUMN alerts TEXT"),
]


def init() -> None:
    global _conn
    _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    with _lock:
        _conn.executescript(SCHEMA)
        existing = {r[1] for r in _conn.execute("PRAGMA table_info(recordings)").fetchall()}
        for col, ddl in _MIGRATIONS:
            if col not in existing:
                _conn.execute(ddl)
        _conn.commit()


def _exec(sql: str, params: tuple = ()) -> sqlite3.Cursor:
    with _lock:
        cur = _conn.execute(sql, params)
        _conn.commit()
        return cur


def add_recording(filename: str, source: str, path: str, size: int, mtime: float) -> Optional[int]:
    """Insert a new pending recording. Returns id, or None if it already exists."""
    try:
        cur = _exec(
            "INSERT INTO recordings (filename, source, path, size, mtime, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            (filename, source, path, size, mtime, time.time()),
        )
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None  # duplicate (same path/size/mtime)


def next_pending() -> Optional[sqlite3.Row]:
    with _lock:
        row = _conn.execute(
            "SELECT * FROM recordings WHERE status='pending' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
    return row


def set_status(rec_id: int, status: str, **fields: Any) -> None:
    cols = ["status=?"]
    vals: list = [status]
    for k, v in fields.items():
        cols.append(f"{k}=?")
        vals.append(v)
    vals.append(rec_id)
    _exec(f"UPDATE recordings SET {', '.join(cols)} WHERE id=?", tuple(vals))


def save_transcript(rec_id: int, text: str, segments: list, duration: float,
                    language: str, model: str, engine: str,
                    entities: Optional[list] = None, alerts: Optional[list] = None) -> None:
    set_status(
        rec_id, "done",
        text=text,
        segments_json=json.dumps(segments),
        entities_json=json.dumps(entities or []),
        alerts=json.dumps(alerts or []),
        duration=duration,
        language=language,
        model=model,
        engine=engine,
        completed_at=time.time(),
        error=None,
    )


def set_entities(rec_id: int, entities: list, alerts: list) -> None:
    """Backfill entities/alerts on an already-transcribed recording."""
    _exec("UPDATE recordings SET entities_json=?, alerts=? WHERE id=?",
          (json.dumps(entities), json.dumps(alerts), rec_id))


def iter_done_segments():
    """Yield (id, filename, segments, text) for every transcribed recording. For backfill."""
    with _lock:
        rows = _conn.execute(
            "SELECT id, filename, segments_json, text FROM recordings WHERE status='done'"
        ).fetchall()
    for r in rows:
        yield r["id"], r["filename"], json.loads(r["segments_json"] or "[]"), (r["text"] or "")


def iter_entities():
    """Yield (id, filename, entities) for recordings that have entities."""
    with _lock:
        rows = _conn.execute(
            "SELECT id, filename, entities_json FROM recordings "
            "WHERE entities_json IS NOT NULL AND entities_json != '[]'"
        ).fetchall()
    for r in rows:
        yield r["id"], r["filename"], json.loads(r["entities_json"] or "[]")


def search_segments(query: str, limit: int = 200) -> list[dict]:
    """Segment-level hits across all transcripts: pre-filter files by text, then
    return the individual matching segments with timestamps."""
    like = f"%{query}%"
    with _lock:
        rows = _conn.execute(
            "SELECT id, filename, segments_json FROM recordings "
            "WHERE status='done' AND text LIKE ? ORDER BY created_at DESC", (like,)
        ).fetchall()
    q = query.lower()
    out = []
    for r in rows:
        for seg in json.loads(r["segments_json"] or "[]"):
            if q in (seg.get("text", "") or "").lower():
                out.append({"id": r["id"], "filename": r["filename"],
                            "start": seg.get("start", 0), "text": seg.get("text", "")})
                if len(out) >= limit:
                    return out
    return out


_LIST_COLS = ("id, filename, source, size, status, duration, language, model, "
              "created_at, completed_at, error, alerts")


def list_recordings(query: Optional[str] = None, alerts_only: bool = False) -> list[dict]:
    where, params = [], []
    if query:
        like = f"%{query}%"
        where.append("(filename LIKE ? OR text LIKE ?)")
        params += [like, like]
    if alerts_only:
        where.append("alerts IS NOT NULL AND alerts != '[]' AND alerts != ''")
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    tag_sub = ("(SELECT GROUP_CONCAT(DISTINCT s.name) FROM segment_tags t "
               "JOIN speakers s ON s.id=t.speaker_id WHERE t.recording_id=recordings.id) AS tag_names")
    with _lock:
        rows = _conn.execute(
            f"SELECT {_LIST_COLS}, {tag_sub} FROM recordings{clause} ORDER BY created_at DESC",
            tuple(params),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["alerts"] = json.loads(d["alerts"]) if d.get("alerts") else []
        d["tags"] = sorted((d.pop("tag_names") or "").split(",")) if d.get("tag_names") else []
        out.append(d)
    return out


def get_recording(rec_id: int) -> Optional[dict]:
    with _lock:
        row = _conn.execute("SELECT * FROM recordings WHERE id=?", (rec_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["segments"] = json.loads(d.pop("segments_json") or "[]")
    d["entities"] = json.loads(d.pop("entities_json", None) or "[]")
    d["alerts"] = json.loads(d["alerts"]) if d.get("alerts") else []
    tags = get_segment_tags(rec_id)
    for i, seg in enumerate(d["segments"]):
        t = tags.get(i)
        if t:
            seg["speaker_id"] = t["speaker_id"]
            seg["speaker"] = t["speaker"]
            seg["speaker_source"] = t["source"]
            seg["speaker_conf"] = t["confidence"]
    return d


def counts() -> dict:
    """Status tally across all recordings, for the backlog progress indicator."""
    with _lock:
        rows = _conn.execute(
            "SELECT status, COUNT(*) AS c FROM recordings GROUP BY status"
        ).fetchall()
    by = {r["status"]: r["c"] for r in rows}
    return {
        "total": sum(by.values()),
        "pending": by.get("pending", 0),
        "processing": by.get("processing", 0),
        "done": by.get("done", 0),
        "error": by.get("error", 0),
    }


def _b(v: Optional[bool]) -> Optional[int]:
    return None if v is None else (1 if v else 0)


def requeue(rec_id: int, model: Optional[str] = None, engine: Optional[str] = None,
            preprocess: Optional[bool] = None, vad: Optional[bool] = None) -> None:
    """Re-queue a recording. model/engine/preprocess/vad override config defaults
    for this job (None = use default)."""
    set_status(rec_id, "pending", error=None, completed_at=None,
               req_model=model, req_engine=engine,
               req_preprocess=_b(preprocess), req_vad=_b(vad))


def delete_recording(rec_id: int) -> None:
    _exec("DELETE FROM recordings WHERE id=?", (rec_id,))


def requeue_stuck() -> int:
    """Return rows orphaned in 'processing' (e.g. container killed mid-transcription)
    back to 'pending'. Safe to call at startup before the worker runs. Returns count."""
    cur = _exec("UPDATE recordings SET status='pending', error=NULL WHERE status='processing'")
    return cur.rowcount


# --- Speakers / voiceprints / tags ---

def create_speaker(name: str) -> int:
    name = name.strip()
    with _lock:
        row = _conn.execute("SELECT id FROM speakers WHERE name=?", (name,)).fetchone()
        if row:
            return row["id"]
        cur = _conn.execute("INSERT INTO speakers (name, created_at) VALUES (?, ?)",
                            (name, time.time()))
        _conn.commit()
        return cur.lastrowid


def list_speakers() -> list[dict]:
    with _lock:
        rows = _conn.execute(
            "SELECT s.id, s.name, "
            "(SELECT COUNT(*) FROM voiceprints v WHERE v.speaker_id=s.id) AS prints, "
            "(SELECT COUNT(*) FROM segment_tags t WHERE t.speaker_id=s.id) AS tags, "
            "(SELECT n FROM speaker_profiles p WHERE p.speaker_id=s.id) AS profile_n "
            "FROM speakers s ORDER BY s.name"
        ).fetchall()
    return [dict(r) for r in rows]


def voiceprint_blobs(speaker_id: int) -> list[bytes]:
    with _lock:
        rows = _conn.execute(
            "SELECT emb FROM voiceprints WHERE speaker_id=?", (speaker_id,)).fetchall()
    return [r["emb"] for r in rows]


def get_speaker_prints() -> list[dict]:
    """[{id, name, blobs:[bytes]}] grouped by speaker — for kNN identification."""
    with _lock:
        rows = _conn.execute(
            "SELECT v.speaker_id AS id, s.name, v.emb "
            "FROM voiceprints v JOIN speakers s ON s.id=v.speaker_id"
        ).fetchall()
    by: dict = {}
    for r in rows:
        d = by.setdefault(r["id"], {"id": r["id"], "name": r["name"], "blobs": []})
        d["blobs"].append(r["emb"])
    return list(by.values())


def iter_voiceprints():
    """Yield (vp_id, recording_id, seg_index, start, end, path) for re-embedding."""
    with _lock:
        rows = _conn.execute(
            "SELECT v.id, v.recording_id, v.seg_index, v.start, v.end, r.path "
            "FROM voiceprints v JOIN recordings r ON r.id=v.recording_id"
        ).fetchall()
    for r in rows:
        yield r["id"], r["recording_id"], r["seg_index"], r["start"], r["end"], r["path"]


def update_voiceprint_emb(vp_id: int, emb: bytes, dim: int) -> None:
    _exec("UPDATE voiceprints SET emb=?, dim=? WHERE id=?", (emb, dim, vp_id))


def manual_tag_recordings() -> list[int]:
    """Distinct recording ids that have at least one manual tag."""
    with _lock:
        rows = _conn.execute(
            "SELECT DISTINCT recording_id FROM segment_tags WHERE source='manual'").fetchall()
    return [r["recording_id"] for r in rows]


def set_profile(speaker_id: int, emb: bytes, dim: int, n: int) -> None:
    _exec("INSERT INTO speaker_profiles (speaker_id, dim, emb, n, updated_at) "
          "VALUES (?, ?, ?, ?, ?) "
          "ON CONFLICT(speaker_id) DO UPDATE SET dim=excluded.dim, emb=excluded.emb, "
          "n=excluded.n, updated_at=excluded.updated_at",
          (speaker_id, dim, emb, n, time.time()))


def get_profiles() -> list[dict]:
    """Return [{id, name, emb(bytes), dim}] for speakers that have a built profile."""
    with _lock:
        rows = _conn.execute(
            "SELECT p.speaker_id AS id, s.name, p.emb, p.dim "
            "FROM speaker_profiles p JOIN speakers s ON s.id=p.speaker_id"
        ).fetchall()
    return [{"id": r["id"], "name": r["name"], "emb": r["emb"], "dim": r["dim"]} for r in rows]


def speaker_name(speaker_id: int) -> Optional[str]:
    with _lock:
        row = _conn.execute("SELECT name FROM speakers WHERE id=?", (speaker_id,)).fetchone()
    return row["name"] if row else None


def delete_speaker(speaker_id: int) -> None:
    _exec("DELETE FROM voiceprints WHERE speaker_id=?", (speaker_id,))
    _exec("DELETE FROM segment_tags WHERE speaker_id=?", (speaker_id,))
    _exec("DELETE FROM speaker_profiles WHERE speaker_id=?", (speaker_id,))
    _exec("DELETE FROM speakers WHERE id=?", (speaker_id,))


def add_voiceprint(speaker_id: int, rec_id: int, seg: int, start: float, end: float,
                   emb: bytes, dim: int) -> None:
    _exec("INSERT INTO voiceprints (speaker_id, recording_id, seg_index, start, end, dim, emb, created_at) "
          "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
          "ON CONFLICT(recording_id, seg_index) DO UPDATE SET "
          "speaker_id=excluded.speaker_id, start=excluded.start, end=excluded.end, "
          "dim=excluded.dim, emb=excluded.emb, created_at=excluded.created_at",
          (speaker_id, rec_id, seg, start, end, dim, emb, time.time()))


def remove_voiceprint(rec_id: int, seg: int) -> None:
    _exec("DELETE FROM voiceprints WHERE recording_id=? AND seg_index=?", (rec_id, seg))


def set_segment_tag(rec_id: int, seg: int, speaker_id: int,
                    source: str = "manual", confidence: Optional[float] = None) -> None:
    _exec("INSERT INTO segment_tags (recording_id, seg_index, speaker_id, source, confidence) "
          "VALUES (?, ?, ?, ?, ?) "
          "ON CONFLICT(recording_id, seg_index) DO UPDATE SET "
          "speaker_id=excluded.speaker_id, source=excluded.source, confidence=excluded.confidence",
          (rec_id, seg, speaker_id, source, confidence))


def remove_segment_tag(rec_id: int, seg: int) -> None:
    _exec("DELETE FROM segment_tags WHERE recording_id=? AND seg_index=?", (rec_id, seg))


def get_segment_tags(rec_id: int) -> dict:
    with _lock:
        rows = _conn.execute(
            "SELECT t.seg_index, t.speaker_id, t.source, t.confidence, s.name "
            "FROM segment_tags t JOIN speakers s ON s.id=t.speaker_id WHERE t.recording_id=?",
            (rec_id,)
        ).fetchall()
    return {r["seg_index"]: {"speaker_id": r["speaker_id"], "speaker": r["name"],
                             "source": r["source"], "confidence": r["confidence"]} for r in rows}


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    with _lock:
        row = _conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    _exec("INSERT INTO settings (key, value) VALUES (?, ?) "
          "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
