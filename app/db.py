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
    with _lock:
        rows = _conn.execute(
            f"SELECT {_LIST_COLS} FROM recordings{clause} ORDER BY created_at DESC",
            tuple(params),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["alerts"] = json.loads(d["alerts"]) if d.get("alerts") else []
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


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    with _lock:
        row = _conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    _exec("INSERT INTO settings (key, value) VALUES (?, ?) "
          "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
