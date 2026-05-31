"""Central logging config: stdout + rotating file + an in-memory ring buffer
that backs the in-UI log panel."""
import collections
import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler

from . import config

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_ring: "RingBufferHandler | None" = None


class RingBufferHandler(logging.Handler):
    """Keeps the most recent N records in memory for retrieval over the API."""

    def __init__(self, capacity: int):
        super().__init__()
        self._buf: collections.deque = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": record.created,
                "level": record.levelname,
                "name": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                entry["msg"] += "\n" + self.format(record).split("\n", 1)[-1]
        except Exception:
            return
        with self._lock:
            self._buf.append(entry)

    def recent(self, limit: int = 300, level: str | None = None) -> list[dict]:
        with self._lock:
            items = list(self._buf)
        if level:
            threshold = _LEVELS.get(level.upper(), 0)
            items = [e for e in items if _LEVELS.get(e["level"], 0) >= threshold]
        return items[-limit:]


def setup() -> None:
    global _ring
    level = _LEVELS.get(config.LOG_LEVEL, logging.INFO)
    fmt = logging.Formatter(_FMT, _DATEFMT)

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):  # idempotent across reloads
        root.removeHandler(h)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        fileh = RotatingFileHandler(
            os.path.join(config.LOG_DIR, "wavereader.log"),
            maxBytes=config.LOG_MAX_BYTES, backupCount=config.LOG_BACKUPS,
        )
        fileh.setFormatter(fmt)
        root.addHandler(fileh)
    except Exception as e:  # disk/permission issue shouldn't kill the app
        root.warning("file logging disabled: %s", e)

    _ring = RingBufferHandler(config.LOG_BUFFER_LINES)
    _ring.setFormatter(fmt)
    root.addHandler(_ring)

    # Route uvicorn's loggers through our handlers; quiet per-request access noise.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        lg.propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    root.info("logging initialized: level=%s dir=%s", config.LOG_LEVEL, config.LOG_DIR)


def recent(limit: int = 300, level: str | None = None) -> list[dict]:
    return _ring.recent(limit, level) if _ring else []
