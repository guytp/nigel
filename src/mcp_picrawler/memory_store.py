"""Persistent key-value memory shared across Nigel's brains and sessions.

Both Claude (via Claude Code → MCP) and gpt-realtime (via the voice agent →
MCP) can read/write here. State survives service restarts.

Storage is SQLite at PICRAWLER_MEMORY_DB (default /var/lib/nigel/memory.db),
with three columns plus a full-text index on value for search.

Schema:
    key         TEXT PRIMARY KEY   -- caller-chosen, e.g. "kitchen_location"
    value       TEXT               -- free-form string, JSON-encoded when structured
    tags        TEXT               -- comma-separated, e.g. "location,layout"
    author      TEXT               -- who wrote it (claude/nigel/etc.)
    ts          REAL               -- unix timestamp of last write

Public API (thread-safe):
    set(key, value, tags=None, author=None)
    get(key) -> dict | None
    search(query, limit=20) -> list[dict]          # substring match on value
    by_tag(tag, limit=100) -> list[dict]
    delete(key) -> bool
    list_keys(limit=500) -> list[str]
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.environ.get(
    "PICRAWLER_MEMORY_DB",
    str(Path.home() / ".local" / "share" / "nigel" / "memory.db"),
)


class MemoryStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self._path,
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS memory (
                       key     TEXT PRIMARY KEY,
                       value   TEXT NOT NULL,
                       tags    TEXT NOT NULL DEFAULT '',
                       author  TEXT NOT NULL DEFAULT '',
                       ts      REAL NOT NULL
                   )"""
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS memory_tags_idx ON memory(tags)"
            )

    # --------------------------------------------------------------- writes

    def set(
        self,
        key: str,
        value: Any,
        tags: list[str] | None = None,
        author: str | None = None,
    ) -> dict:
        if not key:
            raise ValueError("key must be non-empty")
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        tags_str = ",".join(t.strip() for t in (tags or []) if t.strip())
        ts = time.time()
        with self._lock:
            self._conn.execute(
                """INSERT INTO memory (key, value, tags, author, ts)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                       value  = excluded.value,
                       tags   = excluded.tags,
                       author = excluded.author,
                       ts     = excluded.ts""",
                (key, value, tags_str, author or "", ts),
            )
        return {"key": key, "value": value, "tags": tags_str, "author": author or "", "ts": ts}

    def delete(self, key: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM memory WHERE key = ?", (key,))
            return cur.rowcount > 0

    # --------------------------------------------------------------- reads

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memory WHERE key = ?", (key,)
            ).fetchone()
        return _row_to_dict(row) if row else None

    def search(self, query: str, limit: int = 20) -> list[dict]:
        q = f"%{query}%"
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM memory
                   WHERE key LIKE ? OR value LIKE ? OR tags LIKE ?
                   ORDER BY ts DESC
                   LIMIT ?""",
                (q, q, q, limit),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def by_tag(self, tag: str, limit: int = 100) -> list[dict]:
        # tags are stored comma-separated; match with word-ish pattern
        q = f"%{tag}%"
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memory WHERE tags LIKE ? ORDER BY ts DESC LIMIT ?",
                (q, limit),
            ).fetchall()
        # post-filter for exact tag match to avoid false positives
        out = []
        for r in rows:
            tags = [t for t in r["tags"].split(",") if t]
            if tag in tags:
                out.append(_row_to_dict(r))
        return out

    def list_keys(self, limit: int = 500) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key FROM memory ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [r["key"] for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "key": row["key"],
        "value": row["value"],
        "tags": [t for t in row["tags"].split(",") if t],
        "author": row["author"],
        "ts": row["ts"],
    }
