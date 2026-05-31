"""rlm_jobs — coordination table for the autonomous RLM engine.

Control-plane only (dedup + status), never transcript data. The UNIQUE
(transcript_id, project_path) constraint makes claim() idempotent so a
transcript is digested at most once.
"""

from __future__ import annotations

import sqlite3
import time
from typing import TYPE_CHECKING

import simba.db

if TYPE_CHECKING:
    import pathlib


def _init_jobs_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS rlm_jobs (
            transcript_id TEXT,
            project_path  TEXT,
            status        TEXT,
            engine        TEXT,
            started_at    TEXT,
            finished_at   TEXT,
            n_stored      INTEGER DEFAULT 0,
            UNIQUE(transcript_id, project_path)
        )"""
    )


simba.db.register_schema(_init_jobs_schema)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def claim(
    transcript_id: str,
    project_path: str,
    engine: str,
    *,
    cwd: pathlib.Path | None = None,
) -> bool:
    """Insert a 'running' job. Return True if claimed, False if one exists."""
    with simba.db.get_db(cwd) as conn:
        try:
            conn.execute(
                "INSERT INTO rlm_jobs "
                "(transcript_id, project_path, status, engine, started_at) "
                "VALUES (?, ?, 'running', ?, ?)",
                (transcript_id, project_path, engine, _now()),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def complete(
    transcript_id: str,
    project_path: str,
    n_stored: int,
    *,
    cwd: pathlib.Path | None = None,
) -> None:
    with simba.db.get_db(cwd) as conn:
        conn.execute(
            "UPDATE rlm_jobs SET status='done', finished_at=?, n_stored=? "
            "WHERE transcript_id=? AND project_path=?",
            (_now(), n_stored, transcript_id, project_path),
        )
        conn.commit()


