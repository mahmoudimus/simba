"""Watermark tracking for sync pipelines.

Tracks the last-processed cursor (rowid or timestamp) per table per
pipeline so that each sync cycle only processes new rows.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import simba.db

if TYPE_CHECKING:
    import sqlite3

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS sync_watermarks (
    table_name     TEXT NOT NULL,
    pipeline       TEXT NOT NULL,
    last_cursor    TEXT NOT NULL DEFAULT '0',
    last_run_at    TEXT,
    rows_processed INTEGER NOT NULL DEFAULT 0,
    errors         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (table_name, pipeline)
);
"""


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create the sync_watermarks table."""
    conn.executescript(_SCHEMA_SQL)


simba.db.register_schema(_init_schema)


def get_watermark(conn: sqlite3.Connection, table_name: str, pipeline: str) -> str:
    """Return the last processed cursor for *(table_name, pipeline)*.

    Returns ``"0"`` if no watermark has been recorded yet.
    """
    row = conn.execute(
        "SELECT last_cursor FROM sync_watermarks WHERE table_name = ? AND pipeline = ?",
        (table_name, pipeline),
    ).fetchone()
    if row is None:
        return "0"
    return row["last_cursor"]


def set_watermark(
    conn: sqlite3.Connection,
    table_name: str,
    pipeline: str,
    last_cursor: str,
    rows_processed: int = 0,
    errors: int = 0,
) -> None:
    """Upsert the watermark for *(table_name, pipeline)*."""
    now = datetime.datetime.now(tz=datetime.UTC).isoformat()
    conn.execute(
        "INSERT INTO sync_watermarks "
        "(table_name, pipeline, last_cursor, last_run_at, "
        "rows_processed, errors) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(table_name, pipeline) DO UPDATE SET "
        "last_cursor = excluded.last_cursor, "
        "last_run_at = excluded.last_run_at, "
        "rows_processed = rows_processed + excluded.rows_processed, "
        "errors = errors + excluded.errors",
        (table_name, pipeline, last_cursor, now, rows_processed, errors),
    )
    conn.commit()


def get_all_watermarks(conn: sqlite3.Connection) -> list[dict]:
    """Return all watermark rows as dicts."""
    rows = conn.execute(
        "SELECT table_name, pipeline, last_cursor, last_run_at, "
        "rows_processed, errors FROM sync_watermarks "
        "ORDER BY table_name, pipeline"
    ).fetchall()
    return [dict(row) for row in rows]
