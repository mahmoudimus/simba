"""SQLite-backed activity log for tracking tool usage per project.

Ported from claude-turbo-search/scripts/track-activity.sh.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import simba.db

if TYPE_CHECKING:
    import pathlib
    import sqlite3

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_activities_ts ON activities(timestamp DESC);
"""

_MAX_ROWS = 200


def _init_schema(conn: sqlite3.Connection) -> None:
    """Initialize the activities table."""
    conn.executescript(_SCHEMA_SQL)


simba.db.register_schema(_init_schema)


def log_activity(cwd: pathlib.Path, tool_name: str, detail: str) -> None:
    """Insert a timestamped activity entry and rotate if needed."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with simba.db.get_db(cwd) as conn:
        conn.execute(
            "INSERT INTO activities (timestamp, tool_name, detail) VALUES (?, ?, ?)",
            (timestamp, tool_name, detail),
        )
        # Rotation: keep only the last _MAX_ROWS rows
        conn.execute(
            "DELETE FROM activities WHERE id NOT IN "
            "(SELECT id FROM activities ORDER BY id DESC LIMIT ?)",
            (_MAX_ROWS,),
        )
        conn.commit()


def read_activity_log(
    cwd: pathlib.Path,
) -> list[tuple[str, str, str]]:
    """Read activity entries as (timestamp, tool_name, detail) tuples.

    Returns entries in chronological order.
    Returns an empty list when the database does not exist.
    """
    conn = simba.db.get_connection(cwd)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT timestamp, tool_name, detail FROM activities ORDER BY id ASC"
        ).fetchall()
        return [(row["timestamp"], row["tool_name"], row["detail"]) for row in rows]
    finally:
        conn.close()


def clear_activity_log(cwd: pathlib.Path) -> None:
    """Delete all activity entries."""
    conn = simba.db.get_connection(cwd)
    if conn is None:
        return
    try:
        conn.execute("DELETE FROM activities")
        conn.commit()
    finally:
        conn.close()
