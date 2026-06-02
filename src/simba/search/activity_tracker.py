"""SQLite-backed activity log for tracking tool usage per project.

Ported from claude-turbo-search/scripts/track-activity.sh.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import simba._vendor.peewee as pw
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
    """Initialize the activities table (transitional: legacy get_db path)."""
    conn.executescript(_SCHEMA_SQL)


# Transitional: both registrations create the same table IF NOT EXISTS so the
# legacy get_db path and the peewee path interoperate during the migration.
simba.db.register_schema(_init_schema)


class Activity(simba.db.BaseModel):
    timestamp = pw.CharField()
    tool_name = pw.CharField()
    detail = pw.CharField(default="")

    class Meta:
        table_name = "activities"
        indexes = ((("timestamp",), False),)


simba.db.register_model(Activity)


def log_activity(cwd: pathlib.Path, tool_name: str, detail: str) -> None:
    """Insert a timestamped activity entry and rotate if needed."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with simba.db.connect(cwd):
        Activity.create(timestamp=timestamp, tool_name=tool_name, detail=detail)
        # Rotation: keep only the last _MAX_ROWS rows.
        keep = (
            Activity.select(Activity.id).order_by(Activity.id.desc()).limit(_MAX_ROWS)
        )
        Activity.delete().where(Activity.id.not_in(keep)).execute()


def read_activity_log(
    cwd: pathlib.Path,
) -> list[tuple[str, str, str]]:
    """Read activity entries as (timestamp, tool_name, detail) tuples.

    Returns entries in chronological order.
    Returns an empty list when the database does not exist.
    """
    if not simba.db.get_db_path(cwd).exists():
        return []
    with simba.db.connect(cwd):
        return [
            (a.timestamp, a.tool_name, a.detail)
            for a in Activity.select().order_by(Activity.id)
        ]


def clear_activity_log(cwd: pathlib.Path) -> None:
    """Delete all activity entries."""
    if not simba.db.get_db_path(cwd).exists():
        return
    with simba.db.connect(cwd):
        Activity.delete().execute()
