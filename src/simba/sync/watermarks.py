"""Watermark tracking for sync pipelines.

Tracks the last-processed cursor (rowid or timestamp) per table per
pipeline so that each sync cycle only processes new rows.
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
    """Create the sync_watermarks table (transitional: legacy get_db path)."""
    conn.executescript(_SCHEMA_SQL)


simba.db.register_schema(_init_schema)


class SyncWatermark(simba.db.BaseModel):
    table_name = pw.CharField()
    pipeline = pw.CharField()
    last_cursor = pw.CharField(default="0")
    last_run_at = pw.CharField(null=True)
    rows_processed = pw.IntegerField(default=0)
    errors = pw.IntegerField(default=0)

    class Meta:
        table_name = "sync_watermarks"
        primary_key = pw.CompositeKey("table_name", "pipeline")


def get_watermark(
    table_name: str, pipeline: str, *, cwd: pathlib.Path | None = None
) -> str:
    """Return the last processed cursor for *(table_name, pipeline)*.

    Returns ``"0"`` if no watermark has been recorded yet.
    """
    with simba.db.connect(cwd):
        row = SyncWatermark.get_or_none(
            (SyncWatermark.table_name == table_name)
            & (SyncWatermark.pipeline == pipeline)
        )
        return row.last_cursor if row is not None else "0"


def set_watermark(
    table_name: str,
    pipeline: str,
    last_cursor: str,
    rows_processed: int = 0,
    errors: int = 0,
    *,
    cwd: pathlib.Path | None = None,
) -> None:
    """Upsert the watermark for *(table_name, pipeline)*."""
    now = datetime.datetime.now(tz=datetime.UTC).isoformat()
    with simba.db.connect(cwd):
        SyncWatermark.insert(
            table_name=table_name,
            pipeline=pipeline,
            last_cursor=last_cursor,
            last_run_at=now,
            rows_processed=rows_processed,
            errors=errors,
        ).on_conflict(
            conflict_target=[SyncWatermark.table_name, SyncWatermark.pipeline],
            update={
                SyncWatermark.last_cursor: pw.EXCLUDED.last_cursor,
                SyncWatermark.last_run_at: pw.EXCLUDED.last_run_at,
                SyncWatermark.rows_processed: (
                    SyncWatermark.rows_processed + pw.EXCLUDED.rows_processed
                ),
                SyncWatermark.errors: SyncWatermark.errors + pw.EXCLUDED.errors,
            },
        ).execute()


def get_all_watermarks(*, cwd: pathlib.Path | None = None) -> list[dict]:
    """Return all watermark rows as dicts."""
    with simba.db.connect(cwd):
        rows = SyncWatermark.select().order_by(
            SyncWatermark.table_name, SyncWatermark.pipeline
        )
        return list(rows.dicts())
