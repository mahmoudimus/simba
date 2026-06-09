"""Append-only ``memory_conflicts`` store: precomputed write-time conflicts.

The B2 write-time conflict engine moves conflict DETECTION off the answer-time
path. When a memory is written we compare it against its nearest neighbors (see
:func:`simba.memory.conflict.detect_conflicts_on_write`) and PERSIST any real
contradiction here, keyed by the two LanceDB ``id``\\s. At recall we just READ
the precomputed conflict among the recalled set and annotate — zero answer-time
detection latency.

Rows are append-only — never updated or deleted. A pair is normalized to
``(min(a, b), max(a, b))`` so ``(a, b)`` and ``(b, a)`` are the same conflict,
and recording is idempotent per project (the first write for a normalized pair
wins; later duplicates are skipped). All helpers are synchronous and must run
inside a ``simba.db.connect()`` context (mirrors :mod:`simba.memory.usage`).
``record_conflict`` takes ``now`` explicitly — no helper calls ``time.time()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import simba._vendor.peewee as pw
import simba.db

if TYPE_CHECKING:
    import sqlite3


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create the conflicts table + a project_path index before peewee runs.

    Schema initializers run *before* peewee's ``create_tables`` on first connect,
    so we create the table here too (matching the model schema) to guarantee the
    index exists from the very first connect. Both DDL statements use
    ``IF NOT EXISTS``; peewee's later ``create_tables`` is then a no-op.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memory_conflicts ("
        "id INTEGER NOT NULL PRIMARY KEY, "
        "memory_a VARCHAR(64) NOT NULL, "
        "memory_b VARCHAR(64) NOT NULL, "
        "description TEXT NOT NULL, "
        "detected_at REAL NOT NULL DEFAULT 0.0, "
        "project_path VARCHAR(255) NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_conflicts_project "
        "ON memory_conflicts(project_path)"
    )


simba.db.register_schema(_init_schema)


class MemoryConflict(simba.db.BaseModel):
    memory_a = pw.CharField(max_length=64)
    memory_b = pw.CharField(max_length=64)
    description = pw.TextField()
    detected_at = pw.FloatField(default=0.0)
    project_path = pw.CharField()

    class Meta:
        table_name = "memory_conflicts"


simba.db.register_model(MemoryConflict)


def _normalize_pair(memory_a: str, memory_b: str) -> tuple[str, str]:
    """Order a pair so ``(a, b)`` and ``(b, a)`` collapse to one key."""
    return (memory_a, memory_b) if memory_a <= memory_b else (memory_b, memory_a)


def record_conflict(
    memory_a: str,
    memory_b: str,
    description: str,
    *,
    project_path: str,
    now: float,
) -> None:
    """Append a conflict for the normalized pair; idempotent per project.

    The pair is stored as ``(min, max)`` so ``(a, b)`` and ``(b, a)`` are the
    same conflict. If the same normalized pair is already recorded for this
    project the call is a no-op (first write wins — append-only, never updated).
    """
    lo, hi = _normalize_pair(memory_a, memory_b)
    existing = (
        MemoryConflict.select()
        .where(
            (MemoryConflict.memory_a == lo)
            & (MemoryConflict.memory_b == hi)
            & (MemoryConflict.project_path == project_path)
        )
        .exists()
    )
    if existing:
        return
    MemoryConflict.create(
        memory_a=lo,
        memory_b=hi,
        description=description,
        detected_at=now,
        project_path=project_path,
    )


def conflicts_among(ids: list[str], *, project_path: str) -> list[MemoryConflict]:
    """Recorded conflicts where BOTH endpoints are in ``ids`` for this project.

    Returns the conflicts among the recalled set (both ``memory_a`` and
    ``memory_b`` present in ``ids``). FAIL-OPEN: an empty ``ids`` or any
    exception yields ``[]`` — never raises into the recall path.
    """
    if not ids:
        return []
    try:
        rows = MemoryConflict.select().where(
            (MemoryConflict.project_path == project_path)
            & (MemoryConflict.memory_a.in_(ids))
            & (MemoryConflict.memory_b.in_(ids))
        )
        return list(rows)
    except Exception:
        return []
