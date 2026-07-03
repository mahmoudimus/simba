"""Mutable usage sidecar for every LanceDB memory.

The SQLite ``memory_usage`` table is the single source of truth for the mutable
ranking signals — ``access_count``, ``last_accessed``, ``strength``, ``dormant``
and ``feedback_score`` — keyed by the LanceDB ``id``.  LanceDB's vector fragments
are write-once, so all decay/feedback state lives here instead.

Rows are append-on-first-touch (created by ``get_or_create`` on the first recall
or feedback call) and mutated in place thereafter; no row is ever deleted.  All
helpers are synchronous and must run inside a ``simba.db.connect()`` context.
Every time-sensitive helper takes ``now`` explicitly — none call ``time.time()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import simba._vendor.peewee as pw
import simba.db

if TYPE_CHECKING:
    import sqlite3


def _init_index(conn: sqlite3.Connection) -> None:
    """Add the dormancy/strength index used by the decay pass.

    Schema initializers run *before* peewee's ``create_tables`` on first connect,
    so we create the table here too (matching the model schema) to guarantee the
    composite index exists from the very first connect.  Both DDL statements use
    ``IF NOT EXISTS``; peewee's later ``create_tables`` is then a no-op.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memory_usage ("
        "memory_id VARCHAR(64) NOT NULL PRIMARY KEY, "
        "access_count INTEGER NOT NULL DEFAULT 0, "
        "last_accessed REAL NOT NULL DEFAULT 0.0, "
        "strength REAL NOT NULL DEFAULT 1.0, "
        "dormant INTEGER NOT NULL DEFAULT 0, "
        "feedback_score REAL NOT NULL DEFAULT 0.0, "
        "created_at REAL NOT NULL DEFAULT 0.0, "
        "match_count INTEGER NOT NULL DEFAULT 0, "
        "inject_count INTEGER NOT NULL DEFAULT 0, "
        "use_count INTEGER NOT NULL DEFAULT 0, "
        "noise_count INTEGER NOT NULL DEFAULT 0, "
        "save_count INTEGER NOT NULL DEFAULT 0, "
        "last_used REAL NOT NULL DEFAULT 0.0)"
    )
    existing = {row[1] for row in conn.execute("PRAGMA table_info(memory_usage)")}
    for name, ddl in (
        ("match_count", "INTEGER NOT NULL DEFAULT 0"),
        ("inject_count", "INTEGER NOT NULL DEFAULT 0"),
        ("use_count", "INTEGER NOT NULL DEFAULT 0"),
        ("noise_count", "INTEGER NOT NULL DEFAULT 0"),
        ("save_count", "INTEGER NOT NULL DEFAULT 0"),
        ("last_used", "REAL NOT NULL DEFAULT 0.0"),
    ):
        if name not in existing:
            conn.execute(f"ALTER TABLE memory_usage ADD COLUMN {name} {ddl}")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_usage_dormant "
        "ON memory_usage(dormant, strength)"
    )


simba.db.register_schema(_init_index)


class MemoryUsage(simba.db.BaseModel):
    memory_id = pw.CharField(max_length=64, primary_key=True)
    access_count = pw.IntegerField(default=0)
    last_accessed = pw.FloatField(default=0.0)
    strength = pw.FloatField(default=1.0)
    dormant = pw.BooleanField(default=False)
    feedback_score = pw.FloatField(default=0.0)
    created_at = pw.FloatField(default=0.0)
    match_count = pw.IntegerField(default=0)
    inject_count = pw.IntegerField(default=0)
    use_count = pw.IntegerField(default=0)
    noise_count = pw.IntegerField(default=0)
    save_count = pw.IntegerField(default=0)
    # Epoch of the most recent USE (gate fire / citation / explicit good
    # feedback) — the consumption-freshness the rule TTL keys off (spec 33).
    # Distinct from last_accessed, which mere retrieval bumps.
    last_used = pw.FloatField(default=0.0)

    class Meta:
        table_name = "memory_usage"


simba.db.register_model(MemoryUsage)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def get_or_create(memory_id: str, now: float) -> MemoryUsage:
    """Return the existing row, or INSERT a default row with ``created_at=now``."""
    row, _ = MemoryUsage.get_or_create(
        memory_id=memory_id, defaults={"created_at": now}
    )
    return row


def bump_access(memory_id: str, now: float) -> None:
    """Increment ``access_count`` and set ``last_accessed=now``. Upserts if missing."""
    get_or_create(memory_id, now)
    MemoryUsage.update(
        access_count=MemoryUsage.access_count + 1,
        last_accessed=now,
    ).where(MemoryUsage.memory_id == memory_id).execute()


def bump_quality(
    memory_id: str,
    now: float,
    *,
    match: int = 0,
    inject: int = 0,
    use: int = 0,
    noise: int = 0,
    save: int = 0,
) -> None:
    """Increment explicit quality counters. Upserts when missing.

    A ``use`` additionally stamps ``last_used=now`` — freshness follows
    consumption (spec 33), so a rule that keeps firing stays alive.
    """
    get_or_create(memory_id, now)
    updates: dict = {
        "match_count": MemoryUsage.match_count + match,
        "inject_count": MemoryUsage.inject_count + inject,
        "use_count": MemoryUsage.use_count + use,
        "noise_count": MemoryUsage.noise_count + noise,
        "save_count": MemoryUsage.save_count + save,
    }
    if use > 0:
        updates["last_used"] = now
    MemoryUsage.update(**updates).where(MemoryUsage.memory_id == memory_id).execute()


def set_dormant(memory_id: str, *, dormant: bool) -> None:
    """Set the ``dormant`` flag. No-op when the row is missing (silent)."""
    MemoryUsage.update(dormant=dormant).where(
        MemoryUsage.memory_id == memory_id
    ).execute()


def apply_feedback(memory_id: str, delta: float, now: float) -> None:
    """Add ``delta`` to ``feedback_score``, clamped to ``[-1.0, 1.0]``. Upserts."""
    row = get_or_create(memory_id, now)
    clamped = _clamp(row.feedback_score + delta, -1.0, 1.0)
    MemoryUsage.update(feedback_score=clamped).where(
        MemoryUsage.memory_id == memory_id
    ).execute()


def set_strength(memory_id: str, strength: float) -> None:
    """Overwrite ``strength``, clamped to ``[0.0, 1.0]``. No-op when missing."""
    clamped = _clamp(strength, 0.0, 1.0)
    MemoryUsage.update(strength=clamped).where(
        MemoryUsage.memory_id == memory_id
    ).execute()


def get_many(memory_ids: list[str]) -> dict[str, MemoryUsage]:
    """Bulk-fetch rows keyed by ``memory_id``. Missing ids are absent from the dict."""
    if not memory_ids:
        return {}
    rows = MemoryUsage.select().where(MemoryUsage.memory_id.in_(memory_ids))
    return {row.memory_id: row for row in rows}


def get_all_for_decay(*, include_dormant: bool = False) -> list[MemoryUsage]:
    """Return all rows, optionally including already-dormant rows."""
    query = MemoryUsage.select()
    if not include_dormant:
        query = query.where(MemoryUsage.dormant == False)  # noqa: E712
    return list(query)
