"""Append-only supersession audit sidecar for memory rows.

LanceDB rows are kept even when a fresher near-duplicate supersedes them. This
SQLite table records the lineage and lets recall demote old rows without losing
auditability.
"""

from __future__ import annotations

import time

import simba._vendor.peewee as pw
import simba.db

STATUS_ACTIVE = "active"
STATUS_PENDING = "pending_confirmation"
STATUS_REJECTED = "rejected"


def _ensure_column(conn, table: str, name: str, spec: str) -> None:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if name not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {spec}")


def _init_schema(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memory_supersessions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "old_id VARCHAR(64) NOT NULL, "
        "new_id VARCHAR(64) NOT NULL, "
        "project_path TEXT NOT NULL DEFAULT '', "
        "memory_type VARCHAR(64) NOT NULL DEFAULT '', "
        "similarity REAL NOT NULL DEFAULT 0.0, "
        "reason TEXT NOT NULL DEFAULT '', "
        "provenance TEXT NOT NULL DEFAULT '', "
        "status VARCHAR(32) NOT NULL DEFAULT 'active', "
        "pending_event_id INTEGER NOT NULL DEFAULT 0, "
        "old_trust_score REAL NOT NULL DEFAULT 0.0, "
        "new_trust_score REAL NOT NULL DEFAULT 0.0, "
        "created_at REAL NOT NULL DEFAULT 0.0, "
        "created_at_iso VARCHAR(32) NOT NULL DEFAULT '')"
    )
    _ensure_column(
        conn,
        "memory_supersessions",
        "status",
        "VARCHAR(32) NOT NULL DEFAULT 'active'",
    )
    _ensure_column(
        conn,
        "memory_supersessions",
        "pending_event_id",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _ensure_column(
        conn,
        "memory_supersessions",
        "old_trust_score",
        "REAL NOT NULL DEFAULT 0.0",
    )
    _ensure_column(
        conn,
        "memory_supersessions",
        "new_trust_score",
        "REAL NOT NULL DEFAULT 0.0",
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_supersessions_old "
        "ON memory_supersessions(old_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_supersessions_new "
        "ON memory_supersessions(new_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_supersessions_pending "
        "ON memory_supersessions(pending_event_id, status)"
    )


simba.db.register_schema(_init_schema)


class MemorySupersession(simba.db.BaseModel):
    id = pw.AutoField()
    old_id = pw.CharField(max_length=64)
    new_id = pw.CharField(max_length=64)
    project_path = pw.TextField(default="")
    memory_type = pw.CharField(max_length=64, default="")
    similarity = pw.FloatField(default=0.0)
    reason = pw.TextField(default="")
    provenance = pw.TextField(default="")
    status = pw.CharField(max_length=32, default=STATUS_ACTIVE)
    pending_event_id = pw.IntegerField(default=0)
    old_trust_score = pw.FloatField(default=0.0)
    new_trust_score = pw.FloatField(default=0.0)
    created_at = pw.FloatField(default=0.0)
    created_at_iso = pw.CharField(max_length=32, default="")

    class Meta:
        table_name = "memory_supersessions"


simba.db.register_model(MemorySupersession)


def append_event(
    *,
    old_id: str,
    new_id: str,
    project_path: str,
    memory_type: str,
    similarity: float,
    reason: str,
    provenance: str,
    status: str = STATUS_ACTIVE,
    pending_event_id: int = 0,
    old_trust_score: float = 0.0,
    new_trust_score: float = 0.0,
    now: float,
) -> MemorySupersession:
    """Append one supersession event and return it."""
    created_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    return MemorySupersession.create(
        old_id=old_id,
        new_id=new_id,
        project_path=project_path,
        memory_type=memory_type,
        similarity=similarity,
        reason=reason,
        provenance=provenance,
        status=status,
        pending_event_id=pending_event_id,
        old_trust_score=old_trust_score,
        new_trust_score=new_trust_score,
        created_at=now,
        created_at_iso=created_at_iso,
    )


def _decided_pending_ids(pending_ids: list[int]) -> set[int]:
    if not pending_ids:
        return set()
    rows = MemorySupersession.select().where(
        MemorySupersession.pending_event_id.in_(pending_ids),
        MemorySupersession.status.in_([STATUS_ACTIVE, STATUS_REJECTED]),
    )
    return {int(row.pending_event_id) for row in rows}


def latest_successors(memory_ids: list[str]) -> dict[str, MemorySupersession]:
    """Return latest active supersession event keyed by old id."""
    if not memory_ids:
        return {}
    rows = (
        MemorySupersession.select()
        .where(
            MemorySupersession.old_id.in_(memory_ids),
            MemorySupersession.status == STATUS_ACTIVE,
        )
        .order_by(MemorySupersession.created_at.asc(), MemorySupersession.id.asc())
    )
    out: dict[str, MemorySupersession] = {}
    for row in rows:
        out[row.old_id] = row
    return out


def latest_pending(memory_ids: list[str]) -> dict[str, MemorySupersession]:
    """Return latest undecided pending replacement keyed by old id."""
    if not memory_ids:
        return {}
    rows = list(
        MemorySupersession.select()
        .where(
            MemorySupersession.old_id.in_(memory_ids),
            MemorySupersession.status == STATUS_PENDING,
        )
        .order_by(MemorySupersession.created_at.asc(), MemorySupersession.id.asc())
    )
    decided = _decided_pending_ids([int(row.id) for row in rows])
    out: dict[str, MemorySupersession] = {}
    for row in rows:
        if int(row.id) not in decided:
            out[row.old_id] = row
    return out


def pending_by_id(event_id: int) -> MemorySupersession:
    row = MemorySupersession.get_or_none(MemorySupersession.id == event_id)
    if row is None:
        raise KeyError(f"unknown supersession event: {event_id}")
    if row.status != STATUS_PENDING:
        raise ValueError(f"supersession event {event_id} is not pending")
    if _decided_pending_ids([event_id]):
        raise ValueError(f"supersession event {event_id} is already decided")
    return row


def confirm(event_id: int, *, now: float) -> MemorySupersession:
    """Append a confirmation decision for a pending supersession."""
    pending = pending_by_id(event_id)
    return append_event(
        old_id=pending.old_id,
        new_id=pending.new_id,
        project_path=pending.project_path,
        memory_type=pending.memory_type,
        similarity=pending.similarity,
        reason="confirmed_pending_supersession",
        provenance=pending.provenance,
        status=STATUS_ACTIVE,
        pending_event_id=int(pending.id),
        old_trust_score=pending.old_trust_score,
        new_trust_score=pending.new_trust_score,
        now=now,
    )


def reject(event_id: int, *, now: float) -> MemorySupersession:
    """Append a rejection decision for a pending supersession."""
    pending = pending_by_id(event_id)
    return append_event(
        old_id=pending.old_id,
        new_id=pending.new_id,
        project_path=pending.project_path,
        memory_type=pending.memory_type,
        similarity=pending.similarity,
        reason="rejected_pending_supersession",
        provenance=pending.provenance,
        status=STATUS_REJECTED,
        pending_event_id=int(pending.id),
        old_trust_score=pending.old_trust_score,
        new_trust_score=pending.new_trust_score,
        now=now,
    )


def events_for(memory_id: str) -> list[MemorySupersession]:
    """Return direct supersession events where ``memory_id`` is the old row."""
    return list(
        MemorySupersession.select()
        .where(MemorySupersession.old_id == memory_id)
        .order_by(MemorySupersession.created_at.asc(), MemorySupersession.id.asc())
    )


def chain(memory_id: str) -> list[MemorySupersession]:
    """Follow active supersession decisions forward from ``memory_id``."""
    out: list[MemorySupersession] = []
    seen = {memory_id}
    current = memory_id
    while True:
        row = (
            MemorySupersession.select()
            .where(
                MemorySupersession.old_id == current,
                MemorySupersession.status == STATUS_ACTIVE,
            )
            .order_by(
                MemorySupersession.created_at.desc(),
                MemorySupersession.id.desc(),
            )
            .first()
        )
        if row is None:
            return out
        out.append(row)
        if row.new_id in seen:
            return out
        seen.add(row.new_id)
        current = row.new_id
