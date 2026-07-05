"""Append-only per-session usage events (spec 33 v2 rule R2).

The ``memory_usage`` counters roll up totals; events carry WHICH session a
use/noise signal came from — the spec's real promotion trigger is "used in
≥2 DISTINCT sessions", which counters cannot answer. Rows are only ever
appended (the append-only rule applies: this is signal history, not derived
telemetry); the counters remain the fast rollup.
"""

from __future__ import annotations

import simba._vendor.peewee as pw
import simba.db


class UsageEvent(simba.db.BaseModel):
    id = pw.AutoField()
    memory_id = pw.CharField(max_length=64, index=True)
    session_source = pw.CharField(max_length=128)
    kind = pw.CharField(max_length=16)  # "use" | "noise"
    created_at = pw.FloatField(default=0.0)

    class Meta:
        table_name = "usage_events"


simba.db.register_model(UsageEvent)


def record(memory_id: str, session_source: str, kind: str, *, now: float) -> None:
    """Append one event. Must run inside ``simba.db.connect``. No-ops on
    missing attribution — an event without a session answers nothing."""
    if not memory_id or not session_source or kind not in ("use", "noise"):
        return
    UsageEvent.create(
        memory_id=memory_id,
        session_source=session_source,
        kind=kind,
        created_at=now,
    )


def distinct_use_sessions(memory_id: str) -> int:
    """Distinct sessions that produced a ``use`` for ``memory_id``."""
    if not memory_id:
        return 0
    return (
        UsageEvent.select(pw.fn.COUNT(pw.fn.DISTINCT(UsageEvent.session_source)))
        .where((UsageEvent.memory_id == memory_id) & (UsageEvent.kind == "use"))
        .scalar()
        or 0
    )


def use_sessions_for(memory_ids: list[str]) -> dict[str, int]:
    """Bulk distinct-use-session counts; ids with no events are absent."""
    ids = [i for i in memory_ids if i]
    if not ids:
        return {}
    rows = (
        UsageEvent.select(
            UsageEvent.memory_id,
            pw.fn.COUNT(pw.fn.DISTINCT(UsageEvent.session_source)).alias("n"),
        )
        .where((UsageEvent.memory_id.in_(ids)) & (UsageEvent.kind == "use"))
        .group_by(UsageEvent.memory_id)
    )
    return {row.memory_id: int(row.n) for row in rows}
