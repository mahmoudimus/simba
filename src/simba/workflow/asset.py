"""Asset + freshness policy (the dagster model, no daemon/executor).

An *asset* is a named derived artifact (e.g. a materialized view). Its
:class:`FreshnessPolicy` declares staleness on two axes — N new source rows
since the last materialization, and/or T seconds of wall-clock age. ``is_stale``
is the pure decision function the existing scheduler/hooks call to decide
"should I enqueue a refresh?"; ``mark_materialized`` records a fresh
materialization (resetting both axes). No background daemon is introduced.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import simba.db
import simba.workflow._time as _time
from simba.workflow.store import WfAsset

if TYPE_CHECKING:
    import pathlib


@dataclasses.dataclass
class FreshnessPolicy:
    """Declarative staleness thresholds; ``None`` disables that axis."""

    stale_after_events: int | None = None  # N new source rows since materialize
    stale_after_seconds: float | None = None  # wall-clock staleness


def is_stale(
    name: str,
    policy: FreshnessPolicy,
    *,
    current_source_position: int,
    now: str | None = None,
    cwd: pathlib.Path | None = None,
) -> bool:
    """Return whether asset ``name`` is stale under ``policy``.

    Stale if it was never materialized, or (events axis)
    ``current_source_position - last_source_position >= stale_after_events``,
    or (time axis) ``now - last_materialized_at >= stale_after_seconds``.
    Either axis triggering is sufficient.
    """
    now = _time.resolve(now)
    with simba.db.connect(cwd):
        row = WfAsset.get_or_none(WfAsset.name == name)

    if row is None or row.last_materialized_at is None:
        return True

    if policy.stale_after_events is not None:
        new_events = current_source_position - row.last_source_position
        if new_events >= policy.stale_after_events:
            return True

    if policy.stale_after_seconds is not None:
        age = (_time.parse(now) - _time.parse(row.last_materialized_at)).total_seconds()
        if age >= policy.stale_after_seconds:
            return True

    return False


def mark_materialized(
    name: str,
    *,
    source_position: int,
    now: str | None = None,
    cwd: pathlib.Path | None = None,
) -> None:
    """Record a fresh materialization of ``name`` (resets both freshness axes)."""
    now = _time.resolve(now)
    with simba.db.connect(cwd):
        WfAsset.insert(
            name=name,
            last_materialized_at=now,
            last_source_position=source_position,
        ).on_conflict(
            conflict_target=[WfAsset.name],
            update={
                WfAsset.last_materialized_at: now,
                WfAsset.last_source_position: source_position,
            },
        ).execute()
