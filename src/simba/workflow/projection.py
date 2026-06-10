"""Resumable, exactly-once projections (the eventsourcing notification-tracking
idea, native to simba).

A :class:`Projection` reads an ordered event stream and advances a durable
checkpoint cursor in ``wf_checkpoints``. It processes only events whose
position is strictly past the checkpoint, so re-running is a safe resume; the
checkpoint advances per event (inside the same transaction as the caller's
derived write would conceptually sit), giving exactly-once advancement.
``rebuild`` resets the cursor to zero and replays — derived state is therefore
swappable / reconstructable from the source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import simba.db
import simba.workflow._time as _time
from simba.workflow.store import WfCheckpoint

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Iterable


class Projection:
    """A named, resumable projection over a positioned event stream."""

    def __init__(self, name: str, process_fn: Callable[[Any], None]) -> None:
        self.name = name
        self._process_fn = process_fn

    def _position(self, db: Any) -> int:
        row = WfCheckpoint.get_or_none(WfCheckpoint.name == self.name)
        return row.position if row is not None else 0

    def _advance(self, db: Any, position: int, now: str) -> None:
        WfCheckpoint.insert(
            name=self.name, position=position, updated_at=now
        ).on_conflict(
            conflict_target=[WfCheckpoint.name],
            update={WfCheckpoint.position: position, WfCheckpoint.updated_at: now},
        ).execute()

    def run(
        self,
        events: Iterable[tuple[int, Any]],
        *,
        now: str | None = None,
        cwd: pathlib.Path | None = None,
    ) -> int:
        """Process events with ``position`` past the checkpoint; advance it.

        ``events`` is an iterable of ``(position, event)`` sorted ascending.
        Each new event is processed via ``process_fn`` and the checkpoint is
        advanced to its position inside one transaction (so a crash leaves the
        cursor consistent with what was processed). Returns the count
        processed.
        """
        now = _time.resolve(now)
        processed = 0
        with simba.db.connect(cwd) as db:
            cursor = self._position(db)
            for position, event in events:
                if position <= cursor:
                    continue
                with db.atomic():
                    self._process_fn(event)
                    self._advance(db, position, now)
                cursor = position
                processed += 1
        return processed

    def rebuild(
        self,
        all_events: Iterable[tuple[int, Any]],
        *,
        reset_fn: Callable[[], None] | None = None,
        now: str | None = None,
        cwd: pathlib.Path | None = None,
    ) -> int:
        """Reset the checkpoint to 0, clear derived state, then replay all.

        ``reset_fn`` (the caller's "clear my derived table") runs once before
        replay. Rebuildability is the swappability guarantee — derived state is
        reconstructable from the source events alone.
        """
        now = _time.resolve(now)
        with simba.db.connect(cwd) as db, db.atomic():
            self._advance(db, 0, now)
            if reset_fn is not None:
                reset_fn()
        return self.run(all_events, now=now, cwd=cwd)
