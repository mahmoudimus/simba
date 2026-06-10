"""Lease primitive — acquire a ``(queue, key)`` lock at most once, release it.

A lease is *not* a produce/consume queue: it is an acquire-key-at-most-once +
release-by-key lock with optional expiry-based stale-reclaim. It backs the
``rlm_jobs`` / ``episode_jobs`` idempotency ledgers (one transcript digested /
one session consolidated, at most once) on the shared ``WfTask`` table.

Lock state lives in ``WfTask.status``:

* absent row                  -> create ``running`` -> ``True``
* ``done``                    -> ``False`` (durable dedup; never reclaimed)
* ``running``, not stale       -> ``False`` (held by a live worker)
* ``running``, stale           -> reclaim (re-stamp ``started_at``) -> ``True``
  (only when ``stale_after_seconds`` is set and ``started_at`` is older than
  ``now`` - ``stale_after_seconds``)

``stale_after_seconds=None`` means *never reclaim* — a held/dead lock blocks
forever. All time-dependent ops accept an injectable ``now`` ISO string
(defaulting to the real clock) so tests are deterministic.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import simba._vendor.peewee as pw
import simba.db
import simba.workflow._time as _time
from simba.workflow.store import WfTask

if TYPE_CHECKING:
    import pathlib

# A lease never retries; ``max_attempts`` is irrelevant. Store a valid sentinel
# so the shared-table row stays well-formed.
_LEASE_MAX_ATTEMPTS = 1


def acquire(
    queue: str,
    key: str,
    *,
    stale_after_seconds: float | None = None,
    payload: Any = None,
    now: str | None = None,
    cwd: pathlib.Path | None = None,
) -> bool:
    """Try to take the ``(queue, key)`` lock. Return ``True`` iff acquired.

    See the module docstring for the full state table. ``payload`` (JSON) is
    stored on create. ``now`` resolves via :mod:`simba.workflow._time`.
    """
    now = _time.resolve(now)
    encoded_payload = json.dumps(payload) if payload is not None else "{}"

    with simba.db.connect(cwd):
        existing = WfTask.get_or_none(
            (WfTask.queue == queue) & (WfTask.dedup_key == key)
        )
        if existing is not None:
            return _reacquire(existing, stale_after_seconds, now)
        try:
            WfTask.create(
                queue=queue,
                dedup_key=key,
                status="running",
                payload=encoded_payload,
                attempts=0,
                max_attempts=_LEASE_MAX_ATTEMPTS,
                available_at=now,
                created_at=now,
                started_at=now,
            )
            return True
        except pw.IntegrityError:
            # Lost the create race on UNIQUE (queue, dedup_key) — re-read and
            # apply the same lock semantics to the winner's row.
            existing = WfTask.get(
                (WfTask.queue == queue) & (WfTask.dedup_key == key)
            )
            return _reacquire(existing, stale_after_seconds, now)


def _reacquire(row: WfTask, stale_after_seconds: float | None, now: str) -> bool:
    """Decide whether an existing lock row can be (re)acquired at ``now``."""
    if row.status == "done":
        return False  # durable dedup — never reclaimed
    if row.status == "running":
        if stale_after_seconds is None or row.started_at is None:
            return False  # held by a (possibly dead) worker, no expiry
        cutoff = _time.add_seconds(now, -stale_after_seconds)
        if _time.parse(row.started_at) >= _time.parse(cutoff):
            return False  # still fresh — held by a live worker
        # Stale: reclaim in place (keep status='running', re-stamp).
        WfTask.update(started_at=now, attempts=0, finished_at=None).where(
            WfTask.id == row.id
        ).execute()
        return True
    # Any other terminal state (failed/dead/pending) — treat as reclaimable.
    WfTask.update(status="running", started_at=now, attempts=0).where(
        WfTask.id == row.id
    ).execute()
    return True


def release(
    queue: str,
    key: str,
    *,
    result: Any = None,
    now: str | None = None,
    cwd: pathlib.Path | None = None,
) -> None:
    """Release the ``(queue, key)`` lock: ``done`` + ``finished_at`` + result.

    No-op if the row is absent. ``result`` is JSON-encoded when not ``None``.
    """
    now = _time.resolve(now)
    encoded = json.dumps(result) if result is not None else None
    with simba.db.connect(cwd):
        WfTask.update(status="done", finished_at=now, result=encoded).where(
            (WfTask.queue == queue) & (WfTask.dedup_key == key)
        ).execute()
