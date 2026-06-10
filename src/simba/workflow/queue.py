"""Durable task queue — exactly-once enqueue, atomic claim, retry/backoff.

Mirrors the huey SQLite-storage idea (atomic dequeue + retry_delay backoff)
but native to simba's ``simba.db`` peewee layer. All time-dependent ops accept
an injectable ``now`` ISO string (defaulting to the real clock) so tests are
deterministic.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import simba._vendor.peewee as pw
import simba.config
import simba.db
import simba.workflow._time as _time
import simba.workflow.config as _wcfg  # noqa: F401 — registers the section
from simba.workflow.store import WfTask

if TYPE_CHECKING:
    import pathlib


def _config() -> Any:
    return simba.config.load("workflow")


def _backoff_seconds(attempts: int) -> float:
    """Exponential backoff (base ** attempts), capped — read from config."""
    cfg = _config()
    delay = cfg.retry_backoff_base_seconds**attempts
    return min(delay, cfg.retry_backoff_max_seconds)


def _row_to_dict(task: WfTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "queue": task.queue,
        "dedup_key": task.dedup_key,
        "status": task.status,
        "payload": json.loads(task.payload),
        "attempts": task.attempts,
        "max_attempts": task.max_attempts,
        "available_at": task.available_at,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
        "error": task.error,
        "result": json.loads(task.result) if task.result is not None else None,
    }


def enqueue(
    queue: str,
    payload: Any,
    *,
    dedup_key: str | None = None,
    max_attempts: int | None = None,
    delay_seconds: float = 0,
    now: str | None = None,
    cwd: pathlib.Path | None = None,
) -> int:
    """Insert a pending task; idempotent on ``(queue, dedup_key)``.

    With a ``dedup_key``, if a row with that key already exists (in any state)
    its id is returned without inserting. Without a key, always inserts.
    ``available_at`` = ``now`` + ``delay_seconds``. Returns the task id.
    """
    now = _time.resolve(now)
    if max_attempts is None:
        max_attempts = _config().default_max_attempts
    available_at = _time.add_seconds(now, delay_seconds) if delay_seconds else now

    with simba.db.connect(cwd):
        if dedup_key is not None:
            existing = WfTask.get_or_none(
                (WfTask.queue == queue) & (WfTask.dedup_key == dedup_key)
            )
            if existing is not None:
                return existing.id
        try:
            task = WfTask.create(
                queue=queue,
                dedup_key=dedup_key,
                status="pending",
                payload=json.dumps(payload),
                attempts=0,
                max_attempts=max_attempts,
                available_at=available_at,
                created_at=now,
            )
            return task.id
        except pw.IntegrityError:
            # Lost a race on the UNIQUE (queue, dedup_key) — return the winner.
            existing = WfTask.get(
                (WfTask.queue == queue) & (WfTask.dedup_key == dedup_key)
            )
            return existing.id


def claim(
    queue: str,
    *,
    now: str | None = None,
    cwd: pathlib.Path | None = None,
) -> dict[str, Any] | None:
    """Atomically take the oldest available pending task in ``queue``.

    Flips it to ``running`` and stamps ``started_at``. A guarded
    ``UPDATE ... WHERE status='pending'`` makes a concurrent claim of the same
    row impossible (the loser updates 0 rows and retries). Returns the task
    dict, or ``None`` if nothing is available.
    """
    now = _time.resolve(now)
    with simba.db.connect(cwd) as db:
        while True:
            with db.atomic():
                candidate = (
                    WfTask.select()
                    .where(
                        (WfTask.queue == queue)
                        & (WfTask.status == "pending")
                        & (WfTask.available_at <= now)
                    )
                    .order_by(WfTask.available_at, WfTask.id)
                    .first()
                )
                if candidate is None:
                    return None
                updated = (
                    WfTask.update(status="running", started_at=now)
                    .where((WfTask.id == candidate.id) & (WfTask.status == "pending"))
                    .execute()
                )
            if updated:
                return _row_to_dict(WfTask.get_by_id(candidate.id))
            # Someone else claimed it between SELECT and UPDATE — try the next.


def complete(
    task_id: int,
    *,
    result: Any = None,
    now: str | None = None,
    cwd: pathlib.Path | None = None,
) -> None:
    """Mark ``task_id`` done with an optional JSON ``result``."""
    now = _time.resolve(now)
    encoded = json.dumps(result) if result is not None else None
    with simba.db.connect(cwd):
        WfTask.update(status="done", finished_at=now, result=encoded).where(
            WfTask.id == task_id
        ).execute()


def fail(
    task_id: int,
    *,
    error: str,
    now: str | None = None,
    cwd: pathlib.Path | None = None,
) -> str:
    """Record a failure: retry with backoff while attempts remain, else dead.

    ``attempts`` += 1; if ``attempts < max_attempts`` the task returns to
    ``pending`` with ``available_at`` = ``now`` + exponential backoff (capped
    by config); otherwise it becomes ``dead``. Returns the new status.
    """
    now = _time.resolve(now)
    with simba.db.connect(cwd):
        task = WfTask.get_by_id(task_id)
        attempts = task.attempts + 1
        if attempts < task.max_attempts:
            status = "pending"
            available_at = _time.add_seconds(now, _backoff_seconds(attempts))
            WfTask.update(
                status=status,
                attempts=attempts,
                available_at=available_at,
                error=error,
            ).where(WfTask.id == task_id).execute()
        else:
            status = "dead"
            WfTask.update(
                status=status,
                attempts=attempts,
                finished_at=now,
                error=error,
            ).where(WfTask.id == task_id).execute()
        return status


def reclaim_stale(
    queue: str,
    *,
    stale_after_seconds: float | None = None,
    now: str | None = None,
    cwd: pathlib.Path | None = None,
) -> int:
    """Return ``running`` tasks started before the stale cutoff to ``pending``.

    Dead-worker recovery: a detached worker that never closed its task is
    treated as dead once ``started_at`` is older than ``stale_after_seconds``
    (defaults to ``workflow.stale_after_seconds``). Returns the count
    reclaimed.
    """
    now = _time.resolve(now)
    if stale_after_seconds is None:
        stale_after_seconds = _config().stale_after_seconds
    cutoff = _time.add_seconds(now, -stale_after_seconds)
    with simba.db.connect(cwd):
        return (
            WfTask.update(status="pending", started_at=None)
            .where(
                (WfTask.queue == queue)
                & (WfTask.status == "running")
                & (WfTask.started_at < cutoff)
            )
            .execute()
        )
