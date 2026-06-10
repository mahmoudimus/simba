"""Execution helpers for the durable workflow engine.

Four ways to run work, none of them a daemon:

* :func:`run_sync` — execute a claimed task's handler in-process.
* :func:`dispatch_detached` — fire-and-forget ``subprocess.Popen`` (the
  rlm-engine pattern: ``start_new_session=True`` + DEVNULL streams) so a hook
  never blocks on background work.
* :func:`fan_out` — bounded ``ThreadPoolExecutor`` map for within-stage
  parallelism; per-item exceptions are captured (``None`` for failures) so one
  bad item never sinks the batch.
* :func:`worker_loop` — claim → run → complete / fail(retry) until the queue
  drains or ``max_tasks`` is reached (the huey consumer loop, single-shot).
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import simba.config
import simba.workflow._time as _time
import simba.workflow.config as _wcfg  # noqa: F401 — registers the section
import simba.workflow.queue as queue

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Iterable


def run_sync(handler: Callable[[dict[str, Any]], Any], task: dict[str, Any]) -> Any:
    """Execute ``handler(task)`` in-process and return its result."""
    return handler(task)


def dispatch_detached(
    argv: list[str],
    *,
    cwd: str,
    env: dict[str, str] | None = None,
) -> None:
    """Fire-and-forget ``Popen``: detached session, no inherited streams.

    Never blocks the caller — the spawned process runs independently. Mirrors
    :class:`simba.rlm.engine.ClaudeCliEngine.run`.
    """
    subprocess.Popen(
        argv,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        cwd=cwd,
    )


def fan_out(
    items: Iterable[Any],
    fn: Callable[[Any], Any],
    *,
    max_workers: int | None = None,
) -> list[Any]:
    """Concurrent, order-preserving map; failures become ``None``.

    Bounded by ``max_workers`` (defaults to ``workflow.fan_out_max_workers``).
    A per-item exception is captured and yields ``None`` for that slot so the
    rest of the batch still completes.
    """
    if max_workers is None:
        max_workers = simba.config.load("workflow").fan_out_max_workers
    items = list(items)
    if not items:
        return []

    def _safe(item: Any) -> Any:
        try:
            return fn(item)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_safe, items))


def worker_loop(
    queue_name: str,
    handler: Callable[[dict[str, Any]], Any],
    *,
    max_tasks: int | None = None,
    now: str | None = None,
    cwd: pathlib.Path | None = None,
) -> int:
    """Drain ``queue_name``: claim → run_sync → complete / fail(retry).

    Stops when the queue is empty or ``max_tasks`` tasks have been handled.
    A handler exception routes the task through :func:`queue.fail` (retry with
    backoff, or dead once attempts are exhausted). Returns the count handled.
    """
    now = _time.resolve(now)
    handled = 0
    while max_tasks is None or handled < max_tasks:
        task = queue.claim(queue_name, now=now, cwd=cwd)
        if task is None:
            break
        try:
            result = run_sync(handler, task)
        except Exception as exc:
            queue.fail(task["id"], error=str(exc), now=now, cwd=cwd)
        else:
            queue.complete(task["id"], result=result, now=now, cwd=cwd)
        handled += 1
    return handled
