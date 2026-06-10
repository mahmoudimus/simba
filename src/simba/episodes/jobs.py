"""episode consolidation lease — at-most-once consolidation (over the lease).

Control-plane only (a per-``(session_source, project_path)`` lock), never memory
content. A thin delegation over :mod:`simba.workflow.lease`: the
``(queue, dedup_key)`` UNIQUE row makes :func:`claim` idempotent so a session is
consolidated at most once. A still-``running`` lock older than
``stale_after_seconds`` is reclaimed (dead-worker recovery); a ``done`` lock is
never reclaimed — the durable dedup there is the stored EPISODE itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import simba.workflow.lease as lease

if TYPE_CHECKING:
    import pathlib

_QUEUE = "episode_consolidate"


def _key(session_source: str, project_path: str) -> str:
    return f"{session_source}\x00{project_path}"


def claim(
    session_source: str,
    project_path: str,
    *,
    cwd: pathlib.Path | None = None,
    stale_after_seconds: int | None = None,
) -> bool:
    """Acquire the consolidation lock. Return True if claimed, else False.

    A still-``running`` lock older than ``stale_after_seconds`` is treated as a
    dead dispatch (the detached agent never closed it) and reclaimed, so a
    session is never permanently locked out of re-consolidation. ``done`` locks
    are never reclaimed — durable dedup there is the stored EPISODE itself.
    """
    return lease.acquire(
        _QUEUE,
        _key(session_source, project_path),
        stale_after_seconds=stale_after_seconds,
        cwd=cwd,
    )


def complete(
    session_source: str,
    project_path: str,
    *,
    cwd: pathlib.Path | None = None,
) -> None:
    """Release the consolidation lock as done."""
    lease.release(_QUEUE, _key(session_source, project_path), cwd=cwd)
