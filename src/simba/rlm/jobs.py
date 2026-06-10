"""rlm digest lease — at-most-once transcript digestion (over the workflow lease).

Control-plane only (a per-``(transcript_id, project_path)`` lock), never
transcript data. A thin delegation over :mod:`simba.workflow.lease`: the
``(queue, dedup_key)`` UNIQUE row makes :func:`claim` idempotent so a transcript
is digested at most once. Stale-reclaim of a dead digest worker is gated on
``rlm.digest_stale_after_seconds`` (0 => no reclaim => original behavior).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import simba.config
import simba.rlm.config  # registers the "rlm" section
import simba.workflow.lease as lease

if TYPE_CHECKING:
    import pathlib

_QUEUE = "rlm_digest"


def _key(transcript_id: str, project_path: str) -> str:
    return f"{transcript_id}\x00{project_path}"


def _rlm_cfg(cwd: pathlib.Path | None = None):
    return simba.config.load("rlm")


def claim(
    transcript_id: str,
    project_path: str,
    engine: str,
    *,
    cwd: pathlib.Path | None = None,
) -> bool:
    """Acquire the digest lock. Return True if claimed, False if one exists."""
    cfg = _rlm_cfg(cwd)
    return lease.acquire(
        _QUEUE,
        _key(transcript_id, project_path),
        stale_after_seconds=cfg.digest_stale_after_seconds or None,
        payload={"engine": engine},
        cwd=cwd,
    )


def complete(
    transcript_id: str,
    project_path: str,
    n_stored: int,
    *,
    cwd: pathlib.Path | None = None,
) -> None:
    """Release the digest lock as done, recording the stored-memory count."""
    lease.release(
        _QUEUE,
        _key(transcript_id, project_path),
        result={"n_stored": n_stored},
        cwd=cwd,
    )
