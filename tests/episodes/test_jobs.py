"""Tests for the episode consolidation lease (episodes/jobs.py).

The bespoke ``EpisodeJob`` model is gone — these assert *public behavior* over
the ``simba.workflow.lease`` primitive: claim-once / dedup, complete -> done,
and the existing expiry-based stale-reclaim (``stale_after_seconds``). The
durable dedup for a *completed* consolidation is the stored EPISODE itself.
"""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.episodes.jobs as jobs
import simba.workflow.lease as lease
import simba.workflow.store as store

T_OLD = "2000-01-01T00:00:00Z"


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def _lease_row(session_source: str, project_path: str) -> store.WfTask | None:
    key = jobs._key(session_source, project_path)
    with simba.db.connect():
        return store.WfTask.get_or_none(
            (store.WfTask.queue == jobs._QUEUE) & (store.WfTask.dedup_key == key)
        )


def test_claim_is_idempotent() -> None:
    assert jobs.claim("s1", "/proj") is True
    assert jobs.claim("s1", "/proj") is False  # already claimed


def test_claim_distinct_sessions() -> None:
    assert jobs.claim("s1", "/proj") is True
    assert jobs.claim("s2", "/proj") is True


def test_stale_running_job_is_reclaimed() -> None:
    # Simulate a dead dispatch: a 'running' lock with an old started_at.
    lease.acquire(jobs._QUEUE, jobs._key("s1", "/proj"), now=T_OLD)
    # A fresh claim with a timeout reclaims it (the agent never closed it).
    assert jobs.claim("s1", "/proj", stale_after_seconds=3600) is True
    # Without a timeout, a recent running job still blocks.
    assert jobs.claim("s1", "/proj") is False


def test_done_job_is_not_reclaimed() -> None:
    jobs.claim("s1", "/proj")
    jobs.complete("s1", "/proj")
    # A completed job is durable dedup — never reclaimed even past the timeout.
    assert jobs.claim("s1", "/proj", stale_after_seconds=0) is False


def test_complete_marks_done() -> None:
    jobs.claim("s1", "/proj")
    jobs.complete("s1", "/proj")
    row = _lease_row("s1", "/proj")
    assert row.status == "done"
    assert row.finished_at


def test_fresh_running_job_within_window_not_reclaimed() -> None:
    # A running lock younger than the window is NOT reclaimed. ``claim`` has no
    # injectable ``now`` (stable signature), so anchor the lock to the same
    # real clock by taking it via ``claim`` itself.
    assert jobs.claim("s1", "/proj") is True
    assert jobs.claim("s1", "/proj", stale_after_seconds=3600) is False
