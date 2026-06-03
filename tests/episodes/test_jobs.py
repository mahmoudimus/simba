"""Tests for the episode_jobs coordination table."""

from __future__ import annotations

import pathlib
import time

import pytest

import simba.db
import simba.episodes.jobs as jobs


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def test_claim_is_idempotent() -> None:
    assert jobs.claim("s1", "/proj") is True
    assert jobs.claim("s1", "/proj") is False  # already claimed


def test_claim_distinct_sessions() -> None:
    assert jobs.claim("s1", "/proj") is True
    assert jobs.claim("s2", "/proj") is True


def test_stale_running_job_is_reclaimed() -> None:
    # Simulate a dead dispatch: a 'running' job with an old started_at.
    with simba.db.connect():
        jobs.EpisodeJob.create(
            session_source="s1",
            project_path="/proj",
            status="running",
            started_at="2000-01-01T00:00:00Z",
        )
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
    with simba.db.connect():
        row = jobs.EpisodeJob.get(
            (jobs.EpisodeJob.session_source == "s1")
            & (jobs.EpisodeJob.project_path == "/proj")
        )
    assert row.status == "done"
    assert row.finished_at


def test_claim_after_timeout_window(monkeypatch) -> None:
    # A running job younger than the window is NOT reclaimed.
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with simba.db.connect():
        jobs.EpisodeJob.create(
            session_source="s1",
            project_path="/proj",
            status="running",
            started_at=now,
        )
    assert jobs.claim("s1", "/proj", stale_after_seconds=3600) is False
