"""Tests for the rlm_jobs coordination table."""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.rlm.jobs as jobs


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def test_claim_then_dedup():
    assert jobs.claim("t1", "/p", "claude-cli") is True
    assert jobs.claim("t1", "/p", "claude-cli") is False  # already claimed


def test_claim_distinct_keys():
    assert jobs.claim("t1", "/p", "claude-cli") is True
    assert jobs.claim("t1", "/other", "claude-cli") is True  # different project
    assert jobs.claim("t2", "/p", "claude-cli") is True  # different transcript


def test_complete_updates_status():
    jobs.claim("t1", "/p", "claude-cli")
    jobs.complete("t1", "/p", 7)
    with simba.db.get_db() as conn:
        row = conn.execute(
            "SELECT status, n_stored FROM rlm_jobs WHERE transcript_id='t1'"
        ).fetchone()
    assert row["status"] == "done"
    assert row["n_stored"] == 7


