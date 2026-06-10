"""Tests for the rlm digest lease (rlm/jobs.py).

The bespoke ``RlmJob`` model is gone — these assert *public behavior* over the
``simba.workflow.lease`` primitive: claim-once / dedup, complete -> done, and
the config-gated stale-reclaim (off by default).
"""

from __future__ import annotations

import json
import pathlib

import pytest

import simba.db
import simba.rlm.config as rlm_config
import simba.rlm.jobs as jobs
import simba.workflow.lease as lease
import simba.workflow.store as store


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def _set_stale_after(monkeypatch: pytest.MonkeyPatch, seconds: int) -> None:
    """Override the rlm config the lease reads, without touching real TOML."""
    cfg = rlm_config.RlmConfig(digest_stale_after_seconds=seconds)
    monkeypatch.setattr(jobs, "_rlm_cfg", lambda cwd=None: cfg)


def _lease_row(transcript_id: str, project_path: str) -> store.WfTask | None:
    key = jobs._key(transcript_id, project_path)
    with simba.db.connect():
        return store.WfTask.get_or_none(
            (store.WfTask.queue == jobs._QUEUE) & (store.WfTask.dedup_key == key)
        )


def test_claim_then_dedup(monkeypatch):
    _set_stale_after(monkeypatch, 0)
    assert jobs.claim("t1", "/p", "claude-cli") is True
    assert jobs.claim("t1", "/p", "claude-cli") is False  # already claimed


def test_claim_distinct_keys(monkeypatch):
    _set_stale_after(monkeypatch, 0)
    assert jobs.claim("t1", "/p", "claude-cli") is True
    assert jobs.claim("t1", "/other", "claude-cli") is True  # different project
    assert jobs.claim("t2", "/p", "claude-cli") is True  # different transcript


def test_claim_stores_engine_payload(monkeypatch):
    _set_stale_after(monkeypatch, 0)
    jobs.claim("t1", "/p", "claude-cli")
    row = _lease_row("t1", "/p")
    assert row is not None
    assert row.status == "running"
    assert json.loads(row.payload) == {"engine": "claude-cli"}


def test_complete_updates_status(monkeypatch):
    _set_stale_after(monkeypatch, 0)
    jobs.claim("t1", "/p", "claude-cli")
    jobs.complete("t1", "/p", 7)
    row = _lease_row("t1", "/p")
    assert row.status == "done"
    assert json.loads(row.result) == {"n_stored": 7}


def test_no_reclaim_by_default(monkeypatch):
    # digest_stale_after_seconds defaults to 0 => None => never reclaimed: a
    # stranded 'running' digest blocks re-digest forever (current behavior).
    _set_stale_after(monkeypatch, 0)
    key = jobs._key("t1", "/p")
    lease.acquire(jobs._QUEUE, key, now="2000-01-01T00:00:00Z")
    assert jobs.claim("t1", "/p", "claude-cli") is False


def test_reclaim_when_configured(monkeypatch):
    # Flip the field => a stale 'running' digest is re-acquirable.
    _set_stale_after(monkeypatch, 1)
    key = jobs._key("t1", "/p")
    lease.acquire(jobs._QUEUE, key, now="2000-01-01T00:00:00Z")
    assert jobs.claim("t1", "/p", "claude-cli") is True


def test_done_is_never_reclaimed_even_when_configured(monkeypatch):
    _set_stale_after(monkeypatch, 1)
    jobs.claim("t1", "/p", "claude-cli")
    jobs.complete("t1", "/p", 3)
    # done is durable dedup — never reclaimed.
    assert jobs.claim("t1", "/p", "claude-cli") is False
