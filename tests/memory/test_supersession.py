"""Tests for append-only memory supersession audit."""

from __future__ import annotations

import pathlib

import simba.db
import simba.memory.supersession as supersession


def test_append_event_and_chain(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        supersession.append_event(
            old_id="mem_old",
            new_id="mem_new",
            project_path="/repo",
            memory_type="PATTERN",
            similarity=0.91,
            reason="near_duplicate_same_type",
            provenance='{"source":"test"}',
            now=1000.0,
        )
        rows = supersession.chain("mem_old")

    assert len(rows) == 1
    assert rows[0].old_id == "mem_old"
    assert rows[0].new_id == "mem_new"
    assert rows[0].created_at_iso == "1970-01-01T00:16:40Z"


def test_chain_follows_latest_successor(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        supersession.append_event(
            old_id="mem_a",
            new_id="mem_b",
            project_path="/repo",
            memory_type="PATTERN",
            similarity=0.9,
            reason="near_duplicate_same_type",
            provenance="{}",
            now=1000.0,
        )
        supersession.append_event(
            old_id="mem_b",
            new_id="mem_c",
            project_path="/repo",
            memory_type="PATTERN",
            similarity=0.9,
            reason="near_duplicate_same_type",
            provenance="{}",
            now=1001.0,
        )
        rows = supersession.chain("mem_a")

    assert [row.new_id for row in rows] == ["mem_b", "mem_c"]


def test_pending_is_not_active_until_confirmed(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        pending = supersession.append_event(
            old_id="mem_old",
            new_id="mem_new",
            project_path="/repo",
            memory_type="PATTERN",
            similarity=0.9,
            reason="near_duplicate_same_type",
            provenance="{}",
            status=supersession.STATUS_PENDING,
            old_trust_score=1.2,
            new_trust_score=0.6,
            now=1000.0,
        )
        assert supersession.latest_successors(["mem_old"]) == {}
        assert supersession.latest_pending(["mem_old"])["mem_old"].id == pending.id

        decision = supersession.confirm(pending.id, now=1001.0)
        assert decision.status == supersession.STATUS_ACTIVE
        assert decision.pending_event_id == pending.id
        assert supersession.latest_pending(["mem_old"]) == {}
        assert supersession.latest_successors(["mem_old"])["mem_old"].new_id == (
            "mem_new"
        )


def test_rejected_pending_remains_inactive(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        pending = supersession.append_event(
            old_id="mem_old",
            new_id="mem_new",
            project_path="/repo",
            memory_type="PATTERN",
            similarity=0.9,
            reason="near_duplicate_same_type",
            provenance="{}",
            status=supersession.STATUS_PENDING,
            old_trust_score=1.2,
            new_trust_score=0.6,
            now=1000.0,
        )
        decision = supersession.reject(pending.id, now=1001.0)

        assert decision.status == supersession.STATUS_REJECTED
        assert supersession.latest_pending(["mem_old"]) == {}
        assert supersession.latest_successors(["mem_old"]) == {}
