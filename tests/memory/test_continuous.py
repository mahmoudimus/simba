"""Tests for the continuous-extraction Stop-hook orchestration (default-off rails).

on_stop: when enabled, read the incremental transcript window via the cursor,
enqueue it durably, and advance the cursor. Default-off → strict no-op.
"""

from __future__ import annotations

import types

import simba.memory.continuous as cont
import simba.memory.transcript_cursor as tc


def _cfg(enabled=True, max_bytes=1_000_000):
    return types.SimpleNamespace(
        continuous_extraction_enabled=enabled,
        continuous_extraction_max_bytes=max_bytes,
    )


def _hook(t, sid="s1"):
    return {"transcript_path": str(t), "session_id": sid, "project_path": "/proj"}


def test_disabled_is_strict_noop(tmp_path):
    t = tmp_path / "tr.jsonl"
    t.write_text("a meaningful conclusion\n")
    n = cont.on_stop(_hook(t), _cfg(enabled=False), cwd=tmp_path)
    assert n == 0
    assert cont.drain(tmp_path) == []
    assert tc.peek_offset("s1", cwd=tmp_path) == 0  # cursor untouched


def test_enabled_enqueues_and_advances(tmp_path):
    t = tmp_path / "tr.jsonl"
    t.write_text("a meaningful conclusion\n")
    n = cont.on_stop(_hook(t), _cfg(), cwd=tmp_path)
    assert n == 1
    q = cont.drain(tmp_path)
    assert len(q) == 1
    assert q[0]["session_id"] == "s1"
    assert q[0]["start"] == 0
    assert tc.peek_offset("s1", cwd=tmp_path) == q[0]["end"]


def test_only_new_window_on_second_call(tmp_path):
    t = tmp_path / "tr.jsonl"
    t.write_text("first turn\n")
    cont.on_stop(_hook(t), _cfg(), cwd=tmp_path)
    with open(t, "a", encoding="utf-8") as fh:
        fh.write("second turn\n")
    cont.on_stop(_hook(t), _cfg(), cwd=tmp_path)
    q = cont.drain(tmp_path)
    assert len(q) == 2
    assert q[1]["start"] == q[0]["end"]  # picked up only the new window


def test_nothing_new_does_not_enqueue(tmp_path):
    t = tmp_path / "tr.jsonl"
    t.write_text("only turn\n")
    cont.on_stop(_hook(t), _cfg(), cwd=tmp_path)
    n = cont.on_stop(_hook(t), _cfg(), cwd=tmp_path)
    assert n == 0
    assert len(cont.drain(tmp_path)) == 1


def test_whitespace_window_skipped_but_cursor_advances(tmp_path):
    t = tmp_path / "tr.jsonl"
    t.write_text("   \n\n")
    n = cont.on_stop(_hook(t), _cfg(), cwd=tmp_path)
    assert n == 0
    assert cont.drain(tmp_path) == []
    assert (
        tc.peek_offset("s1", cwd=tmp_path) > 0
    )  # advanced past whitespace, won't re-see


def test_missing_transcript_or_session_is_noop(tmp_path):
    assert cont.on_stop({"session_id": "s1"}, _cfg(), cwd=tmp_path) == 0
    assert (
        cont.on_stop(
            {"transcript_path": str(tmp_path / "x.jsonl")}, _cfg(), cwd=tmp_path
        )
        == 0
    )
