"""Tests for the incremental transcript cursor (Continuous-extraction plumbing).

The transcript JSONL is append-only, so a per-session byte offset lets each pass
read only what's new — O(new), not O(whole). Pure logic over a tmp file + a tmp
control-plane DB (cwd with no .git → cwd/.simba/simba.db).
"""

from __future__ import annotations

import simba.memory.transcript_cursor as tc


def _append(p, text):
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(text)


def test_fresh_session_reads_whole_file(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _append(t, "line1\nline2\n")
    w = tc.next_window(str(t), session_id="s1", cwd=tmp_path)
    assert w is not None
    assert w.content == "line1\nline2\n"
    assert w.start == 0
    assert w.end == len(b"line1\nline2\n")


def test_advance_then_only_new(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _append(t, "old\n")
    w1 = tc.next_window(str(t), session_id="s1", cwd=tmp_path)
    tc.advance("s1", w1.end, cwd=tmp_path)
    _append(t, "new1\nnew2\n")
    w2 = tc.next_window(str(t), session_id="s1", cwd=tmp_path)
    assert w2.content == "new1\nnew2\n"
    assert w2.start == w1.end


def test_nothing_new_returns_none(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _append(t, "x\n")
    w = tc.next_window(str(t), session_id="s1", cwd=tmp_path)
    tc.advance("s1", w.end, cwd=tmp_path)
    assert tc.next_window(str(t), session_id="s1", cwd=tmp_path) is None


def test_missing_file_returns_none(tmp_path):
    assert (
        tc.next_window(str(tmp_path / "nope.jsonl"), session_id="s1", cwd=tmp_path)
        is None
    )


def test_file_shrank_resets_to_start(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _append(t, "aaaabbbb\n")
    w = tc.next_window(str(t), session_id="s1", cwd=tmp_path)
    tc.advance("s1", w.end, cwd=tmp_path)
    t.write_text("z\n", encoding="utf-8")  # rotated / replaced shorter
    w2 = tc.next_window(str(t), session_id="s1", cwd=tmp_path)
    assert w2 is not None
    assert w2.start == 0
    assert w2.content == "z\n"


def test_reset_rewinds_to_zero(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _append(t, "hello\n")
    w = tc.next_window(str(t), session_id="s1", cwd=tmp_path)
    tc.advance("s1", w.end, cwd=tmp_path)
    tc.reset("s1", cwd=tmp_path)
    w2 = tc.next_window(str(t), session_id="s1", cwd=tmp_path)
    assert w2.start == 0


def test_max_bytes_caps_the_window(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _append(t, "x" * 100)
    w = tc.next_window(str(t), session_id="s1", cwd=tmp_path, max_bytes=10)
    assert len(w.content.encode()) == 10
    assert w.end == 10


def test_sessions_are_independent(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _append(t, "shared\n")
    w_a = tc.next_window(str(t), session_id="a", cwd=tmp_path)
    tc.advance("a", w_a.end, cwd=tmp_path)
    # session b has its own cursor (still at 0) → sees the whole file
    w_b = tc.next_window(str(t), session_id="b", cwd=tmp_path)
    assert w_b.start == 0
    assert w_b.content == "shared\n"
