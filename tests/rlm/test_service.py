"""Tests for the RLM service composition layer."""

from __future__ import annotations

import pathlib

import simba.rlm.service as service


def _make_transcript(root: pathlib.Path, sid: str, body: str) -> None:
    d = root / sid
    d.mkdir(parents=True)
    (d / "transcript.md").write_text(body)


def _svc(tmp_path):
    # Fresh service rooted at a tmp transcripts dir.
    return service.RlmService(transcripts_root=tmp_path)


def test_grep_returns_match_dicts(tmp_path):
    _make_transcript(tmp_path, "s1", "alpha\nbeta target\ngamma")
    out = _svc(tmp_path).grep("s1", "target")
    assert "matches" in out
    assert out["matches"][0]["match_text"] == "target"
    assert out["matches"][0]["line_number"] == 2


def test_grep_missing_transcript_returns_error(tmp_path):
    out = _svc(tmp_path).grep("missing", "x")
    assert "error" in out
    assert "missing" in out["error"]


def test_peek_and_window(tmp_path):
    _make_transcript(tmp_path, "s1", "0123456789")
    svc = _svc(tmp_path)
    assert svc.peek("s1", 2, 5) == {"text": "234"}
    # window(around=5, radius=2) -> text[3:7]
    assert svc.window("s1", 5, 2) == {"text": "3456"}


def test_head_tail(tmp_path):
    _make_transcript(tmp_path, "s1", "a\nb\nc\nd")
    svc = _svc(tmp_path)
    assert svc.head("s1", 2) == {"text": "a\nb"}
    assert svc.tail("s1", 2) == {"text": "c\nd"}


def test_grep_bad_pattern_returns_error(tmp_path):
    _make_transcript(tmp_path, "s1", "abc")
    out = _svc(tmp_path).grep("s1", "(a+)+")
    assert "error" in out


def test_recall_serializes_pointers(tmp_path, monkeypatch):
    _make_transcript(tmp_path, "s1", "x")
    monkeypatch.setattr(
        "simba.hooks._memory_client.recall_memories",
        lambda *a, **k: [
            {"content": "c", "sessionSource": "s1",
             "projectPath": "/p", "similarity": 0.9},
        ],
    )
    out = _svc(tmp_path).recall("q", cwd="/p")
    assert out["pointers"][0]["transcript_id"] == "s1"
    assert out["pointers"][0]["available"] is True
