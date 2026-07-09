"""Tests for the rlm_* MCP tool wrappers on the neuron server."""

from __future__ import annotations

import json

import simba.neuron.server as server


class _FakeService:
    def recall(self, query, cwd, max_pointers=None):
        return {"pointers": [{"transcript_id": "s1", "available": True}]}

    def grep(self, transcript_id, pattern, max_matches=None):
        return {"matches": [{"match_text": pattern, "line_number": 1}]}

    def peek(self, transcript_id, start_char, end_char):
        return {"text": "sliced"}

    def window(self, transcript_id, around_char, radius=None):
        return {"text": "windowed"}

    def head(self, transcript_id, n_lines=20):
        return {"text": "head"}

    def tail(self, transcript_id, n_lines=20):
        return {"text": "tail"}


def test_rlm_tools_serialize_service_output(monkeypatch):
    monkeypatch.setattr("simba.rlm.service.get_service", lambda: _FakeService())
    assert json.loads(server.rlm_recall("q"))["pointers"][0]["transcript_id"] == "s1"
    grep_out = json.loads(server.rlm_grep("s1", "beta"))
    assert grep_out["matches"][0]["match_text"] == "beta"
    assert json.loads(server.rlm_peek("s1", 0, 6))["text"] == "sliced"
    assert json.loads(server.rlm_window("s1", 3))["text"] == "windowed"
    assert json.loads(server.rlm_head("s1"))["text"] == "head"
    assert json.loads(server.rlm_tail("s1"))["text"] == "tail"
