"""Tests for the recall router."""

from __future__ import annotations

import simba.rlm.recall as recall


class _FakeProvider:
    def __init__(self, available_ids):
        self._available = set(available_ids)

    def available(self, transcript_id):
        return transcript_id in self._available


def test_route_maps_session_source_to_pointer(monkeypatch):
    monkeypatch.setattr(
        "simba.hooks._memory_client.recall_memories",
        lambda *a, **k: [
            {
                "content": "decided X",
                "sessionSource": "s1",
                "projectPath": "/p",
                "similarity": 0.8,
            },
        ],
    )
    pointers = recall.route("what about X", cwd="/p", provider=_FakeProvider({"s1"}))
    assert len(pointers) == 1
    p = pointers[0]
    assert p.transcript_id == "s1"
    assert p.snippet == "decided X"
    assert p.project_path == "/p"
    assert p.available is True


def test_route_marks_unavailable_when_no_transcript(monkeypatch):
    monkeypatch.setattr(
        "simba.hooks._memory_client.recall_memories",
        lambda *a, **k: [
            {"content": "old", "sessionSource": "gone", "similarity": 0.7},
        ],
    )
    pointers = recall.route("q", cwd="/p", provider=_FakeProvider(set()))
    assert pointers[0].available is False


def test_route_handles_missing_session_source(monkeypatch):
    monkeypatch.setattr(
        "simba.hooks._memory_client.recall_memories",
        lambda *a, **k: [{"content": "no session", "similarity": 0.6}],
    )
    pointers = recall.route("q", cwd="/p", provider=_FakeProvider(set()))
    assert pointers[0].transcript_id is None
    assert pointers[0].available is False


def test_route_empty_when_recall_fails(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("daemon down")

    monkeypatch.setattr("simba.hooks._memory_client.recall_memories", _boom)
    assert recall.route("q", cwd="/p", provider=_FakeProvider(set())) == []


def test_pointer_to_dict(monkeypatch):
    monkeypatch.setattr(
        "simba.hooks._memory_client.recall_memories",
        lambda *a, **k: [
            {
                "content": "c",
                "sessionSource": "s1",
                "projectPath": "/p",
                "similarity": 0.9,
            },
        ],
    )
    d = recall.route("q", cwd="/p", provider=_FakeProvider({"s1"}))[0].to_dict()
    assert d == {
        "snippet": "c",
        "transcript_id": "s1",
        "project_path": "/p",
        "similarity": 0.9,
        "available": True,
    }


def test_pointers_from_memories_builds_pointers():
    ps = recall.pointers_from_memories(
        [
            {
                "content": "c",
                "sessionSource": "s1",
                "projectPath": "/p",
                "similarity": 0.9,
            },
            {"content": "no ss", "similarity": 0.5},
        ],
        "/p",
        provider=_FakeProvider({"s1"}),
    )
    assert ps[0].transcript_id == "s1" and ps[0].available is True
    assert ps[1].transcript_id is None and ps[1].available is False
