"""Tests for the kg_* MCP tool wrappers on the neuron server."""

from __future__ import annotations

import json

import pytest

import simba.kg.store
import simba.neuron.server as server


class TestKgAdd:
    def test_serializes_store_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, object] = {}

        def fake_kg_add(subject, predicate, object, proof, **kwargs):
            calls["args"] = (subject, predicate, object, proof)
            calls["kwargs"] = kwargs
            return "added"

        monkeypatch.setattr(simba.kg.store, "kg_add", fake_kg_add)

        result = server.kg_add("simba", "supports", "sqlite", "unit test")

        assert json.loads(result) == "added"
        assert calls["args"] == ("simba", "supports", "sqlite", "unit test")
        assert calls["kwargs"]["project_path"] is None

    def test_forwards_optional_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_kg_add(subject, predicate, object, proof, **kwargs):
            captured.update(kwargs)
            return "added"

        monkeypatch.setattr(simba.kg.store, "kg_add", fake_kg_add)

        server.kg_add(
            "a",
            "b",
            "c",
            "proof",
            subject_type="entity",
            object_type="entity",
            transcript_id="t-1",
            char_start=42,
            occurred_at="2025-03-01",
        )

        assert captured["subject_type"] == "entity"
        assert captured["object_type"] == "entity"
        assert captured["transcript_id"] == "t-1"
        assert captured["char_start"] == 42
        assert captured["occurred_at"] == "2025-03-01"


class TestKgQuery:
    def test_serializes_store_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [{"subject": "simba", "predicate": "supports", "object": "sqlite"}]

        def fake_kg_query(*args, **kwargs):
            return rows

        monkeypatch.setattr(simba.kg.store, "kg_query", fake_kg_query)

        result = server.kg_query(query="simba")

        assert json.loads(result) == rows

    def test_forwards_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_kg_query(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return []

        monkeypatch.setattr(simba.kg.store, "kg_query", fake_kg_query)

        server.kg_query(
            query="q",
            subject="s",
            predicate="p",
            as_of="2026-01-01T00:00:00Z",
            include_expired=True,
            occurred_after="2025-01-01",
            occurred_before="2025-12-31",
            limit=5,
        )

        assert captured["kwargs"]["subject"] == "s"
        assert captured["kwargs"]["predicate"] == "p"
        assert captured["kwargs"]["as_of"] == "2026-01-01T00:00:00Z"
        assert captured["kwargs"]["include_expired"] is True
        assert captured["kwargs"]["occurred_after"] == "2025-01-01"
        assert captured["kwargs"]["occurred_before"] == "2025-12-31"
        assert captured["kwargs"]["limit"] == 5


class TestKgInvalidate:
    def test_serializes_closed_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_kg_invalidate(subject, predicate, object, **kwargs):
            captured["args"] = (subject, predicate, object)
            return 3

        monkeypatch.setattr(simba.kg.store, "kg_invalidate", fake_kg_invalidate)

        result = server.kg_invalidate("simba", "supports", "sqlite")

        assert json.loads(result) == {"closed": 3}
        assert captured["args"] == ("simba", "supports", "sqlite")


class TestKgNeighbors:
    def test_serializes_and_forwards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_neighbors(entity, **kwargs):
            captured["entity"] = entity
            captured["kwargs"] = kwargs
            return [{"subject": "A", "predicate": "uses", "object": "B", "hop": 1}]

        monkeypatch.setattr(simba.kg.store, "kg_neighbors", fake_neighbors)
        result = server.kg_neighbors("A", depth=2, direction="out")

        assert json.loads(result)[0]["hop"] == 1
        assert captured["entity"] == "A"
        assert captured["kwargs"]["depth"] == 2
        assert captured["kwargs"]["direction"] == "out"


class TestKgQueryExpandForwarding:
    def test_expand_hops_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_query(query=None, **kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(simba.kg.store, "kg_query", fake_query)
        server.kg_query(subject="A", expand_hops=2)
        assert captured["expand_hops"] == 2
