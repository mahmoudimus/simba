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

    def test_forwards_optional_arguments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        )

        assert captured["subject_type"] == "entity"
        assert captured["object_type"] == "entity"
        assert captured["transcript_id"] == "t-1"
        assert captured["char_start"] == 42


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
            limit=5,
        )

        assert captured["kwargs"]["subject"] == "s"
        assert captured["kwargs"]["predicate"] == "p"
        assert captured["kwargs"]["as_of"] == "2026-01-01T00:00:00Z"
        assert captured["kwargs"]["include_expired"] is True
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
