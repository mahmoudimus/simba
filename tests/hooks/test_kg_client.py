"""Tests for the KG client — keyword extraction and KG fact lookup."""

from __future__ import annotations

import simba.hooks._kg_client


class TestExtractKeywords:
    def test_basic_extraction(self) -> None:
        result = simba.hooks._kg_client.extract_keywords("simba database module")
        assert result == ["simba", "database", "module"]

    def test_stops_at_max_keywords(self) -> None:
        result = simba.hooks._kg_client.extract_keywords(
            "alpha beta gamma delta", max_keywords=2
        )
        assert len(result) == 2
        assert result == ["alpha", "beta"]

    def test_filters_stop_words(self) -> None:
        result = simba.hooks._kg_client.extract_keywords(
            "the simba is a great tool for testing"
        )
        assert "the" not in result
        assert "is" not in result
        assert "simba" in result

    def test_empty_input(self) -> None:
        assert simba.hooks._kg_client.extract_keywords("") == []

    def test_only_stop_words(self) -> None:
        assert simba.hooks._kg_client.extract_keywords("the is a of to and") == []

    def test_preserves_dotted_identifiers(self) -> None:
        result = simba.hooks._kg_client.extract_keywords(
            "check simba.db module", max_keywords=3
        )
        assert "simba.db" in result


class TestQueryKg:
    def test_returns_empty_for_no_keywords(self) -> None:
        result = simba.hooks._kg_client.query_kg("the is a")
        assert result == ""

    def test_returns_empty_when_no_rows(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "simba.kg.store.kg_query",
            lambda *a, **k: [],
        )
        result = simba.hooks._kg_client.query_kg("simba database module")
        assert result == ""

    def test_returns_kg_facts_block_for_matching_edge(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "simba.kg.store.kg_query",
            lambda *a, **k: [
                {
                    "subject": "simba.db",
                    "predicate": "uses",
                    "object": "sqlite3",
                    "transcript_id": "t-1",
                    "char_start": 42,
                    "valid_to": None,
                }
            ],
        )
        result = simba.hooks._kg_client.query_kg("simba database module")
        assert "<kg-facts>" in result
        assert "</kg-facts>" in result
        assert 'subject="simba.db"' in result
        assert 'predicate="uses"' in result
        assert "sqlite3" in result
        assert 'transcript_id="t-1"' in result
        assert 'char_start="42"' in result

    def test_omits_transcript_attrs_when_absent(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "simba.kg.store.kg_query",
            lambda *a, **k: [
                {
                    "subject": "simba.db",
                    "predicate": "uses",
                    "object": "sqlite3",
                    "transcript_id": None,
                    "char_start": None,
                    "valid_to": None,
                }
            ],
        )
        result = simba.hooks._kg_client.query_kg("simba database module")
        assert "<kg-facts>" in result
        assert "transcript_id=" not in result
        assert "char_start=" not in result

    def test_multiple_facts_returned(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "simba.kg.store.kg_query",
            lambda *a, **k: [
                {
                    "subject": "simba.db",
                    "predicate": "uses",
                    "object": "sqlite3",
                    "transcript_id": None,
                    "char_start": None,
                    "valid_to": None,
                },
                {
                    "subject": "simba.memory",
                    "predicate": "uses",
                    "object": "lancedb",
                    "transcript_id": None,
                    "char_start": None,
                    "valid_to": None,
                },
            ],
        )
        result = simba.hooks._kg_client.query_kg("simba memory database")
        assert result.count("<fact ") == 2

    def test_graceful_on_db_exception(self, monkeypatch) -> None:
        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr("simba.kg.store.kg_query", boom)
        result = simba.hooks._kg_client.query_kg("simba test")
        assert result == ""


class TestQueryKgProjectScoping:
    """Regression guard: query_kg must always scope kg_query to a project.

    Passing ``project_path=None`` straight through to ``kg_query`` would leak
    another repo's facts into this repo's injection — so when no project_path
    is given, query_kg resolves the current repo's stable id and forwards it.
    """

    def test_resolves_project_path_when_none(self, monkeypatch) -> None:
        captured: dict = {}

        def fake_query(*a, **k):
            captured.update(k)
            return []

        monkeypatch.setattr("simba.kg.store.kg_query", fake_query)
        monkeypatch.setattr(
            "simba.db.resolve_project_id", lambda p=None: "proj-resolved"
        )
        simba.hooks._kg_client.query_kg("simba database module", cwd="/some/repo")
        assert captured.get("project_path") == "proj-resolved"

    def test_explicit_project_path_is_forwarded(self, monkeypatch) -> None:
        captured: dict = {}

        def fake_query(*a, **k):
            captured.update(k)
            return []

        monkeypatch.setattr("simba.kg.store.kg_query", fake_query)
        # An explicit project_path must win — resolve_project_id is not consulted.
        monkeypatch.setattr(
            "simba.db.resolve_project_id", lambda p=None: "should-not-be-used"
        )
        simba.hooks._kg_client.query_kg(
            "simba database module", project_path="explicit-proj"
        )
        assert captured.get("project_path") == "explicit-proj"
