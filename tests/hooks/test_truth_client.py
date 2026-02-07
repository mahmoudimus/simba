"""Tests for the truth DB client — keyword extraction and fact lookup."""

from __future__ import annotations

import sqlite3
import unittest.mock

import simba.hooks._truth_client


class TestExtractKeywords:
    def test_basic_extraction(self) -> None:
        result = simba.hooks._truth_client.extract_keywords("simba database module")
        assert result == ["simba", "database", "module"]

    def test_stops_at_max_keywords(self) -> None:
        result = simba.hooks._truth_client.extract_keywords(
            "alpha beta gamma delta", max_keywords=2
        )
        assert len(result) == 2
        assert result == ["alpha", "beta"]

    def test_filters_stop_words(self) -> None:
        result = simba.hooks._truth_client.extract_keywords(
            "the simba is a great tool for testing"
        )
        assert "the" not in result
        assert "is" not in result
        assert "simba" in result

    def test_deduplicates_case_insensitive(self) -> None:
        result = simba.hooks._truth_client.extract_keywords(
            "Simba simba SIMBA other words", max_keywords=5
        )
        # Only first occurrence kept
        assert result.count("Simba") == 1
        assert len([kw for kw in result if kw.lower() == "simba"]) == 1

    def test_empty_input(self) -> None:
        assert simba.hooks._truth_client.extract_keywords("") == []

    def test_only_stop_words(self) -> None:
        assert simba.hooks._truth_client.extract_keywords("the is a of to and") == []

    def test_preserves_dotted_identifiers(self) -> None:
        result = simba.hooks._truth_client.extract_keywords(
            "check simba.db module", max_keywords=3
        )
        assert "simba.db" in result

    def test_short_words_filtered(self) -> None:
        # Single-char words should be filtered (len < 2)
        result = simba.hooks._truth_client.extract_keywords(
            "a x y simba", max_keywords=5
        )
        assert "simba" in result
        assert "x" not in result


class TestQueryTruthDb:
    def test_returns_empty_for_no_keywords(self) -> None:
        result = simba.hooks._truth_client.query_truth_db("the is a")
        assert result == ""

    def test_returns_empty_when_db_unavailable(self) -> None:
        with unittest.mock.patch("simba.db.get_connection", return_value=None):
            result = simba.hooks._truth_client.query_truth_db("simba database")
        assert result == ""

    def test_returns_empty_when_no_facts_found(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE proven_facts "
            "(subject TEXT, predicate TEXT, object TEXT, proof TEXT)"
        )
        with unittest.mock.patch("simba.db.get_connection", return_value=conn):
            result = simba.hooks._truth_client.query_truth_db("simba database")
        assert result == ""

    def test_returns_xml_block_with_matching_facts(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE proven_facts "
            "(subject TEXT, predicate TEXT, object TEXT, proof TEXT)"
        )
        conn.execute(
            "INSERT INTO proven_facts VALUES (?, ?, ?, ?)",
            ("simba.db", "uses", "sqlite3", "direct import in db.py"),
        )
        conn.commit()
        with unittest.mock.patch("simba.db.get_connection", return_value=conn):
            result = simba.hooks._truth_client.query_truth_db("simba database module")
        assert "<proven-facts>" in result
        assert "</proven-facts>" in result
        assert "simba.db" in result
        assert "sqlite3" in result
        assert "direct import in db.py" in result

    def test_multiple_facts_returned(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE proven_facts "
            "(subject TEXT, predicate TEXT, object TEXT, proof TEXT)"
        )
        conn.execute(
            "INSERT INTO proven_facts VALUES (?, ?, ?, ?)",
            ("simba.db", "uses", "sqlite3", "import in db.py"),
        )
        conn.execute(
            "INSERT INTO proven_facts VALUES (?, ?, ?, ?)",
            ("simba.memory", "uses", "lancedb", "import in server.py"),
        )
        conn.commit()
        with unittest.mock.patch("simba.db.get_connection", return_value=conn):
            result = simba.hooks._truth_client.query_truth_db("simba memory database")
        assert result.count("<fact ") == 2

    def test_graceful_on_db_exception(self) -> None:
        with unittest.mock.patch(
            "simba.db.get_connection", side_effect=RuntimeError("boom")
        ):
            result = simba.hooks._truth_client.query_truth_db("simba test")
        assert result == ""
