"""Tests for the neuron truth database module."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import simba.neuron.config
from simba.neuron.truth import get_db_connection, truth_add, truth_query


@pytest.fixture(autouse=True)
def _patch_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point CONFIG.db_path at a temp directory for every test."""
    db_path = tmp_path / "test_truth.db"
    monkeypatch.setattr(simba.neuron.config.CONFIG, "db_path", db_path)


class TestTruthAdd:
    def test_inserts_fact(self, tmp_path: Path) -> None:
        result = truth_add("simba", "supports", "sqlite", "unit test")
        assert result == "Fact recorded: simba supports sqlite"

        # Verify the row is actually in the database
        db_path = simba.neuron.config.CONFIG.resolved_db_path
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT * FROM facts").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0] == ("simba", "supports", "sqlite", "unit test")

    def test_duplicate_returns_already_exists(self) -> None:
        truth_add("simba", "supports", "sqlite", "unit test")
        result = truth_add("simba", "supports", "sqlite", "unit test")
        assert result == "Fact already exists: simba supports sqlite"


class TestTruthQuery:
    def test_query_by_subject(self) -> None:
        truth_add("simba", "supports", "sqlite", "unit test")
        truth_add("neuron", "provides", "verification", "unit test")

        result = truth_query(subject="simba")
        assert "simba supports sqlite" in result
        assert "neuron" not in result

    def test_query_by_predicate(self) -> None:
        truth_add("simba", "supports", "sqlite", "unit test")
        truth_add("neuron", "provides", "verification", "unit test")

        result = truth_query(predicate="provides")
        assert "neuron provides verification" in result
        assert "simba" not in result

    def test_query_no_match(self) -> None:
        truth_add("simba", "supports", "sqlite", "unit test")

        result = truth_query(subject="nonexistent")
        assert result == "No facts found matching criteria."

    def test_query_no_filters_returns_all(self) -> None:
        truth_add("simba", "supports", "sqlite", "unit test")
        truth_add("neuron", "provides", "verification", "unit test")

        result = truth_query()
        assert "simba supports sqlite" in result
        assert "neuron provides verification" in result


class TestGetDbConnection:
    def test_creates_facts_table(self, tmp_path: Path) -> None:
        db_path = simba.neuron.config.CONFIG.resolved_db_path
        assert not db_path.exists()

        with get_db_connection() as conn:
            # Table should exist after entering the context manager
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='facts'"
            )
            tables = cursor.fetchall()

        assert len(tables) == 1
        assert tables[0][0] == "facts"
