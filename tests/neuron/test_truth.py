"""Tests for the neuron truth database module."""

from __future__ import annotations

import pathlib
import sqlite3

import pytest

import simba.db
from simba.neuron.truth import truth_add, truth_query


@pytest.fixture(autouse=True)
def _patch_db_path(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point simba.db.get_db_path at a temp directory for every test."""
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


class TestTruthAdd:
    def test_inserts_fact(self, tmp_path: pathlib.Path) -> None:
        result = truth_add("simba", "supports", "sqlite", "unit test")
        assert result == "Fact recorded: simba supports sqlite"

        # Verify the row is actually in the database
        db_path = simba.db.get_db_path()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT * FROM proven_facts").fetchall()
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


class TestSchema:
    def test_creates_proven_facts_table(self, tmp_path: pathlib.Path) -> None:
        db_path = simba.db.get_db_path()
        assert not db_path.exists()

        with simba.db.get_db() as conn:
            # Table should exist after entering the context manager
            cursor = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='proven_facts'"
            )
            tables = cursor.fetchall()

        assert len(tables) == 1
        assert tables[0][0] == "proven_facts"
