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
        result = truth_add(
            "simba", "supports", "sqlite", "unit test", project_path="proj-1"
        )
        assert result == "Fact recorded: simba supports sqlite"

        # Verify the row is actually in the database (now project-scoped)
        db_path = simba.db.get_db_path()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT * FROM proven_facts").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0] == ("simba", "supports", "sqlite", "unit test", "proj-1")

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

    def test_query_scopes_by_project(self) -> None:
        truth_add("simba", "supports", "sqlite", "t", project_path="proj-A")
        truth_add("simba", "supports", "sqlite", "t", project_path="proj-B")

        # Same fact text in two projects = two distinct rows.
        result_a = truth_query(subject="simba", project_path="proj-A")
        assert "simba supports sqlite" in result_a
        assert truth_query(project_path="proj-B").count("FACT:") == 1


class TestLegacyMigration:
    def test_unscoped_table_is_sidelined(self, tmp_path: pathlib.Path) -> None:
        # Simulate the old (pre-project_path) schema with a junk row.
        db_path = simba.db.get_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE proven_facts (subject TEXT, predicate TEXT, "
            "object TEXT, proof TEXT, UNIQUE(subject, predicate, object))"
        )
        conn.execute(
            "INSERT INTO proven_facts VALUES (?, ?, ?, ?)",
            ("old", "fact", "value", "legacy"),
        )
        conn.commit()
        conn.close()

        # Opening via get_db runs the schema initializer → sideline + recreate.
        with simba.db.get_db() as conn:
            cols = [
                r[1]
                for r in conn.execute("PRAGMA table_info(proven_facts)").fetchall()
            ]
            assert "project_path" in cols
            fresh_count = conn.execute(
                "SELECT COUNT(*) FROM proven_facts"
            ).fetchone()[0]
            assert fresh_count == 0
            legacy = conn.execute("SELECT * FROM proven_facts_legacy").fetchall()
            assert [tuple(r) for r in legacy] == [("old", "fact", "value", "legacy")]


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
