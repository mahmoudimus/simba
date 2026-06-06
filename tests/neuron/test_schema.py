"""Neuron Phase 7 schema: kg_derived_edges + kg_rules + dormant (Task B.1)."""

from __future__ import annotations

import sqlite3

import pytest

import simba.db


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    return db_path


def test_kg_derived_edges_created(tmp_db) -> None:
    import simba.neuron.schema  # triggers registration + migration

    _ = simba.neuron.schema
    with simba.db.connect():
        pass
    conn = sqlite3.connect(str(tmp_db))
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    conn.close()
    assert "kg_derived_edges" in tables
    assert "kg_rules" in tables


def test_dormant_column_added(tmp_db) -> None:
    import simba.neuron.schema

    _ = simba.neuron.schema
    with simba.db.connect():
        pass
    conn = sqlite3.connect(str(tmp_db))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(kg_edges)")}
    conn.close()
    assert "dormant" in cols


def test_migrations_are_idempotent(tmp_db) -> None:
    import simba.neuron.schema

    _ = simba.neuron.schema
    # Run twice — should not raise
    with simba.db.connect():
        pass
    with simba.db.connect():
        pass
