"""Shared fixtures for neuron Phase 7 tests."""

from __future__ import annotations

import pytest


@pytest.fixture()
def planted_contradiction(tmp_path, monkeypatch):
    """Seed the KG with a USES/DOES_NOT_USE pair and return their edge ids."""
    import simba.db
    import simba.kg.store

    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    simba.kg.store.kg_add("ToolA", "uses", "LibB", "test", project_path="/proj")
    simba.kg.store.kg_add("ToolA", "does_not_use", "LibB", "test", project_path="/proj")
    with simba.db.connect():
        ids = [
            e.id
            for e in simba.kg.store.KgEdge.select().where(
                simba.kg.store.KgEdge.project_path == "/proj"
            )
        ]
    return ids, "/proj"
