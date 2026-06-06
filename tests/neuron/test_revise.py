"""Sub-phase REVISE: AGM contraction via entrenchment order (Task B.5)."""

from __future__ import annotations

import time


def test_revise_disabled_returns_empty() -> None:
    from simba.neuron.config import NeuronConfig
    from simba.neuron.revise import revise_unsat_core

    result = revise_unsat_core(
        [1, 2], project_path="/proj", cfg=NeuronConfig(revise_enabled=False)
    )
    assert result.dormant_edge_ids == []


def test_revise_stamps_weaker_edge_dormant(tmp_path, monkeypatch) -> None:
    import simba.kg.store

    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    # Insert: older edge (lower entrenchment) conflicts with newer
    simba.kg.store.kg_add("A", "uses", "B", "test", project_path="/proj")
    time.sleep(1.01)
    simba.kg.store.kg_add("A", "does_not_use", "B", "test", project_path="/proj")

    import simba.db as db
    from simba.neuron.config import NeuronConfig
    from simba.neuron.revise import revise_unsat_core

    with db.connect():
        from simba.kg.store import KgEdge

        ids = [e.id for e in KgEdge.select().where(KgEdge.project_path == "/proj")]
    assert len(ids) == 2

    result = revise_unsat_core(
        ids, project_path="/proj", cfg=NeuronConfig(revise_enabled=True)
    )
    assert len(result.dormant_edge_ids) == 1
    assert len(result.retained_edge_ids) == 1

    with db.connect():
        from simba.kg.store import KgEdge

        dormant_e = KgEdge.get_by_id(result.dormant_edge_ids[0])
        assert dormant_e.dormant == 1
        assert dormant_e.valid_to is not None


def test_revise_tied_entrenchment_skips(tmp_path, monkeypatch) -> None:
    from simba.neuron.revise import entrenchment_score

    edge_a = {
        "id": 1,
        "occurred_at": "2024-01-01T00:00:00Z",
        "valid_from": "2024-01-01T00:00:00Z",
        "confidence": 0.8,
    }
    edge_b = {
        "id": 2,
        "occurred_at": "2024-01-01T00:00:00Z",
        "valid_from": "2024-01-01T00:00:00Z",
        "confidence": 0.8,
    }
    assert entrenchment_score(edge_a) == entrenchment_score(edge_b)
