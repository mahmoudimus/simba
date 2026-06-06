"""Sub-phase DISTILL: write verified derived edges (Task B.6)."""

from __future__ import annotations


def test_distill_disabled_returns_empty() -> None:
    from simba.neuron.config import NeuronConfig
    from simba.neuron.derive import DerivedEdge
    from simba.neuron.distill import distill_edges

    result = distill_edges(
        [DerivedEdge("A", "uses", "B", [1], None)],
        project_path="/proj",
        cfg=NeuronConfig(distill_enabled=False),
    )
    assert result.added == 0


def test_distill_inserts_new_edge(tmp_path, monkeypatch) -> None:
    import simba.db as db
    import simba.neuron.schema  # ensure tables created

    _ = simba.neuron.schema
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(db, "get_db_path", lambda cwd=None: db_path)
    with db.connect():
        pass

    from simba.neuron.config import NeuronConfig
    from simba.neuron.derive import DerivedEdge
    from simba.neuron.distill import distill_edges

    result = distill_edges(
        [DerivedEdge("A", "transitively_uses", "C", [1, 2], None, 0.8)],
        project_path="/proj",
        cfg=NeuronConfig(distill_enabled=True),
    )
    assert result.added == 1
    assert result.duplicates == 0


def test_distill_deduplicates(tmp_path, monkeypatch) -> None:
    import simba.db as db
    import simba.neuron.schema

    _ = simba.neuron.schema
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(db, "get_db_path", lambda cwd=None: db_path)
    with db.connect():
        pass

    from simba.neuron.config import NeuronConfig
    from simba.neuron.derive import DerivedEdge
    from simba.neuron.distill import distill_edges

    cand = DerivedEdge("A", "transitively_uses", "C", [1, 2], None, 0.8)
    distill_edges([cand], project_path="/proj", cfg=NeuronConfig(distill_enabled=True))
    result2 = distill_edges(
        [cand], project_path="/proj", cfg=NeuronConfig(distill_enabled=True)
    )
    assert result2.duplicates == 1
