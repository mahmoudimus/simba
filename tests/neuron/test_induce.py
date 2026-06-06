"""Sub-phase INDUCE: promote recurring patterns to kg_rules (Task B.7)."""

from __future__ import annotations


def test_induce_disabled_returns_empty() -> None:
    from simba.neuron.config import NeuronConfig
    from simba.neuron.induce import induce_rules

    result = induce_rules(project_path="/proj", cfg=NeuronConfig(induce_enabled=False))
    assert result.promoted == 0


def test_induce_promotes_frequent_pattern(tmp_path, monkeypatch) -> None:
    import json
    import sqlite3
    import time

    import simba.db as db
    import simba.neuron.schema

    _ = simba.neuron.schema
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(db, "get_db_path", lambda cwd=None: db_path)
    with db.connect():
        pass

    # Manually seed kg_derived_edges with 4 rows for rule_id=1
    conn = sqlite3.connect(str(db_path))
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for i in range(4):
        conn.execute(
            "INSERT OR IGNORE INTO kg_derived_edges "
            "(subject, predicate, object, proof, source_edge_ids, rule_id, "
            "confidence, valid_from, project_path, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                f"A{i}",
                "transitively_uses",
                f"C{i}",
                "derived:1",
                json.dumps([i, i + 1]),
                1,
                0.8,
                now,
                "/proj",
                now,
            ),
        )
    conn.commit()
    conn.close()

    from simba.neuron.config import NeuronConfig
    from simba.neuron.induce import induce_rules

    result = induce_rules(
        project_path="/proj",
        cfg=NeuronConfig(
            induce_enabled=True,
            induce_min_activations=3,
            induce_min_confidence=0.7,
        ),
    )
    assert result.promoted >= 1


def test_induce_skips_below_threshold(tmp_path, monkeypatch) -> None:
    import simba.db as db
    import simba.neuron.schema

    _ = simba.neuron.schema
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(db, "get_db_path", lambda cwd=None: db_path)
    with db.connect():
        pass

    from simba.neuron.config import NeuronConfig
    from simba.neuron.induce import induce_rules

    # No rows in kg_derived_edges → nothing to promote
    result = induce_rules(
        project_path="/proj",
        cfg=NeuronConfig(induce_enabled=True, induce_min_activations=3),
    )
    assert result.promoted == 0
    assert result.below_threshold == 0
