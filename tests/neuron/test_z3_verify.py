"""Sub-phase VERIFY: Z3 constraint encoding + UNSAT core (Task B.4)."""

from __future__ import annotations

import pytest


def test_sat_empty_graph() -> None:
    from simba.neuron.config import NeuronConfig
    from simba.neuron.z3_verify import run_verify

    result = run_verify("/proj", cfg=NeuronConfig(verify_enabled=False))
    assert result.satisfiable is True
    assert result.unsat_edge_ids == []


def test_build_z3_script_sat_no_conflict() -> None:
    from simba.neuron.z3_verify import build_z3_script

    edges = [
        {
            "id": 1,
            "subject": "A",
            "predicate": "uses",
            "object": "B",
            "valid_from": "2024-01-01T00:00:00Z",
            "valid_to": None,
        },
    ]
    script = build_z3_script(edges)
    assert "e1" in script
    assert "s.check()" in script
    assert "unsat_core" in script


def test_build_z3_script_detects_contradiction() -> None:
    """Two edges: A uses B and A does_not_use B — script should encode UNSAT."""
    from simba.neuron.z3_verify import build_z3_script

    edges = [
        {
            "id": 1,
            "subject": "A",
            "predicate": "uses",
            "object": "B",
            "valid_from": "2024-01-01T00:00:00Z",
            "valid_to": None,
        },
        {
            "id": 2,
            "subject": "A",
            "predicate": "does_not_use",
            "object": "B",
            "valid_from": "2024-01-01T00:00:00Z",
            "valid_to": None,
        },
    ]
    script = build_z3_script(edges)
    assert "Not(And(" in script or "Not(And" in script


def test_planted_contradiction_fixture(monkeypatch, tmp_path) -> None:
    """Regression fixture: verify detects a planted USES/DOES_NOT_USE pair."""
    import simba.kg.store

    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    simba.kg.store.kg_add("X", "uses", "Y", "test", project_path="/proj")
    simba.kg.store.kg_add("X", "does_not_use", "Y", "test", project_path="/proj")

    from simba.neuron.config import NeuronConfig
    from simba.neuron.z3_verify import run_verify

    try:
        import z3  # noqa: F401
    except ImportError:
        pytest.skip("z3 not installed")
    result = run_verify(
        "/proj",
        cfg=NeuronConfig(
            verify_enabled=True,
            contradiction_sample_size=50,
            verify_timeout_seconds=15,
        ),
    )
    assert result.satisfiable is False
    assert len(result.unsat_edge_ids) >= 2
