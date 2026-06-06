"""Contradiction detection + KG density (Task B.9)."""

from __future__ import annotations

import pytest


def test_verify_finds_planted_contradiction(planted_contradiction) -> None:
    try:
        import z3  # noqa: F401
    except ImportError:
        pytest.skip("z3 not installed")
    edge_ids, project_path = planted_contradiction
    _ = edge_ids
    from simba.neuron.config import NeuronConfig
    from simba.neuron.z3_verify import run_verify

    result = run_verify(
        project_path,
        cfg=NeuronConfig(
            verify_enabled=True,
            contradiction_sample_size=10,
            verify_timeout_seconds=15,
        ),
    )
    assert result.satisfiable is False
    assert len(result.unsat_edge_ids) >= 2


def test_kg_density_baseline(planted_contradiction) -> None:
    _, project_path = planted_contradiction
    from simba.kg.store import kg_density

    metrics = kg_density(project_path)
    assert metrics["edge_count"] == 2
    assert metrics["node_count"] >= 2  # ToolA, LibB
    assert 0.0 <= metrics["density"] <= 1.0
    assert metrics["derived_ratio"] == 0.0  # no derived edges yet
