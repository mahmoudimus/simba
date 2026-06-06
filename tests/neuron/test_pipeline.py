"""distillation_pass orchestrator (Task B.8)."""

from __future__ import annotations


def test_disabled_returns_disabled() -> None:
    from simba.neuron.config import NeuronConfig
    from simba.neuron.pipeline import distillation_pass

    result = distillation_pass(project_path="/proj", cfg=NeuronConfig(enabled=False))
    assert result.status == "disabled"


def test_full_pipeline_sat_path(monkeypatch) -> None:
    """All sub-phases called; SAT path — no revise."""
    from simba.neuron import pipeline as p
    from simba.neuron.config import NeuronConfig
    from simba.neuron.derive import DerivedEdge, DeriveResult
    from simba.neuron.distill import DistillResult
    from simba.neuron.induce import InduceResult
    from simba.neuron.z3_verify import VerifyResult

    monkeypatch.setattr(
        p,
        "run_derive",
        lambda *a, **k: DeriveResult(
            candidates=[DerivedEdge("A", "r", "B", [1], None)], edges_fed=1
        ),
    )
    monkeypatch.setattr(
        p,
        "run_verify",
        lambda *a, **k: VerifyResult(
            satisfiable=True, unsat_edge_ids=[], checked_edges=1
        ),
    )
    monkeypatch.setattr(p, "distill_edges", lambda *a, **k: DistillResult(added=1))
    monkeypatch.setattr(p, "induce_rules", lambda *a, **k: InduceResult(promoted=0))

    result = p.distillation_pass(project_path="/proj", cfg=NeuronConfig(enabled=True))
    assert result.status == "ok"
    assert result.candidates == 1
    assert result.distilled == 1
    assert result.dormant == 0


def test_full_pipeline_unsat_path(monkeypatch) -> None:
    """UNSAT path triggers revise."""
    from simba.neuron import pipeline as p
    from simba.neuron.config import NeuronConfig
    from simba.neuron.derive import DeriveResult
    from simba.neuron.distill import DistillResult
    from simba.neuron.induce import InduceResult
    from simba.neuron.revise import ReviseResult
    from simba.neuron.z3_verify import VerifyResult

    monkeypatch.setattr(
        p, "run_derive", lambda *a, **k: DeriveResult(candidates=[], edges_fed=5)
    )
    monkeypatch.setattr(
        p,
        "run_verify",
        lambda *a, **k: VerifyResult(
            satisfiable=False, unsat_edge_ids=[1, 2], checked_edges=5
        ),
    )
    revise_called = []
    monkeypatch.setattr(
        p,
        "revise_unsat_core",
        lambda ids, **k: (
            revise_called.append(ids),
            ReviseResult(dormant_edge_ids=[1], retained_edge_ids=[2]),
        )[1],
    )
    monkeypatch.setattr(p, "distill_edges", lambda *a, **k: DistillResult())
    monkeypatch.setattr(p, "induce_rules", lambda *a, **k: InduceResult())

    result = p.distillation_pass(project_path="/proj", cfg=NeuronConfig(enabled=True))
    assert result.satisfiable is False
    assert result.unsat_core_size == 2
    assert result.dormant == 1
    assert revise_called == [[1, 2]]
