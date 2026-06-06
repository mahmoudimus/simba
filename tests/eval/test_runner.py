"""Tests for the eval runner (dataset + retriever -> report)."""

from __future__ import annotations

import pytest

import simba.eval.dataset as ds
import simba.eval.runner as runner

_DATASET = ds.Dataset(
    name="tiny",
    corpus=[
        ds.Memory(id="m1", content="a"),
        ds.Memory(id="m2", content="b"),
        ds.Memory(id="m3", content="c"),
    ],
    cases=[
        ds.EvalCase(id="c1", query="q1", relevant_ids=["m1"]),
        ds.EvalCase(id="c2", query="q2", relevant_ids=["m2"]),
    ],
)


def _retriever(query: str) -> list[str]:
    # Both queries return the same ranking; c1's gold is at rank 1, c2's at rank 2.
    return ["m1", "m2", "m3"]


def test_report_shape() -> None:
    rep = runner.run_eval(_DATASET, _retriever, ks=(1, 3))
    assert rep.dataset_name == "tiny"
    assert rep.n_cases == 2
    assert rep.ks == (1, 3)
    assert len(rep.per_case) == 2


def test_aggregate_means() -> None:
    rep = runner.run_eval(_DATASET, _retriever, ks=(1, 3))
    agg = rep.aggregate
    assert agg["recall@1"] == pytest.approx(0.5)
    assert agg["recall@3"] == pytest.approx(1.0)
    assert agg["hit@1"] == pytest.approx(0.5)
    assert agg["mrr"] == pytest.approx(0.75)


def test_per_case_metrics_recorded() -> None:
    rep = runner.run_eval(_DATASET, _retriever, ks=(1,))
    by_id = {c.case_id: c for c in rep.per_case}
    assert by_id["c1"].metrics["recall@1"] == pytest.approx(1.0)
    assert by_id["c2"].metrics["recall@1"] == pytest.approx(0.0)
    assert by_id["c2"].ranked[:2] == ["m1", "m2"]


def test_to_dict_is_serializable() -> None:
    import json

    rep = runner.run_eval(_DATASET, _retriever, ks=(1, 3))
    blob = json.dumps(rep.to_dict())
    assert "recall@1" in blob
    assert "tiny" in blob


def test_empty_dataset_safe() -> None:
    empty = ds.Dataset(name="e", corpus=[], cases=[])
    rep = runner.run_eval(empty, _retriever, ks=(1,))
    assert rep.n_cases == 0
    assert rep.aggregate["recall@1"] == 0.0


def test_run_eval_split_filters_cases() -> None:
    import simba.eval.splits as sp

    cases = [
        ds.EvalCase(id="a", query="qa", relevant_ids=["m1"], split="dev"),
        ds.EvalCase(id="b", query="qb", relevant_ids=["m2"], split="test"),
        ds.EvalCase(id="c", query="qc", relevant_ids=["m3"], split="test"),
    ]
    dset = ds.Dataset(
        name="s",
        corpus=[ds.Memory(id=f"m{i}", content="x") for i in (1, 2, 3)],
        cases=cases,
    )
    rep = runner.run_eval(dset, lambda q: ["m1"], ks=(1,), split="test")
    assert rep.n_cases == 2  # only the two 'test' cases scored
    assert {c.case_id for c in rep.per_case} == {"b", "c"}
    # sanity: select agrees
    assert len(sp.select(cases, "test")) == 2


# --- B4: per-query latency ---------------------------------------------------


def test_case_result_has_latency_ms() -> None:
    import time

    def slow_retriever(q: str) -> list[str]:
        time.sleep(0.01)  # 10ms
        return ["m1"]

    rep = runner.run_eval(_DATASET, slow_retriever, ks=(1,))
    for case in rep.per_case:
        assert case.latency_ms >= 5.0  # at least 5ms (generous lower bound)
    assert case.latency_ms < 5000.0  # sanity upper bound


def test_aggregate_has_p50_p95() -> None:
    rep = runner.run_eval(_DATASET, _retriever, ks=(1,))
    assert "p50_ms" in rep.aggregate
    assert "p95_ms" in rep.aggregate
    assert rep.aggregate["p50_ms"] >= 0.0
    assert rep.aggregate["p95_ms"] >= rep.aggregate["p50_ms"]


def test_to_dict_includes_latency_ms() -> None:
    rep = runner.run_eval(_DATASET, _retriever, ks=(1,))
    d = rep.per_case[0].to_dict()
    assert "latency_ms" in d


def test_percentile_correctness() -> None:
    from simba.eval.runner import _percentile

    assert _percentile([], 50) == 0.0
    assert _percentile([10.0], 50) == pytest.approx(10.0)
    assert _percentile([10.0, 20.0, 30.0], 50) == pytest.approx(20.0)
    assert _percentile([10.0, 20.0, 30.0], 100) == pytest.approx(30.0)
    assert _percentile([10.0, 20.0, 30.0], 0) == pytest.approx(10.0)
