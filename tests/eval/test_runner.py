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
