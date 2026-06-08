"""Eval runner: score a retriever against a dataset.

A *retriever* is any ``Callable[[str], list[str]]`` mapping a query to a ranked
list of memory ids (best first). It is given only the query — never the gold
ids — so the harness can't accidentally reward a cheating retriever. Aggregate
metrics are simple means over cases.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import simba.eval.metrics as metrics

if TYPE_CHECKING:
    from simba.eval.dataset import Dataset

Retriever = Callable[[str], list[str]]

_DEFAULT_KS = (1, 3, 5, 10)


@dataclasses.dataclass
class CaseResult:
    case_id: str
    query: str
    ranked: list[str]
    metrics: dict[str, float]
    latency_ms: float = 0.0  # wall-clock ms for the retriever call

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "ranked": self.ranked,
            "metrics": self.metrics,
            "latency_ms": self.latency_ms,
        }


@dataclasses.dataclass
class EvalReport:
    dataset_name: str
    n_cases: int
    ks: tuple[int, ...]
    aggregate: dict[str, float]
    per_case: list[CaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "n_cases": self.n_cases,
            "ks": list(self.ks),
            "aggregate": self.aggregate,
            "per_case": [c.to_dict() for c in self.per_case],
        }


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of values. Returns 0.0 when empty."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (p / 100) * (len(sorted_v) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (idx - lo)


def _case_metrics(
    ranked: Sequence[str], relevant: set[str], ks: tuple[int, ...]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = metrics.recall_at_k(ranked, relevant, k)
        out[f"bridge_recall@{k}"] = metrics.bridge_recall_at_k(ranked, relevant, k)
        out[f"precision@{k}"] = metrics.precision_at_k(ranked, relevant, k)
        out[f"hit@{k}"] = metrics.hit_at_k(ranked, relevant, k)
        out[f"ndcg@{k}"] = metrics.ndcg_at_k(ranked, relevant, k)
    out["mrr"] = metrics.reciprocal_rank(ranked, relevant)
    return out


def run_eval(
    dataset: Dataset,
    retriever: Retriever,
    ks: tuple[int, ...] = _DEFAULT_KS,
    *,
    keep_top: int = 20,
    split: str | None = None,
    test_ratio: float = 0.5,
) -> EvalReport:
    """Run ``retriever`` over the cases (optionally a dev/test split) and report."""
    import simba.eval.splits

    cases = simba.eval.splits.select(dataset.cases, split, test_ratio=test_ratio)
    per_case: list[CaseResult] = []
    for case in cases:
        t0 = time.perf_counter()
        ranked = list(retriever(case.query))
        lat = (time.perf_counter() - t0) * 1000
        cm = _case_metrics(ranked, set(case.relevant_ids), ks)
        per_case.append(
            CaseResult(
                case_id=case.id,
                query=case.query,
                ranked=ranked[:keep_top],
                metrics=cm,
                latency_ms=lat,
            )
        )

    metric_names = [f"recall@{k}" for k in ks]
    metric_names += [f"bridge_recall@{k}" for k in ks]
    metric_names += [f"precision@{k}" for k in ks]
    metric_names += [f"hit@{k}" for k in ks]
    metric_names += [f"ndcg@{k}" for k in ks]
    metric_names.append("mrr")

    n = len(per_case)
    aggregate = {
        name: (sum(c.metrics[name] for c in per_case) / n if n else 0.0)
        for name in metric_names
    }
    latencies = [c.latency_ms for c in per_case]
    aggregate["p50_ms"] = _percentile(latencies, 50)
    aggregate["p95_ms"] = _percentile(latencies, 95)

    return EvalReport(
        dataset_name=dataset.name,
        n_cases=n,
        ks=ks,
        aggregate=aggregate,
        per_case=per_case,
    )
