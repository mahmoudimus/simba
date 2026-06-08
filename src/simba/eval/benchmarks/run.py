"""Run a list of per-conversation Datasets through the recall harness and
aggregate recall@k / MRR overall and per question-category.
"""

from __future__ import annotations

import tempfile
import typing

import simba.eval.recall_adapter
import simba.eval.runner
from simba.eval.runner import _percentile

if typing.TYPE_CHECKING:
    from simba.eval.dataset import Dataset

EmbedFn = typing.Callable[[str], list[float]]


def run_recall(
    datasets: list[Dataset],
    *,
    embed_doc: EmbedFn,
    embed_query: EmbedFn,
    cfg: typing.Any,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    llm_client: typing.Any = None,
) -> dict[str, typing.Any]:
    """Score recall@k per conversation, aggregate overall + by category (intent)."""
    metric_names = (
        [f"recall@{k}" for k in ks]
        + [f"bridge_recall@{k}" for k in ks]
        + [f"ndcg@{k}" for k in ks]
        + ["mrr"]
    )
    by_cat: dict[str, list[dict[str, float]]] = {}
    overall: list[dict[str, float]] = []
    all_latencies: list[float] = []

    for dset in datasets:
        cat_of = {c.id: (c.intent or "?") for c in dset.cases}
        with tempfile.TemporaryDirectory(prefix="simba-bench-") as td:
            retriever = simba.eval.recall_adapter.build_retriever(
                dset,
                cfg,
                embed_doc=embed_doc,
                embed_query=embed_query,
                data_dir=td,
                llm_client=llm_client,
            )
            rep = simba.eval.runner.run_eval(dset, retriever, ks=ks)
        for case in rep.per_case:
            overall.append(case.metrics)
            by_cat.setdefault(cat_of.get(case.case_id, "?"), []).append(case.metrics)
            all_latencies.append(case.latency_ms)

    def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
        n = len(rows)
        return {m: (sum(r[m] for r in rows) / n if n else 0.0) for m in metric_names}

    return {
        "n_conversations": len(datasets),
        "n_cases": len(overall),
        "overall": _mean(overall),
        "by_category": {
            cat: {"n": len(rows), **_mean(rows)} for cat, rows in sorted(by_cat.items())
        },
        "latency": {
            "p50_ms": _percentile(all_latencies, 50),
            "p95_ms": _percentile(all_latencies, 95),
            "n": len(all_latencies),
        },
    }
