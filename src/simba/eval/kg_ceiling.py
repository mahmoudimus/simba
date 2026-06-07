"""Track B recall-ceiling diagnostic — the gate before building PPR.

For each case it compares two recall@k numbers over the labelled gold:

- **baseline**: the gold the live vector/RRF arm already retrieves (top-k), and
- **ceiling**: the gold retrievable if we *also* perfectly recovered every gold
  memory reachable through the corpus KG from the query's seed entities.

``ceiling - baseline`` is the *maximum* recall@k uplift any retrieval-time graph
lever (PPR, community, BFS-fold) could deliver — it's an upper bound, since it
assumes the graph step ranks all reachable gold into the top-k for free. If that
headroom is ~0 (the gold is already retrieved, as C1 found on LoCoMo), PPR cannot
move the needle and the finding is extraction/KG density, not ranking. Pure +
deterministic; the seed extractor and retriever are injected.
"""

from __future__ import annotations

import collections
import dataclasses
import tempfile
import typing

if typing.TYPE_CHECKING:
    from simba.eval.dataset import Dataset
    from simba.eval.kg_corpus import CorpusKG

EntitiesFn = typing.Callable[[str], typing.Iterable[str]]
EmbedFn = typing.Callable[[str], list[float]]


@dataclasses.dataclass
class CeilingReport:
    n_cases: int
    n_with_headroom: int
    density: float
    baseline_recall: float
    ceiling_recall: float
    net_new_fraction: float

    def to_dict(self) -> dict[str, typing.Any]:
        return dataclasses.asdict(self)


def case_ceiling(
    *,
    query: str,
    gold: typing.Iterable[str],
    topk: typing.Iterable[str],
    kg: CorpusKG,
    max_hops: int,
    entities_of: EntitiesFn,
) -> tuple[float, float]:
    """Return ``(baseline_recall, ceiling_recall)`` for one case.

    ``baseline`` = fraction of gold already in ``topk``; ``ceiling`` = fraction of
    gold in ``topk`` *or* graph-reachable from the query's seed entities.
    """
    gold_set = set(gold)
    if not gold_set:
        return (0.0, 0.0)
    topk_set = set(topk)
    baseline = len(gold_set & topk_set) / len(gold_set)
    reachable = kg.reachable_memories(entities_of(query), max_hops)
    ceiling = len(gold_set & (topk_set | reachable)) / len(gold_set)
    return (baseline, ceiling)


def aggregate_ceiling(
    rows: list[tuple[float, float]], *, density: float
) -> CeilingReport:
    """Aggregate per-case ``(baseline, ceiling)`` pairs into a report."""
    n = len(rows)
    if n == 0:
        return CeilingReport(0, 0, density, 0.0, 0.0, 0.0)
    baseline = sum(b for b, _ in rows) / n
    ceiling = sum(c for _, c in rows) / n
    headroom = sum(1 for b, c in rows if c > b + 1e-9)
    return CeilingReport(
        n_cases=n,
        n_with_headroom=headroom,
        density=density,
        baseline_recall=baseline,
        ceiling_recall=ceiling,
        net_new_fraction=ceiling - baseline,
    )


def ppr_topk_memories(
    kg: CorpusKG, seeds: typing.Iterable[str], *, budget: int, damping: float = 0.85
) -> set[str]:
    """The ``budget`` memories with the highest PPR mass, seeded at ``seeds``.

    A memory's score is the max PPR mass over the entities it contains. This is
    the *selective, budgeted* graph fold — the realistic Track B contribution, as
    opposed to the (non-discriminating) "all reachable" set.
    """
    import simba.kg.ppr as ppr

    mass = ppr.personalized_pagerank(kg.adjacency, seeds, damping=damping)
    if not mass:
        return set()
    mem_score: dict[str, float] = {}
    for ent, mems in kg.entity_memories.items():
        m = mass.get(ent, 0.0)
        if m <= 0.0:
            continue
        for mid in mems:
            if m > mem_score.get(mid, 0.0):
                mem_score[mid] = m
    ranked = sorted(mem_score, key=lambda mid: mem_score[mid], reverse=True)
    return set(ranked[:budget])


def run_ceiling(
    datasets: list[Dataset],
    *,
    embed_doc: EmbedFn,
    embed_query: EmbedFn,
    cfg: typing.Any,
    k: int = 10,
    max_hops: int = 2,
    budget: int = 10,
    damping: float = 0.85,
    categories: typing.Iterable[str] | None = None,
) -> dict[str, dict[str, typing.Any]]:
    """Per-category recall@k headroom for a graph fold over a benchmark.

    For each conversation: build the dense co-occurrence corpus KG, run the
    *real* hybrid retriever (no reranker — deterministic, weakest baseline → most
    generous headroom), and per case compute three recall@k numbers over the gold:

    - ``baseline``       — gold already in the vector top-k.
    - ``reach_ceiling``  — gold in top-k *or anywhere graph-reachable* (the
      reachability upper bound; ~1.0 on a near-complete graph → non-discriminating).
    - ``ppr_ceiling``    — gold in top-k *or in the budgeted PPR-top-N* (the
      realistic, selective fold — the number that actually decides Track B).

    The gap ``ppr_ceiling - baseline`` is the max recall@k uplift a PPR fold could
    deliver. ``categories`` filters to the axes of interest; ``None`` = all.
    """
    import simba.eval.kg_corpus as kgc
    import simba.eval.recall_adapter
    import simba.kg.entities

    wanted = set(categories) if categories else None
    by_cat: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
    densities: list[float] = []

    for dset in datasets:
        kg = kgc.build_corpus_kg(dset.corpus, kgc.cooccurrence_extract)
        densities.append(kg.density())
        with tempfile.TemporaryDirectory(prefix="simba-ceiling-") as td:
            retriever = simba.eval.recall_adapter.build_retriever(
                dset,
                cfg,
                embed_doc=embed_doc,
                embed_query=embed_query,
                data_dir=td,
                llm_client=None,  # pure hybrid baseline (no rerank), deterministic
            )
            for case in dset.cases:
                cat = case.intent or "?"
                if wanted is not None and cat not in wanted:
                    continue
                gold = set(case.relevant_ids)
                if not gold:
                    continue
                topk = set(retriever(case.query)[:k])
                seeds_raw = list(kgc.entities_of(case.query))
                seeds_norm = [simba.kg.entities.normalize_entity(s) for s in seeds_raw]
                reach = kg.reachable_memories(seeds_raw, max_hops)
                ppr_set = ppr_topk_memories(
                    kg, seeds_norm, budget=budget, damping=damping
                )
                by_cat[cat].append(
                    {
                        "baseline": len(gold & topk) / len(gold),
                        "reach_ceiling": len(gold & (topk | reach)) / len(gold),
                        "ppr_ceiling": len(gold & (topk | ppr_set)) / len(gold),
                    }
                )

    mean_density = sum(densities) / len(densities) if densities else 0.0
    out: dict[str, dict[str, typing.Any]] = {}
    for cat, rows in sorted(by_cat.items()):
        n = len(rows)

        def _mean(key: str, rows: list[dict[str, float]] = rows, n: int = n) -> float:
            return sum(r[key] for r in rows) / n if n else 0.0

        base = _mean("baseline")
        ppr_c = _mean("ppr_ceiling")
        out[cat] = {
            "n_cases": n,
            "density": mean_density,
            "baseline_recall": base,
            "reach_ceiling": _mean("reach_ceiling"),
            "ppr_ceiling": ppr_c,
            "ppr_net_new": ppr_c - base,
            "n_ppr_headroom": sum(
                1 for r in rows if r["ppr_ceiling"] > r["baseline"] + 1e-9
            ),
        }
    return out
