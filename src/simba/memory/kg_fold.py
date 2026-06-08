"""Fold PPR-ranked graph evidence into the RRF candidate set (Track B).

After ``rrf_fuse``, a personalized-PageRank pass over the KG (seeded by the
query's entities) yields memories ranked by graph proximity. ``ppr_fold`` merges
those as a **third RRF arm** — each folded memory gains ``weight / (rrf_k + rank)``
on its fusion score — so a graph-surfaced memory competes for the top-k instead of
being appended below it. Memories already in the fused set are *boosted*; new ones
are materialized from a record lookup (the corpus in eval, a LanceDB fetch on the
live path). Unmaterializable ids are skipped.

C1 folded raw BFS neighbors with no ranking and lost to displacement; here the
contribution is mass-ranked and budget-capped, and folds *before* composite
rescore + the reranker, so the proven ranker still orders the assembled set.
Pure + deterministic; the PPR ranking is computed by the caller.
"""

from __future__ import annotations

import typing


def ppr_fold(
    fused: list[dict[str, typing.Any]],
    *,
    ppr_ranked_ids: list[str],
    record_lookup: dict[str, dict[str, typing.Any]],
    rrf_k: int,
    weight: float,
) -> list[dict[str, typing.Any]]:
    """Merge PPR-ranked ids into ``fused`` as an RRF arm and re-sort by score.

    ``fused`` records carry ``rrf_score`` (from ``rrf_fuse``). Each id in
    ``ppr_ranked_ids`` (already ordered by PPR mass) contributes
    ``weight / (rrf_k + rank)`` to its score; ids absent from ``fused`` are pulled
    from ``record_lookup`` (skipped if missing). Returns the re-ranked list.
    """
    scores = {r["id"]: r.get("rrf_score", 0.0) for r in fused}
    records = {r["id"]: r for r in fused}

    for rank, mid in enumerate(ppr_ranked_ids, start=1):
        contrib = weight / (rrf_k + rank)
        if mid in records:
            scores[mid] = scores.get(mid, 0.0) + contrib
        else:
            rec = record_lookup.get(mid)
            if rec is None:
                continue  # can't materialize → don't fold a phantom id
            records[mid] = dict(rec)
            scores[mid] = scores.get(mid, 0.0) + contrib

    ordered = sorted(records.values(), key=lambda r: scores[r["id"]], reverse=True)
    for r in ordered:
        r["rrf_score"] = round(scores[r["id"]], 6)
    return ordered
