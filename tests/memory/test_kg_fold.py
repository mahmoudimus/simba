"""The PPR fold: graph-surfaced memories enter the RRF candidate set as a third
arm (score += weight/(rrf_k + ppr_rank)) so they compete for the top-k, rather
than being appended below it. Pure + deterministic."""

from __future__ import annotations

import simba.memory.kg_fold as kgfold


def test_fold_boosts_existing_and_inserts_new():
    fused = [
        {"id": "v1", "rrf_score": 0.05, "content": "a"},
        {"id": "v2", "rrf_score": 0.02, "content": "b"},
    ]
    lookup = {"g1": {"id": "g1", "content": "gold", "context": ""}}
    out = kgfold.ppr_fold(
        fused,
        ppr_ranked_ids=["g1", "v2"],  # g1 new (rank1), v2 existing (rank2)
        record_lookup=lookup,
        rrf_k=20,
        weight=1.0,
    )
    ids = [r["id"] for r in out]
    # v2: 0.02 + 1/22 = 0.0655 → above v1 (0.05); g1: 1/21 = 0.0476 → last.
    assert ids == ["v2", "v1", "g1"]
    assert len(out) == 3  # g1 materialized from the lookup


def test_fold_drops_unmaterializable_ids():
    fused = [{"id": "v1", "rrf_score": 0.05}]
    out = kgfold.ppr_fold(
        fused,
        ppr_ranked_ids=["ghost"],  # not in lookup → can't be added
        record_lookup={},
        rrf_k=20,
        weight=1.0,
    )
    assert [r["id"] for r in out] == ["v1"]


def test_fold_zero_weight_preserves_order():
    fused = [{"id": "v1", "rrf_score": 0.05}, {"id": "v2", "rrf_score": 0.02}]
    out = kgfold.ppr_fold(
        fused, ppr_ranked_ids=["v2"], record_lookup={}, rrf_k=20, weight=0.0
    )
    assert [r["id"] for r in out] == ["v1", "v2"]
