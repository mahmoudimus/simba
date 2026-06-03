"""Pure recall planning: turn a query + config into recall parameters.

This is the single source of truth for the intent-aware floor, broad-query
widening, and the (opt-in) HyDE expansion-term selection. Both the daemon's
``/recall`` route and the eval harness's recall adapter call it, so the
benchmark measures exactly the production decision logic — no drift.
"""

from __future__ import annotations

import dataclasses
import typing

import simba.memory.intent
import simba.memory.keywords


@dataclasses.dataclass
class RecallPlan:
    min_similarity: float
    max_results: int
    candidate_pool: int
    mode: str  # "explicit" | "broad" | "precise"
    expansion_terms: str  # focused-term string for the 2nd HyDE arm; "" if none


def plan_recall(
    query: str,
    cfg: typing.Any,
    *,
    min_similarity: float | None = None,
    max_results: int | None = None,
) -> RecallPlan:
    """Derive the recall parameters for ``query`` under ``cfg``.

    An explicit ``min_similarity`` / ``max_results`` always wins (the client
    escape hatch); otherwise the floor + widths are picked by query intent.
    """
    # Cosine floor + mode.
    if min_similarity is not None:
        min_sim = min_similarity
        mode = "explicit"
    elif cfg.intent_aware:
        mode = simba.memory.intent.classify(query)
        min_sim = (
            cfg.min_similarity_broad if mode == "broad" else cfg.min_similarity
        )
    else:
        min_sim = cfg.min_similarity
        mode = "precise"

    # Broad queries widen the net: more results + a larger RRF candidate pool.
    broad = mode == "broad"
    if max_results is not None:
        max_res = max_results
    elif broad:
        max_res = cfg.max_results_broad
    else:
        max_res = cfg.max_results
    candidate_pool = (
        cfg.fts_candidate_pool_broad if broad else cfg.fts_candidate_pool
    )

    # Multi-arm HyDE (opt-in): a 2nd vector arm over the focused-term string.
    expansion_terms = ""
    if getattr(cfg, "hybrid_enabled", False) and cfg.expansion_enabled:
        terms = simba.memory.keywords.focus_terms(query, max_terms=cfg.fts_max_terms)
        if terms:
            expansion_terms = " ".join(terms)

    return RecallPlan(
        min_similarity=min_sim,
        max_results=max_res,
        candidate_pool=candidate_pool,
        mode=mode,
        expansion_terms=expansion_terms,
    )
