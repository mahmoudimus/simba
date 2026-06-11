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
    hyde_text: str = ""  # the string to embed for the 2nd arm: the LLM answer
    # when hyde_mode=="llm" and non-empty, else expansion_terms (keyword fallback)


def plan_recall(
    query: str,
    cfg: typing.Any,
    *,
    min_similarity: float | None = None,
    max_results: int | None = None,
    llm_client: typing.Any = None,
    hyde_cache: typing.Any = None,
) -> RecallPlan:
    """Derive the recall parameters for ``query`` under ``cfg``.

    An explicit ``min_similarity`` / ``max_results`` always wins (the client
    escape hatch); otherwise the floor + widths are picked by query intent.

    ``llm_client`` + ``hyde_cache`` drive ``hyde_mode == "llm"``: with a cache
    wired (daemon) a miss serves the keyword fallback now and the route warms the
    cache off the hot path; without a cache (eval) the LLM answer is generated
    inline. Any failure collapses to the keyword string, then to ``""``.
    """
    # Cosine floor + mode.
    if min_similarity is not None:
        min_sim = min_similarity
        mode = "explicit"
    elif cfg.intent_aware:
        mode = simba.memory.intent.classify(query)
        min_sim = cfg.min_similarity_broad if mode == "broad" else cfg.min_similarity
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
    candidate_pool = cfg.fts_candidate_pool_broad if broad else cfg.fts_candidate_pool

    # Count queries are recall-breadth-bound: widen the pool + context further than
    # "broad" (a pointwise reranker can't recover gold a narrow pool never fetched).
    # The explicit max_results escape hatch still wins (max_results is not None).
    if (
        getattr(cfg, "count_depth_enabled", False)
        and max_results is None
        and simba.memory.intent.is_count(query)
    ):
        mode = "count"
        max_res = cfg.count_context_k
        candidate_pool = cfg.count_candidate_pool_n

    # Multi-arm HyDE (opt-in): a 2nd vector arm over the focused-term string.
    expansion_terms = ""
    if getattr(cfg, "hybrid_enabled", False) and cfg.expansion_enabled:
        terms = simba.memory.keywords.focus_terms(query, max_terms=cfg.fts_max_terms)
        if terms:
            expansion_terms = " ".join(terms)

    # HyDE text for the 2nd arm. Default = the keyword fallback (current behavior).
    # In "llm" mode: a cache hit serves the answer; a cache miss serves the keyword
    # fallback now (the daemon route warms the cache off the hot path); with no
    # cache wired (eval) the answer is generated inline.
    hyde_text = expansion_terms
    if getattr(cfg, "hyde_mode", "keyword") == "llm" and llm_client is not None:
        key = hyde_cache.signature(query) if hyde_cache is not None else None
        cached = hyde_cache.get(key) if hyde_cache is not None and key else None
        if cached is not None:
            hyde_text = cached
        elif hyde_cache is None:
            from simba.memory.hyde import hypothetical_answer

            text = hypothetical_answer(query, llm_client)
            hyde_text = text if text else expansion_terms
        # else: daemon cache miss → keep keyword fallback; route warms the cache.

    return RecallPlan(
        min_similarity=min_sim,
        max_results=max_res,
        candidate_pool=candidate_pool,
        mode=mode,
        expansion_terms=expansion_terms,
        hyde_text=hyde_text,
    )
