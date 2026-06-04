"""Hybrid recall (L3): fuse the vector and keyword arms via RRF.

``rrf_fuse`` is a pure function over two already-ranked lists; ``hybrid_search``
orchestrates the two arms (LanceDB cosine + the SQLite FTS5 mirror) and returns
the fused top-k.  The keyword arm runs in a worker thread (sync SQLite) and is
fully defensive — any failure degrades the result to vector-only.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import typing

import simba.memory.fts
import simba.memory.keywords
import simba.memory.llm_rerank
import simba.memory.scoring
import simba.memory.vector_db


def _from_vector(item: dict[str, typing.Any]) -> dict[str, typing.Any]:
    return {
        "id": item.get("id"),
        "type": item.get("type"),
        "content": item.get("content"),
        "context": item.get("context", ""),
        "similarity": item.get("similarity", 0.0),
        "confidence": item.get("confidence", 0.0),
        "createdAt": item.get("createdAt"),
        "projectPath": item.get("projectPath", ""),
        "sessionSource": item.get("sessionSource", ""),
    }


def _from_keyword(item: dict[str, typing.Any]) -> dict[str, typing.Any]:
    # Keyword-only hits have no cosine score; similarity defaults to 0.0.
    return {
        "id": item.get("memory_id"),
        "type": item.get("type"),
        "content": item.get("content"),
        "context": item.get("context", ""),
        "similarity": item.get("similarity", 0.0),
        "confidence": item.get("confidence", 0.0),
        "createdAt": item.get("createdAt"),
        "projectPath": item.get("projectPath", ""),
    }


def rrf_fuse(
    vector_results: list[dict[str, typing.Any]],
    keyword_results: list[dict[str, typing.Any]],
    *,
    k: int = 60,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
    extra_vector_results: list[dict[str, typing.Any]] | None = None,
    kg_results: list[dict[str, typing.Any]] | None = None,
    kg_weight: float = 1.0,
) -> list[dict[str, typing.Any]]:
    """Reciprocal Rank Fusion of the ranked arms, deduped by memory id.

    ``score(id) = Σ_arm weight_arm / (k + rank_arm(id))`` with 1-based ranks.
    Vector arms are folded first, so when an id appears in several arms the
    richer vector record (real cosine ``similarity``) is kept while the score
    still accumulates contributions from each.  ``extra_vector_results`` is an
    optional 2nd vector arm (the HyDE expansion arm), weighted like the primary.
    Returns records ordered by fused score (desc), each carrying an ``rrf_score``.
    """
    scores: dict[str, float] = {}
    records: dict[str, dict[str, typing.Any]] = {}

    for arm in (vector_results, extra_vector_results or []):
        for rank, item in enumerate(arm, start=1):
            rid = item.get("id")
            if not rid:
                continue
            scores[rid] = scores.get(rid, 0.0) + vector_weight / (k + rank)
            records.setdefault(rid, _from_vector(item))

    for rank, item in enumerate(keyword_results, start=1):
        rid = item.get("memory_id")
        if not rid:
            continue
        scores[rid] = scores.get(rid, 0.0) + keyword_weight / (k + rank)
        records.setdefault(rid, _from_keyword(item))

    # KG arm: memories bridged in via the knowledge graph (vector-shaped rows,
    # already ranked by the KG walk). Folded like a vector arm so a bridged hit
    # that no direct arm surfaced still earns rank in the fusion.
    for rank, item in enumerate(kg_results or [], start=1):
        rid = item.get("id")
        if not rid:
            continue
        scores[rid] = scores.get(rid, 0.0) + kg_weight / (k + rank)
        records.setdefault(rid, _from_vector(item))

    ordered = sorted(records.values(), key=lambda r: scores[r["id"]], reverse=True)
    for r in ordered:
        r["rrf_score"] = round(scores[r["id"]], 6)
    return ordered


def _keyword_arm(
    fts_path: typing.Any,
    query_text: str,
    project_path: str | None,
    types: list[str] | None,
    limit: int,
) -> list[dict[str, typing.Any]]:
    """Open a per-call connection (thread-affinity safe) and run the bm25 search."""
    with simba.memory.fts.connect(fts_path):
        return simba.memory.fts.search(
            query_text,
            project_path=project_path,
            types=types,
            limit=limit,
        )


async def hybrid_search(
    table: typing.Any,
    fts_path: typing.Any,
    embedding: list[float],
    query_text: str,
    *,
    min_similarity: float,
    max_results: int,
    filters: dict[str, typing.Any] | None,
    cfg: typing.Any,
    candidate_pool: int | None = None,
    extra_embedding: list[float] | None = None,
    llm_client: typing.Any = None,
    rerank_cache: typing.Any = None,
    bg_tasks: set | None = None,
    kg_arm: typing.Any = None,
) -> list[dict[str, typing.Any]]:
    """Run both arms and return the RRF-fused top ``max_results`` memories.

    The vector arm keeps the ``min_similarity`` floor; the keyword arm is *not*
    cosine-gated (that is what widens coverage).  Both arms share the same
    project/type scope via ``filters``.  ``candidate_pool`` overrides the
    per-arm fetch size (broad queries pass a wider pool); it defaults to
    ``cfg.fts_candidate_pool``.  Never raises on the keyword side.
    """
    filters = filters or {}
    pool = candidate_pool if candidate_pool is not None else cfg.fts_candidate_pool
    candidate_pool = max(max_results, pool)

    vector_results = await simba.memory.vector_db.search_memories(
        table, embedding, min_similarity, candidate_pool, filters
    )

    # Optional 2nd vector arm (HyDE expansion): same floor/scope, separate query
    # embedding (the focused-term string), folded into RRF alongside the primary.
    extra_vector_results: list[dict[str, typing.Any]] | None = None
    if extra_embedding is not None:
        extra_vector_results = await simba.memory.vector_db.search_memories(
            table, extra_embedding, min_similarity, candidate_pool, filters
        )

    # The keyword arm is fed high-signal terms, not the whole query: a long
    # thinking block would otherwise OR together ~200 tokens and bm25 would
    # match almost anything.  No usable terms -> skip the arm (vector-only).
    keyword_results: list[dict[str, typing.Any]] = []
    kw_terms = simba.memory.keywords.focus_terms(
        query_text, max_terms=cfg.fts_max_terms
    )
    if fts_path and kw_terms:
        try:
            keyword_results = await asyncio.to_thread(
                _keyword_arm,
                fts_path,
                " ".join(kw_terms),
                filters.get("projectPath"),
                filters.get("types"),
                candidate_pool,
            )
        except Exception:
            keyword_results = []

    # KG arm (multi-hop bridge): seed from the strongest vector hits, hand them to
    # the injected kg_arm (it walks the KG and returns the bridged memory rows).
    # Fail-open: any error / no KG leaves the vector+keyword fusion untouched.
    kg_results: list[dict[str, typing.Any]] | None = None
    if getattr(cfg, "kg_recall_enabled", False) and kg_arm is not None:
        seed_ids = [
            r["id"]
            for r in vector_results[: getattr(cfg, "kg_recall_seed_top_n", 5)]
            if r.get("id")
        ]
        if seed_ids:
            with contextlib.suppress(Exception):
                kg_results = await kg_arm(seed_ids)

    fused = rrf_fuse(
        vector_results,
        keyword_results,
        k=cfg.rrf_k,
        vector_weight=cfg.vector_weight,
        keyword_weight=cfg.keyword_weight,
        extra_vector_results=extra_vector_results,
        kg_results=kg_results,
        kg_weight=getattr(cfg, "kg_recall_weight", 1.0),
    )

    # Optional composite re-scoring: blend RRF relevance with recency +
    # importance over the full fused candidate set, then truncate.
    if getattr(cfg, "scoring_enabled", False):
        fused = simba.memory.scoring.composite_rescore(
            fused, cfg=cfg, now=time.time()
        )

    # Optional LLM rerank of the candidate pool (cross-encoder role) before
    # truncation. Two modes, both fail-open:
    #   - cache wired (daemon): NON-BLOCKING — serve the fast order, rerank off
    #     the hot path, cache the result keyed by (query, candidate-set).
    #   - no cache (eval/CLI): synchronous rerank in a worker thread.
    if getattr(cfg, "llm_rerank_enabled", False) and llm_client is not None:
        max_cands = getattr(cfg, "llm_rerank_candidates", 20)
        if rerank_cache is not None:
            pool_ids = [r.get("id") for r in fused]
            key = rerank_cache.signature(query_text, pool_ids)
            cached = rerank_cache.get(key)
            if cached is not None:
                fused = simba.memory.llm_rerank.reorder_by_ids(fused, cached)
            elif bg_tasks is not None:
                task = asyncio.create_task(
                    _bg_rerank(
                        rerank_cache,
                        key,
                        query_text,
                        list(fused),
                        llm_client,
                        max_cands,
                    )
                )
                bg_tasks.add(task)
                task.add_done_callback(bg_tasks.discard)
            # miss with no task registry → serve the fast order unchanged
        else:
            with contextlib.suppress(Exception):
                fused = await asyncio.to_thread(
                    simba.memory.llm_rerank.rerank,
                    query_text,
                    fused,
                    client=llm_client,
                    max_candidates=max_cands,
                )

    return fused[:max_results]


async def _bg_rerank(
    cache: typing.Any,
    key: str,
    query: str,
    pool: list[dict[str, typing.Any]],
    client: typing.Any,
    max_candidates: int,
) -> None:
    """Rerank ``pool`` off the hot path and store the id order in ``cache``."""
    with contextlib.suppress(Exception):
        reordered = await asyncio.to_thread(
            simba.memory.llm_rerank.rerank,
            query,
            pool,
            client=client,
            max_candidates=max_candidates,
        )
        cache.put(key, [r.get("id") for r in reordered])
