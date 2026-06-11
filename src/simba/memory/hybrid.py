"""Hybrid recall (L3): fuse the vector and keyword arms via RRF.

``rrf_fuse`` is a pure function over two already-ranked lists; ``hybrid_search``
orchestrates the two arms (LanceDB cosine + the SQLite FTS5 mirror) and returns
the fused top-k.  The keyword arm runs in a worker thread (sync SQLite) and is
fully defensive — any failure degrades the result to vector-only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import typing

import simba.db
import simba.kg.ppr
import simba.memory.entity_bridge
import simba.memory.fts
import simba.memory.keywords
import simba.memory.kg_fold
import simba.memory.llm_rerank
import simba.memory.reranker
import simba.memory.scoring
import simba.memory.usage
import simba.memory.vector_db

if typing.TYPE_CHECKING:
    import pathlib

logger = logging.getLogger("simba.memory")


class _FAKE_NON_DORMANT:  # noqa: N801  # sentinel name fixed by the Phase 6 spec
    """Sentinel for records with no usage row — treated as non-dormant."""

    dormant = False


def _rerank_active(cfg: typing.Any, llm_client: typing.Any) -> bool:
    """Whether the rerank stage should fire, given the mode and the client.

    "none" never fires; "llm" needs a client (preserving the prior gate); the
    local GGUF backends ("cross-encoder"/"local-llm") need no client. An unknown
    mode is gated off here (reranker.rerank also fail-opens on it).
    """
    mode = getattr(cfg, "reranker_mode", "none")
    if mode == "none":
        return False
    if mode == "llm":
        return llm_client is not None
    return mode in ("cross-encoder", "local-llm")


def _filter_dormant(
    records: list[dict[str, typing.Any]],
    cwd: pathlib.Path,
) -> list[dict[str, typing.Any]]:
    """Remove records whose ``memory_usage.dormant`` is True.

    Missing rows (no usage record yet) are treated as non-dormant. Runs sync
    sqlite, so callers wrap it in ``asyncio.to_thread``. Fail-open: if the DB
    read raises, return ``records`` unchanged (prefer showing a dormant memory
    over dropping every result).
    """
    ids = [r["id"] for r in records if r.get("id")]
    if not ids:
        return records
    try:
        with simba.db.connect(cwd):
            usage_map = simba.memory.usage.get_many(ids)
    except Exception:
        logger.debug("[recall] dormant filter failed (fail-open)", exc_info=True)
        return records
    return [
        r
        for r in records
        if not usage_map.get(r.get("id", ""), _FAKE_NON_DORMANT).dormant
    ]


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
    cwd: pathlib.Path | None = None,
    entity_bridge_index: typing.Any = None,
    entity_bridge_lookup: dict[str, dict[str, typing.Any]] | None = None,
    kg_adjacency: dict[str, set[str]] | None = None,
    kg_entity_memories: dict[str, set[str]] | None = None,
    kg_record_lookup: dict[str, dict[str, typing.Any]] | None = None,
    kg_seeds: list[str] | None = None,
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

    fused = rrf_fuse(
        vector_results,
        keyword_results,
        k=cfg.rrf_k,
        vector_weight=cfg.vector_weight,
        keyword_weight=cfg.keyword_weight,
        extra_vector_results=extra_vector_results,
    )

    # Optional entity-bridge fold (spec 09): fold memories that share a named
    # entity with the top seeds into the candidates as a third RRF arm, before
    # composite rescore + the reranker (so the ranker still orders the assembled
    # set). No-op unless a caller supplies the index. Fail-open.
    if getattr(cfg, "entity_bridge_enabled", False) and entity_bridge_index is not None:
        with contextlib.suppress(Exception):
            seeds = [r["id"] for r in fused[: getattr(cfg, "entity_bridge_seeds", 5)]]
            bridged = simba.memory.entity_bridge.bridged_ids(
                entity_bridge_index,
                seeds,
                hops=getattr(cfg, "entity_bridge_hops", 1),
                min_shared=getattr(cfg, "entity_bridge_min_shared", 1),
                max_df=getattr(cfg, "entity_bridge_max_df", 0),
                max_out=getattr(cfg, "entity_bridge_max", 10),
            )
            if bridged:
                fused = simba.memory.entity_bridge.fold_into_candidates(
                    fused,
                    bridged,
                    record_lookup=entity_bridge_lookup or {},
                    rrf_k=cfg.rrf_k,
                    weight=getattr(cfg, "entity_bridge_weight", 1.0),
                )

    # Optional retrieval-time GraphRAG fold (Track B): seed PPR with the query's
    # KG entities, rank neighbor memories by stationary mass, and fold the top-N
    # into the candidate set as a third RRF arm — so graph-surfaced evidence
    # competes for the top-k before composite rescore + the reranker. No-op unless
    # a KG is supplied (the caller owns the graph). Fail-open.
    if getattr(cfg, "kg_ppr_enabled", False) and kg_adjacency and kg_seeds:
        with contextlib.suppress(Exception):
            ranked = simba.kg.ppr.rank_memories(
                kg_adjacency,
                kg_entity_memories or {},
                kg_seeds,
                top=getattr(cfg, "kg_ppr_top", 10),
                damping=getattr(cfg, "kg_ppr_damping", 0.85),
            )
            if ranked:
                fused = simba.memory.kg_fold.ppr_fold(
                    fused,
                    ppr_ranked_ids=ranked,
                    record_lookup=kg_record_lookup or {},
                    rrf_k=cfg.rrf_k,
                    weight=getattr(cfg, "kg_ppr_weight", 1.0),
                )

    # Optional composite re-scoring: blend RRF relevance with recency +
    # importance + strength over the full fused candidate set, then truncate.
    if getattr(cfg, "scoring_enabled", False):
        usage_map: dict[str, typing.Any] = {}
        w_str = float(getattr(cfg, "score_weight_strength", 0.0))
        if w_str and cwd is not None:

            def _load_usage() -> dict[str, typing.Any]:
                ids = [r.get("id") for r in fused if r.get("id")]
                try:
                    with simba.db.connect(cwd):
                        return simba.memory.usage.get_many(ids)
                except Exception:
                    logger.debug("[recall] usage load failed", exc_info=True)
                    return {}

            usage_map = await asyncio.to_thread(_load_usage)
        fused = simba.memory.scoring.composite_rescore(
            fused, cfg=cfg, now=time.time(), usage_map=usage_map
        )

    # Optional rerank of the candidate pool (cross-encoder role) before
    # truncation, routed on cfg.reranker_mode (spec 22). Two scheduling modes,
    # both fail-open:
    #   - cache wired (daemon): NON-BLOCKING — serve the fast order, rerank off
    #     the hot path, cache the result keyed by (query, candidate-set).
    #   - no cache (eval/CLI): synchronous rerank in a worker thread.
    # The "llm" backend needs a client (preserves the existing gate); the local
    # GGUF backends ("cross-encoder"/"local-llm") need none; "none" is a no-op.
    if (
        getattr(cfg, "llm_rerank_enabled", False)
        and _rerank_active(cfg, llm_client)
        and simba.memory.reranker.should_rerank(query_text, cfg)
    ):
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
                        cfg,
                    )
                )
                bg_tasks.add(task)
                task.add_done_callback(bg_tasks.discard)
            # miss with no task registry → serve the fast order unchanged
        else:
            with contextlib.suppress(Exception):
                fused = await asyncio.to_thread(
                    simba.memory.reranker.rerank,
                    query_text,
                    fused,
                    cfg=cfg,
                    llm=llm_client,
                    max_candidates=max_cands,
                )

    # Drop dormant memories (forgotten by the decay pass) before truncation.
    if getattr(cfg, "dormant_filter_enabled", True) and cwd is not None:
        fused = await asyncio.to_thread(_filter_dormant, fused, cwd)

    return fused[:max_results]


async def _bg_rerank(
    cache: typing.Any,
    key: str,
    query: str,
    pool: list[dict[str, typing.Any]],
    client: typing.Any,
    max_candidates: int,
    cfg: typing.Any,
) -> None:
    """Rerank ``pool`` off the hot path and store the id order in ``cache``."""
    with contextlib.suppress(Exception):
        reordered = await asyncio.to_thread(
            simba.memory.reranker.rerank,
            query,
            pool,
            cfg=cfg,
            llm=client,
            max_candidates=max_candidates,
        )
        cache.put(key, [r.get("id") for r in reordered])
