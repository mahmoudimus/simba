"""Hybrid recall (L3): fuse the vector and keyword arms via RRF.

``rrf_fuse`` is a pure function over two already-ranked lists; ``hybrid_search``
orchestrates the two arms (LanceDB cosine + the SQLite FTS5 mirror) and returns
the fused top-k.  The keyword arm runs in a worker thread (sync SQLite) and is
fully defensive — any failure degrades the result to vector-only.
"""

from __future__ import annotations

import asyncio
import typing

import simba.memory.fts
import simba.memory.keywords
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
) -> list[dict[str, typing.Any]]:
    """Reciprocal Rank Fusion of two ranked lists, deduped by memory id.

    ``score(id) = Σ_arm weight_arm / (k + rank_arm(id))`` with 1-based ranks.
    The vector arm is folded first, so when an id appears in both arms the
    richer vector record (real cosine ``similarity``) is kept while the score
    still accumulates contributions from both arms.  Returns records ordered by
    fused score (desc), each carrying an ``rrf_score``.
    """
    scores: dict[str, float] = {}
    records: dict[str, dict[str, typing.Any]] = {}

    for rank, item in enumerate(vector_results, start=1):
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
    )
    return fused[:max_results]
