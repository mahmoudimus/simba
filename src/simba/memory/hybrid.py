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
    conn = simba.memory.fts.connect(fts_path)
    try:
        return simba.memory.fts.search(
            conn,
            query_text,
            project_path=project_path,
            types=types,
            limit=limit,
        )
    finally:
        conn.close()


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
) -> list[dict[str, typing.Any]]:
    """Run both arms and return the RRF-fused top ``max_results`` memories.

    The vector arm keeps the ``min_similarity`` floor; the keyword arm is *not*
    cosine-gated (that is what widens coverage).  Both arms share the same
    project/type scope via ``filters``.  Never raises on the keyword side.
    """
    filters = filters or {}
    candidate_pool = max(max_results, cfg.fts_candidate_pool)

    vector_results = await simba.memory.vector_db.search_memories(
        table, embedding, min_similarity, candidate_pool, filters
    )

    keyword_results: list[dict[str, typing.Any]] = []
    if fts_path:
        try:
            keyword_results = await asyncio.to_thread(
                _keyword_arm,
                fts_path,
                query_text,
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
