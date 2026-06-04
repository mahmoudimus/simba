"""Query decomposition (C4) — the reasoning-side multi-hop lever.

LoCoMo's multi-hop difficulty is at reasoning time, not retrieval: the evidence
turns are individually retrievable, but a single query embedding straddles two
facts and surfaces neither cleanly. Decomposition (IRCoT/HippoRAG-style) splits
the question into single-fact sub-queries, retrieves each, and RRF-fuses the
ranked lists — so each evidence piece gets its own targeted retrieval.

Pure + injectable: ``decompose`` takes an llm client (fail-open to the original
query); ``fuse_rankings`` is pure RRF over ranked id-lists. The recall adapter /
daemon wire these around the existing per-query retrieval.
"""

from __future__ import annotations

import typing


def build_decompose_prompt(query: str) -> str:
    return (
        "Break this question into the minimal set of simpler sub-questions, each "
        "answerable on its own and about a single entity or fact. If it is already "
        "simple, return just it. Return ONLY a JSON array of strings.\n\n"
        f"Question: {query}\nJSON:"
    )


def parse_subqueries(
    raw: typing.Any, original: str, *, max_sub: int = 4
) -> list[str]:
    """Original first, then up to ``max_sub`` subs; deduped case-insensitively."""
    subs = [s.strip() for s in (raw or []) if isinstance(s, str) and s.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for q in [original, *subs]:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= max_sub + 1:  # original + at most max_sub
            break
    return out


def decompose(query: str, llm: typing.Any, *, max_sub: int = 4) -> list[str]:
    """Return [original, sub1, ...]; fail-open to ``[query]`` on any problem."""
    if llm is None:
        return [query]
    available = getattr(llm, "available", None)
    if callable(available) and not available():
        return [query]
    try:
        raw = llm.complete_json(build_decompose_prompt(query))
    except Exception:
        return [query]
    return parse_subqueries(raw, query, max_sub=max_sub)


def fuse_rankings(rankings: list[list[str]], *, k: int = 60) -> list[str]:
    """RRF over several ranked id-lists -> one fused id-list (desc by score)."""
    if len(rankings) == 1:
        return list(rankings[0])
    scores: dict[str, float] = {}
    for ids in rankings:
        for rank, rid in enumerate(ids, start=1):
            if rid:
                scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda r: scores[r], reverse=True)
