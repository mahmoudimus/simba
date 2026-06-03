"""LLM reranker — a relevance re-scoring pass over recall candidates.

This is the cross-encoder's role done by an LLM (the deliberately-deferred
reranker): given the query and the top candidate memories, ask the model to
order them by relevance and reorder accordingly. Fail-open — an unavailable
client, a bad reply, or any error leaves the candidates untouched, so recall
degrades to the RRF + composite-score ordering. Pure (client injected) so it is
unit-testable and reusable by both the daemon and the eval harness.
"""

from __future__ import annotations

import typing

_PROMPT = (
    "Rank these memories by how well they answer the query. Return ONLY a JSON "
    "array of the memory ids, most relevant first; include every id exactly "
    "once and add nothing else.\n\n"
    "Query: {query}\n\nMemories:\n{lines}"
)


def reorder_by_ids(
    records: list[dict[str, typing.Any]], order: list
) -> list[dict[str, typing.Any]]:
    """Reorder ``records`` to follow ``order`` (a list of ids); never drop/dup.

    Ids in ``order`` come first (each once), then any record the order omitted in
    its original position. Shared by the live reranker and the rerank cache.
    """
    by_id = {str(r.get("id")): r for r in records}
    ranked: list[dict[str, typing.Any]] = []
    seen: set[str] = set()
    for rid in order:
        key = str(rid)
        rec = by_id.get(key)
        if rec is not None and key not in seen:
            ranked.append(rec)
            seen.add(key)
    for rec in records:
        if str(rec.get("id")) not in seen:
            ranked.append(rec)
    return ranked


def _build_prompt(query: str, candidates: list[dict[str, typing.Any]]) -> str:
    lines = []
    for c in candidates:
        content = (c.get("content") or "").strip()
        ctx = (c.get("context") or "").strip()
        suffix = f" — {ctx[:120]}" if ctx else ""
        lines.append(f"[{c.get('id')}] {content}{suffix}")
    return _PROMPT.format(query=query, lines="\n".join(lines))


def rerank(
    query: str,
    candidates: list[dict[str, typing.Any]],
    *,
    client: typing.Any,
    max_candidates: int = 20,
) -> list[dict[str, typing.Any]]:
    """Reorder ``candidates`` by LLM-judged relevance to ``query``.

    Only the first ``max_candidates`` are sent to the model (prompt-size guard);
    the rest are kept, untouched, at the tail. Returns the candidates unchanged
    on any failure. Never drops or duplicates a candidate.
    """
    if not candidates or client is None or not client.available():
        return candidates

    head = candidates[:max_candidates]
    tail = candidates[max_candidates:]

    order = client.complete_json(_build_prompt(query, head))
    if not isinstance(order, list):
        return candidates

    return reorder_by_ids(head, order) + tail
