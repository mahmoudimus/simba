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

    by_id = {str(c.get("id")): c for c in head}
    ranked: list[dict[str, typing.Any]] = []
    seen: set[str] = set()
    for rid in order:
        key = str(rid)
        cand = by_id.get(key)
        if cand is not None and key not in seen:
            ranked.append(cand)
            seen.add(key)
    # Append any head candidate the model omitted, preserving original order.
    for cand in head:
        if str(cand.get("id")) not in seen:
            ranked.append(cand)
    return ranked + tail
