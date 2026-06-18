"""Session-source expansion for recall candidates.

When normal recall finds one turn from a transcript/session, this fold can add
nearby same-session turns into the candidate pool before scoring/reranking. It
is deliberately non-oracle: only sessions already touched by recall become
eligible.
"""

from __future__ import annotations

import typing


def seed_sessions(
    candidates: list[dict[str, typing.Any]], *, top_sessions: int
) -> list[str]:
    """Return ordered unique ``sessionSource`` values from top candidates."""
    out: list[str] = []
    seen: set[str] = set()
    if top_sessions <= 0:
        return out
    for rec in candidates:
        sid = str(rec.get("sessionSource") or "")
        if not sid or sid in seen:
            continue
        out.append(sid)
        seen.add(sid)
        if len(out) >= top_sessions:
            break
    return out


def fold_session_records(
    fused: list[dict[str, typing.Any]],
    session_records: list[dict[str, typing.Any]],
    *,
    rrf_k: int,
    weight: float,
) -> list[dict[str, typing.Any]]:
    """Add same-session records as a ranked RRF-style arm.

    Existing candidates are left in place except for a small same-session boost;
    new records are materialized from ``session_records`` and scored by their
    order in that list.
    """
    scores = {r["id"]: float(r.get("rrf_score", 0.0) or 0.0) for r in fused}
    records = {r["id"]: r for r in fused}

    for rank, rec in enumerate(session_records, start=1):
        mid = rec.get("id")
        if not mid:
            continue
        contrib = weight / (rrf_k + rank)
        if mid not in records:
            records[mid] = dict(rec)
        scores[mid] = scores.get(mid, 0.0) + contrib

    ordered = sorted(records.values(), key=lambda r: scores[r["id"]], reverse=True)
    for rec in ordered:
        rec["rrf_score"] = round(scores[rec["id"]], 6)
    return ordered
