"""Composite re-scoring of recall candidates (Generative-Agents style).

RRF gives a pure relevance ordering. ``composite_rescore`` optionally blends
that relevance with **recency** (exponential decay on age) and **importance**
(the stored ``confidence``) so the freshest / most-trusted memory can win a
near-tie. Off by default (``scoring_enabled``); with the default weights
(relevance 1, recency 0, importance 0) it is order-preserving, so enabling the
flag alone changes nothing until a recency/importance weight is set.

Pure + deterministic: ``now`` (epoch seconds) is injected so tests don't depend
on the wall clock.
"""

from __future__ import annotations

import datetime
import typing


def _parse_epoch(created_at: str) -> float | None:
    """Parse an ISO8601 ``createdAt`` to epoch seconds, or None if unparseable."""
    if not created_at:
        return None
    try:
        text = created_at.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.UTC)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize to [0, 1]; all-equal ⇒ all 1.0 (no discrimination)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    span = hi - lo
    return [(v - lo) / span for v in values]


def _recency(created_at: str, now: float, halflife_days: float) -> float:
    """Exponential-decay recency in [0, 1]; unparseable/halflife<=0 ⇒ 0.0."""
    epoch = _parse_epoch(created_at)
    if epoch is None or halflife_days <= 0:
        return 0.0
    age_days = max(0.0, (now - epoch) / 86400.0)
    return 0.5 ** (age_days / halflife_days)


def composite_rescore(
    records: list[dict[str, typing.Any]],
    *,
    cfg: typing.Any,
    now: float,
) -> list[dict[str, typing.Any]]:
    """Re-rank ``records`` by a weighted blend of relevance/recency/importance.

    ``records`` are expected pre-sorted by ``rrf_score`` (as ``hybrid_search``
    hands them over). Returns a new list ordered by composite score (desc),
    each record annotated with ``composite_score``. A no-op (returns the input
    unchanged) when ``cfg.scoring_enabled`` is false.
    """
    if not getattr(cfg, "scoring_enabled", False) or not records:
        return records

    w_rel = float(getattr(cfg, "score_weight_relevance", 1.0))
    w_rec = float(getattr(cfg, "score_weight_recency", 0.0))
    w_imp = float(getattr(cfg, "score_weight_importance", 0.0))
    halflife = float(getattr(cfg, "recency_halflife_days", 90.0))

    relevance = _normalize([float(r.get("rrf_score", 0.0)) for r in records])

    scored: list[tuple[float, int, dict[str, typing.Any]]] = []
    for idx, (rec, rel) in enumerate(zip(records, relevance, strict=True)):
        rec_score = _recency(rec.get("createdAt", "") or "", now, halflife)
        imp = max(0.0, min(1.0, float(rec.get("confidence", 0.0) or 0.0)))
        composite = w_rel * rel + w_rec * rec_score + w_imp * imp
        rec["composite_score"] = round(composite, 6)
        # idx as a stable tiebreaker preserves the incoming (rrf) order on ties.
        scored.append((composite, idx, rec))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [rec for _, _, rec in scored]
