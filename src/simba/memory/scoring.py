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


def apply_rejection_gate(
    memories: list[dict[str, typing.Any]],
    *,
    enabled: bool,
    threshold: float,
    score_key: str = "similarity",
) -> list[dict[str, typing.Any]]:
    """Low-confidence abstention gate: suppress the WHOLE recall when the top
    candidate's ``score_key`` is below ``threshold`` (MemX-style).

    Distinct from the per-result ``min_similarity`` floor — that drops individual
    weak hits; this judges the BEST candidate and, if even it is too weak, returns
    nothing (abstain) rather than surfacing spurious context. Off unless ``enabled``;
    fail-open (an empty list passes through unchanged). Pure + order-preserving.
    """
    if not enabled or not memories:
        return memories
    top = memories[0].get(score_key, 0.0) or 0.0
    return [] if top < threshold else memories


def truncate_to_budget(
    records: list[dict[str, typing.Any]],
    *,
    max_results: int,
    token_budget: int,
    chars_per_token: int = 4,
) -> list[dict[str, typing.Any]]:
    """Score-adaptive truncation (SmartSearch, arXiv 2603.15599).

    Off (``token_budget <= 0``) this is the legacy fixed-k cut to ``max_results``.
    On, it ignores the count and returns the longest score-ranked prefix
    that fits ``token_budget`` (estimated as ``len(content+context)/chars_per_token``).
    The top record is always included (never return empty for a non-empty input), so a
    single over-budget hit still surfaces. This targets the completeness gate: when
    there is token room, more co-required evidence survives than a fixed-k would keep;
    when results are long, fewer are returned so the budget isn't overflowed. Pure +
    order-preserving (assumes ``records`` already ranked).
    """
    if token_budget <= 0:
        return records[:max_results]
    out: list[dict[str, typing.Any]] = []
    used = 0
    for r in records:
        text = (r.get("content") or "") + (r.get("context") or "")
        cost = max(1, len(text) // chars_per_token)
        if out and used + cost > token_budget:
            break
        out.append(r)
        used += cost
    return out


def _recency(created_at: str, now: float, halflife_days: float) -> float:
    """Exponential-decay recency in [0, 1]; unparseable/halflife<=0 ⇒ 0.0."""
    epoch = _parse_epoch(created_at)
    if epoch is None or halflife_days <= 0:
        return 0.0
    age_days = max(0.0, (now - epoch) / 86400.0)
    return 0.5 ** (age_days / halflife_days)


def _usage_signal(row: typing.Any) -> float:
    """Usage-quality signal in [-1, 1]: (use_count - noise_count) / max(1, sum).

    A missing row (never recalled/judged) or a row with no use/noise history
    scores 0.0 (neutral) — matches the "no penalty for never-recalled" spirit
    of the strength term.
    """
    if row is None:
        return 0.0
    use = float(getattr(row, "use_count", 0) or 0)
    noise = float(getattr(row, "noise_count", 0) or 0)
    return (use - noise) / max(1.0, use + noise)


def composite_rescore(
    records: list[dict[str, typing.Any]],
    *,
    cfg: typing.Any,
    now: float,
    usage_map: dict[str, typing.Any] | None = None,
) -> list[dict[str, typing.Any]]:
    """Re-rank ``records`` by a weighted blend of relevance/recency/importance.

    ``records`` are expected pre-sorted by ``rrf_score`` (as ``hybrid_search``
    hands them over). Returns a new list ordered by composite score (desc),
    each record annotated with ``composite_score``. A no-op (returns the input
    unchanged) when ``cfg.scoring_enabled`` is false.

    ``usage_map`` (keyed by memory id) optionally supplies a per-memory
    ``strength`` from the sqlite usage sidecar. When ``score_weight_strength``
    is non-zero and a row exists, its strength enters the blend; a missing row
    scores ``1.0`` (no penalty). Backward-compatible: omitting ``usage_map``
    leaves the legacy relevance/recency/importance behaviour unchanged.

    ``usage_influence_weight`` (cognee borrow — feedback-weighted edges, the
    non-redundant idea from cognee's graph-completion stack; see
    docs/plans/08-borrow-survey.md) additionally blends a usage-QUALITY signal
    on top of the relevance/recency/importance/strength composite above:
    ``signal = (use_count - noise_count) / max(1, use_count + noise_count)``
    in ``[-1, 1]``; ``mapped = (signal + 1) / 2`` in ``[0, 1]``; final
    ``score = composite * (1 - w) + mapped * w``. Simple linear interpolation,
    bounded and monotonic in the signal — a heavy-``use`` memory always ranks
    above an otherwise-equal heavy-``noise`` one once ``w > 0``. A missing row
    maps to a neutral 0.5 (same "no penalty" spirit as the strength term).
    DEFAULT 0.0 is an exact SHORT-CIRCUIT: ``usage_map`` is never read for
    this term (no sidecar access), and the composite score is byte-identical
    to the pre-lever function. Data-gated: enable only after >= 1 week of
    accumulated usage signals; graduate default per the SoTA-lever rule via a
    real re-runnable A/B on LME-S + LoCoMo (guard: HaluMem); note somnigraph
    finding — injection-suppression variants must beat always-inject to
    graduate.
    """
    if not getattr(cfg, "scoring_enabled", False) or not records:
        return records

    w_rel = float(getattr(cfg, "score_weight_relevance", 1.0))
    w_rec = float(getattr(cfg, "score_weight_recency", 0.0))
    w_imp = float(getattr(cfg, "score_weight_importance", 0.0))
    w_str = float(getattr(cfg, "score_weight_strength", 0.0))
    w_usage = float(getattr(cfg, "usage_influence_weight", 0.0))
    halflife = float(getattr(cfg, "recency_halflife_days", 90.0))

    relevance = _normalize([float(r.get("rrf_score", 0.0)) for r in records])

    scored: list[tuple[float, int, dict[str, typing.Any]]] = []
    for idx, (rec, rel) in enumerate(zip(records, relevance, strict=True)):
        rec_score = _recency(rec.get("createdAt", "") or "", now, halflife)
        imp = max(0.0, min(1.0, float(rec.get("confidence", 0.0) or 0.0)))
        if w_str and usage_map is not None:
            row = usage_map.get(rec.get("id", ""), None)
            strength_val = float(row.strength) if row is not None else 1.0
        else:
            strength_val = 1.0
        composite = w_rel * rel + w_rec * rec_score + w_imp * imp + w_str * strength_val
        if w_usage > 0 and usage_map is not None:
            usage_row = usage_map.get(rec.get("id", ""), None)
            mapped = (_usage_signal(usage_row) + 1.0) / 2.0
            composite = composite * (1.0 - w_usage) + mapped * w_usage
        rec["composite_score"] = round(composite, 6)
        # idx as a stable tiebreaker preserves the incoming (rrf) order on ties.
        scored.append((composite, idx, rec))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [rec for _, _, rec in scored]
