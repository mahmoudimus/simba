"""Tests for composite re-scoring (relevance + recency + importance)."""

from __future__ import annotations

import datetime

import simba.memory.config as mc
import simba.memory.scoring as scoring

# A fixed "now" so recency is deterministic.
_NOW = datetime.datetime(2026, 6, 1, tzinfo=datetime.UTC).timestamp()


def _cfg(**kw):
    cfg = mc.MemoryConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _rec(rid, rrf, *, created="2026-06-01T00:00:00Z", confidence=0.85):
    return {
        "id": rid,
        "rrf_score": rrf,
        "createdAt": created,
        "confidence": confidence,
    }


def test_disabled_returns_input_order_unchanged() -> None:
    cfg = _cfg(scoring_enabled=False)
    recs = [_rec("a", 0.01), _rec("b", 0.02)]
    out = scoring.composite_rescore(recs, cfg=cfg, now=_NOW)
    assert [r["id"] for r in out] == ["a", "b"]  # untouched


def test_relevance_only_preserves_rrf_order() -> None:
    cfg = _cfg(
        scoring_enabled=True,
        score_weight_relevance=1.0,
        score_weight_recency=0.0,
        score_weight_importance=0.0,
    )
    # Pre-sorted by rrf desc (as hybrid_search hands them over).
    recs = [_rec("hi", 0.05), _rec("mid", 0.03), _rec("lo", 0.01)]
    out = scoring.composite_rescore(recs, cfg=cfg, now=_NOW)
    assert [r["id"] for r in out] == ["hi", "mid", "lo"]


def test_recency_breaks_ties_toward_newer() -> None:
    cfg = _cfg(
        scoring_enabled=True,
        score_weight_relevance=0.0,
        score_weight_recency=1.0,
        score_weight_importance=0.0,
        recency_halflife_days=30.0,
    )
    recs = [
        _rec("old", 0.02, created="2025-01-01T00:00:00Z"),
        _rec("new", 0.02, created="2026-05-30T00:00:00Z"),
    ]
    out = scoring.composite_rescore(recs, cfg=cfg, now=_NOW)
    assert [r["id"] for r in out] == ["new", "old"]


def test_importance_breaks_ties_toward_higher_confidence() -> None:
    cfg = _cfg(
        scoring_enabled=True,
        score_weight_relevance=0.0,
        score_weight_recency=0.0,
        score_weight_importance=1.0,
    )
    recs = [_rec("weak", 0.02, confidence=0.5), _rec("strong", 0.02, confidence=0.99)]
    out = scoring.composite_rescore(recs, cfg=cfg, now=_NOW)
    assert [r["id"] for r in out] == ["strong", "weak"]


def test_bad_created_at_contributes_zero_recency() -> None:
    cfg = _cfg(
        scoring_enabled=True,
        score_weight_relevance=0.0,
        score_weight_recency=1.0,
        score_weight_importance=0.0,
    )
    recs = [
        _rec("good", 0.02, created="2026-05-30T00:00:00Z"),
        _rec("bad", 0.02, created="not-a-date"),
    ]
    out = scoring.composite_rescore(recs, cfg=cfg, now=_NOW)
    assert [r["id"] for r in out] == ["good", "bad"]


def test_annotates_composite_score() -> None:
    cfg = _cfg(scoring_enabled=True)
    out = scoring.composite_rescore([_rec("a", 0.02)], cfg=cfg, now=_NOW)
    assert "composite_score" in out[0]
