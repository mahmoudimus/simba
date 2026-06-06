"""Tests for the strength term in composite_rescore (src/simba/memory/scoring.py)."""

from __future__ import annotations

import types

import simba.memory.config as mc
import simba.memory.scoring as scoring


def _cfg(**kw):
    cfg = mc.MemoryConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _rec(rid, rrf):
    return {"id": rid, "rrf_score": rrf, "createdAt": "", "confidence": 0.0}


def test_composite_rescore_strength_term_lifts_high_strength() -> None:
    cfg = _cfg(
        scoring_enabled=True,
        score_weight_relevance=1.0,
        score_weight_recency=0.0,
        score_weight_importance=0.0,
        score_weight_strength=1.0,
        recency_halflife_days=90.0,
    )
    records = [_rec("r1", 0.5), _rec("r2", 0.5)]
    usage_map = {
        "r1": types.SimpleNamespace(strength=1.0),
        "r2": types.SimpleNamespace(strength=0.1),
    }
    result = scoring.composite_rescore(records, cfg=cfg, now=0.0, usage_map=usage_map)
    assert result[0]["id"] == "r1"


def test_composite_rescore_strength_defaults_to_1_when_no_usage_row() -> None:
    cfg = _cfg(
        scoring_enabled=True,
        score_weight_relevance=1.0,
        score_weight_recency=0.0,
        score_weight_importance=0.0,
        score_weight_strength=1.0,
        recency_halflife_days=90.0,
    )
    records = [_rec("r1", 0.5)]
    result = scoring.composite_rescore(records, cfg=cfg, now=0.0, usage_map={})
    assert result[0]["composite_score"] > 0.0


def test_composite_rescore_backward_compat_no_usage_map() -> None:
    cfg = _cfg(
        scoring_enabled=True,
        score_weight_relevance=1.0,
        score_weight_recency=0.0,
        score_weight_importance=0.0,
        score_weight_strength=1.0,
        recency_halflife_days=90.0,
    )
    records = [_rec("a", 0.01), _rec("b", 0.02)]
    result = scoring.composite_rescore(records, cfg=cfg, now=0.0)
    # With no usage_map, strength is uniform (1.0) → pure rrf order prevails.
    assert [r["id"] for r in result] == ["b", "a"]
