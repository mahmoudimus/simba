"""Ranking-semantics eval for decay/feedback (pure sqlite + scoring layer)."""

from __future__ import annotations

import pathlib
import types

import simba.db
import simba.memory.usage as usage
from simba.memory.decay import run_decay_pass
from simba.memory.hybrid import _filter_dormant
from simba.memory.scoring import composite_rescore
from simba.memory.strength import compute_strength

from .fixtures.decay_feedback_corpus import make_corpus

_DAY = 86400.0
_NOW = 1_000_000_000.0


def _cfg(**kw):
    base = dict(
        decay_half_life_days=30.0,
        reinforcement_scale=0.5,
        feedback_weight=0.2,
        strength_dormancy_threshold=0.1,
        decay_enabled=True,
        decay_capacity_per_type=0,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _strength(entry, cfg):
    return compute_strength(
        created_at_epoch=entry.created_at_epoch,
        now=_NOW,
        access_count=entry.access_count,
        feedback_score=entry.feedback_score,
        half_life=cfg.decay_half_life_days,
        reinforcement_scale=cfg.reinforcement_scale,
        feedback_weight=cfg.feedback_weight,
    )


def _seed(cwd, corpus) -> None:
    with simba.db.connect(cwd):
        for e in corpus:
            usage.get_or_create(e.memory_id, now=e.created_at_epoch)
            usage.MemoryUsage.update(
                access_count=e.access_count,
                feedback_score=e.feedback_score,
            ).where(usage.MemoryUsage.memory_id == e.memory_id).execute()


def test_strength_ordering_matches_expected() -> None:
    cfg = _cfg()
    corpus = make_corpus(_NOW)
    strength_by_id = {e.memory_id: _strength(e, cfg) for e in corpus}

    assert strength_by_id["mem_accessed"] > strength_by_id["mem_fresh"]
    assert strength_by_id["mem_fresh"] > strength_by_id["mem_old"]
    assert strength_by_id["mem_hated"] < strength_by_id["mem_fresh"]
    assert strength_by_id["mem_old"] < cfg.strength_dormancy_threshold * 2


def test_decay_pass_changes_db_strength(tmp_path: pathlib.Path) -> None:
    cfg = _cfg()
    corpus = make_corpus(_NOW)
    _seed(tmp_path, corpus)
    run_decay_pass(now=_NOW, cwd=tmp_path, cfg=cfg)
    with simba.db.connect(tmp_path):
        rows = usage.get_many([e.memory_id for e in corpus])
    assert rows["mem_old"].strength < rows["mem_fresh"].strength
    assert rows["mem_accessed"].strength >= rows["mem_fresh"].strength


def test_decay_pass_marks_mem_old_dormant_at_extreme_age(
    tmp_path: pathlib.Path,
) -> None:
    cfg = _cfg()
    corpus = make_corpus(_NOW)
    for e in corpus:
        if e.memory_id == "mem_old":
            e.created_at_epoch = _NOW - 200 * _DAY
    _seed(tmp_path, corpus)
    run_decay_pass(now=_NOW, cwd=tmp_path, cfg=cfg)
    with simba.db.connect(tmp_path):
        rows = usage.get_many([e.memory_id for e in corpus])
    assert rows["mem_old"].dormant is True


def test_feedback_lifts_strength_above_unfeedback_peer() -> None:
    cfg = _cfg()
    kw = dict(
        created_at_epoch=_NOW - 30 * _DAY,
        now=_NOW,
        access_count=0,
        half_life=cfg.decay_half_life_days,
        reinforcement_scale=cfg.reinforcement_scale,
        feedback_weight=cfg.feedback_weight,
    )
    s_a = compute_strength(feedback_score=0.0, **kw)
    s_b = compute_strength(feedback_score=1.0, **kw)
    assert s_b > s_a


def test_feedback_bad_can_trigger_dormancy(tmp_path: pathlib.Path) -> None:
    cfg = _cfg(decay_half_life_days=5.0, feedback_weight=0.9)
    mid = "mem_punished"
    created = _NOW - 20 * _DAY
    with simba.db.connect(tmp_path):
        usage.get_or_create(mid, now=created)
        usage.MemoryUsage.update(feedback_score=-1.0).where(
            usage.MemoryUsage.memory_id == mid
        ).execute()
    run_decay_pass(now=_NOW, cwd=tmp_path, cfg=cfg)
    with simba.db.connect(tmp_path):
        row = usage.get_many([mid])[mid]
    assert row.dormant is True


def test_dormant_filter_excludes_from_composite_rescore_pipeline(
    tmp_path: pathlib.Path,
) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("r1", now=_NOW)
        usage.get_or_create("r2", now=_NOW)
        usage.set_dormant("r2", dormant=True)
    r1 = {"id": "r1", "rrf_score": 0.4}
    r2 = {"id": "r2", "rrf_score": 0.9}
    filtered = _filter_dormant([r1, r2], cwd=tmp_path)
    assert {r["id"] for r in filtered} == {"r1"}
    assert all(r["id"] != "r2" for r in filtered)


def test_strength_weight_changes_ranking() -> None:
    r1 = {"id": "r1", "rrf_score": 0.5, "createdAt": "", "confidence": 0.0}
    r2 = {"id": "r2", "rrf_score": 0.6, "createdAt": "", "confidence": 0.0}
    usage_map = {
        "r1": types.SimpleNamespace(strength=1.0),
        "r2": types.SimpleNamespace(strength=0.1),
    }
    cfg = _cfg(
        scoring_enabled=True,
        score_weight_relevance=1.0,
        score_weight_recency=0.0,
        score_weight_importance=0.0,
        score_weight_strength=2.0,
        recency_halflife_days=90.0,
    )
    result = composite_rescore([r1, r2], cfg=cfg, now=0.0, usage_map=usage_map)
    assert result[0]["id"] == "r1"

    cfg.score_weight_strength = 0.0
    result2 = composite_rescore(
        [dict(r1), dict(r2)], cfg=cfg, now=0.0, usage_map=usage_map
    )
    assert result2[0]["id"] == "r2"
