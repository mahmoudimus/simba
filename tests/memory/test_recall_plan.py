"""Tests for the pure recall-planning helper (intent floor + widening + HyDE)."""

from __future__ import annotations

import simba.memory.config as mc
import simba.memory.recall_plan as rp


def _cfg(**kw):
    cfg = mc.MemoryConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_precise_query_uses_precise_floor_and_results() -> None:
    cfg = _cfg(intent_aware=True)
    plan = rp.plan_recall("why did the peewee migration break test isolation", cfg)
    assert plan.mode == "precise"
    assert plan.min_similarity == cfg.min_similarity
    assert plan.max_results == cfg.max_results
    assert plan.candidate_pool == cfg.fts_candidate_pool


def test_broad_query_widens_floor_results_and_pool() -> None:
    cfg = _cfg(intent_aware=True)
    plan = rp.plan_recall("summarize everything about the memory daemon", cfg)
    assert plan.mode == "broad"
    assert plan.min_similarity == cfg.min_similarity_broad
    assert plan.max_results == cfg.max_results_broad
    assert plan.candidate_pool == cfg.fts_candidate_pool_broad


def test_explicit_min_similarity_wins() -> None:
    cfg = _cfg(intent_aware=True)
    plan = rp.plan_recall("summarize everything", cfg, min_similarity=0.5)
    assert plan.mode == "explicit"
    assert plan.min_similarity == 0.5
    # explicit ⇒ not broad ⇒ precise widths
    assert plan.max_results == cfg.max_results


def test_explicit_max_results_wins() -> None:
    cfg = _cfg(intent_aware=True)
    plan = rp.plan_recall("summarize everything", cfg, max_results=2)
    assert plan.max_results == 2


def test_intent_off_is_precise() -> None:
    cfg = _cfg(intent_aware=False)
    plan = rp.plan_recall("summarize everything about the daemon", cfg)
    assert plan.mode == "precise"
    assert plan.min_similarity == cfg.min_similarity


def test_expansion_terms_empty_when_disabled() -> None:
    cfg = _cfg(expansion_enabled=False)
    plan = rp.plan_recall("the GITHUB_TOKEN gh auth gotcha", cfg)
    assert plan.expansion_terms == ""


def test_expansion_terms_present_when_enabled() -> None:
    cfg = _cfg(hybrid_enabled=True, expansion_enabled=True)
    plan = rp.plan_recall("the GITHUB_TOKEN gh auth gotcha mahmoudimus", cfg)
    assert plan.expansion_terms  # non-empty focused-term string
    terms = plan.expansion_terms
    assert "GITHUB_TOKEN" in terms or "mahmoudimus" in terms
