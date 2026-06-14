"""Tests for the pure recall-planning helper (intent floor + widening + HyDE)."""

from __future__ import annotations

import simba.memory.config as mc
import simba.memory.intent as intent
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


# ── intent-aware candidate depth for count (recall-breadth-bound) ─────────────
def test_count_query_widens_pool_and_context() -> None:
    cfg = _cfg(
        intent_aware=True,
        count_depth_enabled=True,
        count_candidate_pool_n=80,
        count_context_k=20,
    )
    plan = rp.plan_recall("How many korean restaurants have I tried?", cfg)
    assert plan.mode == "count"
    assert plan.candidate_pool == 80
    assert plan.max_results == 20


def test_count_depth_disabled_uses_normal_sizing() -> None:
    cfg = _cfg(intent_aware=True, count_depth_enabled=False)
    plan = rp.plan_recall("How many korean restaurants have I tried?", cfg)
    assert plan.mode != "count"


def test_count_respects_explicit_max_results() -> None:
    cfg = _cfg(intent_aware=True, count_depth_enabled=True)
    plan = rp.plan_recall("How many trips have I taken?", cfg, max_results=3)
    assert plan.max_results == 3


# ── intent-aware breadth for multi-session / aggregation (MS breadth PR) ──────
def test_aggregation_query_widens_pool_and_context() -> None:
    cfg = _cfg(
        intent_aware=True,
        aggregation_depth_enabled=True,
        aggregation_candidate_pool_n=80,
        aggregation_context_k=80,
    )
    plan = rp.plan_recall("What is the total amount I spent in total this year?", cfg)
    assert plan.mode == "aggregation"
    assert plan.candidate_pool == 80
    assert plan.max_results == 80


def test_aggregation_depth_disabled_uses_normal_sizing() -> None:
    cfg = _cfg(intent_aware=True, aggregation_depth_enabled=False)
    plan = rp.plan_recall("What is the total amount I spent in total this year?", cfg)
    assert plan.mode != "aggregation"


def test_aggregation_respects_explicit_max_results() -> None:
    cfg = _cfg(intent_aware=True, aggregation_depth_enabled=True)
    plan = rp.plan_recall("how often did I travel in total", cfg, max_results=3)
    assert plan.max_results == 3
    assert plan.mode != "aggregation"


def test_count_takes_precedence_over_aggregation() -> None:
    # "How many X across all my trips" matches BOTH is_count and is_aggregation;
    # count must win so its (narrower-context, rerank-skipping) plan is used.
    cfg = _cfg(
        intent_aware=True,
        count_depth_enabled=True,
        aggregation_depth_enabled=True,
        count_candidate_pool_n=80,
        count_context_k=40,
        aggregation_candidate_pool_n=80,
        aggregation_context_k=80,
    )
    q = "How many restaurants did I try across all my trips?"
    assert intent.is_count(q) and intent.is_aggregation(q)  # genuinely both
    plan = rp.plan_recall(q, cfg)
    assert plan.mode == "count"
    assert plan.max_results == 40  # count_context_k, not aggregation_context_k


def test_aggregation_default_config_is_on() -> None:
    # Default-ON (2026-06-14 policy: a measured-SoTA lever graduates to default-on):
    # a fresh config routes an aggregation query to the wider breadth plan.
    cfg = _cfg(intent_aware=True)
    plan = rp.plan_recall("how often did I go to the gym in total this month", cfg)
    assert plan.mode == "aggregation"
    assert plan.max_results == cfg.aggregation_context_k
