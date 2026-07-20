"""Tests for memory daemon config defaults (src/simba/memory/config.py)."""

from __future__ import annotations

import simba.memory.config as config


class TestContentLengthContract:
    def test_default_max_content_length_is_200(self) -> None:
        # CORE contract: "Memory content max 200 chars; use context for details."
        # The session-start extraction prompt and CLI help all assume 200.
        assert config.MemoryConfig().max_content_length == 200


class TestPhase0Defaults:
    def test_intent_aware_on_by_default(self) -> None:
        assert config.MemoryConfig().intent_aware is True

    def test_broad_floor_below_precise_floor(self) -> None:
        cfg = config.MemoryConfig()
        assert cfg.min_similarity_broad < cfg.min_similarity

    def test_fts_max_terms_default(self) -> None:
        assert config.MemoryConfig().fts_max_terms == 12

    def test_broad_widening_defaults(self) -> None:
        # Broad queries should pull a wider net than the precise defaults.
        cfg = config.MemoryConfig()
        assert cfg.max_results_broad > cfg.max_results
        assert cfg.fts_candidate_pool_broad > cfg.fts_candidate_pool

    def test_expansion_on_by_default(self) -> None:
        # Experimental features ship on by default; the 2nd HyDE arm included.
        assert config.MemoryConfig().expansion_enabled is True

    def test_anticipated_query_limit_default(self) -> None:
        assert config.MemoryConfig().anticipated_query_max_per_memory == 5


class TestEmbedderDefaults:
    def test_embed_provider_default_is_gguf(self) -> None:
        assert config.MemoryConfig().embed_provider == "gguf"

    def test_default_embedder_is_bge_large(self) -> None:
        # Default flipped nomic-embed-text (768-d) -> bge-large-en-v1.5 (1024-d)
        # after a cross-dataset bake-off win (docs/plans/07-recall-excellence.md).
        cfg = config.MemoryConfig()
        assert cfg.embedding_model == "bge-large-en-v1.5"
        assert cfg.embedding_dims == 1024
        assert cfg.model_file == "bge-large-en-v1.5-q8_0.gguf"
        assert cfg.embed_doc_prefix == ""
        assert cfg.embed_query_prefix == (
            "Represent this sentence for searching relevant passages: "
        )


class TestRerankModeDefault:
    def test_default_is_async_nonblocking(self) -> None:
        assert config.MemoryConfig().llm_rerank_mode == "async"


class TestScoringDefaults:
    def test_scoring_on_by_default(self) -> None:
        assert config.MemoryConfig().scoring_enabled is True

    def test_llm_rerank_on_by_default(self) -> None:
        assert config.MemoryConfig().llm_rerank_enabled is True

    def test_outcome_quality_weight_default_off(self) -> None:
        assert config.MemoryConfig().outcome_quality_weight == 0.0

    def test_relevance_dominant_blend(self) -> None:
        # Enabling the flag should activate the measured-good blend: relevance
        # dominant, recency + importance as tie-breakers (never the sole signal).
        cfg = config.MemoryConfig()
        assert cfg.score_weight_relevance == 1.0
        assert 0.0 < cfg.score_weight_recency < cfg.score_weight_relevance
        assert 0.0 < cfg.score_weight_importance < cfg.score_weight_relevance
        assert cfg.recency_halflife_days > 0


class TestSupersedeDefaults:
    def test_supersede_on_by_default(self) -> None:
        assert config.MemoryConfig().supersede_enabled is True

    def test_supersede_band_below_duplicate(self) -> None:
        # The supersede band sits below the duplicate threshold.
        cfg = config.MemoryConfig()
        assert 0 < cfg.supersede_threshold < cfg.duplicate_threshold

    def test_supersede_trust_gate_on_by_default(self) -> None:
        cfg = config.MemoryConfig()
        assert cfg.supersede_trust_gate_enabled is True
        assert cfg.supersede_trust_margin > 0


class TestUsageInfluenceDefaults:
    def test_usage_influence_weight_default_off(self) -> None:
        # Data-gated (cognee borrow): ships inert until >= 1 week of usage
        # signals accumulate and a real A/B measures it. 0.0 is a no-op.
        assert config.MemoryConfig().usage_influence_weight == 0.0


class TestGraduationReadinessDefaults:
    """Spec 33 Part 8 rule R1: the DATA criteria for `maintenance_apply`."""

    def test_min_days_default(self) -> None:
        assert config.MemoryConfig().maintenance_graduation_min_days == 14.0

    def test_min_used_ratio_default(self) -> None:
        assert config.MemoryConfig().maintenance_graduation_min_used_ratio == 0.6


class TestHierarchicalRecallDefaults:
    def test_hierarchical_recall_off_by_default(self) -> None:
        # UNMEASURED lever (precision-dilution risk) -> default OFF so recall stays
        # byte-identical to today's strict exact-match scoping.
        assert config.MemoryConfig().hierarchical_recall is False

    def test_include_global_on_by_default(self) -> None:
        # Global memories are the root of the tree; this is a separate lever so it
        # can be measured independently, but it defaults ON (no effect while
        # hierarchical_recall is OFF).
        assert config.MemoryConfig().hierarchical_recall_include_global is True


class TestRssWatchdogDefaults:
    """RSS watchdog + recall admission control: UNMEASURED levers, both OFF by
    repo policy, so a fresh MemoryConfig() is byte-identical to pre-watchdog
    behavior (no polling task starts, no /recall gating)."""

    def test_soft_and_hard_limits_default_disabled(self) -> None:
        cfg = config.MemoryConfig()
        assert cfg.rss_soft_limit_mb == 0
        assert cfg.rss_hard_limit_mb == 0

    def test_check_interval_default(self) -> None:
        assert config.MemoryConfig().rss_check_interval_seconds == 30.0

    def test_restart_min_uptime_default(self) -> None:
        assert config.MemoryConfig().rss_restart_min_uptime_seconds == 300.0

    def test_max_concurrent_recalls_default_unlimited(self) -> None:
        # 0 = unlimited: no asyncio.Semaphore is constructed, /recall is
        # byte-identical to pre-admission-control behavior.
        assert config.MemoryConfig().max_concurrent_recalls == 0

    def test_rss_history_samples_default(self) -> None:
        # At the default 30s check interval, 240 samples is ~2h of history.
        assert config.MemoryConfig().rss_history_samples == 240


class TestBindProbeGraceDefault:
    def test_bind_probe_grace_seconds_default(self) -> None:
        assert config.MemoryConfig().bind_probe_grace_seconds == 45.0


class TestMallocStackLoggingDefault:
    """Diagnostic-only lever (2026-07-19 16.7GB RSS burst with no attributable
    stacks) — adds allocator overhead, so it stays an explicit opt-in."""

    def test_malloc_stack_logging_off_by_default(self) -> None:
        assert config.MemoryConfig().malloc_stack_logging is False
