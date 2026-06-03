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


class TestEmbedderDefaults:
    def test_embed_provider_default_is_gguf(self) -> None:
        assert config.MemoryConfig().embed_provider == "gguf"

    def test_nomic_prefixes_preserved_by_default(self) -> None:
        cfg = config.MemoryConfig()
        assert cfg.embed_doc_prefix == "search_document: "
        assert cfg.embed_query_prefix == "search_query: "


class TestRerankModeDefault:
    def test_default_is_async_nonblocking(self) -> None:
        assert config.MemoryConfig().llm_rerank_mode == "async"


class TestScoringDefaults:
    def test_scoring_on_by_default(self) -> None:
        assert config.MemoryConfig().scoring_enabled is True

    def test_llm_rerank_on_by_default(self) -> None:
        assert config.MemoryConfig().llm_rerank_enabled is True

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
