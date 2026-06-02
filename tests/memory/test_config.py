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
