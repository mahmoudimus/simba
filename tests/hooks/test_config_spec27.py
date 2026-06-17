"""Config default-assertions for the reasoning-layer levers (spec 27)."""

from __future__ import annotations

import simba.hooks.config as config


class TestEngagementMarkerDefault:
    def test_engagement_marker_off_by_default(self) -> None:
        # UNMEASURED observability lever -> default OFF (byte-identical to today).
        assert config.HooksConfig().engagement_marker_enabled is False


class TestReasoningVerifyDefault:
    def test_reasoning_verify_off_by_default(self) -> None:
        # Costs an LLM judgment + can block-to-reconsider -> default OFF.
        assert config.HooksConfig().reasoning_verify_enabled is False
