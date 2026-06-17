"""Config default-assertions for intent priming + preflight mandate (spec 28)."""

from __future__ import annotations

import simba.hooks.config as config


class TestIntentPrimingDefaults:
    def test_intent_priming_off_by_default(self) -> None:
        # UNMEASURED false-prime rate -> default OFF (byte-identical to today).
        assert config.HooksConfig().intent_priming_enabled is False

    def test_priming_floor_stricter_than_recall(self) -> None:
        # A primed doctrine steers the agent, so its floor is above recall's 0.35.
        cfg = config.HooksConfig()
        assert cfg.intent_priming_min_similarity >= 0.5

    def test_max_doctrines_default(self) -> None:
        assert config.HooksConfig().intent_priming_max_doctrines == 3


class TestPreflightMandateDefaults:
    def test_mandate_off_by_default(self) -> None:
        assert config.HooksConfig().preflight_mandate_enabled is False

    def test_risk_only_on_by_default(self) -> None:
        # The over-fire guard defaults ON: mandate only for risk-tier intents.
        assert config.HooksConfig().preflight_mandate_risk_only is True

    def test_mutating_tool_set_default(self) -> None:
        cfg = config.HooksConfig()
        tools = {t.strip() for t in cfg.preflight_mandate_tools.split(",")}
        assert tools == {"Edit", "Write", "Bash"}
