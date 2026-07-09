"""Wiring of the 🦁☑ engagement marker into UserPromptSubmit + PreToolUse (spec 27).

Off (default) ⇒ byte-identical to today (characterization). On ⇒ the marker is
emitted into additional_context and recorded for the Stop echo-verify.
"""

from __future__ import annotations

import json
import unittest.mock

import simba.hooks.pre_tool_use
import simba.hooks.user_prompt_submit


class _MarkerCfg:
    """HooksConfig stub with the marker lever ON and everything else off."""

    prompt_min_similarity = 0.45
    prompt_min_length = 10
    guardian_signal_gated = False
    intent_priming_enabled = False
    preflight_mandate_enabled = False
    preflight_mandate_risk_only = True
    engagement_marker_enabled = True


class TestUserPromptSubmitMarker:
    def test_off_by_default_no_marker(self, tmp_path, monkeypatch) -> None:
        # Characterization: lever OFF (default) → no 🦁☑. Pin _cfg to the real
        # default so an ambient (dogfood) .simba/config.toml enabling the marker
        # can't leak in (the hook loads config from the process cwd, not payload cwd).
        import simba.hooks.config

        monkeypatch.setattr(
            simba.hooks.user_prompt_submit,
            "_cfg",
            lambda: simba.hooks.config.HooksConfig(),
            raising=False,
        )
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            out = simba.hooks.user_prompt_submit.main(
                {"prompt": "a sufficiently long prompt", "cwd": str(tmp_path)}
            )
        ctx = json.loads(out)["hookSpecificOutput"].get("additionalContext", "")
        assert "🦁☑" not in ctx

    def test_on_emits_recalled_marker(self, tmp_path, monkeypatch) -> None:
        import simba.guardian.engagement_flag as ef

        monkeypatch.setattr(ef, "_TMP_DIR", tmp_path)
        monkeypatch.setattr(
            simba.hooks.user_prompt_submit, "_cfg", lambda: _MarkerCfg(), raising=False
        )
        memories = [
            {"type": "GOTCHA", "content": "x", "similarity": 0.74},
            {"type": "PATTERN", "content": "y", "similarity": 0.5},
        ]
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=memories
        ):
            out = simba.hooks.user_prompt_submit.main(
                {
                    "prompt": "a sufficiently long prompt",
                    "cwd": str(tmp_path),
                    "session_id": "sess-marker",
                }
            )
        ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        assert "🦁☑ recalled 2 (top 0.74)" in ctx
        # Recorded for the Stop echo-verify.
        assert ef.engaged("sess-marker") is True

    def test_on_zero_tool_turn_emits_idle(self, tmp_path, monkeypatch) -> None:
        # Even a zero-recall turn emits a marker (every turn, tools or not).
        import simba.guardian.engagement_flag as ef

        monkeypatch.setattr(ef, "_TMP_DIR", tmp_path)
        monkeypatch.setattr(
            simba.hooks.user_prompt_submit, "_cfg", lambda: _MarkerCfg(), raising=False
        )
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            out = simba.hooks.user_prompt_submit.main(
                {
                    "prompt": "a sufficiently long prompt",
                    "cwd": str(tmp_path),
                    "session_id": "sess-idle",
                }
            )
        ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        assert "🦁☑ idle (nothing matched)" in ctx
        assert ef.engaged("sess-idle") is True


class TestPreToolUseMarkerAppend:
    def test_off_by_default_no_marker(self, tmp_path, monkeypatch) -> None:
        # Characterization: a redirect deny output carries no marker when off.
        # tool_input is truthy here, so PreToolUse also drives the unrelated
        # TOOL_RULE gate (_recall_tool_rules); pinning cwd to tmp_path already
        # keeps its project-id resolution isolated, but _project_has_tool_rules
        # still asks the real daemon whether that (freshly-minted) project has
        # rules. Stub it False so this test never depends on daemon reachability.
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_project_has_tool_rules", lambda *a, **k: False
        )
        out = simba.hooks.pre_tool_use.main(
            {
                "tool_name": "Read",
                "tool_input": {"file_path": str(tmp_path / "f")},
                "cwd": str(tmp_path),
            }
        )
        assert "🦁☑" not in out

    def test_on_appends_gate_action_for_rewrite(self, tmp_path, monkeypatch) -> None:
        import simba.guardian.engagement_flag as ef
        import simba.redirect.check as rc

        monkeypatch.setattr(ef, "_TMP_DIR", tmp_path)
        ef.record_engagement("sess-rw", ledger="🦁☑ idle (nothing matched)")

        class _Decision:
            action = "rewrite"
            command = "uv run pytest"
            reason = "use uv"

        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_hooks_cfg", lambda: _MarkerCfg(), raising=False
        )
        monkeypatch.setattr(rc, "check_command", lambda *a, **k: _Decision())
        simba.hooks.pre_tool_use.main(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "pytest"},
                "cwd": str(tmp_path),
                "session_id": "sess-rw",
            }
        )
        # The marker record now carries the appended gate action.
        assert "rewrote: uv run pytest" in ef.last_ledger("sess-rw")
