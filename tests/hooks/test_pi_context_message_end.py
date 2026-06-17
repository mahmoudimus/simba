"""pi-native context re-injection + message_end doctrine-verify (spec 27, Tier 2).

These are pi-only canonical hooks. Both default-OFF → empty result (the pi bridge
applies nothing). On → context returns a ledger/doctrine to re-inject; message_end
returns a block_reason (the correction) on a doctrine violation.
"""

from __future__ import annotations

import simba.hooks.context as ctx_hook
import simba.hooks.message_end as me_hook


class _OffCfg:
    engagement_marker_enabled = False
    reasoning_verify_enabled = False


class _OnCfg:
    engagement_marker_enabled = True
    reasoning_verify_enabled = True
    prompt_min_similarity = 0.45
    prompt_min_length = 10
    pitfall_gate_types = "FAILURE,PREFERENCE,GOTCHA"
    pitfall_gate_max_results = 5
    pitfall_gate_mode = "violation"
    pitfall_gate_topical_floor = 0.70
    pitfall_gate_max_checks = 3
    pitfall_gate_min_similarity = 0.78
    pitfall_gate_fallback = "failure_only"


class TestContextHook:
    def test_off_by_default_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(ctx_hook, "_hooks_cfg", lambda: _OffCfg(), raising=False)
        result = ctx_hook.run({"messages_text": "let me edit the schema", "cwd": "/p"})
        assert result.additional_context == ""

    def test_on_reinjects_ledger(self, monkeypatch) -> None:
        monkeypatch.setattr(ctx_hook, "_hooks_cfg", lambda: _OnCfg(), raising=False)
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories",
            lambda *a, **k: [{"type": "GOTCHA", "content": "x", "similarity": 0.8}],
        )
        result = ctx_hook.run(
            {"messages_text": "let me edit the generated schema", "cwd": "/p"}
        )
        assert "🦁☑" in result.additional_context

    def test_on_no_query_text_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(ctx_hook, "_hooks_cfg", lambda: _OnCfg(), raising=False)
        result = ctx_hook.run({"messages_text": "", "cwd": "/p"})
        assert result.additional_context == ""

    def test_fail_open_on_recall_error(self, monkeypatch) -> None:
        monkeypatch.setattr(ctx_hook, "_hooks_cfg", lambda: _OnCfg(), raising=False)

        def boom(*a, **k):
            raise RuntimeError("daemon down")

        monkeypatch.setattr("simba.hooks._memory_client.recall_memories", boom)
        # Fail-open: never raises; ledger still emits idle (recall failed → 0).
        result = ctx_hook.run({"messages_text": "edit schema by hand", "cwd": "/p"})
        assert "🦁☑" in result.additional_context


class TestMessageEndHook:
    def test_off_by_default_no_block(self, monkeypatch) -> None:
        monkeypatch.setattr(me_hook, "_hooks_cfg", lambda: _OffCfg(), raising=False)
        result = me_hook.run(
            {"message_text": "I will just skip the failing test", "cwd": "/p"}
        )
        assert result.block_reason is None

    def test_on_violation_returns_block_reason(self, monkeypatch) -> None:
        monkeypatch.setattr(me_hook, "_hooks_cfg", lambda: _OnCfg(), raising=False)
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories",
            lambda *a, **k: [{"type": "FAILURE", "content": "don't skip tests"}],
        )
        monkeypatch.setattr(
            "simba.memory.pitfall.pitfall_note",
            lambda *a, **k: "<pitfall-warning>do not skip tests</pitfall-warning>",
        )
        result = me_hook.run(
            {"message_text": "I will just skip the failing test", "cwd": "/p"}
        )
        assert result.block_reason is not None
        assert "skip" in result.block_reason.lower()

    def test_on_no_violation_no_block(self, monkeypatch) -> None:
        monkeypatch.setattr(me_hook, "_hooks_cfg", lambda: _OnCfg(), raising=False)
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories", lambda *a, **k: []
        )
        monkeypatch.setattr("simba.memory.pitfall.pitfall_note", lambda *a, **k: "")
        result = me_hook.run({"message_text": "all good", "cwd": "/p"})
        assert result.block_reason is None


class TestCanonicalRegistration:
    def test_context_and_message_end_registered(self) -> None:
        import simba.harness.core as core

        assert core._EVENT_MODULES["context"] == "simba.hooks.context"
        assert core._EVENT_MODULES["message_end"] == "simba.hooks.message_end"
