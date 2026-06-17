"""Stop / SubagentStop doctrine-verify + block-to-reconsider (spec 27, Phase E)."""

from __future__ import annotations

import json

import simba.harness.adapters.claude as claude
import simba.harness.core
import simba.hooks.stop
import simba.hooks.subagent_stop
from simba.harness.core import CanonicalResult


class _VerifyCfg:
    engagement_marker_enabled = False
    reasoning_verify_enabled = True
    pitfall_gate_types = "FAILURE,PREFERENCE,GOTCHA"
    pitfall_gate_max_results = 5
    pitfall_gate_mode = "violation"
    pitfall_gate_topical_floor = 0.70
    pitfall_gate_max_checks = 3
    pitfall_gate_min_similarity = 0.78
    pitfall_gate_fallback = "failure_only"


class _OffCfg:
    engagement_marker_enabled = False
    reasoning_verify_enabled = False


class TestStopBlockToReconsider:
    def test_off_by_default_no_block(self, tmp_path, monkeypatch) -> None:
        # Characterization: lever OFF (default) → Stop never sets block_reason.
        monkeypatch.setattr(
            simba.hooks.stop, "_hooks_cfg", lambda: _OffCfg(), raising=False
        )
        result = simba.hooks.stop.run(
            {"response": "I will just skip the failing test", "cwd": str(tmp_path)}
        )
        assert result.block_reason is None

    def test_on_violation_sets_block_reason(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            simba.hooks.stop, "_hooks_cfg", lambda: _VerifyCfg(), raising=False
        )
        # The doctrine-verify recalls scars + asks pitfall.pitfall_note; stub both.
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories",
            lambda *a, **k: [{"type": "FAILURE", "content": "don't skip tests"}],
        )
        monkeypatch.setattr(
            "simba.memory.pitfall.pitfall_note",
            lambda *a, **k: "<pitfall-warning>do not skip</pitfall-warning>",
        )
        result = simba.hooks.stop.run(
            {"response": "I will just skip the failing test", "cwd": str(tmp_path)}
        )
        assert result.block_reason is not None
        assert "skip" in result.block_reason.lower()

    def test_on_no_violation_no_block(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            simba.hooks.stop, "_hooks_cfg", lambda: _VerifyCfg(), raising=False
        )
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories", lambda *a, **k: []
        )
        monkeypatch.setattr("simba.memory.pitfall.pitfall_note", lambda *a, **k: "")
        result = simba.hooks.stop.run(
            {"response": "All done correctly", "cwd": str(tmp_path)}
        )
        assert result.block_reason is None

    def test_adapter_maps_stop_block_to_decision_block(self) -> None:
        out = claude.render("Stop", CanonicalResult(block_reason="reconsider this"))
        assert json.loads(out) == {"decision": "block", "reason": "reconsider this"}


class TestSubagentStop:
    def test_canonical_event_registered(self) -> None:
        assert "subagent_stop" in simba.harness.core._EVENT_MODULES
        assert claude.NATIVE_TO_CANONICAL["SubagentStop"] == "subagent_stop"

    def test_off_by_default_no_block(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            simba.hooks.subagent_stop, "_hooks_cfg", lambda: _OffCfg(), raising=False
        )
        result = simba.hooks.subagent_stop.run(
            {"response": "subagent answer", "cwd": str(tmp_path)}
        )
        assert result.block_reason is None

    def test_on_violation_blocks(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            simba.hooks.subagent_stop, "_hooks_cfg", lambda: _VerifyCfg(), raising=False
        )
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories",
            lambda *a, **k: [{"type": "FAILURE", "content": "x"}],
        )
        monkeypatch.setattr(
            "simba.memory.pitfall.pitfall_note",
            lambda *a, **k: "<pitfall-warning>violated</pitfall-warning>",
        )
        result = simba.hooks.subagent_stop.run(
            {"response": "I edited the generated schema by hand", "cwd": str(tmp_path)}
        )
        assert result.block_reason is not None

    def test_adapter_renders_subagent_stop_block(self) -> None:
        out = claude.render("SubagentStop", CanonicalResult(block_reason="redo"))
        assert json.loads(out) == {"decision": "block", "reason": "redo"}

    def test_adapter_subagent_stop_empty_is_empty_object(self) -> None:
        out = claude.render("SubagentStop", CanonicalResult())
        assert json.loads(out) == {}
