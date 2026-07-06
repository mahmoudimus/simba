"""Tests for the PreToolUse hook module."""

from __future__ import annotations

import dataclasses
import json
import unittest.mock

import simba.hooks.config
import simba.hooks.pre_tool_use


class TestPreToolUseHook:
    def test_skips_unsupported_tools(self):
        result = json.loads(simba.hooks.pre_tool_use.main({"tool_name": "Write"}))
        assert result["hookSpecificOutput"] == {"hookEventName": "PreToolUse"}

    def test_skips_missing_transcript(self, tmp_path):
        result = json.loads(
            simba.hooks.pre_tool_use.main(
                {
                    "tool_name": "Read",
                    "transcript_path": str(tmp_path / "nonexistent.jsonl"),
                }
            )
        )
        assert result["hookSpecificOutput"] == {"hookEventName": "PreToolUse"}

    def test_returns_memories_for_valid_tool(self, tmp_path):
        # Create transcript with thinking block
        transcript = tmp_path / "transcript.jsonl"
        entry = {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I need to check the auth module",
                    }
                ],
            }
        }
        transcript.write_text(json.dumps(entry) + "\n")

        memories = [{"type": "GOTCHA", "content": "auth uses JWT", "similarity": 0.5}]
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories",
                return_value=memories,
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            result = json.loads(
                simba.hooks.pre_tool_use.main(
                    {
                        "tool_name": "Read",
                        "transcript_path": str(transcript),
                    }
                )
            )
        ctx = result["hookSpecificOutput"].get("additionalContext", "")
        assert "auth uses JWT" in ctx

    def test_dedup_skips_repeated_thinking(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        entry = {
            "message": {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "repeated thought"}],
            }
        }
        transcript.write_text(json.dumps(entry) + "\n")

        with unittest.mock.patch(
            "simba.hooks.pre_tool_use._check_dedup", return_value=True
        ):
            result = json.loads(
                simba.hooks.pre_tool_use.main(
                    {
                        "tool_name": "Grep",
                        "transcript_path": str(transcript),
                    }
                )
            )
        assert result["hookSpecificOutput"] == {"hookEventName": "PreToolUse"}

    def test_passes_project_path(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        entry = {
            "message": {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "exploring the codebase"}],
            }
        }
        transcript.write_text(json.dumps(entry) + "\n")

        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories",
                return_value=[],
            ) as mock_recall,
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
        ):
            simba.hooks.pre_tool_use.main(
                {
                    "tool_name": "Read",
                    "transcript_path": str(transcript),
                    "cwd": "/my/project",
                }
            )
        mock_recall.assert_called_once()
        _, kwargs = mock_recall.call_args
        assert kwargs["project_path"] == "/my/project"


def _thinking_transcript(tmp_path, text):
    transcript = tmp_path / "transcript.jsonl"
    entry = {
        "message": {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": text}],
        }
    }
    transcript.write_text(json.dumps(entry) + "\n")
    return transcript


class TestRecallMinQueryChars:
    """MemOS borrow: parity with UserPromptSubmit's ``prompt_min_length`` floor.

    The general thinking-block recall previously had no length floor at all
    (deferred entirely to the daemon's similarity gate) — ``hooks.
    recall_min_query_chars`` adds the same cheap short-circuit UserPromptSubmit
    already has, so a near-empty thinking block never reaches the daemon.
    """

    def test_default_matches_prompt_min_length(self):
        cfg = simba.hooks.config.HooksConfig()
        assert cfg.recall_min_query_chars == 10
        assert cfg.recall_min_query_chars == cfg.prompt_min_length

    def test_short_thinking_skips_recall_entirely(self, tmp_path):
        # "ok" is 2 chars, well under the 10-char default floor.
        transcript = _thinking_transcript(tmp_path, "ok")
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories"
            ) as mock_recall,
            unittest.mock.patch("simba.hooks.pre_tool_use._check_dedup") as mock_dedup,
        ):
            result = json.loads(
                simba.hooks.pre_tool_use.main(
                    {"tool_name": "Read", "transcript_path": str(transcript)}
                )
            )
        mock_recall.assert_not_called()
        mock_dedup.assert_not_called()  # gate short-circuits before dedup work
        assert result["hookSpecificOutput"] == {"hookEventName": "PreToolUse"}

    def test_long_enough_thinking_still_recalls(self, tmp_path):
        # Exactly at the floor (10 chars) — inclusive boundary still recalls.
        transcript = _thinking_transcript(tmp_path, "a" * 10)
        memories = [{"type": "GOTCHA", "content": "at the floor", "similarity": 0.5}]
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories",
                return_value=memories,
            ) as mock_recall,
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            result = json.loads(
                simba.hooks.pre_tool_use.main(
                    {"tool_name": "Read", "transcript_path": str(transcript)}
                )
            )
        mock_recall.assert_called_once()
        ctx = result["hookSpecificOutput"].get("additionalContext", "")
        assert "at the floor" in ctx

    def test_custom_floor_configurable(self, tmp_path, monkeypatch):
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), recall_min_query_chars=50
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        # 22 chars — clears the default floor but not this raised one.
        transcript = _thinking_transcript(tmp_path, "a fairly short thought")
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories"
            ) as mock_recall,
            unittest.mock.patch("simba.hooks.pre_tool_use._check_dedup") as mock_dedup,
        ):
            simba.hooks.pre_tool_use.main(
                {"tool_name": "Read", "transcript_path": str(transcript)}
            )
        mock_recall.assert_not_called()
        mock_dedup.assert_not_called()

    def test_zero_floor_disables_gate(self, tmp_path, monkeypatch):
        # 0 = off: even a 1-char thinking block recalls (parity with
        # prompt_min_length's own semantics, where 0 would mean "no floor").
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), recall_min_query_chars=0
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        transcript = _thinking_transcript(tmp_path, "x")
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories",
                return_value=[],
            ) as mock_recall,
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
        ):
            simba.hooks.pre_tool_use.main(
                {"tool_name": "Read", "transcript_path": str(transcript)}
            )
        mock_recall.assert_called_once()


class _ViolatingLLM:
    def complete_json(self, prompt):
        return {
            "violates": True,
            "why": "reverting+xfail violates the no-weakening rule",
        }


class _NonViolatingLLM:
    def complete_json(self, prompt):
        return {"violates": False}


class TestPitfallGate:
    def test_disabled_by_default_no_fire(self, tmp_path, monkeypatch):
        # Default config has the gate OFF — even a strong doctrine match stays silent.
        # Pin the dataclass default so the assertion holds regardless of any ambient
        # .simba/config.toml (e.g. a parent repo enabling the gate); the rest of this
        # class uses the same _hooks_cfg pin.
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=False
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        transcript = _thinking_transcript(tmp_path, "I'll revert and xfail the test")
        hit = [
            {
                "type": "PREFERENCE",
                "content": "No assertion weakening",
                "similarity": 0.9,
            }
        ]
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=hit
        ):
            result = json.loads(
                simba.hooks.pre_tool_use.main(
                    {"tool_name": "Edit", "transcript_path": str(transcript)}
                )
            )
        # Edit is not an _ENABLED_TOOL and the gate is off → no injection at all.
        assert result["hookSpecificOutput"] == {"hookEventName": "PreToolUse"}

    def test_violation_fires_for_mutating_tool(self, tmp_path, monkeypatch):
        # violation mode (default) + LLM says the move violates → fires. Patch the LLM
        # client so the test never shells out to a real provider.
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=True
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: _ViolatingLLM()
        )
        transcript = _thinking_transcript(tmp_path, "I'll revert and xfail the test")
        hit = [
            {
                "type": "PREFERENCE",
                "content": "No assertion weakening",
                "similarity": 0.82,
            }
        ]
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=hit
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            # "Edit" is NOT in _ENABLED_TOOLS — the gate must still fire on it.
            result = json.loads(
                simba.hooks.pre_tool_use.main(
                    {"tool_name": "Edit", "transcript_path": str(transcript)}
                )
            )
        ctx = result["hookSpecificOutput"].get("additionalContext", "")
        assert "pitfall-warning" in ctx
        assert "No assertion weakening" in ctx

    def test_silent_when_topical_but_no_violation(self, tmp_path, monkeypatch):
        # The key fix: topically-close candidate the LLM judges NON-violating → silent.
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=True
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use,
            "_pitfall_llm_client",
            lambda _c: _NonViolatingLLM(),
        )
        transcript = _thinking_transcript(tmp_path, "storing concrete memories now")
        hit = [
            {
                "type": "PREFERENCE",
                "content": "store concrete memories",
                "similarity": 0.86,
            }
        ]
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=hit
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            result = json.loads(
                simba.hooks.pre_tool_use.main(
                    {"tool_name": "Edit", "transcript_path": str(transcript)}
                )
            )
        assert result["hookSpecificOutput"] == {"hookEventName": "PreToolUse"}

    def test_fallback_failure_only_when_no_llm(self, tmp_path, monkeypatch):
        # No LLM available → fall back to FAILURE-only similarity gate.
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=True
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: None
        )
        transcript = _thinking_transcript(tmp_path, "applying the same fix again")
        hit = [
            {
                "type": "FAILURE",
                "content": "that fix broke the build",
                "similarity": 0.84,
            }
        ]
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=hit
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            result = json.loads(
                simba.hooks.pre_tool_use.main(
                    {"tool_name": "Edit", "transcript_path": str(transcript)}
                )
            )
        ctx = result["hookSpecificOutput"].get("additionalContext", "")
        assert "pitfall-warning" in ctx and "that fix broke the build" in ctx

    def test_recalls_only_doctrine_types(self, tmp_path, monkeypatch):
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=True
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: None
        )
        transcript = _thinking_transcript(tmp_path, "thinking about a risky move")
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=[]
            ) as mock_recall,
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            simba.hooks.pre_tool_use.main(
                {"tool_name": "Edit", "transcript_path": str(transcript), "cwd": "/p"}
            )
        _, kwargs = mock_recall.call_args
        assert kwargs["filters"] == {"types": ["FAILURE", "PREFERENCE", "GOTCHA"]}
        assert kwargs["project_path"] == "/p"

    def test_does_not_fire_on_read_tool(self, tmp_path, monkeypatch):
        # The gate runs only before mutating tools (pitfall_gate_tools). On a read tool
        # (Grep) it must NOT fire even though the LLM would say "violates" — the
        # tool-type gate is what stops it (general recall may still inject context).
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=True
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: _ViolatingLLM()
        )
        transcript = _thinking_transcript(tmp_path, "I'll revert and xfail the test")
        hit = [
            {
                "type": "PREFERENCE",
                "content": "No assertion weakening",
                "similarity": 0.9,
            }
        ]
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=hit
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            result = json.loads(
                simba.hooks.pre_tool_use.main(
                    {"tool_name": "Grep", "transcript_path": str(transcript)}
                )
            )
        ctx = result["hookSpecificOutput"].get("additionalContext", "")
        assert "pitfall-warning" not in ctx  # gate suppressed on a read tool

    def test_check_pitfall_disabled_returns_none(self):
        # Unit: default config (gate off) → None regardless of input.
        assert simba.hooks.pre_tool_use._check_pitfall("anything", None) is None


class TestPitfallGatePayloadThinking:
    """The v2.1 pi path: ``thinking`` arrives in the payload (no transcript file).

    A fired pitfall must escalate to ``CanonicalResult.escalated_block`` (so pi's
    block-only tool gate enforces it) while still appearing in ``additional_context``
    (so Claude/Codex see it unchanged).
    """

    def _hit(self):
        return [
            {
                "type": "PREFERENCE",
                "content": "No assertion weakening",
                "similarity": 0.82,
            }
        ]

    def test_payload_thinking_escalates_pitfall(self, tmp_path, monkeypatch):
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=True
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: _ViolatingLLM()
        )
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=self._hit()
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            # No transcript_path — only payload thinking, as pi sends it.
            result = simba.hooks.pre_tool_use.run(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "git revert"},
                    "thinking": "I'll revert and xfail the test",
                }
            )
        assert result.escalated_block is not None
        assert "pitfall-warning" in result.escalated_block
        # The directive also rides in additional_context (Claude/Codex channel).
        assert "pitfall-warning" in result.additional_context
        assert result.escalated_block in result.additional_context

    def test_no_thinking_no_escalation(self, tmp_path, monkeypatch):
        # No transcript, no payload thinking → the gate never runs → no escalation.
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=True
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: _ViolatingLLM()
        )
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=self._hit()
        ):
            result = simba.hooks.pre_tool_use.run(
                {"tool_name": "Bash", "tool_input": {"command": "git revert"}}
            )
        assert result.escalated_block is None
        assert "pitfall-warning" not in result.additional_context

    def test_disabled_gate_no_escalation_with_thinking(self, tmp_path, monkeypatch):
        # Gate OFF (default) → payload thinking is ignored, no escalation.
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=False
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: _ViolatingLLM()
        )
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=self._hit()
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            result = simba.hooks.pre_tool_use.run(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "git revert"},
                    "thinking": "I'll revert and xfail the test",
                }
            )
        assert result.escalated_block is None

    def test_payload_thinking_preferred_over_transcript(self, tmp_path, monkeypatch):
        # When both are present, the payload thinking wins (pi never sends both, but
        # the precedence must be deterministic). Transcript holds a NON-violating
        # thought; the payload holds the violating one → the gate fires on the payload.
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=True
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: _ViolatingLLM()
        )
        seen = {}

        def _check(thinking, cwd):
            seen["thinking"] = thinking
            return None  # don't matter; we only assert which text was passed

        monkeypatch.setattr(simba.hooks.pre_tool_use, "_check_pitfall", _check)
        transcript = _thinking_transcript(tmp_path, "harmless transcript thought")
        with (
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            simba.hooks.pre_tool_use.run(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "git revert"},
                    "transcript_path": str(transcript),
                    "thinking": "payload violating thought",
                }
            )
        assert seen["thinking"] == "payload violating thought"


class TestCheckContextLow:
    def test_below_threshold_returns_none(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("small file\n")
        assert simba.hooks.pre_tool_use._check_context_low(transcript) is None

    def test_above_threshold_returns_warning(self, tmp_path, monkeypatch):
        low_cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), context_low_bytes=100
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: low_cfg)
        flag = tmp_path / "flag.json"
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_CONTEXT_LOW_FLAG", flag)

        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("x" * 200 + "\n")

        result = simba.hooks.pre_tool_use._check_context_low(transcript)
        assert result is not None
        assert "context-low-warning" in result
        assert "0.0MB" in result

    def test_warns_only_once(self, tmp_path, monkeypatch):
        low_cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), context_low_bytes=100
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: low_cfg)
        flag = tmp_path / "flag.json"
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_CONTEXT_LOW_FLAG", flag)

        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("x" * 200 + "\n")

        first = simba.hooks.pre_tool_use._check_context_low(transcript)
        assert first is not None

        second = simba.hooks.pre_tool_use._check_context_low(transcript)
        assert second is None

    def test_different_transcript_resets(self, tmp_path, monkeypatch):
        low_cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), context_low_bytes=100
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: low_cfg)
        flag = tmp_path / "flag.json"
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_CONTEXT_LOW_FLAG", flag)

        t1 = tmp_path / "t1.jsonl"
        t1.write_text("x" * 200 + "\n")
        t2 = tmp_path / "t2.jsonl"
        t2.write_text("x" * 200 + "\n")

        assert simba.hooks.pre_tool_use._check_context_low(t1) is not None
        assert simba.hooks.pre_tool_use._check_context_low(t1) is None
        assert simba.hooks.pre_tool_use._check_context_low(t2) is not None

    def test_nonexistent_transcript(self, tmp_path):
        assert simba.hooks.pre_tool_use._check_context_low(tmp_path / "nope") is None

    def test_fires_for_non_enabled_tool(self, tmp_path, monkeypatch):
        """Context warning fires even for tools not in _ENABLED_TOOLS."""
        low_cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), context_low_bytes=100
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: low_cfg)
        flag = tmp_path / "flag.json"
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_CONTEXT_LOW_FLAG", flag)

        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("x" * 200 + "\n")

        result = json.loads(
            simba.hooks.pre_tool_use.main(
                {"tool_name": "Write", "transcript_path": str(transcript)}
            )
        )
        ctx = result["hookSpecificOutput"].get("additionalContext", "")
        assert "context-low-warning" in ctx


class TestPreflightGate:
    """PreToolUse blocks a mutating tool with no preflight this turn (spec 28)."""

    @staticmethod
    def _cfg(monkeypatch, **over):
        import simba.guardian.preflight_flag as pf

        cfg = dataclasses.replace(simba.hooks.config.HooksConfig(), **over)
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        return cfg, pf

    def test_disabled_by_default_no_block(self, tmp_path, monkeypatch):
        # Gate OFF (default) → a mutating tool with no preflight is NOT blocked
        # (byte-identical to today: no preflight machinery on the path).
        _, pf = self._cfg(monkeypatch)
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        result = simba.hooks.pre_tool_use.run(
            {
                "tool_name": "Edit",
                "tool_input": {"file_path": "/x"},
                "session_id": "s",
            }
        )
        assert result.block_reason is None

    def test_mutating_tool_without_preflight_blocks(self, tmp_path, monkeypatch):
        _, pf = self._cfg(
            monkeypatch,
            preflight_mandate_enabled=True,
            preflight_mandate_risk_only=False,
        )
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        result = simba.hooks.pre_tool_use.run(
            {
                "tool_name": "Edit",
                "tool_input": {"file_path": "/x"},
                "session_id": "s",
            }
        )
        assert result.block_reason is not None
        assert "preflight" in result.block_reason.lower()

    def test_read_only_tool_allowed_without_preflight(self, tmp_path, monkeypatch):
        _, pf = self._cfg(
            monkeypatch,
            preflight_mandate_enabled=True,
            preflight_mandate_risk_only=False,
        )
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            result = simba.hooks.pre_tool_use.run(
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/x"},
                    "session_id": "s",
                }
            )
        assert result.block_reason is None  # read-only never gated

    def test_preflight_clears_the_gate(self, tmp_path, monkeypatch):
        _, pf = self._cfg(
            monkeypatch,
            preflight_mandate_enabled=True,
            preflight_mandate_risk_only=False,
        )
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        pf.set_preflight("s", task="edit the file")  # preflight ran this turn
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            result = simba.hooks.pre_tool_use.run(
                {
                    "tool_name": "Edit",
                    "tool_input": {"file_path": "/x"},
                    "session_id": "s",
                }
            )
        assert result.block_reason is None

    def test_risk_only_not_armed_without_risk_prime(self, tmp_path, monkeypatch):
        # risk_only=True (default) and no risk-tier prime this turn → gate is NOT
        # armed, so a mutating tool with no preflight is allowed.
        _, pf = self._cfg(
            monkeypatch,
            preflight_mandate_enabled=True,
            preflight_mandate_risk_only=True,
        )
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            result = simba.hooks.pre_tool_use.run(
                {
                    "tool_name": "Edit",
                    "tool_input": {"file_path": "/x"},
                    "session_id": "s",
                }
            )
        assert result.block_reason is None

    def test_risk_only_armed_blocks_without_preflight(self, tmp_path, monkeypatch):
        _, pf = self._cfg(
            monkeypatch,
            preflight_mandate_enabled=True,
            preflight_mandate_risk_only=True,
        )
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        pf.arm_mandate("s")  # a risk-tier doctrine was primed this turn
        result = simba.hooks.pre_tool_use.run(
            {
                "tool_name": "Edit",
                "tool_input": {"file_path": "/x"},
                "session_id": "s",
            }
        )
        assert result.block_reason is not None
