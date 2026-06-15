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
