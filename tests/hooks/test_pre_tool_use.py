"""Tests for the PreToolUse hook module."""

from __future__ import annotations

import json
import unittest.mock

import simba.hooks.pre_tool_use


class TestPreToolUseHook:
    def test_skips_unsupported_tools(self):
        result = json.loads(simba.hooks.pre_tool_use.main({"tool_name": "Write"}))
        assert result["hookSpecificOutput"] == {}

    def test_skips_missing_transcript(self, tmp_path):
        result = json.loads(
            simba.hooks.pre_tool_use.main(
                {
                    "tool_name": "Read",
                    "transcript_path": str(tmp_path / "nonexistent.jsonl"),
                }
            )
        )
        assert result["hookSpecificOutput"] == {}

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
        assert result["hookSpecificOutput"] == {}

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


class TestCheckContextLow:
    def test_below_threshold_returns_none(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("small file\n")
        assert simba.hooks.pre_tool_use._check_context_low(transcript) is None

    def test_above_threshold_returns_warning(self, tmp_path, monkeypatch):
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_CONTEXT_LOW_BYTES", 100)
        flag = tmp_path / "flag.json"
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_CONTEXT_LOW_FLAG", flag)

        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("x" * 200 + "\n")

        result = simba.hooks.pre_tool_use._check_context_low(transcript)
        assert result is not None
        assert "context-low-warning" in result
        assert "0.0MB" in result

    def test_warns_only_once(self, tmp_path, monkeypatch):
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_CONTEXT_LOW_BYTES", 100)
        flag = tmp_path / "flag.json"
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_CONTEXT_LOW_FLAG", flag)

        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("x" * 200 + "\n")

        first = simba.hooks.pre_tool_use._check_context_low(transcript)
        assert first is not None

        second = simba.hooks.pre_tool_use._check_context_low(transcript)
        assert second is None

    def test_different_transcript_resets(self, tmp_path, monkeypatch):
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_CONTEXT_LOW_BYTES", 100)
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
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_CONTEXT_LOW_BYTES", 100)
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
