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
                "simba.hooks.pre_tool_use._recall_memories",
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
                "simba.hooks.pre_tool_use._recall_memories",
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
