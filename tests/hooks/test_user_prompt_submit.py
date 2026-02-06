"""Tests for the UserPromptSubmit hook module."""

from __future__ import annotations

import json
import unittest.mock

import simba.hooks.user_prompt_submit


class TestUserPromptSubmitHook:
    def test_returns_valid_json(self, tmp_path):
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            result = json.loads(
                simba.hooks.user_prompt_submit.main({"cwd": str(tmp_path)})
            )
        assert "hookSpecificOutput" in result
        assert result["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"

    def test_skips_short_prompts(self, tmp_path):
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories"
        ) as mock_recall:
            simba.hooks.user_prompt_submit.main({"prompt": "hi", "cwd": str(tmp_path)})
        mock_recall.assert_not_called()

    def test_recalls_with_long_prompt(self, tmp_path):
        memories = [{"type": "GOTCHA", "content": "test memory", "similarity": 0.8}]
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories",
            return_value=memories,
        ):
            result = json.loads(
                simba.hooks.user_prompt_submit.main(
                    {
                        "prompt": "a sufficiently long prompt",
                        "cwd": str(tmp_path),
                    }
                )
            )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "recalled-memories" in ctx
        assert "test memory" in ctx

    def test_includes_core_blocks(self, tmp_path):
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("# Rules\n<!-- CORE -->\nImportant rule\n<!-- /CORE -->\n")

        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            result = json.loads(
                simba.hooks.user_prompt_submit.main({"cwd": str(tmp_path)})
            )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "Important rule" in ctx

    def test_passes_project_path_to_recall(self, tmp_path):
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories",
            return_value=[],
        ) as mock_recall:
            simba.hooks.user_prompt_submit.main(
                {
                    "prompt": "a sufficiently long prompt here",
                    "cwd": str(tmp_path),
                }
            )
        mock_recall.assert_called_once()
        _, kwargs = mock_recall.call_args
        assert kwargs["project_path"] == str(tmp_path)

    def test_includes_search_context(self, tmp_path):
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories",
                return_value=[],
            ),
            unittest.mock.patch(
                "simba.search.rag_context.build_context",
                return_value="<relevant-context>test search context</relevant-context>",
            ),
        ):
            result = json.loads(
                simba.hooks.user_prompt_submit.main(
                    {
                        "prompt": "a sufficiently long prompt for search",
                        "cwd": str(tmp_path),
                    }
                )
            )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "test search context" in ctx

    def test_search_context_error_does_not_crash(self, tmp_path):
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories",
                return_value=[],
            ),
            unittest.mock.patch(
                "simba.search.rag_context.build_context",
                side_effect=RuntimeError("search failed"),
            ),
        ):
            result = json.loads(
                simba.hooks.user_prompt_submit.main(
                    {
                        "prompt": "a sufficiently long prompt for search",
                        "cwd": str(tmp_path),
                    }
                )
            )
        assert "hookSpecificOutput" in result

    def test_empty_input(self):
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            result = json.loads(simba.hooks.user_prompt_submit.main({}))
        assert "hookSpecificOutput" in result
