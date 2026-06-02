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
        claude_md.write_text(
            "# Rules\n"
            "<!-- BEGIN SIMBA:core -->\n"
            "Important rule\n"
            "<!-- END SIMBA:core -->\n"
        )

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

    def test_hooks_config_exposes_prompt_floor(self):
        import simba.hooks.config

        cfg = simba.hooks.config.HooksConfig()
        assert cfg.prompt_min_similarity == 0.45
        assert cfg.prompt_min_length == 10

    def test_recall_floor_comes_from_config(self, tmp_path, monkeypatch):
        class _Stub:
            prompt_min_similarity = 0.6
            prompt_min_length = 5

        monkeypatch.setattr(
            simba.hooks.user_prompt_submit, "_cfg", lambda: _Stub(), raising=False
        )
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ) as mock_recall:
            simba.hooks.user_prompt_submit.main(
                {"prompt": "a long enough prompt", "cwd": str(tmp_path)}
            )
        _, kwargs = mock_recall.call_args
        assert kwargs["min_similarity"] == 0.6

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


class TestRlmPointerInjection:
    class _Cfg:
        def __init__(self, inject):
            self.inject_pointers = inject

    def test_off_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            "simba.config.load", lambda section, *a, **k: self._Cfg(False)
        )
        out = simba.hooks.user_prompt_submit._rlm_pointer_context(
            [{"content": "x"}], "/p"
        )
        assert out == ""

    def test_on_surfaces_only_available(self, monkeypatch):
        import simba.rlm.recall as rlm_recall

        monkeypatch.setattr(
            "simba.config.load", lambda section, *a, **k: self._Cfg(True)
        )
        monkeypatch.setattr(
            "simba.rlm.recall.pointers_from_memories",
            lambda mems, cwd, **k: [
                rlm_recall.Pointer("decided X", "sid-1", "/p", 0.8, True),
                rlm_recall.Pointer("no transcript", None, "/p", 0.7, False),
            ],
        )
        out = simba.hooks.user_prompt_submit._rlm_pointer_context(
            [{"content": "x"}], "/p"
        )
        assert "<rlm-pointers>" in out
        assert "sid-1" in out
        assert "no transcript" not in out  # unavailable filtered out

    def test_on_but_none_available(self, monkeypatch):
        monkeypatch.setattr(
            "simba.config.load", lambda section, *a, **k: self._Cfg(True)
        )
        monkeypatch.setattr(
            "simba.rlm.recall.pointers_from_memories", lambda mems, cwd, **k: []
        )
        out = simba.hooks.user_prompt_submit._rlm_pointer_context(
            [{"content": "x"}], "/p"
        )
        assert out == ""
