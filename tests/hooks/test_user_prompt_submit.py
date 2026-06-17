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

    def test_guardian_signal_gated_defaults_off(self):
        import simba.hooks.config

        # Default-OFF preserves today's behavior (CORE block every prompt).
        assert simba.hooks.config.HooksConfig().guardian_signal_gated is False


class TestIntentPriming:
    """Spec 28 Phase B: intent-primed injection at UserPromptSubmit."""

    def test_off_by_default_no_priming(self, tmp_path):
        # intent_priming_enabled defaults OFF → no <intent-priming> block, and the
        # priming machinery is never invoked (byte-identical to today).
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=[]
            ),
            unittest.mock.patch("simba.doctrine.store.list_doctrines") as mock_list,
        ):
            result = json.loads(
                simba.hooks.user_prompt_submit.main(
                    {"prompt": "please review PR #42", "cwd": str(tmp_path)}
                )
            )
        ctx = result["hookSpecificOutput"].get("additionalContext", "")
        assert "intent-priming" not in ctx
        mock_list.assert_not_called()  # off → not even loaded

    def test_on_injects_matched_doctrine(self, tmp_path, monkeypatch):
        import simba.doctrine.store as store

        class _Cfg:
            prompt_min_similarity = 0.45
            prompt_min_length = 10
            guardian_signal_gated = False
            intent_priming_enabled = True
            intent_priming_min_similarity = 0.55
            intent_priming_max_doctrines = 3
            preflight_mandate_enabled = False
            preflight_mandate_risk_only = True

        monkeypatch.setattr(
            simba.hooks.user_prompt_submit, "_cfg", lambda: _Cfg(), raising=False
        )
        doc = store.Doctrine(
            id="pr",
            doctrine="Use the worktree skill for PR review.",
            triggers=["review PR"],
            trigger_embeddings=[[1.0, 0.0]],
            risk_tier=True,
            applicable_rules=["redirect: git show pr-N -> worktree"],
            project_path=str(tmp_path),
        )
        monkeypatch.setattr("simba.doctrine.store.list_doctrines", lambda **kw: [doc])
        monkeypatch.setattr(
            "simba.hooks._memory_client.embed_text", lambda _t: [1.0, 0.0]
        )
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            result = json.loads(
                simba.hooks.user_prompt_submit.main(
                    {"prompt": "please review PR #42", "cwd": str(tmp_path)}
                )
            )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "intent-priming" in ctx
        assert "Use the worktree skill" in ctx

    def test_risk_prime_arms_mandate_and_injects_instruction(
        self, tmp_path, monkeypatch
    ):
        import simba.doctrine.store as store
        import simba.guardian.preflight_flag as pf

        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)

        class _Cfg:
            prompt_min_similarity = 0.45
            prompt_min_length = 10
            guardian_signal_gated = False
            intent_priming_enabled = True
            intent_priming_min_similarity = 0.55
            intent_priming_max_doctrines = 3
            preflight_mandate_enabled = True
            preflight_mandate_risk_only = True

        monkeypatch.setattr(
            simba.hooks.user_prompt_submit, "_cfg", lambda: _Cfg(), raising=False
        )
        doc = store.Doctrine(
            id="pr",
            doctrine="Use the worktree skill for PR review.",
            triggers=["review PR"],
            trigger_embeddings=[[1.0, 0.0]],
            risk_tier=True,
            applicable_rules=[],
            project_path=str(tmp_path),
        )
        monkeypatch.setattr("simba.doctrine.store.list_doctrines", lambda **kw: [doc])
        monkeypatch.setattr(
            "simba.hooks._memory_client.embed_text", lambda _t: [1.0, 0.0]
        )
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            result = json.loads(
                simba.hooks.user_prompt_submit.main(
                    {
                        "prompt": "please review PR #42",
                        "cwd": str(tmp_path),
                        "session_id": "sess-P",
                    }
                )
            )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "simba preflight" in ctx  # the mandate instruction
        assert pf.mandate_armed("sess-P") is True  # gate armed for the turn

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


class TestNoHardcodedThresholds:
    def test_prompt_min_similarity_reads_from_config(self, tmp_path, monkeypatch):
        """The hook passes cfg.prompt_min_similarity, not a literal."""

        class _Stub:
            prompt_min_similarity = 0.77
            prompt_min_length = 10

        captured: dict[str, float] = {}

        def fake_recall(query, *, project_path=None, min_similarity=None):
            captured["min_similarity"] = min_similarity
            return []

        monkeypatch.setattr(
            simba.hooks.user_prompt_submit, "_cfg", lambda: _Stub(), raising=False
        )
        monkeypatch.setattr("simba.hooks._memory_client.recall_memories", fake_recall)
        simba.hooks.user_prompt_submit.main({"prompt": "x" * 20, "cwd": str(tmp_path)})
        assert captured.get("min_similarity") == 0.77

    def test_no_hardcoded_similarity_in_hooks_source(self):
        """Regression: no magic 0.45 literal outside of HooksConfig default."""
        import pathlib
        import re

        src = pathlib.Path(__file__).resolve().parents[2] / "src" / "simba" / "hooks"
        for py in src.glob("*.py"):
            if py.name == "config.py":
                continue  # defaults are allowed here
            text = py.read_text()
            matches = re.findall(r'(?<!["\'\w])0\.45(?!["\'\w])', text)
            assert matches == [], f"Hardcoded 0.45 found in {py}: {matches}"


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


def _claude_md_with_core(tmp_path):
    (tmp_path / "CLAUDE.md").write_text(
        "# Rules\n<!-- BEGIN SIMBA:core -->\nImportant rule\n<!-- END SIMBA:core -->\n"
    )


class _GatedCfg:
    """HooksConfig stub with the signal-gating lever ON."""

    prompt_min_similarity = 0.45
    prompt_min_length = 10
    guardian_signal_gated = True


class TestGuardianSignalGating:
    """Proposal A (spec 25): conditional CORE re-injection."""

    def test_default_off_injects_every_prompt(self, tmp_path, monkeypatch):
        """Characterization: lever OFF (default) → CORE present every prompt,
        regardless of any recorded signal flag."""
        import simba.guardian.signal_flag as sf
        import simba.hooks.config

        _claude_md_with_core(tmp_path)
        monkeypatch.setattr(
            simba.hooks.user_prompt_submit,
            "_cfg",
            lambda: simba.hooks.config.HooksConfig(),
            raising=False,
        )
        # Even with a 'signal present' flag for this session, default-off ignores it.
        with (
            unittest.mock.patch.object(sf, "should_inject", return_value=False),
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=[]
            ),
        ):
            result = json.loads(
                simba.hooks.user_prompt_submit.main(
                    {"cwd": str(tmp_path), "session_id": "s-default"}
                )
            )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "Important rule" in ctx

    def test_gated_first_prompt_injects(self, tmp_path, monkeypatch):
        import simba.guardian.signal_flag as sf

        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        _claude_md_with_core(tmp_path)
        monkeypatch.setattr(
            simba.hooks.user_prompt_submit, "_cfg", lambda: _GatedCfg(), raising=False
        )
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            result = json.loads(
                simba.hooks.user_prompt_submit.main(
                    {"cwd": str(tmp_path), "session_id": "s-first"}
                )
            )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "Important rule" in ctx

    def test_gated_prior_signal_present_skips(self, tmp_path, monkeypatch):
        import simba.guardian.signal_flag as sf

        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        _claude_md_with_core(tmp_path)
        sf.record_signal("s-keep", present=True)
        monkeypatch.setattr(
            simba.hooks.user_prompt_submit, "_cfg", lambda: _GatedCfg(), raising=False
        )
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            result = json.loads(
                simba.hooks.user_prompt_submit.main(
                    {"cwd": str(tmp_path), "session_id": "s-keep"}
                )
            )
        # No CORE block (and no memories) → empty context envelope (no
        # additionalContext key), and certainly no "Important rule"/"✓ rules" tag.
        ctx = result["hookSpecificOutput"].get("additionalContext", "")
        assert "Important rule" not in ctx
        assert "✓ rules" not in ctx

    def test_gated_prior_signal_missing_injects(self, tmp_path, monkeypatch):
        import simba.guardian.signal_flag as sf

        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        _claude_md_with_core(tmp_path)
        sf.record_signal("s-decayed", present=False)
        monkeypatch.setattr(
            simba.hooks.user_prompt_submit, "_cfg", lambda: _GatedCfg(), raising=False
        )
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            result = json.loads(
                simba.hooks.user_prompt_submit.main(
                    {"cwd": str(tmp_path), "session_id": "s-decayed"}
                )
            )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "Important rule" in ctx

    def test_gated_fails_open_on_error(self, tmp_path, monkeypatch):
        """Any error in the decision → inject (never silently drop the rules)."""
        import simba.guardian.signal_flag as sf

        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        _claude_md_with_core(tmp_path)
        monkeypatch.setattr(
            simba.hooks.user_prompt_submit, "_cfg", lambda: _GatedCfg(), raising=False
        )

        def boom(_session_id):
            raise RuntimeError("flag read failed")

        monkeypatch.setattr(sf, "should_inject", boom)
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            result = json.loads(
                simba.hooks.user_prompt_submit.main(
                    {"cwd": str(tmp_path), "session_id": "s-err"}
                )
            )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "Important rule" in ctx
