"""The memory content cap is a single source of truth.

``memory.max_content_length`` drives both enforcement and every "keep content
under N chars" guidance string the daemon emits. These tests pin that the
resolver reads the config (with safe fallbacks) and that each prompt builder
hydrates the number it is given — so raising the config raises the guidance, and
nobody silently hardcodes 200 again.
"""

from __future__ import annotations

import simba.config
import simba.episodes.consolidate as cons
import simba.memory.config as mcfg
import simba.reflection.prompt as rp
import simba.rlm.engine as eng


class TestResolveMaxContentLength:
    def test_default_is_200(self, monkeypatch) -> None:
        monkeypatch.setattr(simba.config, "load", lambda *a, **k: mcfg.MemoryConfig())
        assert mcfg.resolve_max_content_length() == 200

    def test_reads_custom_value(self, monkeypatch) -> None:
        monkeypatch.setattr(
            simba.config,
            "load",
            lambda *a, **k: mcfg.MemoryConfig(max_content_length=750),
        )
        assert mcfg.resolve_max_content_length() == 750

    def test_fallback_on_load_error(self, monkeypatch) -> None:
        def _boom(*a, **k):
            raise RuntimeError("no config")

        monkeypatch.setattr(simba.config, "load", _boom)
        assert mcfg.resolve_max_content_length() == 200

    def test_nonpositive_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setattr(
            simba.config,
            "load",
            lambda *a, **k: mcfg.MemoryConfig(max_content_length=0),
        )
        assert mcfg.resolve_max_content_length() == 200


class TestPromptsHydrate:
    """Each builder must emit the cap it is handed (no hardcoded 200)."""

    def test_reflection_prompt(self) -> None:
        out = rp.build_reflection_prompt(
            [], project="p", existing_reflections=[], max_content_length=750
        )
        assert "≤750-char" in out
        assert "≤750 characters" in out
        assert "200" not in out

    def test_claude_digest_prompt(self) -> None:
        out = eng._build_digest_prompt("tid", "cwd", maxlen=750)
        assert "<<=750 chars>" in out
        assert "200" not in out

    def test_llm_digest_prompt(self) -> None:
        out = eng._LLM_DIGEST_PROMPT.format(
            transcript="x", cwd="c", tid="t", maxlen=750
        )
        assert "at most 750 characters" in out

    def test_agentic_episode_prompt(self) -> None:
        out = cons._build_episode_prompt("s", "c", [], 10, maxlen=750)
        assert "<=750-char summary>" in out
        assert "under 750 characters" in out
        assert "200" not in out

    def test_llm_episode_prompt(self) -> None:
        out = cons._LLM_EPISODE_PROMPT.format(sid="s", cwd="c", members="m", maxlen=750)
        assert "<=750-char summary>" in out
        assert "at most 750" in out
