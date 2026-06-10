"""Tests for the episodic consolidation orchestrator."""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.episodes.config as ecfg_mod
import simba.episodes.consolidate as ec
import simba.rlm.engine


@pytest.fixture(autouse=True)
def _patch_db_path(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


class FakeEngine:
    def __init__(self) -> None:
        self.runs: list[tuple[str, str]] = []
        self.sources: list[str] = []

    def run(self, prompt: str, *, cwd: str, session_source: str = "") -> None:
        self.runs.append((prompt, cwd))
        self.sources.append(session_source)

    def digest(self, *a, **k) -> None:  # pragma: no cover - protocol stub
        pass


def _cfg(**kw):
    return ecfg_mod.EpisodesConfig(
        min_memories=kw.get("min_memories", 2),
        **{k: v for k, v in kw.items() if k != "min_memories"},
    )


def _mem(mid, sid, mtype="GOTCHA", project="/proj", content="did x"):
    return {
        "id": mid,
        "type": mtype,
        "sessionSource": sid,
        "projectPath": project,
        "content": content,
        "context": "",
    }


class TestConsolidateSession:
    def test_dispatches_eligible(self) -> None:
        engine = FakeEngine()
        group = [_mem("m1", "s1"), _mem("m2", "s1"), _mem("m3", "s1")]
        status = ec.consolidate_session(
            "s1", cwd="/proj", group=group, ecfg=_cfg(), engine=engine
        )
        assert status == "dispatched"
        assert len(engine.runs) == 1
        prompt, cwd = engine.runs[0]
        assert "s1" in prompt
        assert "did x" in prompt
        assert cwd == "/proj"
        assert engine.sources == ["s1"]  # provenance threaded to the engine

    def test_too_few_not_dispatched(self) -> None:
        engine = FakeEngine()
        status = ec.consolidate_session(
            "s1", cwd="/proj", group=[_mem("m1", "s1")], ecfg=_cfg(), engine=engine
        )
        assert status == "too_few"
        assert engine.runs == []

    def test_existing_episode_skipped(self) -> None:
        engine = FakeEngine()
        group = [
            _mem("m1", "s1"),
            _mem("m2", "s1"),
            _mem("e1", "s1", mtype="EPISODE", content="summary"),
        ]
        status = ec.consolidate_session(
            "s1", cwd="/proj", group=group, ecfg=_cfg(), engine=engine
        )
        assert status == "exists"
        assert engine.runs == []

    def test_in_progress_dedup(self) -> None:
        engine = FakeEngine()
        group = [_mem("m1", "s1"), _mem("m2", "s1")]
        assert (
            ec.consolidate_session(
                "s1", cwd="/proj", group=group, ecfg=_cfg(), engine=engine
            )
            == "dispatched"
        )
        # second dispatch is blocked by the claimed episode_jobs row
        assert (
            ec.consolidate_session(
                "s1", cwd="/proj", group=group, ecfg=_cfg(), engine=engine
            )
            == "in_progress"
        )

    def test_no_engine(self, monkeypatch) -> None:
        monkeypatch.setattr(simba.rlm.engine, "get_engine", lambda cfg: None)
        group = [_mem("m1", "s1"), _mem("m2", "s1")]
        status = ec.consolidate_session("s1", cwd="/proj", group=group, ecfg=_cfg())
        assert status == "no_engine"


class TestConsolidateEligible:
    def _patch_list(self, monkeypatch, memories):
        monkeypatch.setattr(ec, "_list_memories", lambda *a, **k: memories)

    def test_scopes_to_project(self, monkeypatch) -> None:
        engine = FakeEngine()
        self._patch_list(
            monkeypatch,
            [
                _mem("a1", "s1", project="/proj"),
                _mem("a2", "s1", project="/proj"),
                _mem("b1", "s2", project="/other"),
                _mem("b2", "s2", project="/other"),
            ],
        )
        result = ec.consolidate_eligible(
            "/proj", ecfg=_cfg(), engine=engine, daemon_url="http://test"
        )
        assert result["dispatched"] == ["s1"]
        assert len(engine.runs) == 1

    def test_all_projects(self, monkeypatch) -> None:
        engine = FakeEngine()
        self._patch_list(
            monkeypatch,
            [
                _mem("a1", "s1", project="/proj"),
                _mem("a2", "s1", project="/proj"),
                _mem("b1", "s2", project="/other"),
                _mem("b2", "s2", project="/other"),
            ],
        )
        result = ec.consolidate_eligible(
            "/proj",
            all_projects=True,
            ecfg=_cfg(),
            engine=engine,
            daemon_url="http://test",
        )
        assert set(result["dispatched"]) == {"s1", "s2"}


class _RlmCfg:
    digest_prompt = ""
    engine_model = "deepseek-v4-flash"


class TestEpisodeTemplate:
    def test_default_is_agentic_for_claude_engine(self) -> None:
        claude = simba.rlm.engine.ClaudeCliEngine(_RlmCfg())
        assert ec._episode_template(_cfg(), claude) is ec._EPISODE_PROMPT

    def test_default_is_json_for_llm_engine(self) -> None:
        llm = simba.rlm.engine.LlmCliEngine(_RlmCfg())
        assert ec._episode_template(_cfg(), llm) is ec._LLM_EPISODE_PROMPT

    def test_config_override_wins_for_any_engine(self) -> None:
        llm = simba.rlm.engine.LlmCliEngine(_RlmCfg())
        tmpl = "CUSTOM {sid} {cwd} {members}"
        assert ec._episode_template(_cfg(episode_prompt=tmpl), llm) == tmpl

    def test_llm_engine_dispatches_json_prompt(self) -> None:
        engine = simba.rlm.engine.LlmCliEngine(_RlmCfg())
        captured = {}
        import subprocess

        orig = subprocess.Popen
        try:
            subprocess.Popen = (  # type: ignore[assignment]
                lambda argv, **kw: captured.update(argv=argv) or object()
            )
            group = [_mem("m1", "s1"), _mem("m2", "s1")]
            status = ec.consolidate_session(
                "s1", cwd="/proj", group=group, ecfg=_cfg(), engine=engine
            )
        finally:
            subprocess.Popen = orig  # type: ignore[assignment]
        assert status == "dispatched"
        pf = captured["argv"][captured["argv"].index("--prompt-file") + 1]
        body = pathlib.Path(pf).read_text()
        assert "JSON" in body and "s1" in body
        pathlib.Path(pf).unlink()


class TestEpisodeType:
    def test_episode_is_a_valid_memory_type(self) -> None:
        import simba.__main__ as cli
        import simba.memory.routes as routes

        assert "EPISODE" in routes.VALID_TYPES
        assert "EPISODE" in cli._VALID_MEMORY_TYPES
