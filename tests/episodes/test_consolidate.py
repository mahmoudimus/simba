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

    def run(self, prompt: str, *, cwd: str) -> None:
        self.runs.append((prompt, cwd))

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


class TestEpisodeType:
    def test_episode_is_a_valid_memory_type(self) -> None:
        import simba.__main__ as cli
        import simba.memory.routes as routes

        assert "EPISODE" in routes.VALID_TYPES
        assert "EPISODE" in cli._VALID_MEMORY_TYPES
