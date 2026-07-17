"""Tests for the episodic consolidation orchestrator."""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.episodes.config as ecfg_mod
import simba.episodes.consolidate as ec
import simba.episodes.watermark as ewm
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


def _mem(mid, sid, mtype="GOTCHA", project="/proj", content="did x", created_at=None):
    return {
        "id": mid,
        "type": mtype,
        "sessionSource": sid,
        "projectPath": project,
        "content": content,
        "context": "",
        "createdAt": created_at or "2026-01-01T00:00:00Z",
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
        """Back-compat shim: wire both `_discover` (the projected discovery
        scan) and `_fetch_session` (the full, session-scoped fetch) off the
        same flat memory list -- `_fetch_session` groups it server-side-
        style by `sessionSource`, mirroring what a real `sessionSource=`
        filter on `/list` would return. Sufficient for tests that don't
        exercise incremental-discovery semantics directly (see
        TestIncrementalDiscovery for those)."""
        groups = ec._group_by_session(memories)
        monkeypatch.setattr(ec, "_discover", lambda *a, **k: memories)
        monkeypatch.setattr(
            ec, "_fetch_session", lambda daemon_url, sid: groups.get(sid, [])
        )

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


class TestIncrementalDiscovery:
    """Spec: 2026-07-17 RSS-storm fix -- `consolidate_eligible` discovers via
    a projected (no content/context) scan, optionally bounded by a
    per-project watermark, then fetches full session content only for
    sessions that survive the eligibility pre-check."""

    def _wire(self, monkeypatch, discovered, sessions):
        calls: dict[str, list] = {"discover_since": [], "fetch": []}

        def _discover(daemon_url, *, since=None):
            calls["discover_since"].append(since)
            return discovered

        def _fetch_session(daemon_url, sid):
            calls["fetch"].append(sid)
            return sessions.get(sid, [])

        monkeypatch.setattr(ec, "_discover", _discover)
        monkeypatch.setattr(ec, "_fetch_session", _fetch_session)
        return calls

    def test_watermark_advances_on_clean_sweep(self, monkeypatch) -> None:
        engine = FakeEngine()
        discovered = [
            _mem("m1", "s1", created_at="2026-01-01T00:00:00Z"),
            _mem("m2", "s1", created_at="2026-01-02T00:00:00Z"),
        ]
        sessions = {"s1": discovered}
        self._wire(monkeypatch, discovered, sessions)

        result = ec.consolidate_eligible(
            "/proj", ecfg=_cfg(), engine=engine, daemon_url="http://test"
        )

        assert result["dispatched"] == ["s1"]
        assert ewm.get("/proj") == "2026-01-02T00:00:00Z"

    def test_watermark_does_not_advance_on_error(self, monkeypatch) -> None:
        class BrokenEngine:
            def run(self, *a, **k):
                raise RuntimeError("boom")

        # First sweep: succeeds, watermark advances to V1.
        engine = FakeEngine()
        first = [
            _mem("m1", "s1", created_at="2026-01-01T00:00:00Z"),
            _mem("m2", "s1", created_at="2026-01-02T00:00:00Z"),
        ]
        self._wire(monkeypatch, first, {"s1": first})
        ec.consolidate_eligible(
            "/proj", ecfg=_cfg(), engine=engine, daemon_url="http://test"
        )
        v1 = ewm.get("/proj")
        assert v1 == "2026-01-02T00:00:00Z"

        # Second sweep: a NEW, later-timestamped session dispatch errors --
        # watermark must NOT advance past V1, so the span is retried.
        second = [
            _mem("m3", "s2", created_at="2026-03-01T00:00:00Z"),
            _mem("m4", "s2", created_at="2026-03-02T00:00:00Z"),
        ]
        self._wire(monkeypatch, second, {"s2": second})
        result = ec.consolidate_eligible(
            "/proj", ecfg=_cfg(), engine=BrokenEngine(), daemon_url="http://test"
        )
        assert result["dispatched"] == []
        assert ewm.get("/proj") == v1

    def test_incremental_discovery_disabled_ignores_watermark(
        self, monkeypatch
    ) -> None:
        ewm.advance("/proj", "2026-01-01T00:00:00Z")
        engine = FakeEngine()
        discovered = [
            _mem("m1", "s1", created_at="2026-02-01T00:00:00Z"),
            _mem("m2", "s1", created_at="2026-02-02T00:00:00Z"),
        ]
        calls = self._wire(monkeypatch, discovered, {"s1": discovered})

        ec.consolidate_eligible(
            "/proj",
            ecfg=_cfg(incremental_discovery=False),
            engine=engine,
            daemon_url="http://test",
        )

        assert calls["discover_since"] == [None]
        assert ewm.get("/proj") == "2026-01-01T00:00:00Z"  # untouched

    def test_rediscovered_session_with_pre_and_post_watermark_memory_fetched_in_full(
        self, monkeypatch
    ) -> None:
        """The core correctness guarantee: a session already has 1 member
        before the watermark (too few, min_memories=2) and gains exactly 1
        NEW member after it. The incremental discovery scan (since=) only
        returns the new row -- but the session must still be `_fetch_session`-
        ed in FULL (recovering the pre-watermark member too) and become
        eligible, not wrongly skipped as "too few" off the partial batch."""
        ewm.advance("/proj", "2026-01-01T00:00:00Z")
        engine = FakeEngine()
        # Discovery (since=watermark) sees only the NEW post-watermark row.
        discovered = [_mem("m2", "s1", created_at="2026-01-02T00:00:00Z")]
        # The full session actually has 2 members (pre- + post-watermark).
        full_session = [
            _mem("m1", "s1", created_at="2025-06-01T00:00:00Z"),
            _mem("m2", "s1", created_at="2026-01-02T00:00:00Z"),
        ]
        calls = self._wire(monkeypatch, discovered, {"s1": full_session})

        result = ec.consolidate_eligible(
            "/proj", ecfg=_cfg(min_memories=2), engine=engine, daemon_url="http://test"
        )

        assert calls["discover_since"] == ["2026-01-01T00:00:00Z"]
        assert calls["fetch"] == ["s1"]
        assert result["dispatched"] == ["s1"]
        assert len(engine.runs) == 1
        prompt, _cwd = engine.runs[0]
        assert "m1" in prompt or "did x" in prompt  # pre-watermark member baked in
        assert ewm.get("/proj") == "2026-01-02T00:00:00Z"

    def test_all_projects_uses_all_projects_watermark_key(self, monkeypatch) -> None:
        engine = FakeEngine()
        discovered = [
            _mem("m1", "s1", project="/proj", created_at="2026-01-01T00:00:00Z"),
            _mem("m2", "s1", project="/proj", created_at="2026-01-02T00:00:00Z"),
        ]
        self._wire(monkeypatch, discovered, {"s1": discovered})

        ec.consolidate_eligible(
            "/proj",
            all_projects=True,
            ecfg=_cfg(),
            engine=engine,
            daemon_url="http://test",
        )

        assert ewm.get("/proj", all_projects=True) == "2026-01-02T00:00:00Z"
        assert ewm.get("/proj") is None  # the per-project key is untouched


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
