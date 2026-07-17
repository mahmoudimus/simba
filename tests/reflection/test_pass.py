"""reflect_pass orchestrator (Phase 5, Task A.4)."""

from __future__ import annotations


class FakeEngine:
    def __init__(self):
        self.runs = []

    def run(self, prompt, *, cwd):
        self.runs.append((prompt, cwd))


def _make_mems(n, mtype="GOTCHA", project="/proj"):
    return [
        {
            "id": f"m{i}",
            "type": mtype,
            "content": f"c{i}",
            "context": "",
            "projectPath": project,
        }
        for i in range(n)
    ]


def _wire(monkeypatch, mems):
    """Back-compat shim: wire both `_discover` (the projected, content-free
    scan used only to count eligible memories) and `_fetch_source` (the full
    content+context fetch, only called once discovery proves the pass
    eligible) off the same flat memory list -- sufficient for tests that
    don't exercise the discovery/fetch split directly (see
    TestDiscoveryFetchSplit for those). 2026-07-17 RSS-storm fix: see
    simba/reflection/pass_.py's module docstring."""
    import simba.reflection.pass_ as rp

    monkeypatch.setattr(rp, "_discover", lambda *a, **k: mems)
    monkeypatch.setattr(rp, "_fetch_source", lambda *a, **k: mems)


def test_disabled_returns_early(monkeypatch):
    from simba.reflection.config import ReflectionConfig
    from simba.reflection.pass_ import reflect_pass

    result = reflect_pass(
        cwd="/proj", rcfg=ReflectionConfig(enabled=False), engine=FakeEngine()
    )
    assert result.status == "disabled"
    assert not result.dispatched


def test_interval_gate(monkeypatch):
    from simba.reflection.config import ReflectionConfig
    from simba.reflection.pass_ import reflect_pass

    _wire(monkeypatch, _make_mems(20))
    result = reflect_pass(
        cwd="/proj",
        cycle_count=1,
        rcfg=ReflectionConfig(interval_cycles=5),
        engine=FakeEngine(),
    )
    assert result.status == "skipped_interval"


def test_too_few_memories(monkeypatch):
    from simba.reflection.config import ReflectionConfig
    from simba.reflection.pass_ import reflect_pass

    _wire(monkeypatch, _make_mems(3))
    result = reflect_pass(
        cwd="/proj",
        rcfg=ReflectionConfig(min_source_memories=10),
        engine=FakeEngine(),
    )
    assert result.status == "too_few"


def test_dispatches_when_eligible(monkeypatch):
    from simba.reflection.config import ReflectionConfig
    from simba.reflection.pass_ import reflect_pass

    _wire(monkeypatch, _make_mems(15))
    engine = FakeEngine()
    result = reflect_pass(
        cwd="/proj",
        cycle_count=0,
        rcfg=ReflectionConfig(min_source_memories=5, interval_cycles=0),
        engine=engine,
    )
    assert result.status == "dispatched"
    assert result.dispatched
    assert len(engine.runs) == 1
    assert "/proj" in engine.runs[0][0]


def test_no_engine_returns_no_engine(monkeypatch):
    import simba.rlm.engine
    from simba.reflection.config import ReflectionConfig
    from simba.reflection.pass_ import reflect_pass

    _wire(monkeypatch, _make_mems(20))
    monkeypatch.setattr(simba.rlm.engine, "get_engine", lambda cfg: None)
    result = reflect_pass(cwd="/proj", rcfg=ReflectionConfig(min_source_memories=5))
    assert result.status == "no_engine"


def test_engine_error_returns_error(monkeypatch):
    from simba.reflection.config import ReflectionConfig
    from simba.reflection.pass_ import reflect_pass

    _wire(monkeypatch, _make_mems(15))

    class BrokenEngine:
        def run(self, *a, **k):
            raise RuntimeError("boom")

    result = reflect_pass(
        cwd="/proj",
        rcfg=ReflectionConfig(min_source_memories=5, interval_cycles=0),
        engine=BrokenEngine(),
    )
    assert result.status == "error"
    assert result.errors == 1


class TestDiscoveryFetchSplit:
    """2026-07-17 RSS-storm fix: `reflect_pass` must never pay for a
    content+context fetch when the projected discovery scan alone already
    proves the pass ineligible, and a project-scoped pass must bound both
    calls server-side to its own project instead of the whole corpus."""

    def test_too_few_gate_never_fetches_full_content(self, monkeypatch):
        import simba.reflection.pass_ as rp
        from simba.reflection.config import ReflectionConfig
        from simba.reflection.pass_ import reflect_pass

        fetch_calls = []
        monkeypatch.setattr(rp, "_discover", lambda *a, **k: _make_mems(3))
        monkeypatch.setattr(
            rp, "_fetch_source", lambda *a, **k: fetch_calls.append(1) or []
        )

        result = reflect_pass(
            cwd="/proj",
            rcfg=ReflectionConfig(min_source_memories=10),
            engine=FakeEngine(),
        )

        assert result.status == "too_few"
        assert fetch_calls == []  # never paid for content+context

    def test_eligible_pass_scopes_both_calls_to_project(self, monkeypatch):
        import simba.reflection.pass_ as rp
        from simba.reflection.config import ReflectionConfig
        from simba.reflection.pass_ import reflect_pass

        seen = {"discover": [], "fetch": []}

        def _discover(daemon_url, *, project_path=None):
            seen["discover"].append(project_path)
            return _make_mems(15)

        def _fetch_source(daemon_url, *, project_path=None):
            seen["fetch"].append(project_path)
            return _make_mems(15)

        monkeypatch.setattr(rp, "_discover", _discover)
        monkeypatch.setattr(rp, "_fetch_source", _fetch_source)

        engine = FakeEngine()
        result = reflect_pass(
            cwd="/proj",
            rcfg=ReflectionConfig(
                min_source_memories=5, interval_cycles=0, project_scoped=True
            ),
            engine=engine,
        )

        assert result.status == "dispatched"
        assert seen["discover"] == ["/proj"]
        assert seen["fetch"] == ["/proj"]

    def test_global_pass_does_not_scope_by_project(self, monkeypatch):
        import simba.reflection.pass_ as rp
        from simba.reflection.config import ReflectionConfig
        from simba.reflection.pass_ import reflect_pass

        seen = {"discover": [], "fetch": []}

        def _discover(daemon_url, *, project_path=None):
            seen["discover"].append(project_path)
            return _make_mems(15)

        def _fetch_source(daemon_url, *, project_path=None):
            seen["fetch"].append(project_path)
            return _make_mems(15)

        monkeypatch.setattr(rp, "_discover", _discover)
        monkeypatch.setattr(rp, "_fetch_source", _fetch_source)

        engine = FakeEngine()
        reflect_pass(
            cwd="/proj",
            rcfg=ReflectionConfig(
                min_source_memories=5, interval_cycles=0, project_scoped=False
            ),
            engine=engine,
        )

        assert seen["discover"] == [None]
        assert seen["fetch"] == [None]
