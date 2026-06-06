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


def test_disabled_returns_early(monkeypatch):
    from simba.reflection.config import ReflectionConfig
    from simba.reflection.pass_ import reflect_pass

    result = reflect_pass(
        cwd="/proj", rcfg=ReflectionConfig(enabled=False), engine=FakeEngine()
    )
    assert result.status == "disabled"
    assert not result.dispatched


def test_interval_gate(monkeypatch):
    import simba.reflection.pass_ as rp
    from simba.reflection.config import ReflectionConfig
    from simba.reflection.pass_ import reflect_pass

    monkeypatch.setattr(rp, "_list_memories", lambda *a, **k: _make_mems(20))
    result = reflect_pass(
        cwd="/proj",
        cycle_count=1,
        rcfg=ReflectionConfig(interval_cycles=5),
        engine=FakeEngine(),
    )
    assert result.status == "skipped_interval"


def test_too_few_memories(monkeypatch):
    import simba.reflection.pass_ as rp
    from simba.reflection.config import ReflectionConfig
    from simba.reflection.pass_ import reflect_pass

    monkeypatch.setattr(rp, "_list_memories", lambda *a, **k: _make_mems(3))
    result = reflect_pass(
        cwd="/proj",
        rcfg=ReflectionConfig(min_source_memories=10),
        engine=FakeEngine(),
    )
    assert result.status == "too_few"


def test_dispatches_when_eligible(monkeypatch):
    import simba.reflection.pass_ as rp
    from simba.reflection.config import ReflectionConfig
    from simba.reflection.pass_ import reflect_pass

    monkeypatch.setattr(rp, "_list_memories", lambda *a, **k: _make_mems(15))
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
    import simba.reflection.pass_ as rp
    import simba.rlm.engine
    from simba.reflection.config import ReflectionConfig
    from simba.reflection.pass_ import reflect_pass

    monkeypatch.setattr(rp, "_list_memories", lambda *a, **k: _make_mems(20))
    monkeypatch.setattr(simba.rlm.engine, "get_engine", lambda cfg: None)
    result = reflect_pass(cwd="/proj", rcfg=ReflectionConfig(min_source_memories=5))
    assert result.status == "no_engine"


def test_engine_error_returns_error(monkeypatch):
    import simba.reflection.pass_ as rp
    from simba.reflection.config import ReflectionConfig
    from simba.reflection.pass_ import reflect_pass

    monkeypatch.setattr(rp, "_list_memories", lambda *a, **k: _make_mems(15))

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
