"""Answer-time conflict surfacing: detect a real contradiction among retrieved
memories with ONE LLM call, then emit a directive that NAMES it and tells the
consumer to surface (not pick a side). Gated + fail-open. Mirrors the fake-llm
style of test_entity_bridge / judge tests (no live model)."""

from __future__ import annotations

import dataclasses

import simba.memory.conflict as conflict


class FakeLlm:
    """Canned llm_client: returns a fixed JSON-ish reply (or raises)."""

    def __init__(self, reply, *, raises: bool = False):
        self._reply = reply
        self._raises = raises
        self.calls = 0

    def complete_json(self, prompt: str):
        self.calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return self._reply

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return ""


@dataclasses.dataclass
class FakeCfg:
    conflict_surfacing_enabled: bool = True
    conflict_surfacing_min_memories: int = 2


MEMS = [
    "Alice lives in Paris.",
    "Alice lives in Berlin.",
    "Bob likes coffee.",
]


def test_detect_conflict_resolves_indices_to_texts():
    llm = FakeLlm({"conflict": True, "a": 0, "b": 1, "description": "two cities"})
    res = conflict.detect_conflict(MEMS, "Where does Alice live?", llm_client=llm)
    assert res is not None
    assert res.a == "Alice lives in Paris."
    assert res.b == "Alice lives in Berlin."
    assert res.description == "two cities"
    assert llm.calls == 1  # exactly one LLM call


def test_detect_conflict_returns_none_when_no_conflict():
    llm = FakeLlm({"conflict": False, "a": 0, "b": 1, "description": ""})
    assert conflict.detect_conflict(MEMS, "q", llm_client=llm) is None


def test_detect_conflict_fail_open_on_garbage():
    llm = FakeLlm("not json at all")
    assert conflict.detect_conflict(MEMS, "q", llm_client=llm) is None


def test_detect_conflict_fail_open_on_exception():
    llm = FakeLlm(None, raises=True)
    assert conflict.detect_conflict(MEMS, "q", llm_client=llm) is None


def test_detect_conflict_fail_open_on_empty_input():
    llm = FakeLlm({"conflict": True, "a": 0, "b": 1, "description": "x"})
    assert conflict.detect_conflict([], "q", llm_client=llm) is None
    assert llm.calls == 0  # never call the LLM with nothing to compare


def test_detect_conflict_fail_open_on_out_of_range_indices():
    llm = FakeLlm({"conflict": True, "a": 0, "b": 99, "description": "x"})
    assert conflict.detect_conflict(MEMS, "q", llm_client=llm) is None


def test_detect_conflict_fail_open_on_same_index():
    llm = FakeLlm({"conflict": True, "a": 1, "b": 1, "description": "x"})
    assert conflict.detect_conflict(MEMS, "q", llm_client=llm) is None


def test_surface_directive_names_both_and_description():
    res = conflict.ConflictResult(
        a="Alice lives in Paris.",
        b="Alice lives in Berlin.",
        description="two cities",
    )
    d = conflict.surface_directive(res)
    assert "Alice lives in Paris." in d
    assert "Alice lives in Berlin." in d
    assert "two cities" in d
    # It must steer toward surfacing, not picking a side.
    assert "confirm" in d.lower()


def test_conflict_note_empty_when_disabled():
    llm = FakeLlm({"conflict": True, "a": 0, "b": 1, "description": "x"})
    cfg = FakeCfg(conflict_surfacing_enabled=False)
    assert conflict.conflict_note(MEMS, "q", cfg=cfg, llm_client=llm) == ""
    assert llm.calls == 0  # disabled → no LLM cost


def test_conflict_note_empty_when_below_min_memories():
    llm = FakeLlm({"conflict": True, "a": 0, "b": 1, "description": "x"})
    cfg = FakeCfg(conflict_surfacing_enabled=True, conflict_surfacing_min_memories=2)
    assert conflict.conflict_note(["only one"], "q", cfg=cfg, llm_client=llm) == ""
    assert llm.calls == 0  # gated before the LLM


def test_conflict_note_empty_when_no_llm_client():
    cfg = FakeCfg(conflict_surfacing_enabled=True)
    assert conflict.conflict_note(MEMS, "q", cfg=cfg, llm_client=None) == ""


def test_conflict_note_returns_directive_when_enabled_and_conflict():
    llm = FakeLlm({"conflict": True, "a": 0, "b": 1, "description": "two cities"})
    cfg = FakeCfg(conflict_surfacing_enabled=True, conflict_surfacing_min_memories=2)
    note = conflict.conflict_note(MEMS, "q", cfg=cfg, llm_client=llm)
    assert "Alice lives in Paris." in note
    assert "Alice lives in Berlin." in note
    assert "two cities" in note


def test_conflict_note_empty_when_enabled_but_no_conflict():
    llm = FakeLlm({"conflict": False, "a": 0, "b": 1, "description": ""})
    cfg = FakeCfg(conflict_surfacing_enabled=True)
    assert conflict.conflict_note(MEMS, "q", cfg=cfg, llm_client=llm) == ""
