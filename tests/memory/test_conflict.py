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
    conflict_detect_strategy: str = "single"


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


# --- Pairwise strategy: isolate the conflicting pair --------------------------


class ScriptedLlm:
    """llm_client whose verdict depends on the number of memories in the prompt.

    The single all-at-once prompt numbers every memory, so its prompt mentions
    the buried Bob/coffee line; a focused pair prompt only mentions two. We key
    off a marker substring to decide what the model "sees" and answers.
    """

    def __init__(self, *, pair_marker: str, conflict_desc: str = "two cities"):
        self._pair_marker = pair_marker
        self._conflict_desc = conflict_desc
        self.calls = 0

    def complete_json(self, prompt: str):
        self.calls += 1
        # The model only flags a conflict when BOTH conflicting lines are present
        # AND the unrelated line is absent (the focused pair). The all-at-once
        # prompt also contains the distractor, so the conflict is "buried" → false.
        has_paris = "Alice lives in Paris." in prompt
        has_berlin = "Alice lives in Berlin." in prompt
        has_distractor = self._pair_marker in prompt
        if has_paris and has_berlin and not has_distractor:
            return {"conflict": True, "description": self._conflict_desc}
        return {"conflict": False}


# A larger pool so the conflicting pair is genuinely buried among distractors.
BURIED_MEMS = [
    "Bob likes coffee.",
    "Carol enjoys hiking.",
    "Alice lives in Paris.",
    "Dave plays guitar.",
    "Alice lives in Berlin.",
    "Eve studies physics.",
]


def test_pairwise_finds_buried_conflict_that_single_pass_misses():
    # Single pass: the all-at-once prompt buries the pair among distractors → miss.
    llm_single = ScriptedLlm(pair_marker="Bob likes coffee.")
    assert (
        conflict.detect_conflict(
            BURIED_MEMS, "Where does Alice live?", llm_client=llm_single
        )
        is None
    )
    # Pairwise: isolating Paris/Berlin (no distractor) → the conflict is found.
    llm_pair = ScriptedLlm(pair_marker="Bob likes coffee.")
    res = conflict.detect_conflict_pairwise(
        BURIED_MEMS, "Where does Alice live?", llm_client=llm_pair
    )
    assert res is not None
    assert res.a == "Alice lives in Paris."
    assert res.b == "Alice lives in Berlin."
    assert res.description == "two cities"


def test_pairwise_returns_none_when_no_pair_conflicts():
    llm = FakeLlm({"conflict": False, "description": ""})
    assert conflict.detect_conflict_pairwise(MEMS, "q", llm_client=llm) is None


def test_pairwise_fail_open_on_empty_or_single():
    llm = FakeLlm({"conflict": True, "description": "x"})
    assert conflict.detect_conflict_pairwise([], "q", llm_client=llm) is None
    assert conflict.detect_conflict_pairwise(["solo"], "q", llm_client=llm) is None
    assert llm.calls == 0  # never call the LLM with nothing to compare


def test_pairwise_fail_open_on_no_client():
    assert conflict.detect_conflict_pairwise(MEMS, "q", llm_client=None) is None


def test_pairwise_fail_open_on_exception():
    llm = FakeLlm(None, raises=True)
    assert conflict.detect_conflict_pairwise(MEMS, "q", llm_client=llm) is None


def test_pairwise_short_circuits_on_first_hit():
    # First pair (0,1) conflicts → must stop immediately (one call only).
    llm = FakeLlm({"conflict": True, "description": "first pair"})
    res = conflict.detect_conflict_pairwise(MEMS, "q", llm_client=llm)
    assert res is not None
    assert res.a == "Alice lives in Paris."
    assert res.b == "Alice lives in Berlin."
    assert llm.calls == 1  # short-circuit on the first flagged pair


def test_pairwise_respects_max_pairs_cap():
    # No pair conflicts so it would otherwise check every pair. With 4 memories
    # there are 6 unordered pairs; cap at 2 → exactly 2 LLM calls.
    four = ["m0", "m1", "m2", "m3"]
    llm = FakeLlm({"conflict": False, "description": ""})
    res = conflict.detect_conflict_pairwise(four, "q", llm_client=llm, max_pairs=2)
    assert res is None
    assert llm.calls == 2  # honored the cap, did not check all 6 pairs


def test_conflict_note_routes_to_pairwise_when_strategy_pairwise():
    # All-at-once misses, pairwise finds it → note must be non-empty under pairwise.
    llm = ScriptedLlm(pair_marker="Bob likes coffee.")
    cfg = FakeCfg(
        conflict_surfacing_enabled=True,
        conflict_surfacing_min_memories=2,
        conflict_detect_strategy="pairwise",
    )
    note = conflict.conflict_note(
        BURIED_MEMS, "Where does Alice live?", cfg=cfg, llm_client=llm
    )
    assert "Alice lives in Paris." in note
    assert "Alice lives in Berlin." in note


def test_conflict_note_routes_to_single_when_strategy_single():
    # Same buried set; single strategy misses → empty note.
    llm = ScriptedLlm(pair_marker="Bob likes coffee.")
    cfg = FakeCfg(
        conflict_surfacing_enabled=True,
        conflict_surfacing_min_memories=2,
        conflict_detect_strategy="single",
    )
    note = conflict.conflict_note(
        BURIED_MEMS, "Where does Alice live?", cfg=cfg, llm_client=llm
    )
    assert note == ""


def test_default_config_strategy_is_single():
    from simba.memory.config import MemoryConfig

    assert MemoryConfig().conflict_detect_strategy == "single"
