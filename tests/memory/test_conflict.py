"""Answer-time conflict surfacing: detect a real contradiction among retrieved
memories with ONE LLM call, then emit a directive that NAMES it and tells the
consumer to surface (not pick a side). Gated + fail-open. Mirrors the fake-llm
style of test_entity_bridge / judge tests (no live model)."""

from __future__ import annotations

import dataclasses
import pathlib

import simba.db
import simba.memory.conflict as conflict
import simba.memory.conflict_store as cstore


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
    conflict_detect_parallel: int = 1
    conflict_skip_on_current_value: bool = True


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


def test_default_config_strategy_is_pairwise():
    # Superseded "single" on the 0.7.0 ship: pairwise is the measured-good
    # detector (single's ~0.15 fire rate was the spec-14 detection wall).
    from simba.memory.config import MemoryConfig

    assert MemoryConfig().conflict_detect_strategy == "pairwise"


# ── 0.7.1 query-intent gate: skip surfacing on current-value queries ──────────
# v0.7.0 default-ON surfacing REGRESSED knowledge-update QA (LME-S KU OFF=0.958
# vs directive=0.25): a "what is X now?" query retrieves both the old and new
# value of a fact, the pairwise detector flags that as a conflict and the
# directive tells the answerer not to pick a side — exactly wrong when the
# correct answer is most-recent-wins. The fix gates by QUERY INTENT (the ARM3
# date-disjoint carve-out FAILED its SubtleMemory gate: date-disjointness does
# not discriminate update-vs-conflict — genuine preference conflicts are also
# date-disjoint). So: current-value query → skip (return "", no detection cost);
# everything else → unchanged strict pairwise path.
def test_conflict_note_skips_current_value_query_even_with_conflict():
    # Conflicting memories ARE present, but the query asks for the present value
    # → must return "" and never pay the detection LLM cost (recency handles it).
    llm = FakeLlm({"conflict": True, "a": 0, "b": 1, "description": "two cities"})
    cfg = FakeCfg(
        conflict_surfacing_enabled=True,
        conflict_skip_on_current_value=True,
        conflict_detect_strategy="pairwise",
    )
    note = conflict.conflict_note(
        MEMS, "Where does Alice live now?", cfg=cfg, llm_client=llm
    )
    assert note == ""
    assert llm.calls == 0  # skipped BEFORE detection → zero LLM cost


def test_conflict_note_genuine_conflict_query_still_surfaces():
    # No current-value marker ("which do I prefer") → strict path is untouched:
    # the conflict is detected and surfaced exactly as before.
    llm = FakeLlm({"conflict": True, "a": 0, "b": 1, "description": "two cities"})
    cfg = FakeCfg(
        conflict_surfacing_enabled=True,
        conflict_skip_on_current_value=True,
        conflict_detect_strategy="single",
    )
    note = conflict.conflict_note(
        MEMS, "Which city do I prefer, Paris or Berlin?", cfg=cfg, llm_client=llm
    )
    assert "Alice lives in Paris." in note
    assert "Alice lives in Berlin." in note
    assert llm.calls == 1  # genuine path still runs detection


def test_conflict_note_skip_can_be_disabled_via_config():
    # Flag off → current-value query falls through to the strict path (the old
    # 0.7.0 behaviour), proving the gate is fully config-controlled.
    llm = FakeLlm({"conflict": True, "a": 0, "b": 1, "description": "two cities"})
    cfg = FakeCfg(
        conflict_surfacing_enabled=True,
        conflict_skip_on_current_value=False,
        conflict_detect_strategy="single",
    )
    note = conflict.conflict_note(
        MEMS, "Where does Alice live now?", cfg=cfg, llm_client=llm
    )
    assert "Alice lives in Paris." in note
    assert llm.calls == 1


def test_default_config_skip_on_current_value_is_on():
    # This is the regression fix → must default ON.
    from simba.memory.config import MemoryConfig

    assert MemoryConfig().conflict_skip_on_current_value is True


# --- Write-time detection (B2): compare a NEW memory vs its neighbors ---------


def test_detect_on_write_flags_conflicting_neighbor():
    # The fake flags any pair as a conflict; we get back (neighbor_id, desc).
    llm = FakeLlm({"conflict": True, "description": "two cities"})
    neighbors = [
        ("n_paris", "Alice lives in Paris."),
        ("n_coffee", "Bob likes coffee."),
    ]
    out = conflict.detect_conflicts_on_write(
        "new_berlin", "Alice lives in Berlin.", neighbors, llm_client=llm
    )
    assert ("n_paris", "two cities") in out
    assert ("n_coffee", "two cities") in out  # this fake flags everything
    assert llm.calls == 2  # one call per neighbor


def test_detect_on_write_returns_only_conflicting_neighbors():
    # ScriptedLlm only flags Paris vs Berlin (no distractor in a focused pair).
    llm = ScriptedLlm(pair_marker="Bob likes coffee.")
    neighbors = [
        ("n_paris", "Alice lives in Paris."),
        ("n_coffee", "Bob likes coffee."),
    ]
    out = conflict.detect_conflicts_on_write(
        "new_berlin", "Alice lives in Berlin.", neighbors, llm_client=llm
    )
    ids = [nid for nid, _ in out]
    assert ids == ["n_paris"]


def test_detect_on_write_respects_max_neighbors():
    llm = FakeLlm({"conflict": False, "description": ""})
    neighbors = [(f"n{i}", f"mem {i}") for i in range(10)]
    conflict.detect_conflicts_on_write(
        "new", "new text", neighbors, llm_client=llm, max_neighbors=3
    )
    assert llm.calls == 3  # capped at max_neighbors


def test_detect_on_write_fail_open_on_empty_or_no_client():
    llm = FakeLlm({"conflict": True, "description": "x"})
    assert conflict.detect_conflicts_on_write("new", "t", [], llm_client=llm) == []
    assert llm.calls == 0
    assert (
        conflict.detect_conflicts_on_write("new", "t", [("n", "m")], llm_client=None)
        == []
    )


def test_detect_on_write_fail_open_on_exception():
    llm = FakeLlm(None, raises=True)
    out = conflict.detect_conflicts_on_write("new", "t", [("n", "m")], llm_client=llm)
    assert out == []


# --- Recall-read (B2): read a precomputed conflict from the store -------------


def test_conflict_note_from_store_reads_recorded_conflict(
    tmp_path: pathlib.Path,
) -> None:
    cfg = FakeCfg(conflict_surfacing_enabled=True)
    with simba.db.connect(tmp_path):
        cstore.record_conflict(
            "mem_a", "mem_b", "two cities", project_path="proj", now=1.0
        )
        note = conflict.conflict_note_from_store(
            ["mem_a", "mem_b", "mem_c"], project_path="proj", cfg=cfg
        )
    assert "two cities" in note
    assert "confirm" in note.lower()  # steers toward surfacing


def test_conflict_note_from_store_uses_description_not_raw_ids(
    tmp_path: pathlib.Path,
) -> None:
    # Regression (B2 smoke): the recall-read directive must lead with the LLM
    # description, NOT the opaque memory IDs (which are noise to the answerer and
    # were why the write-time path underperformed the live B1 directive).
    cfg = FakeCfg(conflict_surfacing_enabled=True)
    desc = "Memory A says the user prefers reusable bags; Memory B says never"
    with simba.db.connect(tmp_path):
        cstore.record_conflict(
            "related-persona_0-export-deadbeef-s1",
            "related-persona_0-export-deadbeef-s2",
            desc,
            project_path="proj",
            now=1.0,
        )
        note = conflict.conflict_note_from_store(
            [
                "related-persona_0-export-deadbeef-s1",
                "related-persona_0-export-deadbeef-s2",
            ],
            project_path="proj",
            cfg=cfg,
        )
    assert desc in note
    assert "deadbeef" not in note  # raw memory IDs must NOT leak into the directive
    assert "confirm" in note.lower()


def test_conflict_note_from_store_empty_when_disabled(
    tmp_path: pathlib.Path,
) -> None:
    cfg = FakeCfg(conflict_surfacing_enabled=False)
    with simba.db.connect(tmp_path):
        cstore.record_conflict(
            "mem_a", "mem_b", "two cities", project_path="proj", now=1.0
        )
        note = conflict.conflict_note_from_store(
            ["mem_a", "mem_b"], project_path="proj", cfg=cfg
        )
    assert note == ""


def test_conflict_note_from_store_empty_when_none_recorded(
    tmp_path: pathlib.Path,
) -> None:
    cfg = FakeCfg(conflict_surfacing_enabled=True)
    with simba.db.connect(tmp_path):
        note = conflict.conflict_note_from_store(
            ["mem_a", "mem_b"], project_path="proj", cfg=cfg
        )
    assert note == ""


# --- B2b: generous write-time pre-filter + query-aware recall re-check --------


class PromptKeyedLlm:
    """llm_client whose verdict depends on a marker in the prompt text.

    Lets a test assert that the GENEROUS prompt (not the strict one) was the one
    that flagged the borderline pair: the generous prompt carries a distinctive
    marker substring, so we return ``conflict: True`` only when it is present.
    """

    def __init__(self, *, only_if_contains: str, conflict_desc: str = "borderline"):
        self._marker = only_if_contains
        self._desc = conflict_desc
        self.calls = 0
        self.prompts: list[str] = []

    def complete_json(self, prompt: str):
        self.calls += 1
        self.prompts.append(prompt)
        if self._marker in prompt:
            return {"conflict": True, "description": self._desc}
        return {"conflict": False}


def test_generous_pair_prompt_differs_from_strict():
    # The generous prompt must instruct toward RECALL ("could", "plausibly")
    # whereas the strict prompt insists on mutual exclusion.
    strict = conflict.build_pair_detect_prompt("A", "B", "q")
    generous = conflict.build_pair_detect_prompt("A", "B", "q", generous=True)
    assert strict != generous
    assert "plausibl" in generous.lower() or "could" in generous.lower()
    # Default keyword keeps the strict prompt identical to the legacy signature.
    assert conflict.build_pair_detect_prompt("A", "B", "q", generous=False) == strict


def test_detect_on_write_generous_flags_borderline_pair_strict_skips():
    # A borderline pair the STRICT prompt would skip but the GENEROUS one flags.
    # The fake only flags when the generous marker is present in the prompt.
    marker = "could plausibly"  # appears only in the generous prompt
    neighbors = [("n_old", "The user usually drinks tea in the morning.")]

    strict_llm = PromptKeyedLlm(only_if_contains=marker)
    strict_out = conflict.detect_conflicts_on_write(
        "new", "The user had coffee this morning.", neighbors, llm_client=strict_llm
    )
    assert strict_out == []  # strict prompt lacks the marker → no flag

    generous_llm = PromptKeyedLlm(only_if_contains=marker)
    generous_out = conflict.detect_conflicts_on_write(
        "new",
        "The user had coffee this morning.",
        neighbors,
        llm_client=generous_llm,
        generous=True,
    )
    assert [nid for nid, _ in generous_out] == ["n_old"]


def test_detect_on_write_generous_default_false_unchanged():
    # Default (generous=False) sends the strict prompt — verify by the marker.
    marker = "could plausibly"
    llm = PromptKeyedLlm(only_if_contains=marker)
    conflict.detect_conflicts_on_write("new", "t", [("n", "m")], llm_client=llm)
    assert all(marker not in p for p in llm.prompts)


class RecheckCfg(FakeCfg):
    """FakeCfg extended with the B2b recall re-check flag."""

    def __init__(self, *, conflict_recall_recheck: bool = False, **kw):
        super().__init__(**kw)
        self.conflict_recall_recheck = conflict_recall_recheck


def test_recall_recheck_confirms_relevant_candidate(
    tmp_path: pathlib.Path,
) -> None:
    cfg = RecheckCfg(conflict_surfacing_enabled=True, conflict_recall_recheck=True)
    desc = "Memory A says Paris; Memory B says Berlin"
    # Confirm-LLM says candidate index 0 is the real, relevant conflict.
    llm = FakeLlm({"relevant": True, "index": 0})
    with simba.db.connect(tmp_path):
        cstore.record_conflict("mem_a", "mem_b", desc, project_path="proj", now=1.0)
        note = conflict.conflict_note_from_store(
            ["mem_a", "mem_b", "mem_c"],
            project_path="proj",
            cfg=cfg,
            query="Where does Alice live?",
            llm_client=llm,
        )
    assert desc in note
    assert "confirm" in note.lower()
    assert llm.calls == 1  # one query-aware confirm call


def test_recall_recheck_filters_irrelevant_candidate(
    tmp_path: pathlib.Path,
) -> None:
    # The precision win: a stored candidate the confirm-LLM deems not relevant for
    # THIS question is dropped → empty directive.
    cfg = RecheckCfg(conflict_surfacing_enabled=True, conflict_recall_recheck=True)
    llm = FakeLlm({"relevant": False})
    with simba.db.connect(tmp_path):
        cstore.record_conflict(
            "mem_a", "mem_b", "some conflict", project_path="proj", now=1.0
        )
        note = conflict.conflict_note_from_store(
            ["mem_a", "mem_b"],
            project_path="proj",
            cfg=cfg,
            query="Totally unrelated question?",
            llm_client=llm,
        )
    assert note == ""
    assert llm.calls == 1


def test_recall_recheck_falls_back_without_llm_or_query(
    tmp_path: pathlib.Path,
) -> None:
    # recheck=True but no llm_client / no query → non-recheck path (first stored).
    cfg = RecheckCfg(conflict_surfacing_enabled=True, conflict_recall_recheck=True)
    desc = "first stored conflict"
    with simba.db.connect(tmp_path):
        cstore.record_conflict("mem_a", "mem_b", desc, project_path="proj", now=1.0)
        # No llm_client.
        note_no_llm = conflict.conflict_note_from_store(
            ["mem_a", "mem_b"],
            project_path="proj",
            cfg=cfg,
            query="some question?",
        )
        # No query.
        llm = FakeLlm({"relevant": True, "index": 0})
        note_no_query = conflict.conflict_note_from_store(
            ["mem_a", "mem_b"],
            project_path="proj",
            cfg=cfg,
            llm_client=llm,
        )
    assert desc in note_no_llm
    assert desc in note_no_query
    assert llm.calls == 0  # no query → never reached the confirm call


def test_recall_recheck_disabled_default_unchanged(
    tmp_path: pathlib.Path,
) -> None:
    # recheck=False (default): current behavior, no LLM call even if one is wired.
    cfg = RecheckCfg(conflict_surfacing_enabled=True, conflict_recall_recheck=False)
    desc = "first stored conflict"
    llm = FakeLlm({"relevant": True, "index": 0})
    with simba.db.connect(tmp_path):
        cstore.record_conflict("mem_a", "mem_b", desc, project_path="proj", now=1.0)
        note = conflict.conflict_note_from_store(
            ["mem_a", "mem_b"],
            project_path="proj",
            cfg=cfg,
            query="some question?",
            llm_client=llm,
        )
    assert desc in note
    assert llm.calls == 0  # disabled → no confirm call


def test_recall_recheck_fail_open_on_confirm_exception(
    tmp_path: pathlib.Path,
) -> None:
    # Any exception in the confirm call → fall back to the non-recheck path.
    cfg = RecheckCfg(conflict_surfacing_enabled=True, conflict_recall_recheck=True)
    desc = "first stored conflict"
    llm = FakeLlm(None, raises=True)
    with simba.db.connect(tmp_path):
        cstore.record_conflict("mem_a", "mem_b", desc, project_path="proj", now=1.0)
        note = conflict.conflict_note_from_store(
            ["mem_a", "mem_b"],
            project_path="proj",
            cfg=cfg,
            query="some question?",
            llm_client=llm,
        )
    assert desc in note  # fell back to first stored candidate, never raised


def test_default_config_conflict_recall_recheck_is_false():
    from simba.memory.config import MemoryConfig

    assert MemoryConfig().conflict_recall_recheck is False


# --- parallel pairwise (waves) -----------------------------------------------


class OrderedLlm:
    """Flags configured pair indices; records call order. Thread-safe enough for
    the wave executor (appends only)."""

    def __init__(self, flag_at: set[int]):
        self._flag_at = flag_at
        self.seen: list[int] = []
        self._n = 0

    def complete_json(self, prompt: str):
        import threading

        with getattr(self, "_lock", threading.Lock()):
            i = self._n
            self._n += 1
        self.seen.append(i)
        if i in self._flag_at:
            return {"conflict": True, "description": f"pair {i}"}
        return {"conflict": False}

    @property
    def calls(self) -> int:
        return self._n


def test_pairwise_parallel_returns_lowest_index_flagged_pair():
    # Pairs over MEMS in order: (0,1), (0,2), (1,2). Flag (0,1) AND (1,2): with
    # the whole set in one wave, the result must be the LOWEST-index pair —
    # deterministic, identical to the sequential answer.
    llm = OrderedLlm(flag_at={0, 2})
    res = conflict.detect_conflict_pairwise(MEMS, "q", llm_client=llm, parallel=3)
    assert res is not None
    assert res.a == "Alice lives in Paris."
    assert res.b == "Alice lives in Berlin."
    assert res.description == "pair 0"


def test_pairwise_parallel_short_circuits_between_waves():
    # 4 memories -> 6 pairs; first pair flagged; parallel=2 -> only the first
    # wave (2 calls) runs, never the remaining 4 pairs.
    four = ["m0", "m1", "m2", "m3"]
    llm = OrderedLlm(flag_at={0})
    res = conflict.detect_conflict_pairwise(four, "q", llm_client=llm, parallel=2)
    assert res is not None
    assert llm.calls == 2  # one wave, not all 6
    assert res.a == "m0" and res.b == "m1"


def test_pairwise_parallel_respects_max_pairs_cap():
    four = ["m0", "m1", "m2", "m3"]
    llm = FakeLlm({"conflict": False, "description": ""})
    res = conflict.detect_conflict_pairwise(
        four, "q", llm_client=llm, max_pairs=3, parallel=2
    )
    assert res is None
    assert llm.calls == 3  # cap honored across waves


def test_pairwise_parallel_one_is_sequential():
    llm = FakeLlm({"conflict": True, "description": "first"})
    res = conflict.detect_conflict_pairwise(MEMS, "q", llm_client=llm, parallel=1)
    assert res is not None
    assert llm.calls == 1


def test_conflict_note_threads_parallel_from_cfg():
    # conflict_note must pass cfg.conflict_detect_parallel to the detector:
    # first pair flagged + parallel=2 -> exactly one 2-call wave.
    llm = OrderedLlm(flag_at={0})
    cfg = FakeCfg(
        conflict_surfacing_enabled=True,
        conflict_surfacing_min_memories=2,
        conflict_detect_strategy="pairwise",
        conflict_detect_parallel=2,
    )
    note = conflict.conflict_note(
        ["m0", "m1", "m2", "m3"], "q", cfg=cfg, llm_client=llm
    )
    assert "pair 0" in note
    assert llm.calls == 2


def test_shipped_defaults_pairwise_surfacing_on():
    # 2026-06-11 ship decision: surfacing default-ON with the measured-good
    # pairwise detector (both-sides 0.111->0.944, net-positive harm check).
    import simba.memory.config as mc

    cfg = mc.MemoryConfig()
    assert cfg.conflict_surfacing_enabled is True
    assert cfg.conflict_detect_strategy == "pairwise"
    assert cfg.conflict_detect_parallel >= 4  # hot path pays ~1 LLM latency/wave
