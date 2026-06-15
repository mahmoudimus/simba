"""Tests for the pitfall/doctrine enforcement gate (src/simba/memory/pitfall.py).

The retrieval half (entropy-boost) surfaces the right scar; this is the ENFORCEMENT
half: when the agent's pending move matches a stored doctrine/scar (FAILURE /
PREFERENCE / GOTCHA), surface it as a STOP-and-confirm DIRECTIVE — "you're about to
take the workaround you told me not to" — not passive context.

Pure + fail-open. The recall + type filtering happens in the daemon/hook; this module
only decides whether the *top-ranked* candidate clears the directive FLOOR (a stricter
bar than recall's min_similarity, so an interrupt fires only on a strong, specific
match) and frames it. Checking ONLY the top-ranked candidate is deliberate: the
measured no-false-positive guarantee (probe: benign top <= 0.73, labeled fires >= 0.82,
floor 0.78) holds for the top candidate, not for a deep scan.
"""

from __future__ import annotations

import types

from simba.memory.pitfall import (
    detect_violation,
    pitfall_directive,
    pitfall_note,
    select_failure_fallback,
    select_pitfall,
    select_violation,
    surface_pitfall_directive,
)


class _FakeLLM:
    """Stand-in for the llm_client: ``complete_json`` says the move violates a doctrine
    iff ``violates_when`` is a substring of the prompt (i.e. of the doctrine text)."""

    def __init__(self, violates_when=None, *, reply=None, raise_=False):
        self.violates_when = violates_when
        self.reply = reply  # override the dict reply entirely when set
        self.raise_ = raise_
        self.calls = 0

    def complete_json(self, prompt):
        self.calls += 1
        if self.raise_:
            raise RuntimeError("boom")
        if self.reply is not None:
            return self.reply
        if self.violates_when and self.violates_when in prompt:
            return {"violates": True, "why": "would do the forbidden thing"}
        return {"violates": False}


def _cfg(**kw):
    return types.SimpleNamespace(**kw)


class TestSelectPitfall:
    def test_empty_returns_none(self) -> None:
        assert select_pitfall([], min_similarity=0.78) is None

    def test_top_clears_floor_returns_top(self) -> None:
        mems = [
            {"id": "mem_a", "type": "PREFERENCE", "similarity": 0.82, "content": "x"},
            {"id": "mem_b", "type": "GOTCHA", "similarity": 0.50, "content": "y"},
        ]
        got = select_pitfall(mems, min_similarity=0.78)
        assert got is not None and got["id"] == "mem_a"

    def test_top_below_floor_returns_none(self) -> None:
        mems = [{"id": "mem_a", "type": "FAILURE", "similarity": 0.73, "content": "x"}]
        assert select_pitfall(mems, min_similarity=0.78) is None

    def test_only_top_ranked_is_considered(self) -> None:
        # A lower-ranked high-similarity memory must NOT fire — the measured FP=0
        # guarantee was established on the TOP candidate only. The top is below the
        # floor, so the gate stays silent even though mem_b is above it.
        mems = [
            {"id": "mem_a", "type": "GOTCHA", "similarity": 0.40, "content": "x"},
            {"id": "mem_b", "type": "PREFERENCE", "similarity": 0.95, "content": "y"},
        ]
        assert select_pitfall(mems, min_similarity=0.78) is None

    def test_garbage_similarity_fails_open(self) -> None:
        assert (
            select_pitfall(
                [{"id": "m", "type": "FAILURE", "similarity": None, "content": "x"}],
                min_similarity=0.78,
            )
            is None
        )
        assert (
            select_pitfall(
                [{"id": "m", "type": "FAILURE", "similarity": "high", "content": "x"}],
                min_similarity=0.78,
            )
            is None
        )

    def test_missing_similarity_treated_as_zero(self) -> None:
        assert (
            select_pitfall(
                [{"id": "m", "type": "FAILURE", "content": "x"}], min_similarity=0.78
            )
            is None
        )


class TestSurfaceDirective:
    def test_includes_content_and_confirm_instruction(self) -> None:
        d = surface_pitfall_directive(
            {"type": "PREFERENCE", "content": "No assertion weakening; fix behavior."}
        )
        assert "No assertion weakening" in d
        assert "pitfall-warning" in d
        # It must instruct a stop/confirm, not merely inform.
        assert "confirm" in d.lower()

    def test_preference_framed_as_your_doctrine(self) -> None:
        d = surface_pitfall_directive(
            {"type": "PREFERENCE", "content": "Never use --no-verify."}
        )
        assert "doctrine" in d.lower() or "you stated" in d.lower()

    def test_failure_framed_as_already_tried(self) -> None:
        d = surface_pitfall_directive(
            {"type": "FAILURE", "content": "copy_block after exit crashes IDA."}
        )
        assert "failed" in d.lower() or "already" in d.lower()

    def test_gotcha_framed_as_known_trap(self) -> None:
        d = surface_pitfall_directive(
            {"type": "GOTCHA", "content": "deleting a block before its successor."}
        )
        assert "trap" in d.lower() or "known" in d.lower()

    def test_empty_content_still_safe(self) -> None:
        d = surface_pitfall_directive({"type": "FAILURE", "content": ""})
        assert isinstance(d, str)


class TestPitfallDirectiveEntry:
    def test_fires_when_top_clears_floor(self) -> None:
        mems = [
            {
                "id": "m",
                "type": "PREFERENCE",
                "similarity": 0.82,
                "content": "Own your bugs.",
            }
        ]
        out = pitfall_directive(mems, min_similarity=0.78)
        assert "Own your bugs" in out and "pitfall-warning" in out

    def test_silent_when_below_floor(self) -> None:
        mems = [{"id": "m", "type": "GOTCHA", "similarity": 0.73, "content": "x"}]
        assert pitfall_directive(mems, min_similarity=0.78) == ""

    def test_silent_on_empty(self) -> None:
        assert pitfall_directive([], min_similarity=0.78) == ""

    def test_fail_open_on_bad_input(self) -> None:
        # Never raise into the hook path.
        assert pitfall_directive(None, min_similarity=0.78) == ""  # type: ignore[arg-type]
        assert pitfall_directive([{"bogus": 1}], min_similarity=0.78) == ""


class TestDetectViolation:
    def test_violation_true_returns_reason(self) -> None:
        llm = _FakeLLM(violates_when="no assertion weakening")
        out = detect_violation(
            "I'll xfail the failing test", "no assertion weakening", llm_client=llm
        )
        assert out is not None and out[0] is True and out[1]

    def test_consistent_move_not_a_violation(self) -> None:
        llm = _FakeLLM(violates_when="ZZZ")  # never matches
        out = detect_violation(
            "storing concrete memories", "store concrete memories", llm_client=llm
        )
        assert out == (False, "")

    def test_no_client_fails_open(self) -> None:
        assert detect_violation("m", "d", llm_client=None) is None

    def test_garbage_reply_fails_open(self) -> None:
        assert detect_violation("m", "d", llm_client=_FakeLLM(reply={"foo": 1})) is None

    def test_exception_fails_open(self) -> None:
        assert detect_violation("m", "d", llm_client=_FakeLLM(raise_=True)) is None


class TestSelectViolation:
    def test_returns_first_violating_candidate_above_floor(self) -> None:
        mems = [
            {
                "id": "a",
                "type": "GOTCHA",
                "similarity": 0.85,
                "content": "topical only",
            },
            {
                "id": "b",
                "type": "PREFERENCE",
                "similarity": 0.82,
                "content": "no weakening",
            },
        ]
        llm = _FakeLLM(violates_when="no weakening")
        got = select_violation(
            mems, "xfail it", llm_client=llm, topical_floor=0.70, max_checks=3
        )
        assert got is not None and got[0]["id"] == "b"

    def test_none_when_no_violation(self) -> None:
        mems = [{"id": "a", "type": "GOTCHA", "similarity": 0.85, "content": "topical"}]
        llm = _FakeLLM(violates_when="ZZZ")
        assert (
            select_violation(
                mems, "move", llm_client=llm, topical_floor=0.70, max_checks=3
            )
            is None
        )

    def test_skips_candidates_below_topical_floor(self) -> None:
        mems = [
            {"id": "a", "type": "FAILURE", "similarity": 0.5, "content": "no weakening"}
        ]
        llm = _FakeLLM(violates_when="no weakening")
        got = select_violation(
            mems, "m", llm_client=llm, topical_floor=0.70, max_checks=3
        )
        assert (
            got is None and llm.calls == 0
        )  # never LLM-checked the sub-floor candidate

    def test_respects_max_checks(self) -> None:
        mems = [
            {"id": "a", "type": "GOTCHA", "similarity": 0.9, "content": "x"},
            {"id": "b", "type": "GOTCHA", "similarity": 0.9, "content": "y"},
            {
                "id": "c",
                "type": "PREFERENCE",
                "similarity": 0.9,
                "content": "no weakening",
            },
        ]
        llm = _FakeLLM(violates_when="no weakening")
        got = select_violation(
            mems, "m", llm_client=llm, topical_floor=0.70, max_checks=2
        )
        assert got is None and llm.calls == 2  # stopped before reaching candidate c

    def test_no_client_fails_open(self) -> None:
        mems = [{"id": "a", "type": "FAILURE", "similarity": 0.9, "content": "x"}]
        assert (
            select_violation(
                mems, "m", llm_client=None, topical_floor=0.7, max_checks=3
            )
            is None
        )


class TestFailureFallback:
    def test_top_failure_above_floor(self) -> None:
        mems = [
            {"id": "p", "type": "PREFERENCE", "similarity": 0.90, "content": "x"},
            {
                "id": "f",
                "type": "FAILURE",
                "similarity": 0.81,
                "content": "broke before",
            },
        ]
        got = select_failure_fallback(mems, min_similarity=0.78)
        assert got is not None and got["id"] == "f"

    def test_no_failure_type_returns_none(self) -> None:
        mems = [{"id": "p", "type": "PREFERENCE", "similarity": 0.95, "content": "x"}]
        assert select_failure_fallback(mems, min_similarity=0.78) is None

    def test_failure_below_floor_returns_none(self) -> None:
        mems = [{"id": "f", "type": "FAILURE", "similarity": 0.6, "content": "x"}]
        assert select_failure_fallback(mems, min_similarity=0.78) is None


class TestPitfallNote:
    def test_violation_mode_fires_with_reason(self) -> None:
        mems = [
            {
                "id": "b",
                "type": "PREFERENCE",
                "similarity": 0.82,
                "content": "no weakening",
            }
        ]
        cfg = _cfg(pitfall_gate_mode="violation")
        out = pitfall_note(
            mems, "xfail it", cfg=cfg, llm_client=_FakeLLM(violates_when="no weakening")
        )
        assert "pitfall-warning" in out and "Why this fires" in out

    def test_violation_mode_silent_when_consistent(self) -> None:
        mems = [
            {
                "id": "b",
                "type": "PREFERENCE",
                "similarity": 0.95,
                "content": "store concrete",
            }
        ]
        cfg = _cfg(pitfall_gate_mode="violation")
        out = pitfall_note(
            mems, "storing concrete", cfg=cfg, llm_client=_FakeLLM(violates_when="ZZZ")
        )
        assert out == ""  # topically close but no violation → no fire (the key fix)

    def test_fallback_failure_only_fires_on_failure(self) -> None:
        mems = [
            {
                "id": "f",
                "type": "FAILURE",
                "similarity": 0.82,
                "content": "broke before",
            }
        ]
        cfg = _cfg(pitfall_gate_mode="violation", pitfall_gate_fallback="failure_only")
        out = pitfall_note(
            mems, "do it again", cfg=cfg, llm_client=None
        )  # no LLM → fallback
        assert "pitfall-warning" in out and "broke before" in out

    def test_fallback_failure_only_ignores_preference(self) -> None:
        mems = [
            {"id": "p", "type": "PREFERENCE", "similarity": 0.95, "content": "doctrine"}
        ]
        cfg = _cfg(pitfall_gate_mode="violation", pitfall_gate_fallback="failure_only")
        assert pitfall_note(mems, "m", cfg=cfg, llm_client=None) == ""

    def test_fallback_off_fires_nothing(self) -> None:
        mems = [{"id": "f", "type": "FAILURE", "similarity": 0.95, "content": "broke"}]
        cfg = _cfg(pitfall_gate_mode="violation", pitfall_gate_fallback="off")
        assert pitfall_note(mems, "m", cfg=cfg, llm_client=None) == ""

    def test_similarity_mode_top_over_floor(self) -> None:
        mems = [{"id": "a", "type": "GOTCHA", "similarity": 0.82, "content": "trap"}]
        cfg = _cfg(pitfall_gate_mode="similarity")
        assert "pitfall-warning" in pitfall_note(mems, "m", cfg=cfg, llm_client=None)

    def test_empty_is_silent(self) -> None:
        assert (
            pitfall_note(
                [], "m", cfg=_cfg(pitfall_gate_mode="violation"), llm_client=None
            )
            == ""
        )
