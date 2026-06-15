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

from simba.memory.pitfall import (
    pitfall_directive,
    select_pitfall,
    surface_pitfall_directive,
)


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
