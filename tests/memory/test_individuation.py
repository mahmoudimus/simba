"""Tests for the typed-IR deterministic evaluator (src/simba/memory/individuation.py).

The pure evaluators carry the possible-worlds contract for independent choices;
``clingo_check`` is a differential checker (skipped when clingo is absent — it is
never a production dependency).
"""

from __future__ import annotations

import pytest

import simba.memory.individuation as ind
from simba.memory.individuation import IR, Row


def _ir(rows, intent="count", constraints=None):
    return IR(rows=rows, intent=intent, constraints=constraints or [])


class TestRow:
    def test_rejects_bad_status(self) -> None:
        with pytest.raises(ValueError):
            Row("r0", "a", "Maybe")

    def test_default_status_in(self) -> None:
        assert Row("r0", "a").status == "In"


class TestEvaluateCount:
    def test_all_in_distinct(self) -> None:
        ir = _ir([Row("r0", "a", "In"), Row("r1", "b", "In"), Row("r2", "c", "In")])
        assert ind.evaluate_count(ir) == (3, 3)

    def test_strict_loose_interval(self) -> None:
        # 2 In + 2 Possible distinct -> certain 2, possible 4 (the 6d550036 shape)
        ir = _ir(
            [
                Row("r0", "p1", "In"),
                Row("r1", "p2", "In"),
                Row("r2", "c1", "Possible"),
                Row("r3", "c2", "Possible"),
            ]
        )
        assert ind.evaluate_count(ir) == (2, 4)

    def test_dedup_collapses_duplicate_extraction(self) -> None:
        # peace lily extracted twice (dated + undated) is ONE individual
        ir = _ir(
            [
                Row("r0", "peace lily", "In"),
                Row("r1", "peace lily", "In"),
                Row("r2", "succulent", "In"),
                Row("r3", "snake plant", "In"),
            ]
        )
        assert ind.evaluate_count(ir) == (3, 3)

    def test_any_in_member_makes_group_certain(self) -> None:
        ir = _ir(
            [
                Row("r0", "a", "In"),
                Row("r1", "a", "Possible"),
                Row("r2", "b", "Possible"),
            ]
        )
        assert ind.evaluate_count(ir) == (1, 2)

    def test_excluded_never_counts(self) -> None:
        ir = _ir(
            [
                Row("r0", "a", "In"),
                Row("r1", "b", "Excluded"),
                Row("r2", "c", "Possible"),
            ]
        )
        assert ind.evaluate_count(ir) == (1, 2)

    def test_empty(self) -> None:
        assert ind.evaluate_count(_ir([])) == (0, 0)


class TestEvaluateSum:
    def test_in_certain_possible_adds_maybes(self) -> None:
        ir = _ir(
            [Row("r0", "a", "In", value=100.0), Row("r1", "b", "Possible", value=50.0)],
            intent="sum",
        )
        assert ind.evaluate_sum(ir) == (100.0, 150.0)

    def test_duplicate_group_counted_once(self) -> None:
        ir = _ir(
            [Row("r0", "trip", "In", value=8.0), Row("r1", "trip", "In", value=8.0)],
            intent="sum",
        )
        assert ind.evaluate_sum(ir) == (8.0, 8.0)

    def test_evaluate_dispatches_on_intent(self) -> None:
        ir = _ir([Row("r0", "a", "In", value=5.0)], intent="sum")
        assert ind.evaluate(ir) == (5.0, 5.0)


class TestIR:
    def test_independent_when_no_constraints(self) -> None:
        assert _ir([Row("r0", "a")]).is_independent()

    def test_constrained_flag(self) -> None:
        ir = _ir([Row("r0", "a")], constraints=[("mutex", "r0", "r1")])
        assert not ir.is_independent()


class TestClingoDifferential:
    """clingo must agree with the Python evaluator on independent inputs."""

    def setup_method(self) -> None:
        pytest.importorskip("clingo")

    def test_agrees_on_interval(self) -> None:
        ir = _ir(
            [
                Row("r0", "p1", "In"),
                Row("r1", "p2", "In"),
                Row("r2", "c1", "Possible"),
                Row("r3", "c2", "Possible"),
            ]
        )
        agree, res = ind.clingo_check(ir)
        assert agree, f"clingo {res} != python {ind.evaluate_count(ir)}"

    def test_agrees_on_dedup_possible_excluded_mix(self) -> None:
        ir = _ir(
            [
                Row("r0", "a", "In"),
                Row("r1", "a", "Possible"),
                Row("r2", "b", "Possible"),
                Row("r3", "c", "Excluded"),
            ]
        )
        agree, res = ind.clingo_check(ir)
        assert agree, f"clingo {res} != python {ind.evaluate_count(ir)}"

    def test_agrees_all_in(self) -> None:
        ir = _ir([Row("r0", "a", "In"), Row("r1", "b", "In")])
        agree, res = ind.clingo_check(ir)
        assert agree and res == (2, 2)
