from __future__ import annotations

import json
import pathlib

import pytest

import simba.eval.ambiguity as ambiguity
import simba.eval.ambiguity_backends as ambiguity_backends
import simba.eval.ambiguity_codegen as ambiguity_codegen
import simba.eval.ambiguity_fail18 as ambiguity_fail18
import simba.eval.world_lexicon as world_lexicon

FIXTURE = pathlib.Path("src/simba/eval/datasets/ambiguity.json")


class _ScriptedLlm:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._replies.pop(0) if self._replies else ""


def test_load_ambiguity_cases_fixture() -> None:
    cases = ambiguity.load_cases(FIXTURE)

    assert [case.id for case in cases] == [
        "amb_recent_births",
        "amb_lot_products",
        "amb_nearby_users",
        "amb_apple_purchases",
    ]
    assert {case.category for case in cases} == {
        "temporal",
        "quantifier",
        "geographic",
        "entity",
    }


def test_fixture_expected_answers_match_executor() -> None:
    for case in ambiguity.load_cases(FIXTURE):
        report = ambiguity.evaluate_case(case)
        by_id = {result.interpretation_id: result for result in report.interpretations}
        for interp in case.interpretations:
            assert by_id[interp.id].answer == interp.expected_answer
        assert report.answer_space == case.expected_answer_space


def test_temporal_case_preserves_missing_date_range() -> None:
    case = _case("amb_recent_births")
    report = ambiguity.evaluate_case(case)

    answers = {
        result.interpretation_id: result.answer
        for result in report.interpretations
    }
    assert answers["few_2_known_only"] == {"count": 2}
    assert answers["few_3_known_only"] == {"count": 4}
    assert answers["few_6_include_possible"] == {"lower": 6, "upper": 7}
    assert report.answer_space == {"lower": 2, "upper": 7}


def test_dual_ceiling_caps_are_independent() -> None:
    case = _case("amb_recent_births")
    report = ambiguity.evaluate_case(case)
    by_id = {result.interpretation_id: result for result in report.interpretations}

    # F3 formality cannot route around the L0 layer cap.
    assert by_id["few_2_known_only"].reliability == 0.35
    # L1 + F2 is still capped by the weakest assumption, not by the raw score.
    assert by_id["few_3_known_only"].reliability == 0.75


def test_z3_proves_answer_space_bounds_for_each_case() -> None:
    for case in ambiguity.load_cases(FIXTURE):
        report = ambiguity.evaluate_case(case)
        assert ambiguity.prove_answer_space_with_z3(report)


def test_souffle_backend_matches_python_when_available() -> None:
    if not ambiguity_backends.backend_available("souffle"):
        pytest.skip("souffle executable not installed")
    case = _case("amb_recent_births")

    python_report = ambiguity.evaluate_case(case, backend="python")
    souffle_report = ambiguity.evaluate_case(case, backend="souffle")

    assert souffle_report.answer_space == python_report.answer_space
    assert [r.answer for r in souffle_report.interpretations] == [
        r.answer for r in python_report.interpretations
    ]


def test_clingo_backend_matches_python_when_available() -> None:
    if not ambiguity_backends.backend_available("clingo"):
        pytest.skip("clingo Python module or executable not installed")
    case = _case("amb_recent_births")

    python_report = ambiguity.evaluate_case(case, backend="python")
    clingo_report = ambiguity.evaluate_case(case, backend="clingo")

    assert clingo_report.answer_space == python_report.answer_space
    assert [r.answer for r in clingo_report.interpretations] == [
        r.answer for r in python_report.interpretations
    ]


def test_fail18_summary_parses_human_gold_answer_first(
    tmp_path: pathlib.Path,
) -> None:
    manifest = tmp_path / "fail18.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "60036106",
                    "question": "What was the total reach?",
                    "gold_answer": "12,000",
                    "gold_count": 12,
                    "failure_mode": "B_sum_value_readerfixed",
                    "clingo_certain": 0,
                    "clingo_possible": 14000,
                },
                {
                    "question_id": "6d550036",
                    "question": "How many projects have I led?",
                    "gold_answer": 2,
                    "gold_count": 2,
                    "failure_mode": "A_overcount_predicate_led",
                    "clingo_certain": 12,
                    "clingo_possible": 23,
                },
            ]
        ),
        encoding="utf-8",
    )

    summary = ambiguity_fail18.summarize(manifest, backend="python")

    assert summary.total == 2
    assert summary.gold_known == 2
    assert summary.contains_gold == 1
    assert summary.misses_gold == 1
    by_id = {item.question_id: item for item in summary.results}
    assert by_id["60036106"].gold_numeric == 12000
    assert by_id["60036106"].contains_gold is True
    assert by_id["6d550036"].answer_space == {"lower": 12, "upper": 23}
    assert by_id["6d550036"].contains_gold is False


def test_fail18_numeric_gold_prefers_earliest_human_number() -> None:
    row = {
        "gold_answer": (
            "I have worked on or bought five model kits. The scales are: "
            "Revell F-15 Eagle and 1/72 scale B-29 bomber."
        ),
        "gold_count": None,
    }

    assert ambiguity_fail18.numeric_gold(row) == 5


def test_fail18_numeric_gold_preserves_decimal_human_answer() -> None:
    row = {
        "gold_answer": "0.5 hours",
        "gold_count": 0,
    }

    assert ambiguity_fail18.numeric_gold(row) == 0.5


def test_fail18_answer_type_router() -> None:
    cases = {
        "How many points do I need to earn to redeem a free skincare product?":
            "threshold_lookup",
        "How many musical instruments do I currently own?": "current_inventory",
        "How many times did I bake something in the past two weeks?":
            "temporal_event_count",
        "How many model kits have I worked on or bought?":
            "canonical_entity_count",
        "How many projects have I led or am currently leading?":
            "role_filtered_count",
    }

    for question, answer_type in cases.items():
        classified = ambiguity_fail18.classify_answer_type({"question": question})
        assert classified == answer_type


def test_world_lexicon_resolves_typed_concepts_and_frames() -> None:
    lexicon = world_lexicon.default_world_lexicon()

    model_matches = {
        match.concept_id
        for match in lexicon.resolve_concepts("a 1/72 scale B-29 bomber")
    }
    instrument_matches = {
        match.concept_id
        for match in lexicon.resolve_concepts("my acoustic guitar")
    }

    assert "scale_model_kit" in model_matches
    assert "musical_instrument" in instrument_matches
    assert "working on" in lexicon.lexical_units_for_frame("worked_or_bought")
    assert "led" in lexicon.lexical_units_for_frame("leadership_role")


def test_local_fail18_fixture_shows_old_clingo_range_coverage() -> None:
    if not ambiguity_fail18.DEFAULT_MANIFEST.exists():
        pytest.skip("local clingo_fail18 fixture not present")
    summary = ambiguity_fail18.summarize(backend="python")

    assert summary.total == 18
    assert summary.gold_known == 18
    assert summary.contains_gold == 14
    assert summary.misses_gold == 4
    by_id = {item.question_id: item for item in summary.results}
    assert by_id["60036106"].gold_numeric == 12000
    assert by_id["60036106"].contains_gold is True
    assert by_id["6d550036"].answer_space == {"lower": 12, "upper": 23}
    assert by_id["6d550036"].contains_gold is False
    assert by_id["gpt4_59c863d7"].gold_numeric == 5
    assert by_id["gpt4_59c863d7"].contains_gold is True


def test_local_fail18_repair_lifts_remaining_misses() -> None:
    if (
        not ambiguity_fail18.DEFAULT_MANIFEST.exists()
        or not ambiguity_fail18.DEFAULT_CORPUS.exists()
    ):
        pytest.skip("local clingo_fail18 fixtures not present")
    summary = ambiguity_fail18.summarize(backend="python", repair=True)

    assert summary.total == 18
    assert summary.gold_known == 18
    assert summary.contains_gold == 18
    assert summary.misses_gold == 0
    repaired = {
        item.question_id: item
        for item in summary.results
        if item.repair_applied
    }
    assert repaired["6d550036"].answer_space == {"count": 2}
    assert repaired["88432d0a"].answer_space == {"count": 4}
    assert repaired["9ee3ecd6"].answer_space == {"count": 100}
    assert repaired["gpt4_194be4b3"].answer_space == {"count": 4}
    assert repaired["gpt4_59c863d7"].answer_space == {"count": 5}


def test_codegen_prompt_requires_llm_to_write_executable_program() -> None:
    prompt = ambiguity_codegen.build_codegen_prompt(
        _case("amb_recent_births"), "python"
    )

    assert "writing executable code" in prompt
    assert "Do not collapse vague language to one interpretation" in prompt
    assert "ANSWER_SPACE" in prompt
    assert "amb_recent_births" in prompt


def test_llm_generated_python_program_is_executed() -> None:
    code = """
counts = []
counts.append(len([r for r in RECORDS if r.get("birth_date", "") >= "2026-05-01"]))
counts.append(len([r for r in RECORDS if r.get("birth_date", "") >= "2026-03-01"]))
ANSWER_SPACE = {"lower": min(counts), "upper": max(counts) + 4}
PROVENANCE = [r["id"] for r in RECORDS]
"""
    llm = _ScriptedLlm([f"```python\n{code}\n```"])
    case = _case("amb_recent_births")

    program, run = ambiguity_codegen.generate_and_run(
        case, language="python", client=llm
    )

    assert program.source == "llm"
    assert run.ok is True
    assert run.answer_space == {"lower": 2, "upper": 7}
    assert set(run.provenance) == {record["id"] for record in case.records}
    assert len(llm.prompts) == 1


def test_generated_python_blocks_file_and_os_access() -> None:
    code = "import os\nANSWER_SPACE = {'count': len(os.listdir('.'))}"

    run = ambiguity_codegen.run_generated_python(_case("amb_recent_births"), code)

    assert run.ok is False
    assert "not allowed" in run.stderr


def test_generated_souffle_program_runs_when_available() -> None:
    if not ambiguity_backends.backend_available("souffle"):
        pytest.skip("souffle executable not installed")
    code = """
.decl field(id:symbol, key:symbol, value:symbol)
.input field

.decl answer_space(kind:symbol, value:number)
.output answer_space

answer_space("lower", 2).
answer_space("upper", 3).
"""
    program = ambiguity_codegen.GeneratedProgram(
        case_id="amb_apple_purchases",
        language="souffle",
        code=code,
    )

    run = ambiguity_codegen.run_generated_program(_case("amb_apple_purchases"), program)

    assert run.ok is True
    assert run.answer_space == {"lower": 2, "upper": 3}


def test_generated_clingo_program_runs_when_available() -> None:
    if not ambiguity_backends.backend_available("clingo"):
        pytest.skip("clingo Python module or executable not installed")
    code = """
answer_space("lower", 2).
answer_space("upper", 3).
#show answer_space/2.
"""
    program = ambiguity_codegen.GeneratedProgram(
        case_id="amb_apple_purchases",
        language="clingo",
        code=code,
    )

    run = ambiguity_codegen.run_generated_program(_case("amb_apple_purchases"), program)

    assert run.ok is True
    assert run.answer_space == {"lower": 2, "upper": 3}


def test_quantifier_case_exposes_threshold_choice() -> None:
    case = _case("amb_lot_products")
    report = ambiguity.evaluate_case(case)
    answers = {
        result.interpretation_id: result.answer
        for result in report.interpretations
    }

    assert answers == {
        "absolute_10_items": {"count": 3},
        "top_quartile": {"count": 2},
        "above_average": {"count": 3},
    }
    assert report.answer_space == {"lower": 2, "upper": 3}


def _case(case_id: str) -> ambiguity.AmbiguityCase:
    return next(case for case in ambiguity.load_cases(FIXTURE) if case.id == case_id)
