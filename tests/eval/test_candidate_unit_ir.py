from __future__ import annotations

import json

import simba.eval.candidate_unit_ir as candidate_unit_ir


def test_fail9_candidate_unit_fixture_scores_9_of_9() -> None:
    fixture = candidate_unit_ir.load_fixture()
    candidate_unit_ir.validate_fixture(fixture)
    score = candidate_unit_ir.score_fixture(fixture)

    assert fixture.name == "fail9-candidate-unit-claude"
    assert fixture.tool == "claude -p"
    assert score.matches == 9
    assert score.total == 9
    assert score.mismatches == ()


def test_fail9_prompt_contract_stays_domain_general() -> None:
    fixture = candidate_unit_ir.load_fixture()
    contract = "\n".join(fixture.compiler_contract).lower()

    # These would indicate the prompt is memorizing the fail9 answers instead
    # of expressing the reusable answer-variable / individuation-policy protocol.
    forbidden_terms = {
        "zara",
        "boots",
        "blazer",
        "tame impala",
        "midnight sky",
        "wedding",
        "citrus",
        "furniture",
        "hawaii",
        "new york",
    }

    assert not any(term in contract for term in forbidden_terms)
    assert "answer_variable" in contract
    assert "individuation_policy" in contract
    assert "candidate_units" in contract
    assert "action(subject, object, verb)" in contract


def test_candidate_units_record_included_excluded_and_merged_decisions() -> None:
    fixture = candidate_unit_ir.load_fixture()
    by_id = {case.id: case for case in fixture.cases}

    clothing = by_id["0a995998"]
    assert clothing.computed_answer == 3
    assert len(clothing.units_with_status("included")) == 3
    assert clothing.units_with_status("excluded")
    assert clothing.units_with_status("merged")

    music = by_id["bf659f65"]
    included_music = {unit.unit_id for unit in music.units_with_status("included")}
    assert "tame_impala_vinyl" in included_music
    assert "midnight_sky_s1" in included_music
    assert any(
        "assistant recommendations" in unit.reason.lower()
        for unit in music.candidate_units
        if unit.status == "excluded"
    )

    weddings = by_id["gpt4_2f8be40d"]
    assert weddings.individuation_policy == "event_instance"
    assert len(weddings.units_with_status("included")) == 3
    assert weddings.units_with_status("merged")


def test_build_prompt_payload_round_trips_as_json() -> None:
    fixture = candidate_unit_ir.load_fixture()
    payload = candidate_unit_ir.build_prompt_payload(
        case_id="q",
        question="How many things did I acquire?",
        evidence_sessions=[
            {
                "session_id": "s1",
                "text": "user: I bought a thing.\nassistant: Nice.",
            }
        ],
        compiler_contract=fixture.compiler_contract,
    )

    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded["case"]["id"] == "q"
    assert decoded["compiler_contract"] == list(fixture.compiler_contract)
    assert decoded["output_schema"]["computed_answer"] == "integer only"
