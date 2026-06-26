from __future__ import annotations

import pytest

import simba.eval.ambiguity_taxonomy as ambiguity_taxonomy
import simba.eval.interpretation_ir as interpretation_ir


def test_interpretation_record_round_trips_external_labels() -> None:
    record = interpretation_ir.InterpretationRecord.from_dict(
        {
            "interpretation_id": "i1",
            "natural_language_interpretation": (
                "Count only events the user actually attended."
            ),
            "ambiguity_types": ["AmbiView", "scope"],
            "assumptions": ["assistant recommendations are not attended events"],
            "expected_answer_shape": "count",
        }
    )

    assert record.ambiguity_types == (
        ambiguity_taxonomy.AmbiguityType.AGGREGATION_VIEW_AMBIGUOUS,
        ambiguity_taxonomy.AmbiguityType.SCOPE_AMBIGUOUS,
    )
    assert record.to_dict() == {
        "interpretation_id": "i1",
        "natural_language_interpretation": (
            "Count only events the user actually attended."
        ),
        "ambiguity_types": [
            "aggregation_view_ambiguous",
            "scope_ambiguous",
        ],
        "assumptions": ["assistant recommendations are not attended events"],
        "expected_answer_shape": "count",
    }


def test_stable_interpretation_id_ignores_case_and_spacing_noise() -> None:
    first = interpretation_ir.stable_interpretation_id(
        "Q1",
        " Count only completed events. ",
    )
    second = interpretation_ir.stable_interpretation_id(
        " q1 ",
        "count   only completed EVENTS.",
    )

    assert first == second
    assert first.startswith("interp_")


def test_interpretation_record_requires_an_ambiguity_label() -> None:
    with pytest.raises(ValueError, match="at least one ambiguity type"):
        interpretation_ir.InterpretationRecord.from_dict(
            {
                "interpretation_id": "i1",
                "natural_language_interpretation": "Count relevant rows.",
                "ambiguity_types": [],
                "expected_answer_shape": "count",
            }
        )
