from __future__ import annotations

import pytest

import simba.eval.ambiguity_taxonomy as ambiguity_taxonomy


def test_normalizes_ambisql_and_ambrosia_labels() -> None:
    assert (
        ambiguity_taxonomy.normalize_ambiguity_type("AmbiSchema")
        == ambiguity_taxonomy.AmbiguityType.SCHEMA_LINK_AMBIGUOUS
    )
    assert (
        ambiguity_taxonomy.normalize_ambiguity_type("AmbiView")
        == ambiguity_taxonomy.AmbiguityType.AGGREGATION_VIEW_AMBIGUOUS
    )
    assert (
        ambiguity_taxonomy.normalize_ambiguity_type("scope")
        == ambiguity_taxonomy.AmbiguityType.SCOPE_AMBIGUOUS
    )
    assert (
        ambiguity_taxonomy.normalize_ambiguity_type("vagueness")
        == ambiguity_taxonomy.AmbiguityType.VAGUE_PREDICATE
    )


def test_normalize_ambiguity_types_deduplicates_in_order() -> None:
    normalized = ambiguity_taxonomy.normalize_ambiguity_types(
        ["AmbiRef", "reference_resolution_ambiguous", "AmbiContext"]
    )

    assert normalized == (
        ambiguity_taxonomy.AmbiguityType.REFERENCE_RESOLUTION_AMBIGUOUS,
        ambiguity_taxonomy.AmbiguityType.CONTEXT_PARAMETER_AMBIGUOUS,
    )


def test_unknown_ambiguity_type_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown ambiguity type"):
        ambiguity_taxonomy.normalize_ambiguity_type("domain_specific_magic")
