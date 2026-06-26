"""Domain-general ambiguity labels for ambiguous NLIDB evals."""

from __future__ import annotations

import enum


class AmbiguityType(enum.StrEnum):
    SCHEMA_LINK_AMBIGUOUS = "schema_link_ambiguous"
    VALUE_MAPPING_AMBIGUOUS = "value_mapping_ambiguous"
    AGGREGATION_VIEW_AMBIGUOUS = "aggregation_view_ambiguous"
    SOURCE_OF_TRUTH_AMBIGUOUS = "source_of_truth_ambiguous"
    CONTEXT_PARAMETER_AMBIGUOUS = "context_parameter_ambiguous"
    FALSE_ASSUMPTION = "false_assumption"
    REFERENCE_RESOLUTION_AMBIGUOUS = "reference_resolution_ambiguous"
    SCOPE_AMBIGUOUS = "scope_ambiguous"
    ATTACHMENT_AMBIGUOUS = "attachment_ambiguous"
    VAGUE_PREDICATE = "vague_predicate"


_ALIASES = {
    "ambischema": AmbiguityType.SCHEMA_LINK_AMBIGUOUS,
    "ambi_schema": AmbiguityType.SCHEMA_LINK_AMBIGUOUS,
    "schema": AmbiguityType.SCHEMA_LINK_AMBIGUOUS,
    "ambivalue": AmbiguityType.VALUE_MAPPING_AMBIGUOUS,
    "ambi_value": AmbiguityType.VALUE_MAPPING_AMBIGUOUS,
    "value": AmbiguityType.VALUE_MAPPING_AMBIGUOUS,
    "ambiview": AmbiguityType.AGGREGATION_VIEW_AMBIGUOUS,
    "ambi_view": AmbiguityType.AGGREGATION_VIEW_AMBIGUOUS,
    "view": AmbiguityType.AGGREGATION_VIEW_AMBIGUOUS,
    "ambisource": AmbiguityType.SOURCE_OF_TRUTH_AMBIGUOUS,
    "ambi_source": AmbiguityType.SOURCE_OF_TRUTH_AMBIGUOUS,
    "source": AmbiguityType.SOURCE_OF_TRUTH_AMBIGUOUS,
    "ambicontext": AmbiguityType.CONTEXT_PARAMETER_AMBIGUOUS,
    "ambi_context": AmbiguityType.CONTEXT_PARAMETER_AMBIGUOUS,
    "context": AmbiguityType.CONTEXT_PARAMETER_AMBIGUOUS,
    "ambifallacy": AmbiguityType.FALSE_ASSUMPTION,
    "ambi_fallacy": AmbiguityType.FALSE_ASSUMPTION,
    "fallacy": AmbiguityType.FALSE_ASSUMPTION,
    "ambiref": AmbiguityType.REFERENCE_RESOLUTION_AMBIGUOUS,
    "ambi_ref": AmbiguityType.REFERENCE_RESOLUTION_AMBIGUOUS,
    "reference": AmbiguityType.REFERENCE_RESOLUTION_AMBIGUOUS,
    "scope": AmbiguityType.SCOPE_AMBIGUOUS,
    "attachment": AmbiguityType.ATTACHMENT_AMBIGUOUS,
    "vagueness": AmbiguityType.VAGUE_PREDICATE,
    "vague": AmbiguityType.VAGUE_PREDICATE,
}


def normalize_ambiguity_type(value: str | AmbiguityType) -> AmbiguityType:
    """Map external taxonomy names onto Simba's stable ambiguity labels."""
    if isinstance(value, AmbiguityType):
        return value
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return AmbiguityType(key)
    except ValueError:
        pass
    compact_key = key.replace("_", "")
    if compact_key in _ALIASES:
        return _ALIASES[compact_key]
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"unknown ambiguity type: {value!r}")


def normalize_ambiguity_types(
    values: list[str] | tuple[str, ...] | tuple[AmbiguityType, ...],
) -> tuple[AmbiguityType, ...]:
    """Normalize labels while preserving first-seen order and removing repeats."""
    seen: set[AmbiguityType] = set()
    normalized: list[AmbiguityType] = []
    for value in values:
        item = normalize_ambiguity_type(value)
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    return tuple(normalized)


def ambiguity_type_values() -> tuple[str, ...]:
    return tuple(item.value for item in AmbiguityType)
