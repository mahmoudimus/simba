"""Natural-language interpretation records for ambiguous NLIDB evals."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import typing

from simba.eval.ambiguity_taxonomy import (
    AmbiguityType,
    normalize_ambiguity_types,
)


class ExpectedAnswerShape(enum.StrEnum):
    COUNT = "count"
    SUM = "sum"
    LOOKUP = "lookup"
    RANGE = "range"
    SET = "set"


@dataclasses.dataclass(frozen=True)
class InterpretationRecord:
    interpretation_id: str
    natural_language_interpretation: str
    ambiguity_types: tuple[AmbiguityType, ...]
    assumptions: tuple[str, ...]
    expected_answer_shape: ExpectedAnswerShape

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> InterpretationRecord:
        return cls(
            interpretation_id=str(raw["interpretation_id"]),
            natural_language_interpretation=str(raw["natural_language_interpretation"]),
            ambiguity_types=normalize_ambiguity_types(
                tuple(str(item) for item in raw.get("ambiguity_types", ()))
            ),
            assumptions=tuple(str(item) for item in raw.get("assumptions", ())),
            expected_answer_shape=ExpectedAnswerShape(
                str(raw["expected_answer_shape"])
            ),
        )

    def __post_init__(self) -> None:
        if not self.interpretation_id.strip():
            raise ValueError("interpretation_id is required")
        if not self.natural_language_interpretation.strip():
            raise ValueError("natural_language_interpretation is required")
        if not self.ambiguity_types:
            raise ValueError("at least one ambiguity type is required")

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "interpretation_id": self.interpretation_id,
            "natural_language_interpretation": (self.natural_language_interpretation),
            "ambiguity_types": [item.value for item in self.ambiguity_types],
            "assumptions": list(self.assumptions),
            "expected_answer_shape": self.expected_answer_shape.value,
        }


def stable_interpretation_id(
    case_id: str,
    natural_language_interpretation: str,
    *,
    prefix: str = "interp",
) -> str:
    """Return a deterministic short id for a proposed interpretation."""
    body = "\n".join(
        (
            _normalize_for_hash(case_id),
            _normalize_for_hash(natural_language_interpretation),
        )
    )
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


def _normalize_for_hash(value: str) -> str:
    return " ".join(value.lower().strip().split())
