"""Recorded candidate-unit IR eval fixtures.

The fixture in ``eval/datasets/fail9_candidate_unit_claude.json`` is a
provider-recorded behavioral spec: Claude was given only the generic
candidate-unit prompt plus fail18 evidence sessions, then returned
``candidate_units`` and an executable-style answer rule.

This module deliberately does not call a model. It validates and scores the
recorded JSON so the behavior can become a stable target for deterministic
compiler/verifier work.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import typing

DEFAULT_FIXTURE_PATH = (
    pathlib.Path(__file__).parent / "datasets" / "fail9_candidate_unit_claude.json"
)

_ALLOWED_ANSWER_VARIABLES = {
    "entity",
    "action_obligation",
    "event",
    "semantic_type",
    "scalar_value",
    "duration",
}
_ALLOWED_POLICIES = {
    "canonical_entity",
    "action_obligation",
    "event_instance",
    "semantic_type",
    "scalar_value",
    "duration_sum",
}
_ALLOWED_UNIT_STATUSES = {"included", "excluded", "merged"}


@dataclasses.dataclass(frozen=True)
class CandidateUnit:
    unit_id: str
    status: str
    merge_target: str | None
    reason: str

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> CandidateUnit:
        return cls(
            unit_id=str(raw["unit_id"]),
            status=str(raw["status"]),
            merge_target=(
                str(raw["merge_target"])
                if raw.get("merge_target") is not None
                else None
            ),
            reason=str(raw.get("reason", "")),
        )


@dataclasses.dataclass(frozen=True)
class CandidateUnitCase:
    id: str
    answer_variable: str
    individuation_policy: str
    aggregation: str
    computed_answer: int
    gold: int
    match: bool
    candidate_units: tuple[CandidateUnit, ...]
    facts: tuple[str, ...]
    query: str
    rationale: str

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> CandidateUnitCase:
        return cls(
            id=str(raw["id"]),
            answer_variable=str(raw["answer_variable"]),
            individuation_policy=str(raw["individuation_policy"]),
            aggregation=str(raw.get("aggregation", "")),
            computed_answer=int(raw["computed_answer"]),
            gold=int(raw["gold"]),
            match=bool(raw["match"]),
            candidate_units=tuple(
                CandidateUnit.from_dict(unit) for unit in raw.get("candidate_units", [])
            ),
            facts=tuple(str(fact) for fact in raw.get("facts", [])),
            query=str(raw.get("query", "")),
            rationale=str(raw.get("rationale", "")),
        )

    @property
    def recomputed_match(self) -> bool:
        return self.computed_answer == self.gold

    def units_with_status(self, status: str) -> tuple[CandidateUnit, ...]:
        return tuple(unit for unit in self.candidate_units if unit.status == status)


@dataclasses.dataclass(frozen=True)
class CandidateUnitFixture:
    name: str
    prompt_version: str
    tool: str
    compiler_contract: tuple[str, ...]
    cases: tuple[CandidateUnitCase, ...]
    stored_score: dict[str, int]

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> CandidateUnitFixture:
        return cls(
            name=str(raw["name"]),
            prompt_version=str(raw["prompt_version"]),
            tool=str(raw["tool"]),
            compiler_contract=tuple(str(rule) for rule in raw["compiler_contract"]),
            cases=tuple(CandidateUnitCase.from_dict(case) for case in raw["cases"]),
            stored_score={
                "matches": int(raw.get("score", {}).get("matches", 0)),
                "total": int(raw.get("score", {}).get("total", 0)),
            },
        )


@dataclasses.dataclass(frozen=True)
class CandidateUnitScore:
    matches: int
    total: int
    mismatches: tuple[str, ...]

    @property
    def accuracy(self) -> float:
        return self.matches / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "matches": self.matches,
            "total": self.total,
            "accuracy": self.accuracy,
            "mismatches": list(self.mismatches),
        }


def load_fixture(
    path: str | pathlib.Path = DEFAULT_FIXTURE_PATH,
) -> CandidateUnitFixture:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return CandidateUnitFixture.from_dict(raw)


def score_fixture(fixture: CandidateUnitFixture) -> CandidateUnitScore:
    mismatches = tuple(case.id for case in fixture.cases if not case.recomputed_match)
    return CandidateUnitScore(
        matches=len(fixture.cases) - len(mismatches),
        total=len(fixture.cases),
        mismatches=mismatches,
    )


def validate_fixture(fixture: CandidateUnitFixture) -> None:
    score = score_fixture(fixture)
    if fixture.stored_score != {"matches": score.matches, "total": score.total}:
        raise ValueError(
            "stored score does not match recomputed score: "
            f"stored={fixture.stored_score} recomputed={score.to_dict()}"
        )

    seen_ids: set[str] = set()
    for case in fixture.cases:
        if case.id in seen_ids:
            raise ValueError(f"duplicate case id: {case.id}")
        seen_ids.add(case.id)
        if case.answer_variable not in _ALLOWED_ANSWER_VARIABLES:
            raise ValueError(
                f"{case.id}: unknown answer_variable {case.answer_variable!r}"
            )
        if case.individuation_policy not in _ALLOWED_POLICIES:
            raise ValueError(
                f"{case.id}: unknown individuation_policy {case.individuation_policy!r}"
            )
        if case.match != case.recomputed_match:
            raise ValueError(
                f"{case.id}: stored match={case.match} but "
                f"computed_answer={case.computed_answer} gold={case.gold}"
            )
        if not case.candidate_units:
            raise ValueError(f"{case.id}: missing candidate_units")
        if not case.query.strip():
            raise ValueError(f"{case.id}: missing query")
        if not case.facts:
            raise ValueError(f"{case.id}: missing facts")
        _validate_units(case)


def build_prompt_payload(
    *,
    case_id: str,
    question: str,
    evidence_sessions: list[dict[str, str]],
    compiler_contract: tuple[str, ...],
) -> dict[str, typing.Any]:
    """Build the live-provider prompt shape captured by the fixture."""
    return {
        "task": (
            "Compile evidence into recursive action(subject, object, verb) IR "
            "and compute the numeric answer. Do not use tools."
        ),
        "compiler_contract": list(compiler_contract),
        "output_schema": {
            "id": case_id,
            "answer_variable": (
                "entity|action_obligation|event|semantic_type|scalar_value|duration"
            ),
            "individuation_policy": (
                "canonical_entity|action_obligation|event_instance|semantic_type|"
                "scalar_value|duration_sum"
            ),
            "candidate_units": [
                {
                    "unit_id": "string",
                    "status": "included|excluded|merged",
                    "merge_target": "unit_id or null",
                    "reason": "short",
                }
            ],
            "aggregation": "count_distinct|sum|lookup",
            "facts": ["prolog-like fact strings"],
            "query": "prolog-like answer(N) rule",
            "computed_answer": "integer only",
            "rationale": "one terse sentence",
        },
        "case": {
            "id": case_id,
            "question": question,
            "evidence_sessions": evidence_sessions,
        },
    }


def _validate_units(case: CandidateUnitCase) -> None:
    unit_ids = {unit.unit_id for unit in case.candidate_units}
    for unit in case.candidate_units:
        if unit.status not in _ALLOWED_UNIT_STATUSES:
            raise ValueError(
                f"{case.id}: unknown candidate unit status {unit.status!r}"
            )
        if unit.status == "merged":
            if not unit.merge_target:
                raise ValueError(f"{case.id}: merged unit lacks merge_target")
            if unit.merge_target not in unit_ids:
                raise ValueError(
                    f"{case.id}: merge target {unit.merge_target!r} does not exist"
                )
        if unit.status != "merged" and unit.merge_target is not None:
            raise ValueError(
                f"{case.id}: non-merged unit has merge_target {unit.merge_target!r}"
            )
        if not unit.reason.strip():
            raise ValueError(f"{case.id}: unit {unit.unit_id!r} lacks reason")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=pathlib.Path,
        default=DEFAULT_FIXTURE_PATH,
        help="Candidate-unit fixture JSON path.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON score.")
    args = parser.parse_args(argv)

    fixture = load_fixture(args.fixture)
    validate_fixture(fixture)
    score = score_fixture(fixture)
    if args.json:
        print(json.dumps(score.to_dict(), sort_keys=True))
    else:
        print(f"{fixture.name}: {score.matches}/{score.total} ({score.accuracy:.0%})")
    return 0 if not score.mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
