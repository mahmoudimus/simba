"""Adapter for the saved clingo_fail18 failure fixture."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import typing

import simba.eval.ambiguity as ambiguity

DEFAULT_MANIFEST = pathlib.Path(".simba/fixtures/clingo_fail18_manifest.json")

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


@dataclasses.dataclass(frozen=True)
class Fail18Result:
    question_id: str
    question: str
    failure_mode: str
    gold_numeric: int | None
    answer_space: dict[str, int]
    contains_gold: bool | None
    backend: str


@dataclasses.dataclass(frozen=True)
class Fail18Summary:
    backend: str
    total: int
    gold_known: int
    contains_gold: int
    misses_gold: int
    results: list[Fail18Result]

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "backend": self.backend,
            "total": self.total,
            "gold_known": self.gold_known,
            "contains_gold": self.contains_gold,
            "misses_gold": self.misses_gold,
            "results": [
                {
                    "question_id": item.question_id,
                    "question": item.question,
                    "failure_mode": item.failure_mode,
                    "gold_numeric": item.gold_numeric,
                    "answer_space": item.answer_space,
                    "contains_gold": item.contains_gold,
                    "backend": item.backend,
                }
                for item in self.results
            ],
        }


def load_manifest(
    path: str | pathlib.Path = DEFAULT_MANIFEST,
) -> list[dict[str, typing.Any]]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def manifest_row_to_case(row: dict[str, typing.Any]) -> ambiguity.AmbiguityCase:
    qid = str(row["question_id"])
    certain = max(0, int(row.get("clingo_certain") or 0))
    possible = max(certain, int(row.get("clingo_possible") or certain))
    records = [
        {"id": f"{qid}_certain_{idx}", "status": "certain"}
        for idx in range(certain)
    ]
    records.extend(
        {"id": f"{qid}_possible_{idx}", "status": "possible"}
        for idx in range(possible - certain)
    )
    return ambiguity.AmbiguityCase(
        id=f"fail18_{qid}",
        category="clingo_fail18",
        source_dataset="clingo_fail18_manifest",
        question=str(row.get("question", "")),
        program="count_candidate_rows",
        records=records,
        expected_answer_space={"lower": certain, "upper": possible},
        interpretations=[
            ambiguity.Interpretation(
                id="certain_only",
                label="old clingo certain rows only",
                params={"statuses": ["certain"]},
                assumptions=[
                    ambiguity.Assumption(
                        id="candidate_policy",
                        value="certain",
                        reliability=0.60,
                    )
                ],
                raw_reliability=0.60,
                layer="L1",
                formality="F1",
                expected_answer={"count": certain},
            ),
            ambiguity.Interpretation(
                id="certain_possible_range",
                label="old clingo certain lower bound and possible upper bound",
                params={
                    "lower_statuses": ["certain"],
                    "upper_statuses": ["certain", "possible"],
                },
                assumptions=[
                    ambiguity.Assumption(
                        id="candidate_policy",
                        value="certain..possible",
                        reliability=0.55,
                    )
                ],
                raw_reliability=0.55,
                layer="L1",
                formality="F1",
                expected_answer={"lower": certain, "upper": possible},
            ),
        ],
    )


def load_cases(
    path: str | pathlib.Path = DEFAULT_MANIFEST,
) -> list[ambiguity.AmbiguityCase]:
    return [manifest_row_to_case(row) for row in load_manifest(path)]


def summarize(
    path: str | pathlib.Path = DEFAULT_MANIFEST, *, backend: str = "python"
) -> Fail18Summary:
    rows = load_manifest(path)
    results: list[Fail18Result] = []
    for row in rows:
        case = manifest_row_to_case(row)
        report = ambiguity.evaluate_case(case, backend=backend)
        gold = numeric_gold(row)
        contains = _contains(report.answer_space, gold) if gold is not None else None
        results.append(
            Fail18Result(
                question_id=str(row["question_id"]),
                question=str(row.get("question", "")),
                failure_mode=str(row.get("failure_mode", "")),
                gold_numeric=gold,
                answer_space=report.answer_space,
                contains_gold=contains,
                backend=backend,
            )
        )
    known = [item for item in results if item.contains_gold is not None]
    hits = [item for item in known if item.contains_gold]
    return Fail18Summary(
        backend=backend,
        total=len(results),
        gold_known=len(known),
        contains_gold=len(hits),
        misses_gold=len(known) - len(hits),
        results=results,
    )


def numeric_gold(row: dict[str, typing.Any]) -> int | None:
    text = str(row.get("gold_answer", "")).lower()
    match = re.search(r"\d[\d,]*", text)
    if match:
        return int(match.group().replace(",", ""))
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return value
    raw_count = row.get("gold_count")
    if isinstance(raw_count, int):
        return raw_count
    if isinstance(raw_count, str) and raw_count.strip().isdigit():
        return int(raw_count)
    return None


def _contains(answer: dict[str, int], gold: int) -> bool:
    if "count" in answer:
        return int(answer["count"]) == gold
    return int(answer["lower"]) <= gold <= int(answer["upper"])
