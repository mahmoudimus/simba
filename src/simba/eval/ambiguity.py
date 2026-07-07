"""Tiny executable ambiguity cases for structured-data reasoning experiments."""

from __future__ import annotations

import calendar
import dataclasses
import datetime as dt
import json
import math
import pathlib
import typing

FORMALITY_CEILINGS = {
    "F0": 0.70,
    "F1": 0.85,
    "F2": 0.95,
    "F3": 0.99,
}

LAYER_CEILINGS = {
    "L0": 0.35,
    "L1": 0.75,
    "L2": 1.00,
}

Answer = dict[str, int]


@dataclasses.dataclass(frozen=True)
class Assumption:
    id: str
    value: typing.Any
    reliability: float

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> Assumption:
        return cls(
            id=str(raw["id"]),
            value=raw.get("value"),
            reliability=float(raw.get("reliability", 1.0)),
        )


@dataclasses.dataclass(frozen=True)
class Interpretation:
    id: str
    label: str
    params: dict[str, typing.Any]
    assumptions: list[Assumption]
    raw_reliability: float
    layer: str = "L0"
    formality: str = "F1"
    expected_answer: Answer = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> Interpretation:
        return cls(
            id=str(raw["id"]),
            label=str(raw.get("label", raw["id"])),
            params=dict(raw.get("params", {})),
            assumptions=[
                Assumption.from_dict(item) for item in raw.get("assumptions", [])
            ],
            raw_reliability=float(raw.get("raw_reliability", 1.0)),
            layer=str(raw.get("layer", "L0")),
            formality=str(raw.get("formality", "F1")),
            expected_answer=_normalize_answer(raw.get("expected_answer", {})),
        )

    def effective_reliability(self) -> float:
        """Weakest-link reliability with independent layer/formality ceilings."""
        scores = [
            self.raw_reliability,
            LAYER_CEILINGS[self.layer],
            FORMALITY_CEILINGS[self.formality],
        ]
        scores.extend(assumption.reliability for assumption in self.assumptions)
        return min(scores)


@dataclasses.dataclass(frozen=True)
class AmbiguityCase:
    id: str
    category: str
    question: str
    program: str
    records: list[dict[str, typing.Any]]
    interpretations: list[Interpretation]
    expected_answer_space: Answer
    source_dataset: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> AmbiguityCase:
        return cls(
            id=str(raw["id"]),
            category=str(raw.get("category", "")),
            question=str(raw["question"]),
            program=str(raw["program"]),
            records=list(raw.get("records", [])),
            interpretations=[
                Interpretation.from_dict(item)
                for item in raw.get("interpretations", [])
            ],
            expected_answer_space=_normalize_answer(
                raw.get("expected_answer_space", {})
            ),
            source_dataset=str(raw.get("source_dataset", "")),
        )


@dataclasses.dataclass(frozen=True)
class InterpretationResult:
    interpretation_id: str
    answer: Answer
    reliability: float
    evidence_ids: list[str]


@dataclasses.dataclass(frozen=True)
class AmbiguityReport:
    case_id: str
    question: str
    answer_space: Answer
    interpretations: list[InterpretationResult]


def load_cases(path: str | pathlib.Path) -> list[AmbiguityCase]:
    """Load and validate a JSON ambiguity-case fixture."""
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    cases = [AmbiguityCase.from_dict(item) for item in raw.get("cases", [])]
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"duplicate ambiguity case id: {case.id}")
        seen.add(case.id)
        _validate_case(case)
    return cases


def evaluate_case(case: AmbiguityCase, *, backend: str = "python") -> AmbiguityReport:
    """Evaluate every interpretation and return the preserved answer space."""
    from simba.eval.ambiguity_backends import resolve_backend

    runner = resolve_backend(backend)
    results: list[InterpretationResult] = []
    for interp in case.interpretations:
        backend_result = runner.evaluate(case, interp)
        results.append(
            InterpretationResult(
                interpretation_id=interp.id,
                answer=backend_result.answer,
                reliability=interp.effective_reliability(),
                evidence_ids=backend_result.evidence_ids,
            )
        )
    return AmbiguityReport(
        case_id=case.id,
        question=case.question,
        answer_space=answer_space(results),
        interpretations=results,
    )


def evaluate_all(
    cases: list[AmbiguityCase], *, backend: str = "python"
) -> list[AmbiguityReport]:
    return [evaluate_case(case, backend=backend) for case in cases]


def evaluate_interpretation_python(
    case: AmbiguityCase, interp: Interpretation
) -> tuple[Answer, list[str]]:
    """Evaluate one interpretation with Simba's in-process Python semantics."""
    return _evaluate_interpretation(case, interp)


def answer_space(results: list[InterpretationResult]) -> Answer:
    lows: list[int] = []
    highs: list[int] = []
    for result in results:
        low, high = _answer_bounds(result.answer)
        lows.append(low)
        highs.append(high)
    if not lows:
        return {"lower": 0, "upper": 0}
    return {"lower": min(lows), "upper": max(highs)}


def prove_answer_space_with_z3(report: AmbiguityReport) -> bool:
    """Use Z3 to prove every interpretation answer lies inside the report bounds."""
    import z3

    lo, hi = _answer_bounds(report.answer_space)
    answer = z3.Int("answer")
    solver = z3.Solver()
    solver.add(
        z3.Or(
            *[
                z3.And(answer >= low, answer <= high)
                for low, high in (
                    _answer_bounds(r.answer) for r in report.interpretations
                )
            ]
        )
    )
    solver.add(z3.Or(answer < lo, answer > hi))
    return solver.check() == z3.unsat


def _validate_case(case: AmbiguityCase) -> None:
    if not case.interpretations:
        raise ValueError(f"case {case.id!r} has no interpretations")
    interp_ids = [interp.id for interp in case.interpretations]
    if len(interp_ids) != len(set(interp_ids)):
        raise ValueError(f"case {case.id!r} has duplicate interpretation ids")
    for interp in case.interpretations:
        if interp.layer not in LAYER_CEILINGS:
            raise ValueError(f"unknown layer {interp.layer!r} in case {case.id!r}")
        if interp.formality not in FORMALITY_CEILINGS:
            raise ValueError(
                f"unknown formality {interp.formality!r} in case {case.id!r}"
            )


def _evaluate_interpretation(
    case: AmbiguityCase, interp: Interpretation
) -> tuple[Answer, list[str]]:
    if case.program == "count_recent_births":
        return _count_recent_births(case.records, interp.params)
    if case.program == "count_lot_products":
        return _count_lot_products(case.records, interp.params)
    if case.program == "count_nearby_users":
        return _count_nearby_users(case.records, interp.params)
    if case.program == "count_apple_purchases":
        return _count_apple_purchases(case.records, interp.params)
    if case.program == "count_candidate_rows":
        return _count_candidate_rows(case.records, interp.params)
    raise ValueError(f"unknown ambiguity program: {case.program}")


def _count_recent_births(
    records: list[dict[str, typing.Any]], params: dict[str, typing.Any]
) -> tuple[Answer, list[str]]:
    anchor = _parse_date(str(params["anchor_date"]))
    months = int(params["months"])
    cutoff = anchor - dt.timedelta(days=31 * months)
    missing_policy = str(params.get("missing_policy", "exclude_unknown"))
    must_ids: list[str] = []
    possible_ids: list[str] = []

    for record in records:
        status = _date_window_status(record, cutoff=cutoff, anchor=anchor)
        if status == "must":
            must_ids.append(str(record["id"]))
        elif status == "possible":
            possible_ids.append(str(record["id"]))

    if missing_policy == "include_possible_range":
        ids = must_ids + possible_ids
        return {"lower": len(must_ids), "upper": len(ids)}, ids
    return {"count": len(must_ids)}, must_ids


def _count_lot_products(
    records: list[dict[str, typing.Any]], params: dict[str, typing.Any]
) -> tuple[Answer, list[str]]:
    kind = str(params["threshold"])
    counts = [int(record["items"]) for record in records]
    threshold = 0.0
    if kind == "min_items":
        threshold = float(params["min_items"])
    elif kind == "above_average":
        threshold = sum(counts) / len(counts)
    elif kind == "top_percentile":
        top_n = max(1, math.ceil(len(counts) * float(params["percentile"]) / 100.0))
        threshold = sorted(counts, reverse=True)[top_n - 1]
    else:
        raise ValueError(f"unknown product threshold: {kind}")
    ids = [str(record["id"]) for record in records if int(record["items"]) >= threshold]
    if kind == "above_average":
        ids = [
            str(record["id"]) for record in records if int(record["items"]) > threshold
        ]
    return {"count": len(ids)}, ids


def _count_nearby_users(
    records: list[dict[str, typing.Any]], params: dict[str, typing.Any]
) -> tuple[Answer, list[str]]:
    mode = str(params["mode"])
    if mode == "radius_miles":
        radius = float(params["radius_miles"])
        ids = [
            str(record["id"])
            for record in records
            if float(record["distance_miles"]) <= radius
        ]
    elif mode == "same_city":
        city = str(params["city"])
        ids = [str(record["id"]) for record in records if record.get("city") == city]
    elif mode == "same_metro":
        metro = str(params["metro"])
        ids = [str(record["id"]) for record in records if record.get("metro") == metro]
    else:
        raise ValueError(f"unknown nearby mode: {mode}")
    return {"count": len(ids)}, ids


def _count_apple_purchases(
    records: list[dict[str, typing.Any]], params: dict[str, typing.Any]
) -> tuple[Answer, list[str]]:
    entity_kind = str(params["entity_kind"])
    if entity_kind == "merchant_apple_inc":
        ids = [
            str(record["id"]) for record in records if record.get("merchant") == "Apple"
        ]
    elif entity_kind == "apple_product_brand":
        ids = [
            str(record["id"]) for record in records if record.get("brand") == "Apple"
        ]
    elif entity_kind == "grocery_apples":
        ids = [
            str(record["id"])
            for record in records
            if record.get("category") == "grocery"
            and "apple" in str(record.get("item", "")).lower()
        ]
    else:
        raise ValueError(f"unknown Apple entity kind: {entity_kind}")
    return {"count": len(ids)}, ids


def _count_candidate_rows(
    records: list[dict[str, typing.Any]], params: dict[str, typing.Any]
) -> tuple[Answer, list[str]]:
    lower_statuses = {str(item) for item in params.get("lower_statuses", [])}
    upper_statuses = {str(item) for item in params.get("upper_statuses", [])}
    statuses = {str(item) for item in params.get("statuses", [])}
    if statuses:
        lower_statuses = statuses
        upper_statuses = statuses
    lower_ids = [
        str(record["id"])
        for record in records
        if str(record.get("status", "")) in lower_statuses
    ]
    upper_ids = [
        str(record["id"])
        for record in records
        if str(record.get("status", "")) in upper_statuses
    ]
    if len(lower_ids) == len(upper_ids):
        return {"count": len(lower_ids)}, lower_ids
    return {"lower": len(lower_ids), "upper": len(upper_ids)}, upper_ids


def _date_window_status(
    record: dict[str, typing.Any], *, cutoff: dt.date, anchor: dt.date
) -> str:
    if record.get("birth_date"):
        date = _parse_date(str(record["birth_date"]))
        return "must" if cutoff <= date <= anchor else "outside"
    if record.get("birth_month"):
        first, last = _month_bounds(str(record["birth_month"]))
        if last < cutoff or first > anchor:
            return "outside"
        if first >= cutoff and last <= anchor:
            return "must"
        return "possible"
    return "possible"


def _month_bounds(value: str) -> tuple[dt.date, dt.date]:
    year, month = (int(part) for part in value.split("-", 1))
    last_day = calendar.monthrange(year, month)[1]
    return dt.date(year, month, 1), dt.date(year, month, last_day)


def _parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def _normalize_answer(raw: typing.Any) -> Answer:
    if isinstance(raw, dict):
        if "count" in raw:
            return {"count": int(raw["count"])}
        if "lower" in raw and "upper" in raw:
            return {"lower": int(raw["lower"]), "upper": int(raw["upper"])}
    raise ValueError(f"answer must be {{count}} or {{lower, upper}}, got {raw!r}")


def _answer_bounds(answer: Answer) -> tuple[int, int]:
    if "count" in answer:
        count = int(answer["count"])
        return count, count
    return int(answer["lower"]), int(answer["upper"])
