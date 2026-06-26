"""Verifier-enumeration probe for aggregate fail18 interpretation misses."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import pathlib
import re
import typing

from simba.eval import ambiguity_fail18, interpretation_infill_diagnostics

DEFAULT_VERIFIER_ENUMERATION_PROBE_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_verifier_enumeration_probe.json"
)
DEFAULT_PAYLOADS_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_payloads.json"
)
DEFAULT_PAYLOAD_PROVENANCE_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_payloads_provenance.json"
)

_DAY_RE = re.compile(
    r"\b(?P<number>\d+(?:,\d{3})*(?:\.\d+)?|zero|one|two|three|four|five|"
    r"six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty)\s*(?:-| )?days?\b",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(r"\$\s*(?P<number>\d[\d,]*(?:\.\d+)?)")
_USER_SEGMENT_RE = re.compile(
    r"user:\s*(?P<content>.*?)(?=(?:\n?assistant:|\n?user:)|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_SPACE_RE = re.compile(r"\s+")
_NUMBER_WORDS = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "thirteen": 13.0,
    "fourteen": 14.0,
    "fifteen": 15.0,
    "sixteen": 16.0,
    "seventeen": 17.0,
    "eighteen": 18.0,
    "nineteen": 19.0,
    "twenty": 20.0,
}


@dataclasses.dataclass(frozen=True)
class EvidenceSegment:
    case_id: str
    source_scope: str
    session_id: str
    raw_session_id: str | None
    evidence_date: str | None
    segment_index: int
    text: str
    is_answer_session: bool


@dataclasses.dataclass(frozen=True)
class CandidateUnit:
    case_id: str
    source_scope: str
    unit_id: str
    value: float
    unit: str
    status: str
    reason_code: str
    reason: str
    aggregation_key: str
    evidence_session_id: str
    raw_session_id: str | None
    evidence_date: str | None
    segment_index: int
    evidence_span: str
    is_answer_session: bool

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "case_id": self.case_id,
            "source_scope": self.source_scope,
            "unit_id": self.unit_id,
            "value": self.value,
            "unit": self.unit,
            "status": self.status,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "aggregation_key": self.aggregation_key,
            "evidence_session_id": self.evidence_session_id,
            "raw_session_id": self.raw_session_id,
            "evidence_date": self.evidence_date,
            "segment_index": self.segment_index,
            "evidence_span": self.evidence_span,
            "is_answer_session": self.is_answer_session,
        }


def build_fail18_verifier_enumeration_probe(
    *,
    diagnostics_path: str | pathlib.Path = (
        interpretation_infill_diagnostics.DEFAULT_INFILL_DIAGNOSTICS_PATH
    ),
    payloads_path: str | pathlib.Path = DEFAULT_PAYLOADS_PATH,
    payload_provenance_path: str | pathlib.Path = DEFAULT_PAYLOAD_PROVENANCE_PATH,
    corpus_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_CORPUS,
) -> dict[str, typing.Any]:
    diagnostics = _load_json(diagnostics_path)
    payloads = _load_payloads(payloads_path)
    provenance = _load_optional_json(payload_provenance_path).get(
        "evidence_provenance", {}
    )
    corpus_rows = ambiguity_fail18.load_corpus(corpus_path)
    corpus_by_id = {str(row["question_id"]): row for row in corpus_rows}
    target_cases = [
        typing.cast("dict[str, typing.Any]", case)
        for case in diagnostics.get("blocked_cases", [])
        if case.get("recommended_next_intervention") == "verifier_enumeration_probe"
    ]
    case_results = [
        _probe_case(
            diagnosis=case,
            payload=payloads.get(str(case.get("case_id", "")), {}),
            payload_provenance=typing.cast(
                "dict[str, dict[str, typing.Any]]",
                provenance.get(str(case.get("case_id", "")), {}),
            ),
            corpus_row=corpus_by_id.get(str(case.get("case_id", "")), {}),
        )
        for case in target_cases
    ]
    verdict_counts: dict[str, int] = {}
    for case in case_results:
        verdict = str(case["verdict"])
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    return {
        "name": "fail18-ambiguous-nlidb-gate1-verifier-enumeration-probe",
        "artifact_kind": "interpretation_verifier_enumeration_probe",
        "gate": "gate1",
        "gate_status": "slice2c_verifier_probe_complete",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_diagnostics": str(diagnostics_path),
        "source_payloads": str(payloads_path),
        "source_payload_provenance": str(payload_provenance_path),
        "source_corpus": str(corpus_path),
        "summary": {
            "target_rows": len(case_results),
            "payload_rows_matching_gold": sum(
                1 for case in case_results if case["payload_result"]["matches_gold"]
            ),
            "full_corpus_rows_matching_gold": sum(
                1
                for case in case_results
                if case["full_corpus_result"]["matches_gold"]
            ),
            "verdict_counts": dict(sorted(verdict_counts.items())),
        },
        "decision": {
            "candidate_unit_compilation_should_start": False,
            "next_slice": "targeted_payload_retrieval_or_unit_compiler",
            "reason": (
                "The verifier probe can enumerate aggregate units, but Gate 1 "
                "still needs a non-oracle compiler path and retrieval coverage "
                "checks before candidate-unit compilation is credited."
            ),
        },
        "cases": case_results,
    }


def _probe_case(
    *,
    diagnosis: dict[str, typing.Any],
    payload: dict[str, typing.Any],
    payload_provenance: dict[str, dict[str, typing.Any]],
    corpus_row: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    case_id = str(diagnosis.get("case_id", ""))
    question = str(diagnosis.get("question") or corpus_row.get("question", ""))
    gold_value = _as_float(diagnosis.get("gold_value"))
    probe_kind = _probe_kind(question)
    payload_segments = _payload_segments(
        case_id=case_id,
        payload=payload,
        provenance=payload_provenance,
        answer_session_ids=set(corpus_row.get("answer_session_ids", [])),
        question_date=_optional_str(corpus_row.get("question_date")),
    )
    corpus_segments = _corpus_segments(case_id=case_id, corpus_row=corpus_row)
    payload_units = _enumerate_units(
        case_id=case_id,
        probe_kind=probe_kind,
        segments=payload_segments,
    )
    corpus_units = _enumerate_units(
        case_id=case_id,
        probe_kind=probe_kind,
        segments=corpus_segments,
    )
    payload_result = _summarize_units(payload_units, gold_value)
    corpus_result = _summarize_units(corpus_units, gold_value)
    full_corpus_units_not_in_payload = _full_corpus_units_not_in_payload(
        payload_units=payload_units,
        corpus_units=corpus_units,
    )
    missing_payload_units = (
        []
        if payload_result["matches_gold"]
        else full_corpus_units_not_in_payload
    )
    verdict, verdict_reason = _verdict(
        payload_result=payload_result,
        corpus_result=corpus_result,
        missing_payload_units=missing_payload_units,
    )
    return {
        "case_id": case_id,
        "question": question,
        "failure_mode": diagnosis.get("failure_mode"),
        "probe_kind": probe_kind,
        "gold_value": gold_value,
        "payload_result": payload_result,
        "full_corpus_result": corpus_result,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "full_corpus_included_units_not_in_payload": [
            unit.to_dict() for unit in full_corpus_units_not_in_payload
        ],
        "missing_payload_units_from_full_corpus": [
            unit.to_dict() for unit in missing_payload_units
        ],
        "payload_candidate_units": [unit.to_dict() for unit in payload_units],
        "full_corpus_candidate_units": [unit.to_dict() for unit in corpus_units],
    }


def _probe_kind(question: str) -> str:
    lowered = question.lower()
    if "charity" in lowered and ("money" in lowered or "raise" in lowered):
        return "charity_money_sum"
    if "camping" in lowered and "days" in lowered:
        return "camping_trip_days_sum"
    if "hawaii" in lowered and "new york city" in lowered and "days" in lowered:
        return "destination_travel_days_sum"
    return "unsupported"


def _payload_segments(
    *,
    case_id: str,
    payload: dict[str, typing.Any],
    provenance: dict[str, dict[str, typing.Any]],
    answer_session_ids: set[str],
    question_date: str | None,
) -> list[EvidenceSegment]:
    evidence_sessions = (
        payload.get("case", {}).get("evidence_sessions", [])
        if isinstance(payload.get("case"), dict)
        else []
    )
    segments: list[EvidenceSegment] = []
    for evidence in evidence_sessions:
        if not isinstance(evidence, dict):
            continue
        session_id = str(evidence.get("session_id", ""))
        meta = provenance.get(session_id, {})
        raw_session_id = _optional_str(meta.get("raw_session_id"))
        evidence_date = _optional_str(evidence.get("date"))
        if _is_after_question_date(evidence_date, question_date):
            continue
        for segment_index, text in enumerate(
            _user_segments_from_text(str(evidence.get("text", ""))), start=1
        ):
            segments.append(
                EvidenceSegment(
                    case_id=case_id,
                    source_scope="payload",
                    session_id=session_id,
                    raw_session_id=raw_session_id,
                    evidence_date=evidence_date,
                    segment_index=segment_index,
                    text=text,
                    is_answer_session=bool(
                        raw_session_id and raw_session_id in answer_session_ids
                    ),
                )
            )
    return segments


def _corpus_segments(
    *,
    case_id: str,
    corpus_row: dict[str, typing.Any],
) -> list[EvidenceSegment]:
    answer_session_ids = set(corpus_row.get("answer_session_ids", []))
    sessions = corpus_row.get("haystack_sessions", [])
    session_ids = corpus_row.get("haystack_session_ids", [])
    dates = corpus_row.get("haystack_dates", [])
    question_date = _optional_str(corpus_row.get("question_date"))
    segments: list[EvidenceSegment] = []
    for session_index, session in enumerate(sessions):
        session_id = (
            str(session_ids[session_index])
            if session_index < len(session_ids)
            else ""
        )
        evidence_date = (
            str(dates[session_index]) if session_index < len(dates) else None
        )
        if _is_after_question_date(evidence_date, question_date):
            continue
        session_segments = _user_segments_from_session(session)
        for segment_index, text in enumerate(session_segments, start=1):
            segments.append(
                EvidenceSegment(
                    case_id=case_id,
                    source_scope="full_corpus",
                    session_id=session_id,
                    raw_session_id=session_id,
                    evidence_date=evidence_date,
                    segment_index=segment_index,
                    text=text,
                    is_answer_session=session_id in answer_session_ids,
                )
            )
    return segments


def _user_segments_from_session(session: typing.Any) -> list[str]:
    if isinstance(session, list):
        segments = []
        for message in session:
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).lower() != "user":
                continue
            content = _clean_text(str(message.get("content", "")))
            if content:
                segments.append(content)
        return segments
    return _user_segments_from_text(str(session))


def _user_segments_from_text(text: str) -> list[str]:
    matches = [
        _clean_text(match.group("content"))
        for match in _USER_SEGMENT_RE.finditer(text)
    ]
    return [match for match in matches if match]


def _enumerate_units(
    *,
    case_id: str,
    probe_kind: str,
    segments: list[EvidenceSegment],
) -> list[CandidateUnit]:
    if probe_kind == "charity_money_sum":
        return _enumerate_charity_money(case_id=case_id, segments=segments)
    if probe_kind == "camping_trip_days_sum":
        return _enumerate_camping_days(case_id=case_id, segments=segments)
    if probe_kind == "destination_travel_days_sum":
        return _enumerate_destination_days(case_id=case_id, segments=segments)
    return []


def _enumerate_charity_money(
    *,
    case_id: str,
    segments: list[EvidenceSegment],
) -> list[CandidateUnit]:
    units: list[CandidateUnit] = []
    included_keys: set[str] = set()
    for segment in segments:
        lowered = segment.text.lower()
        if not any(
            term in lowered
            for term in ("charity", "fundrais", "donat", "raised", "raise")
        ):
            continue
        for match in _MONEY_RE.finditer(segment.text):
            value = _number_value(match.group("number"))
            key = _charity_event_key(segment.text, value, match_end=match.end())
            raised_terms = ("raised", "helped raise", "managed to raise")
            if not any(term in lowered for term in raised_terms):
                status = "excluded"
                reason_code = "not_reported_as_raised"
                reason = (
                    "Dollar amount is present, but the user does not report "
                    "it as money already raised."
                )
            elif key in included_keys:
                status = "excluded"
                reason_code = "duplicate_same_charity_event"
                reason = (
                    "Same amount and beneficiary already appeared for the "
                    "charity event."
                )
            else:
                status = "included"
                reason_code = "reported_charity_amount"
                reason = "User reports this dollar amount as money raised for charity."
                included_keys.add(key)
            units.append(
                _candidate_unit(
                    case_id=case_id,
                    segment=segment,
                    units=units,
                    value=value,
                    unit="dollars",
                    status=status,
                    reason_code=reason_code,
                    reason=reason,
                    aggregation_key=key,
                    evidence_span=_span(segment.text, match.start(), match.end()),
                )
            )
    return units


def _enumerate_camping_days(
    *,
    case_id: str,
    segments: list[EvidenceSegment],
) -> list[CandidateUnit]:
    units: list[CandidateUnit] = []
    included_keys: set[str] = set()
    for segment in segments:
        lowered = segment.text.lower()
        broad_terms = ("trip", "camp", "backpack", "trek")
        if not any(term in lowered for term in broad_terms):
            continue
        for match in _DAY_RE.finditer(segment.text):
            value = _number_value(match.group("number"))
            key = _destination_event_key(segment.text, value)
            camping_terms = ("camping", "camped", "backpacking")
            if "not camping" in lowered:
                status = "excluded"
                reason_code = "explicit_not_camping"
                reason = "The user explicitly says this trip was not camping."
            elif not any(term in lowered for term in camping_terms):
                status = "excluded"
                reason_code = "not_camping_trip"
                reason = (
                    "The duration belongs to a trip, but the user does not "
                    "describe it as camping."
                )
            elif not _contains_us_destination(lowered):
                status = "excluded"
                reason_code = "outside_united_states"
                reason = (
                    "The camping or trekking duration is not grounded to a "
                    "United States destination."
                )
            elif _is_future_or_planned(segment.text):
                status = "excluded"
                reason_code = "planned_future_trip"
                reason = (
                    "The duration belongs to a planned future trip rather "
                    "than a completed trip this year."
                )
            elif key in included_keys:
                status = "excluded"
                reason_code = "duplicate_same_trip"
                reason = (
                    "Same destination and duration already appeared for the "
                    "camping trip."
                )
            else:
                status = "included"
                reason_code = "completed_us_camping_trip"
                reason = (
                    "User reports a completed camping trip in a United "
                    "States destination."
                )
                included_keys.add(key)
            units.append(
                _candidate_unit(
                    case_id=case_id,
                    segment=segment,
                    units=units,
                    value=value,
                    unit="days",
                    status=status,
                    reason_code=reason_code,
                    reason=reason,
                    aggregation_key=key,
                    evidence_span=_span(segment.text, match.start(), match.end()),
                )
            )
    return units


def _enumerate_destination_days(
    *,
    case_id: str,
    segments: list[EvidenceSegment],
) -> list[CandidateUnit]:
    units: list[CandidateUnit] = []
    included_keys: set[str] = set()
    target_context_by_session: dict[str, str] = {}
    for segment in segments:
        lowered = segment.text.lower()
        context_key = segment.raw_session_id or segment.session_id
        has_travel_context = any(
            term in lowered for term in ("trip", "travel", "staying", "spend")
        )
        has_existing_target_context = context_key in target_context_by_session
        if not has_travel_context and not has_existing_target_context:
            continue
        explicit_destination = _target_travel_destination(lowered)
        if explicit_destination is not None:
            target_context_by_session[context_key] = explicit_destination
        elif _mentions_non_target_destination(lowered):
            target_context_by_session.pop(context_key, None)
        for match in _DAY_RE.finditer(segment.text):
            value = _number_value(match.group("number"))
            destination = explicit_destination or target_context_by_session.get(
                context_key
            )
            key = f"{destination}:{value:g}:{_short_key(segment.text)}"
            if destination is None:
                status = "excluded"
                reason_code = "non_target_destination"
                reason = "Duration belongs to travel outside Hawaii and New York City."
            elif _is_future_or_planned(segment.text) and "got back" not in lowered:
                status = "excluded"
                reason_code = "planned_future_trip"
                reason = (
                    "Duration belongs to a planned future trip rather than "
                    "completed travel."
                )
            elif key in included_keys:
                status = "excluded"
                reason_code = "duplicate_same_trip"
                reason = "Same target destination and duration already appeared."
            else:
                status = "included"
                reason_code = "completed_target_destination_trip"
                reason = (
                    "User reports completed travel in "
                    f"{destination.replace('_', ' ')}."
                )
                included_keys.add(key)
            units.append(
                _candidate_unit(
                    case_id=case_id,
                    segment=segment,
                    units=units,
                    value=value,
                    unit="days",
                    status=status,
                    reason_code=reason_code,
                    reason=reason,
                    aggregation_key=key,
                    evidence_span=_span(segment.text, match.start(), match.end()),
                )
            )
    return units


def _candidate_unit(
    *,
    case_id: str,
    segment: EvidenceSegment,
    units: list[CandidateUnit],
    value: float,
    unit: str,
    status: str,
    reason_code: str,
    reason: str,
    aggregation_key: str,
    evidence_span: str,
) -> CandidateUnit:
    return CandidateUnit(
        case_id=case_id,
        source_scope=segment.source_scope,
        unit_id=f"{case_id}:{segment.source_scope}:{len(units) + 1:03d}",
        value=value,
        unit=unit,
        status=status,
        reason_code=reason_code,
        reason=reason,
        aggregation_key=aggregation_key,
        evidence_session_id=segment.session_id,
        raw_session_id=segment.raw_session_id,
        evidence_date=segment.evidence_date,
        segment_index=segment.segment_index,
        evidence_span=evidence_span,
        is_answer_session=segment.is_answer_session,
    )


def _summarize_units(
    units: list[CandidateUnit],
    gold_value: float | None,
) -> dict[str, typing.Any]:
    included = [unit for unit in units if unit.status == "included"]
    total = sum(unit.value for unit in included)
    return {
        "candidate_unit_count": len(units),
        "included_unit_count": len(included),
        "excluded_unit_count": len(units) - len(included),
        "computed_total": total,
        "gold_value": gold_value,
        "matches_gold": _matches_gold(total, gold_value),
        "included_aggregation_keys": [unit.aggregation_key for unit in included],
        "included_raw_session_ids": sorted(
            {unit.raw_session_id for unit in included if unit.raw_session_id}
        ),
        "included_answer_session_count": sum(
            1 for unit in included if unit.is_answer_session
        ),
    }


def _full_corpus_units_not_in_payload(
    *,
    payload_units: list[CandidateUnit],
    corpus_units: list[CandidateUnit],
) -> list[CandidateUnit]:
    payload_keys = {
        unit.aggregation_key for unit in payload_units if unit.status == "included"
    }
    return [
        unit
        for unit in corpus_units
        if unit.status == "included" and unit.aggregation_key not in payload_keys
    ]


def _verdict(
    *,
    payload_result: dict[str, typing.Any],
    corpus_result: dict[str, typing.Any],
    missing_payload_units: list[CandidateUnit],
) -> tuple[str, str]:
    if payload_result["matches_gold"]:
        return (
            "payload_enumeration_closes_gold",
            "The bounded provider payload contains enough units for "
            "deterministic enumeration.",
        )
    if corpus_result["matches_gold"] and missing_payload_units:
        return (
            "payload_missing_required_units",
            "Full-corpus enumeration reaches gold, but the bounded provider "
            "payload omits required units.",
        )
    if corpus_result["matches_gold"]:
        return (
            "payload_policy_or_dedup_gap",
            "Full-corpus enumeration reaches gold but the payload computation "
            "still misses without an obvious missing unit.",
        )
    return (
        "verifier_policy_gap",
        "Even full-corpus enumeration does not reach gold, so the "
        "deterministic verifier policy needs work.",
    )


def _charity_event_key(text: str, value: float, *, match_end: int = 0) -> str:
    lowered = text.lower()
    after_amount = lowered[match_end:]
    beneficiary = None
    match = re.search(
        r"for (?:the |a |an )?(?P<beneficiary>[^,.!]+?)(?: on | at |[,.!]|$)",
        after_amount,
    )
    if match:
        beneficiary = _short_key(match.group("beneficiary"))
    if beneficiary is None:
        beneficiary = _short_key(lowered)
    return f"charity:{value:g}:{beneficiary}"


def _destination_event_key(text: str, value: float) -> str:
    lowered = text.lower()
    destinations = [
        "yellowstone",
        "big sur",
        "utah",
        "moab",
        "colorado",
        "rocky mountains",
        "new zealand",
        "hawaii",
        "new york city",
        "nyc",
    ]
    destination = next((item for item in destinations if item in lowered), "unknown")
    return f"{destination.replace(' ', '_')}:{value:g}:{_short_key(text)}"


def _target_travel_destination(lowered: str) -> str | None:
    if "hawaii" in lowered or "island-hopping" in lowered:
        return "hawaii"
    if "new york city" in lowered or re.search(r"\bnyc\b", lowered):
        return "new_york_city"
    return None


def _mentions_non_target_destination(lowered: str) -> bool:
    return any(
        term in lowered
        for term in (
            "europe",
            "paris",
            "rome",
            "barcelona",
            "amsterdam",
            "japan",
            "louvre",
            "colosseum",
            "berlin",
        )
    )


def _contains_us_destination(lowered: str) -> bool:
    return any(
        term in lowered
        for term in (
            "big sur",
            "yellowstone",
            "utah",
            "moab",
            "colorado",
            "rocky mountains",
            "united states",
            "u.s.",
        )
    )


def _is_future_or_planned(text: str) -> bool:
    lowered = text.lower()
    return any(
        term in lowered
        for term in (
            "planning",
            "thinking of",
            "would like",
            "want to",
            "next trip",
            "in november",
            "soon",
        )
    ) and not any(term in lowered for term in ("just got back", "recently got back"))


def _span(text: str, start: int, end: int, *, radius: int = 160) -> str:
    return _clean_text(text[max(0, start - radius) : min(len(text), end + radius)])


def _short_key(text: str, *, limit: int = 80) -> str:
    cleaned = _clean_text(text.lower())
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    return cleaned[:limit] or "unknown"


def _number_value(value: str) -> float:
    normalized = value.replace(",", "").strip().lower()
    if normalized in _NUMBER_WORDS:
        return _NUMBER_WORDS[normalized]
    return float(normalized)


def _as_float(value: typing.Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"\d[\d,]*(?:\.\d+)?", value)
        if match:
            return float(match.group(0).replace(",", ""))
    return None


def _matches_gold(value: float, gold_value: float | None) -> bool:
    return gold_value is not None and abs(value - gold_value) < 0.000001


def _is_after_question_date(
    evidence_date: str | None,
    question_date: str | None,
) -> bool:
    evidence = _date_value(evidence_date)
    question = _date_value(question_date)
    return bool(evidence and question and evidence > question)


def _date_value(value: str | None) -> dt.date | None:
    if not value:
        return None
    match = re.search(r"(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})", value)
    if not match:
        return None
    return dt.date(
        int(match.group("year")),
        int(match.group("month")),
        int(match.group("day")),
    )


def _clean_text(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _optional_str(value: typing.Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _load_payloads(path: str | pathlib.Path) -> dict[str, dict[str, typing.Any]]:
    payload_artifact = _load_json(path)
    return {
        str(payload.get("case", {}).get("id", "")): payload
        for payload in payload_artifact.get("payloads", [])
        if isinstance(payload, dict) and isinstance(payload.get("case"), dict)
    }


def _load_json(path: str | pathlib.Path) -> dict[str, typing.Any]:
    return typing.cast(
        "dict[str, typing.Any]",
        json.loads(pathlib.Path(path).read_text(encoding="utf-8")),
    )


def _load_optional_json(path: str | pathlib.Path) -> dict[str, typing.Any]:
    json_path = pathlib.Path(path)
    if not json_path.exists():
        return {}
    return _load_json(json_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics",
        type=pathlib.Path,
        default=interpretation_infill_diagnostics.DEFAULT_INFILL_DIAGNOSTICS_PATH,
    )
    parser.add_argument("--payloads", type=pathlib.Path, default=DEFAULT_PAYLOADS_PATH)
    parser.add_argument(
        "--payload-provenance",
        type=pathlib.Path,
        default=DEFAULT_PAYLOAD_PROVENANCE_PATH,
    )
    parser.add_argument(
        "--corpus",
        type=pathlib.Path,
        default=ambiguity_fail18.DEFAULT_CORPUS,
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_VERIFIER_ENUMERATION_PROBE_PATH,
    )
    args = parser.parse_args(argv)

    artifact = build_fail18_verifier_enumeration_probe(
        diagnostics_path=args.diagnostics,
        payloads_path=args.payloads,
        payload_provenance_path=args.payload_provenance,
        corpus_path=args.corpus,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"{json.dumps(artifact, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
