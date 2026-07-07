"""Adjudicator-first ambiguity mining for LongMemEval-S rows.

This probe is deliberately separate from answer generation. It asks whether the
question plus gold answer-session evidence admits multiple defensible readings,
then gates the disambiguation-witness thesis on that population existing.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import pathlib
import re
import typing

from simba.eval import ambiguity_fail18, interpretation_runner

PROMPT_VERSION = "ambiguity_mining_adjudicator_v1"
DEFAULT_DATASET_PATH = pathlib.Path(".simba/benchmarks/longmemeval_s.json")
DEFAULT_HELDOUT_MANIFEST_PATH = pathlib.Path(
    "_gitless/longmemeval_s_answer_unit_witness_heldout_manifest_8k.json"
)
DEFAULT_PAYLOADS_PATH = pathlib.Path(
    "_gitless/longmemeval_s_ambiguity_mining_payloads.json"
)
DEFAULT_OUTPUTS_PATH = pathlib.Path(
    "_gitless/longmemeval_s_ambiguity_mining_outputs.jsonl"
)
DEFAULT_REPORT_PATH = pathlib.Path(
    "_gitless/longmemeval_s_ambiguity_mining_report.json"
)
DEFAULT_PROVIDER_COMMAND = (
    "claude -p --no-session-persistence --safe-mode --tools '' "
    "--model 'claude-opus-4-8[1m]' --effort medium --output-format json "
    "--system-prompt 'Return exactly one strict JSON object and no markdown.'"
)
DEFAULT_TIMEOUT_SECONDS = 240
DEFAULT_SESSION_CHAR_LIMIT = 8000
DEFAULT_MAX_ANSWER_SESSIONS = 6
DEFAULT_WIDER_LIMIT = 32
DEFAULT_GO_MIN_CASES = 3
DEFAULT_GO_MIN_RATE = 0.05

PARSE_STATUS_PARSED = "parsed"
PARSE_STATUS_EMPTY = "empty"
PARSE_STATUS_INVALID_JSON = "invalid_json"
PARSE_STATUS_INVALID_SCHEMA = "invalid_schema"

ALLOWED_VERDICTS = {
    "contestable",
    "not_contestable",
    "insufficient_evidence",
}
ALLOWED_DEFENSIBILITY = {"strong", "moderate", "weak"}
ALLOWED_BUCKETS = {
    "semantic_collapsed_gold_contestable",
    "gold_already_articulated",
    "inclusive_exclusive_arithmetic",
    "data_conflict",
    "knowledge_update",
    "enumeration_variance",
    "retrieval_context_gap",
    "not_contestable",
    "insufficient_evidence",
}
ALLOWED_AXIS_TYPES = {
    "category_boundary",
    "set_scope",
    "temporal_scope",
    "multi_entity_collapse",
    "multi_location_collapse",
    "underspecification_inference",
    "inclusive_exclusive_arithmetic",
    "data_conflict",
    "knowledge_update",
    "enumeration_variance",
    "retrieval_context_gap",
    "not_ambiguous",
}
THESIS_AXIS_TYPES = {
    "category_boundary",
    "set_scope",
    "temporal_scope",
    "multi_entity_collapse",
    "multi_location_collapse",
    "underspecification_inference",
}
EXCLUDED_AXIS_TYPES = ALLOWED_AXIS_TYPES - THESIS_AXIS_TYPES
SEMANTIC_BUCKET = "semantic_collapsed_gold_contestable"
GOLD_ALREADY_ARTICULATED_BUCKET = "gold_already_articulated"
INCLUSIVE_EXCLUSIVE_BUCKET = "inclusive_exclusive_arithmetic"


@dataclasses.dataclass(frozen=True)
class PivotSpan:
    evidence_session_id: str
    span: str
    why_pivot: str

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "evidence_session_id": self.evidence_session_id,
            "span": self.span,
            "why_pivot": self.why_pivot,
        }


@dataclasses.dataclass(frozen=True)
class CandidateReading:
    reading_id: str
    interpretation: str
    answer_value: str
    compatible_with_official_answer: bool
    pivot_spans: tuple[PivotSpan, ...]
    assumptions: tuple[str, ...]
    defensibility: str

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "reading_id": self.reading_id,
            "interpretation": self.interpretation,
            "answer_value": self.answer_value,
            "compatible_with_official_answer": self.compatible_with_official_answer,
            "pivot_spans": [span.to_dict() for span in self.pivot_spans],
            "assumptions": list(self.assumptions),
            "defensibility": self.defensibility,
        }


@dataclasses.dataclass(frozen=True)
class AdjudicationParseResult:
    case_id: str
    parse_status: str
    verdict: str
    bucket: str
    axis_type: str
    readings: tuple[CandidateReading, ...]
    contestability_reason: str
    parse_errors: tuple[str, ...]
    provider_result_text: str = ""

    def to_output_dict(
        self,
        *,
        provider: str,
        raw_output: str,
    ) -> dict[str, typing.Any]:
        return {
            "case_id": self.case_id,
            "provider": provider,
            "prompt_version": PROMPT_VERSION,
            "raw_output": raw_output,
            "provider_result_text": self.provider_result_text,
            "parse_status": self.parse_status,
            "verdict": self.verdict,
            "bucket": self.bucket,
            "axis_type": self.axis_type,
            "readings": [reading.to_dict() for reading in self.readings],
            "contestability_reason": self.contestability_reason,
            "parse_errors": list(self.parse_errors),
        }


def build_payload_artifact(
    *,
    dataset_path: str | pathlib.Path = DEFAULT_DATASET_PATH,
    heldout_manifest_path: str | pathlib.Path = DEFAULT_HELDOUT_MANIFEST_PATH,
    fail18_manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    wider_limit: int = DEFAULT_WIDER_LIMIT,
    session_char_limit: int = DEFAULT_SESSION_CHAR_LIMIT,
    max_answer_sessions: int = DEFAULT_MAX_ANSWER_SESSIONS,
) -> dict[str, typing.Any]:
    dataset_rows = _load_dataset_rows(dataset_path)
    rows_by_id = {str(row.get("question_id", "")): row for row in dataset_rows}
    heldout_ids = _heldout_ids(heldout_manifest_path)
    fail18_ids = _fail18_ids(fail18_manifest_path)
    selected: list[tuple[str, dict[str, typing.Any]]] = []
    for question_id in heldout_ids:
        row = rows_by_id.get(question_id)
        if row is not None and _row_has_answer_sessions(row):
            selected.append(("heldout", row))

    excluded = set(heldout_ids) | fail18_ids
    wider_added = 0
    for row in dataset_rows:
        question_id = str(row.get("question_id", ""))
        if question_id in excluded or question_id.endswith("_abs"):
            continue
        if not _row_has_answer_sessions(row):
            continue
        selected.append(("wider_lme_s", row))
        wider_added += 1
        if wider_added >= wider_limit:
            break

    payloads = [
        build_payload(
            source_split=source_split,
            row=row,
            session_char_limit=session_char_limit,
            max_answer_sessions=max_answer_sessions,
        )
        for source_split, row in selected
    ]
    return {
        "name": "longmemeval-s-ambiguity-mining-payloads",
        "artifact_kind": "ambiguity_mining_provider_payloads",
        "prompt_version": PROMPT_VERSION,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_dataset": str(dataset_path),
        "source_heldout_manifest": str(heldout_manifest_path),
        "source_fail18_manifest": str(fail18_manifest_path),
        "provider_visibility": {
            "official_answer_visible": True,
            "system_answer_visible": False,
            "witness_outputs_visible": False,
            "failure_mode_visible": False,
        },
        "selection": {
            "heldout_count": sum(
                1 for payload in payloads if _split(payload) == "heldout"
            ),
            "wider_limit": wider_limit,
            "wider_count": sum(
                1 for payload in payloads if _split(payload) == "wider_lme_s"
            ),
            "answer_sessions_only": True,
            "session_char_limit": session_char_limit,
            "max_answer_sessions": max_answer_sessions,
            "excluded_fail18_count": len(fail18_ids),
        },
        "total": len(payloads),
        "payloads": payloads,
    }


def build_payload(
    *,
    source_split: str,
    row: dict[str, typing.Any],
    session_char_limit: int = DEFAULT_SESSION_CHAR_LIMIT,
    max_answer_sessions: int = DEFAULT_MAX_ANSWER_SESSIONS,
) -> dict[str, typing.Any]:
    answer_session_ids = [str(item) for item in row.get("answer_session_ids", [])]
    sessions = dict(
        zip(
            [str(item) for item in row.get("haystack_session_ids", [])],
            row.get("haystack_sessions", []),
            strict=False,
        )
    )
    dates = dict(
        zip(
            [str(item) for item in row.get("haystack_session_ids", [])],
            [str(item) for item in row.get("haystack_dates", [])],
            strict=False,
        )
    )
    evidence_sessions = []
    for session_id in answer_session_ids[:max_answer_sessions]:
        messages = sessions.get(session_id)
        if not isinstance(messages, list):
            continue
        evidence_sessions.append(
            {
                "session_id": session_id,
                "date": dates.get(session_id, ""),
                "text": _session_text(messages, char_limit=session_char_limit),
            }
        )
    return {
        "task": (
            "Adjudicate whether this dataset row has genuinely contestable "
            "answer semantics. Do not answer as a retrieval system and do not "
            "evaluate any model output."
        ),
        "prompt_version": PROMPT_VERSION,
        "contract": [
            "Use only the provided question, official_answer, and evidence_sessions.",
            (
                "Return contestable only if at least two textually defensible "
                "readings yield different answer values."
            ),
            (
                "Choose exactly one bucket from the output schema. Only "
                "semantic_collapsed_gold_contestable means the official answer "
                "collapsed a genuine semantic ambiguity."
            ),
            (
                "Use gold_already_articulated when official_answer already lists "
                "or accepts multiple values."
            ),
            (
                "Use inclusive_exclusive_arithmetic for ordinary inclusive versus "
                "exclusive interval/day counting, even when one value is official."
            ),
            (
                "Use data_conflict for conflicting evidence, and knowledge_update "
                "when latest-known-state convention resolves the conflict."
            ),
            (
                "At least one contestable reading must be compatible with the "
                "official_answer; otherwise mark insufficient_evidence or "
                "not_contestable and explain."
            ),
            (
                "Do not count missing retrieval, provider uncertainty, or a model "
                "mistake as ambiguity."
            ),
            (
                "Every reading must cite short exact pivot spans copied from the "
                "evidence sessions."
            ),
            "Prefer not_contestable when the alternative reading is weak.",
        ],
        "output_schema": {
            "case_id": str(row.get("question_id", "")),
            "verdict": "contestable|not_contestable|insufficient_evidence",
            "bucket": (
                "semantic_collapsed_gold_contestable|gold_already_articulated|"
                "inclusive_exclusive_arithmetic|data_conflict|knowledge_update|"
                "enumeration_variance|retrieval_context_gap|not_contestable|"
                "insufficient_evidence"
            ),
            "axis_type": (
                "category_boundary|set_scope|temporal_scope|"
                "multi_entity_collapse|multi_location_collapse|"
                "underspecification_inference|inclusive_exclusive_arithmetic|"
                "data_conflict|knowledge_update|enumeration_variance|"
                "retrieval_context_gap|not_ambiguous"
            ),
            "readings": [
                {
                    "reading_id": "stable string unique within this response",
                    "interpretation": "natural-language reading",
                    "answer_value": "answer under this reading as a string",
                    "compatible_with_official_answer": True,
                    "pivot_spans": [
                        {
                            "evidence_session_id": "one provided session_id",
                            "span": "short exact copied evidence span",
                            "why_pivot": (
                                "why this span supports or separates the reading"
                            ),
                        }
                    ],
                    "assumptions": ["explicit interpretive assumption"],
                    "defensibility": "strong|moderate|weak",
                }
            ],
            "contestability_reason": "one terse reason for the verdict",
        },
        "case": {
            "id": str(row.get("question_id", "")),
            "source_split": source_split,
            "question": str(row.get("question", "")),
            "question_type": str(row.get("question_type", "")),
            "question_date": str(row.get("question_date", "")),
            "official_answer": str(row.get("answer", "")),
            "evidence_sessions": evidence_sessions,
        },
    }


def build_provider_prompt(payload: dict[str, typing.Any]) -> str:
    return (
        "Complete this ambiguity-mining adjudication task.\n"
        "Return exactly one strict JSON object matching output_schema. "
        "No markdown, comments, or trailing text.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def run_payloads(
    *,
    payload_artifact: dict[str, typing.Any],
    provider_command: str = DEFAULT_PROVIDER_COMMAND,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    limit: int = 0,
    case_ids: set[str] | None = None,
    stream_outputs_path: str | pathlib.Path | None = None,
) -> list[dict[str, typing.Any]]:
    payloads = list(payload_artifact.get("payloads", []))
    if case_ids:
        payloads = [
            payload
            for payload in payloads
            if str(payload.get("case", {}).get("id", "")) in case_ids
        ]
    if limit > 0:
        payloads = payloads[:limit]
    rows: list[dict[str, typing.Any]] = []
    stream_handle: typing.TextIO | None = None
    if stream_outputs_path is not None:
        output_path = pathlib.Path(stream_outputs_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stream_handle = output_path.open("w", encoding="utf-8")
    try:
        for payload in payloads:
            case_id = str(payload.get("case", {}).get("id", ""))
            provider_result = interpretation_runner.run_provider(
                command=provider_command,
                prompt=build_provider_prompt(payload),
                timeout_seconds=timeout_seconds,
            )
            parsed = parse_adjudication_response(
                provider_result.raw_output,
                expected_case_id=case_id,
            )
            row = parsed.to_output_dict(
                provider=provider_command,
                raw_output=provider_result.raw_output,
            )
            row.update(
                {
                    "provider_exit_code": provider_result.exit_code,
                    "provider_stderr": provider_result.stderr,
                    "provider_timed_out": provider_result.timed_out,
                    "latency_seconds": round(provider_result.latency_seconds, 3),
                }
            )
            if provider_result.exit_code != 0:
                row["parse_errors"] = [
                    *row["parse_errors"],
                    f"provider exited with code {provider_result.exit_code}",
                ]
            rows.append(row)
            if stream_handle is not None:
                stream_handle.write(f"{json.dumps(row, sort_keys=True)}\n")
                stream_handle.flush()
    finally:
        if stream_handle is not None:
            stream_handle.close()
    return rows


def parse_adjudication_response(
    raw_output: str,
    *,
    expected_case_id: str | None = None,
) -> AdjudicationParseResult:
    if not raw_output.strip():
        return _invalid_result(
            case_id=expected_case_id or "",
            status=PARSE_STATUS_EMPTY,
            errors=("empty provider output",),
        )
    result_text = _provider_result_text(raw_output)
    decoded = _decode_first_json_object(result_text)
    if decoded is None:
        return _invalid_result(
            case_id=expected_case_id or "",
            status=PARSE_STATUS_INVALID_JSON,
            errors=("could not parse JSON object from provider output",),
            provider_result_text=result_text,
        )
    if not isinstance(decoded, dict):
        return _invalid_result(
            case_id=expected_case_id or "",
            status=PARSE_STATUS_INVALID_SCHEMA,
            errors=("root output must be a JSON object",),
            provider_result_text=result_text,
        )
    return parse_adjudication_object(
        decoded,
        expected_case_id=expected_case_id,
        provider_result_text=result_text,
    )


def parse_adjudication_object(
    raw: dict[str, typing.Any],
    *,
    expected_case_id: str | None = None,
    provider_result_text: str = "",
) -> AdjudicationParseResult:
    errors: list[str] = []
    fallback_case_id = expected_case_id or ""
    case_id = _string_field(raw, "case_id", errors)
    if expected_case_id is not None and case_id and case_id != expected_case_id:
        errors.append(
            f"case_id {case_id!r} does not match expected {expected_case_id!r}"
        )
    verdict = _string_field(raw, "verdict", errors)
    if verdict and verdict not in ALLOWED_VERDICTS:
        errors.append(f"unknown verdict {verdict!r}")
    bucket = _string_field(raw, "bucket", errors)
    if bucket and bucket not in ALLOWED_BUCKETS:
        errors.append(f"unknown bucket {bucket!r}")
    axis_type = _string_field(raw, "axis_type", errors)
    if axis_type and axis_type not in ALLOWED_AXIS_TYPES:
        errors.append(f"unknown axis_type {axis_type!r}")
    readings = _readings(raw.get("readings"), errors)
    _validate_reading_ids(readings, errors)
    contestability_reason = _string_field(raw, "contestability_reason", errors)
    if errors:
        return _invalid_result(
            case_id=case_id or fallback_case_id,
            status=PARSE_STATUS_INVALID_SCHEMA,
            errors=tuple(errors),
            provider_result_text=provider_result_text,
        )
    return AdjudicationParseResult(
        case_id=case_id,
        parse_status=PARSE_STATUS_PARSED,
        verdict=verdict,
        bucket=bucket,
        axis_type=axis_type,
        readings=tuple(readings),
        contestability_reason=contestability_reason,
        parse_errors=(),
        provider_result_text=provider_result_text,
    )


def build_report(
    *,
    rows: list[dict[str, typing.Any]],
    payload_artifact: dict[str, typing.Any],
    outputs_path: str | pathlib.Path = DEFAULT_OUTPUTS_PATH,
    payloads_path: str | pathlib.Path = DEFAULT_PAYLOADS_PATH,
    go_min_cases: int = DEFAULT_GO_MIN_CASES,
    go_min_rate: float = DEFAULT_GO_MIN_RATE,
) -> dict[str, typing.Any]:
    payload_by_id = {
        str(payload.get("case", {}).get("id", "")): payload
        for payload in payload_artifact.get("payloads", [])
        if isinstance(payload, dict)
    }
    expected_case_ids = sorted(payload_by_id)
    reviews = [
        review_adjudication_row(row, payload_by_id.get(str(row.get("case_id", ""))))
        for row in rows
    ]
    grouped = _group_reviews(reviews, expected_case_ids)
    provider_failed_rows = [review for review in reviews if review["provider_failed"]]
    parse_status_parsed_rows = [
        review for review in reviews if review["parse_status"] == PARSE_STATUS_PARSED
    ]
    accepted_rows = [
        review
        for review in parse_status_parsed_rows
        if review["parse_status"] == PARSE_STATUS_PARSED
        and not review["provider_failed"]
    ]
    semantic_count = sum(
        1 for review in reviews if review["semantic_collapsed_gold_contestable"]
    )
    denominator = len(accepted_rows)
    semantic_rate = semantic_count / denominator if denominator else 0.0
    go = semantic_count >= go_min_cases and semantic_rate >= go_min_rate
    total_latency = sum(float(row.get("latency_seconds", 0.0) or 0.0) for row in rows)
    return {
        "name": "longmemeval-s-ambiguity-mining-report",
        "artifact_kind": "ambiguity_mining_report",
        "prompt_version": PROMPT_VERSION,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "provider": str(rows[0].get("provider", "")) if rows else "",
        "source_payload_artifact": str(payloads_path),
        "source_outputs_artifact": str(outputs_path),
        "rows_total": len(rows),
        "rows_expected": len(expected_case_ids),
        "rows_parse_status_parsed": len(parse_status_parsed_rows),
        "rows_parsed": len(accepted_rows),
        "rows_failed_parse": len(rows) - len(parse_status_parsed_rows),
        "rows_provider_failed": len(provider_failed_rows),
        "rows_provider_timed_out": sum(
            1 for row in rows if bool(row.get("provider_timed_out", False))
        ),
        "verdict_counts": dict(
            collections.Counter(review["verdict"] for review in reviews)
        ),
        "axis_type_counts": dict(
            collections.Counter(review["axis_type"] for review in reviews)
        ),
        "bucket_counts": dict(
            collections.Counter(review["bucket"] for review in reviews)
        ),
        "raw_contestable_count": sum(
            1 for review in reviews if review["verdict"] == "contestable"
        ),
        "semantic_collapsed_gold_contestable_count": semantic_count,
        "semantic_collapsed_gold_contestable_rate": round(semantic_rate, 4),
        "confirmed_contestable_gold_count": semantic_count,
        "confirmed_contestable_gold_rate": round(semantic_rate, 4),
        "gold_already_articulated_count": sum(
            1
            for review in reviews
            if review["bucket"] == GOLD_ALREADY_ARTICULATED_BUCKET
        ),
        "inclusive_exclusive_arithmetic_count": sum(
            1
            for review in reviews
            if review["bucket"] == INCLUSIVE_EXCLUSIVE_BUCKET
            or review["axis_type"] == "inclusive_exclusive_arithmetic"
        ),
        "excluded_non_thesis_count": sum(
            1
            for review in reviews
            if review["bucket"]
            in {
                "data_conflict",
                "knowledge_update",
                "enumeration_variance",
                "retrieval_context_gap",
            }
        ),
        "split_summary": _split_summary(reviews),
        "stratum_summary": _stratum_summary(reviews),
        "go_no_go": {
            "decision": "go" if go else "no_go",
            "min_cases": go_min_cases,
            "min_rate": go_min_rate,
            "reason": (
                "semantic collapsed-gold contestable population found"
                if go
                else "semantic collapsed-gold contestable population below threshold"
            ),
        },
        "provider_cost_or_latency_if_available": {
            "total_latency_seconds": round(total_latency, 3),
            "average_latency_seconds": round(total_latency / len(rows), 3)
            if rows
            else 0.0,
        },
        "acceptance": {
            "outputs_cover_expected_cases": len(rows) == len(expected_case_ids),
            "raw_provider_output_retained": all("raw_output" in row for row in rows),
            "system_outputs_hidden_from_provider_payload": (
                _payloads_hide_system_outputs(payload_artifact)
            ),
            "official_answer_visible_to_adjudicator": bool(
                payload_artifact.get("provider_visibility", {}).get(
                    "official_answer_visible"
                )
            ),
        },
        "cases": grouped,
        "row_reviews": reviews,
    }


def review_adjudication_row(
    row: dict[str, typing.Any],
    payload: dict[str, typing.Any] | None,
) -> dict[str, typing.Any]:
    issues: list[str] = []
    provider_failed = _provider_failed(row)
    if provider_failed:
        issues.append("provider_failed")
    parse_status = str(row.get("parse_status", ""))
    if parse_status != PARSE_STATUS_PARSED:
        issues.append("parse_failed")
    readings = [item for item in row.get("readings", []) if isinstance(item, dict)]
    span_results = _span_results(readings, payload)
    readings_with_resolved_pivot = {
        str(item["reading_id"]) for item in span_results if bool(item["span_resolves"])
    }
    answer_values = {
        _normalize_answer_value(str(reading.get("answer_value", "")))
        for reading in readings
        if str(reading.get("answer_value", "")).strip()
    }
    answer_values.discard("")
    official_compatible_count = sum(
        1
        for reading in readings
        if bool(reading.get("compatible_with_official_answer"))
    )
    official_incompatible_count = len(readings) - official_compatible_count
    all_readings_have_resolved_pivot = all(
        str(reading.get("reading_id", "")) in readings_with_resolved_pivot
        for reading in readings
    )
    if readings and not all_readings_have_resolved_pivot:
        issues.append("reading_pivot_span_unresolved")
    inferred_axis_type = _classify_axis_type(row, readings)
    inferred_bucket = _classify_bucket(
        row=row,
        axis_type=inferred_axis_type,
        readings=readings,
        official_compatible_count=official_compatible_count,
        official_incompatible_count=official_incompatible_count,
    )
    semantic_confirmed = (
        not provider_failed
        and parse_status == PARSE_STATUS_PARSED
        and str(row.get("verdict", "")) == "contestable"
        and inferred_bucket == SEMANTIC_BUCKET
        and inferred_axis_type in THESIS_AXIS_TYPES
        and len(readings) >= 2
        and len(answer_values) >= 2
        and official_compatible_count >= 1
        and official_incompatible_count >= 1
        and all_readings_have_resolved_pivot
    )
    if str(row.get("verdict", "")) == "contestable" and not semantic_confirmed:
        issues.append("contestable_verdict_failed_confirmation_gate")
    return {
        "case_id": str(row.get("case_id", "")),
        "source_split": str((payload or {}).get("case", {}).get("source_split", "")),
        "question": str((payload or {}).get("case", {}).get("question", "")),
        "official_answer": str(
            (payload or {}).get("case", {}).get("official_answer", "")
        ),
        "parse_status": parse_status,
        "provider_failed": provider_failed,
        "verdict": str(row.get("verdict", "")),
        "bucket": inferred_bucket,
        "axis_type": inferred_axis_type,
        "reading_count": len(readings),
        "distinct_answer_value_count": len(answer_values),
        "official_compatible_reading_count": official_compatible_count,
        "official_incompatible_reading_count": official_incompatible_count,
        "all_readings_have_resolved_pivot": all_readings_have_resolved_pivot,
        "semantic_collapsed_gold_contestable": semantic_confirmed,
        "confirmed_contestable_gold": semantic_confirmed,
        "contestability_reason": str(row.get("contestability_reason", "")),
        "span_results": span_results,
        "quality_issues": sorted(set(issues)),
    }


def write_jsonl(
    path: str | pathlib.Path,
    rows: typing.Iterable[dict[str, typing.Any]],
) -> None:
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{json.dumps(row, sort_keys=True)}\n")


def load_jsonl(path: str | pathlib.Path) -> list[dict[str, typing.Any]]:
    rows: list[dict[str, typing.Any]] = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-payloads", action="store_true")
    parser.add_argument("--run-provider", action="store_true")
    parser.add_argument("--build-report", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--heldout-manifest",
        type=pathlib.Path,
        default=DEFAULT_HELDOUT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--fail18-manifest",
        type=pathlib.Path,
        default=ambiguity_fail18.DEFAULT_MANIFEST,
    )
    parser.add_argument("--payloads", type=pathlib.Path, default=DEFAULT_PAYLOADS_PATH)
    parser.add_argument("--outputs", type=pathlib.Path, default=DEFAULT_OUTPUTS_PATH)
    parser.add_argument("--report", type=pathlib.Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--wider-limit", type=int, default=DEFAULT_WIDER_LIMIT)
    parser.add_argument(
        "--session-char-limit",
        type=int,
        default=DEFAULT_SESSION_CHAR_LIMIT,
    )
    parser.add_argument(
        "--max-answer-sessions",
        type=int,
        default=DEFAULT_MAX_ANSWER_SESSIONS,
    )
    parser.add_argument("--provider", default=DEFAULT_PROVIDER_COMMAND)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--case-ids",
        default="",
        help="Comma-separated case ids to run, for targeted retries.",
    )
    parser.add_argument("--go-min-cases", type=int, default=DEFAULT_GO_MIN_CASES)
    parser.add_argument("--go-min-rate", type=float, default=DEFAULT_GO_MIN_RATE)
    args = parser.parse_args(argv)
    if not any(
        (args.build_payloads, args.run_provider, args.build_report, args.parse_only)
    ):
        parser.print_help()
        return 2

    payload_artifact: dict[str, typing.Any] | None = None
    if args.build_payloads:
        payload_artifact = build_payload_artifact(
            dataset_path=args.dataset,
            heldout_manifest_path=args.heldout_manifest,
            fail18_manifest_path=args.fail18_manifest,
            wider_limit=args.wider_limit,
            session_char_limit=args.session_char_limit,
            max_answer_sessions=args.max_answer_sessions,
        )
        _write_json(args.payloads, payload_artifact)

    if args.run_provider:
        payload_artifact = payload_artifact or _load_json(args.payloads)
        rows = run_payloads(
            payload_artifact=payload_artifact,
            provider_command=args.provider,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
            case_ids=_case_ids_from_arg(args.case_ids),
            stream_outputs_path=args.outputs,
        )

    if args.build_report:
        payload_artifact = payload_artifact or _load_json(args.payloads)
        rows = load_jsonl(args.outputs)
        report = build_report(
            rows=rows,
            payload_artifact=payload_artifact,
            outputs_path=args.outputs,
            payloads_path=args.payloads,
            go_min_cases=args.go_min_cases,
            go_min_rate=args.go_min_rate,
        )
        _write_json(args.report, report)

    if args.parse_only and not args.build_report:
        payload_artifact = payload_artifact or _load_json(args.payloads)
        rows = load_jsonl(args.outputs)
        report = build_report(
            rows=rows,
            payload_artifact=payload_artifact,
            outputs_path=args.outputs,
            payloads_path=args.payloads,
            go_min_cases=args.go_min_cases,
            go_min_rate=args.go_min_rate,
        )
        _write_json(args.report, report)

    return 0


def _case_ids_from_arg(raw: str) -> set[str] | None:
    case_ids = {item.strip() for item in raw.split(",") if item.strip()}
    return case_ids or None


def _readings(raw_readings: typing.Any, errors: list[str]) -> list[CandidateReading]:
    if not isinstance(raw_readings, list):
        errors.append("readings must be a list")
        return []
    readings: list[CandidateReading] = []
    for index, raw in enumerate(raw_readings):
        if not isinstance(raw, dict):
            errors.append(f"readings[{index}] must be a JSON object")
            continue
        reading_errors: list[str] = []
        reading_id = _string_field(raw, "reading_id", reading_errors)
        interpretation = _string_field(raw, "interpretation", reading_errors)
        answer_value = _string_field(raw, "answer_value", reading_errors)
        compatible = _bool_field(
            raw,
            "compatible_with_official_answer",
            reading_errors,
        )
        pivot_spans = _pivot_spans(raw.get("pivot_spans"), reading_errors)
        assumptions = _string_list_field(raw, "assumptions", reading_errors)
        defensibility = _string_field(raw, "defensibility", reading_errors)
        if defensibility and defensibility not in ALLOWED_DEFENSIBILITY:
            reading_errors.append(f"unknown defensibility {defensibility!r}")
        if reading_errors:
            errors.extend(f"readings[{index}]: {error}" for error in reading_errors)
            continue
        readings.append(
            CandidateReading(
                reading_id=reading_id,
                interpretation=interpretation,
                answer_value=answer_value,
                compatible_with_official_answer=compatible,
                pivot_spans=tuple(pivot_spans),
                assumptions=tuple(assumptions),
                defensibility=defensibility,
            )
        )
    return readings


def _pivot_spans(raw_spans: typing.Any, errors: list[str]) -> list[PivotSpan]:
    if not isinstance(raw_spans, list) or not raw_spans:
        errors.append("pivot_spans must be a non-empty list")
        return []
    spans: list[PivotSpan] = []
    for index, raw in enumerate(raw_spans):
        if not isinstance(raw, dict):
            errors.append(f"pivot_spans[{index}] must be a JSON object")
            continue
        span_errors: list[str] = []
        evidence_session_id = _string_field(raw, "evidence_session_id", span_errors)
        span = _string_field(raw, "span", span_errors)
        why_pivot = _string_field(raw, "why_pivot", span_errors)
        if span_errors:
            errors.extend(f"pivot_spans[{index}]: {error}" for error in span_errors)
            continue
        spans.append(
            PivotSpan(
                evidence_session_id=evidence_session_id,
                span=span,
                why_pivot=why_pivot,
            )
        )
    return spans


def _span_results(
    readings: list[dict[str, typing.Any]],
    payload: dict[str, typing.Any] | None,
) -> list[dict[str, typing.Any]]:
    sessions = {
        str(session.get("session_id", "")): str(session.get("text", ""))
        for session in (payload or {}).get("case", {}).get("evidence_sessions", [])
        if isinstance(session, dict)
    }
    results: list[dict[str, typing.Any]] = []
    for reading in readings:
        reading_id = str(reading.get("reading_id", ""))
        for span in reading.get("pivot_spans", []):
            if not isinstance(span, dict):
                continue
            session_id = str(span.get("evidence_session_id", ""))
            text = sessions.get(session_id, "")
            evidence_span = str(span.get("span", ""))
            results.append(
                {
                    "reading_id": reading_id,
                    "evidence_session_id": session_id,
                    "session_exists": session_id in sessions,
                    "span": evidence_span,
                    "span_resolves": bool(text)
                    and _normalized_contains(text, evidence_span),
                }
            )
    return results


def _group_reviews(
    reviews: list[dict[str, typing.Any]],
    expected_case_ids: list[str],
) -> list[dict[str, typing.Any]]:
    by_case: dict[str, dict[str, typing.Any] | None] = {
        case_id: None for case_id in expected_case_ids
    }
    for review in reviews:
        by_case[str(review.get("case_id", ""))] = review
    grouped = []
    for case_id in sorted(by_case):
        review = by_case[case_id]
        if review is None:
            grouped.append(
                {
                    "case_id": case_id,
                    "missing_output": True,
                    "confirmed_contestable_gold": False,
                    "quality_issues": ["missing_output"],
                }
            )
        else:
            grouped.append(review)
    return grouped


def _split_summary(reviews: list[dict[str, typing.Any]]) -> dict[str, dict[str, int]]:
    by_split: dict[str, collections.Counter[str]] = {}
    for review in reviews:
        split = str(review.get("source_split", "")) or "unknown"
        counter = by_split.setdefault(split, collections.Counter())
        counter["rows"] += 1
        if review.get("semantic_collapsed_gold_contestable"):
            counter["semantic_collapsed_gold_contestable"] += 1
        counter[f"verdict_{review.get('verdict', '')}"] += 1
        counter[f"bucket_{review.get('bucket', '')}"] += 1
    return {split: dict(counter) for split, counter in sorted(by_split.items())}


def _stratum_summary(reviews: list[dict[str, typing.Any]]) -> dict[str, dict[str, int]]:
    strata = {
        "all_scanned_rows": reviews,
        "count_how_many_rows": [
            review for review in reviews if _is_count_how_many(review)
        ],
        "temporal_duration_rows": [
            review for review in reviews if _is_temporal_duration(review)
        ],
        "wider_non_count_rows": [
            review
            for review in reviews
            if review.get("source_split") == "wider_lme_s"
            and not _is_count_how_many(review)
        ],
    }
    summary: dict[str, dict[str, int]] = {}
    for name, rows in strata.items():
        counter: collections.Counter[str] = collections.Counter()
        counter["rows"] = len(rows)
        counter["semantic_collapsed_gold_contestable"] = sum(
            1 for row in rows if row.get("semantic_collapsed_gold_contestable")
        )
        counter["raw_contestable"] = sum(
            1 for row in rows if row.get("verdict") == "contestable"
        )
        for row in rows:
            counter[f"bucket_{row.get('bucket', '')}"] += 1
        summary[name] = dict(counter)
    return summary


def _classify_axis_type(
    row: dict[str, typing.Any],
    readings: list[dict[str, typing.Any]],
) -> str:
    axis_type = str(row.get("axis_type", "")).strip()
    if axis_type in ALLOWED_AXIS_TYPES:
        return axis_type
    text = _classification_text(row, readings)
    if "inclusive" in text and "exclusive" in text:
        return "inclusive_exclusive_arithmetic"
    if "latest" in text or "later self-report" in text or "latest wins" in text:
        return "knowledge_update"
    if "conflict" in text or "conflicting" in text or "self-reported" in text:
        return "data_conflict"
    if "missed" in text or "enumeration" in text:
        return "enumeration_variance"
    if "retrieval" in text or "truncated" in text or "context" in text:
        return "retrieval_context_gap"
    if "count as" in text or "counts as" in text or "qualifies as" in text:
        return "category_boundary"
    if "faith-related" in text or "category" in text:
        return "category_boundary"
    if "full set" in text or "singular" in text or "set" in text:
        return "set_scope"
    if "location" in text or "where" in text or "studio" in text:
        return "multi_location_collapse"
    if "specific" in text or "inferred" in text or "underspecified" in text:
        return "underspecification_inference"
    if str(row.get("verdict", "")) == "contestable":
        return "underspecification_inference"
    return "not_ambiguous"


def _classify_bucket(
    *,
    row: dict[str, typing.Any],
    axis_type: str,
    readings: list[dict[str, typing.Any]],
    official_compatible_count: int,
    official_incompatible_count: int,
) -> str:
    bucket = str(row.get("bucket", "")).strip()
    if bucket in ALLOWED_BUCKETS:
        return bucket
    verdict = str(row.get("verdict", ""))
    if verdict == "insufficient_evidence":
        return "insufficient_evidence"
    if verdict != "contestable":
        return "not_contestable"
    if official_incompatible_count == 0 and len(readings) >= 2:
        return GOLD_ALREADY_ARTICULATED_BUCKET
    if axis_type == "inclusive_exclusive_arithmetic":
        return INCLUSIVE_EXCLUSIVE_BUCKET
    if axis_type in {
        "data_conflict",
        "knowledge_update",
        "enumeration_variance",
        "retrieval_context_gap",
    }:
        return axis_type
    if (
        official_compatible_count >= 1
        and official_incompatible_count >= 1
        and axis_type in THESIS_AXIS_TYPES
    ):
        return SEMANTIC_BUCKET
    return "not_contestable"


def _classification_text(
    row: dict[str, typing.Any],
    readings: list[dict[str, typing.Any]],
) -> str:
    parts = [str(row.get("contestability_reason", ""))]
    for reading in readings:
        parts.append(str(reading.get("interpretation", "")))
        parts.append(" ".join(str(item) for item in reading.get("assumptions", [])))
    return " ".join(parts).casefold()


def _is_count_how_many(review: dict[str, typing.Any]) -> bool:
    question = str(review.get("question", "")).casefold()
    return question.startswith("how many") or " how many " in question


def _is_temporal_duration(review: dict[str, typing.Any]) -> bool:
    question = str(review.get("question", "")).casefold()
    temporal_markers = (
        "how long",
        "how many days",
        "how many months",
        "how many years",
        "how many hours",
        "days ago",
        "months ago",
        "years ago",
        "passed between",
        "when did",
    )
    return any(marker in question for marker in temporal_markers)


def _provider_failed(row: dict[str, typing.Any]) -> bool:
    return (
        bool(row.get("provider_timed_out", False))
        or int(row.get("provider_exit_code", 0) or 0) != 0
    )


def _payloads_hide_system_outputs(payload_artifact: dict[str, typing.Any]) -> bool:
    rendered = json.dumps(payload_artifact.get("payloads", [])).casefold()
    forbidden = (
        "raw_output",
        "provider_answer_number",
        "recomputed_answer",
        "gold_in_answer_support",
        "witness_outputs",
        "witness output",
        "answer_unit_witness",
        "system answer",
    )
    return not any(term in rendered for term in forbidden)


def _session_text(
    messages: list[dict[str, typing.Any]],
    *,
    char_limit: int,
) -> str:
    chunks = [
        f"{str(message.get('role', '')).upper()}: {message.get('content', '')!s}"
        for message in messages
        if isinstance(message, dict)
    ]
    text = "\n".join(chunks)
    if len(text) > char_limit:
        return f"{text[:char_limit]}\n...[truncated]"
    return text


def _normalized_contains(text: str, span: str) -> bool:
    if not span.strip():
        return False
    return _normalize_for_span(span) in _normalize_for_span(text)


def _normalize_for_span(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold().strip()


def _normalize_answer_value(value: str) -> str:
    normalized = value.casefold().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[,]", "", normalized)
    return normalized


def _provider_result_text(raw_output: str) -> str:
    decoded = _decode_json(raw_output)
    if isinstance(decoded, dict) and isinstance(decoded.get("result"), str):
        return str(decoded["result"])
    return raw_output


def _decode_first_json_object(text: str) -> typing.Any:
    stripped = text.strip()
    if not stripped:
        return None
    decoded = _decode_json(stripped)
    if decoded is not None:
        return decoded
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S | re.I)
    if fence:
        decoded = _decode_json(fence.group(1))
        if decoded is not None:
            return decoded
    brace = re.search(r"\{.*\}", stripped, flags=re.S)
    if brace:
        return _decode_json(brace.group(0))
    return None


def _decode_json(text: str) -> typing.Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _invalid_result(
    *,
    case_id: str,
    status: str,
    errors: tuple[str, ...],
    provider_result_text: str = "",
) -> AdjudicationParseResult:
    return AdjudicationParseResult(
        case_id=case_id,
        parse_status=status,
        verdict="",
        bucket="",
        axis_type="",
        readings=(),
        contestability_reason="",
        parse_errors=errors,
        provider_result_text=provider_result_text,
    )


def _validate_reading_ids(
    readings: list[CandidateReading],
    errors: list[str],
) -> None:
    counts = collections.Counter(reading.reading_id for reading in readings)
    duplicate_ids = sorted(
        reading_id for reading_id, count in counts.items() if count > 1
    )
    if duplicate_ids:
        errors.append("duplicate reading_id values: " + ", ".join(duplicate_ids))


def _string_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty string")
        return ""
    return value.strip()


def _bool_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        errors.append(f"{key} must be a boolean")
        return False
    return value


def _string_list_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list):
        errors.append(f"{key} must be a list")
        return []
    if not all(isinstance(item, str) for item in value):
        errors.append(f"{key} must contain only strings")
        return []
    return [item.strip() for item in value if item.strip()]


def _load_dataset_rows(path: str | pathlib.Path) -> list[dict[str, typing.Any]]:
    rows = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("LongMemEval-S dataset must be a JSON list")
    return [row for row in rows if isinstance(row, dict)]


def _heldout_ids(path: str | pathlib.Path) -> list[str]:
    rows = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return []
    return [
        str(row.get("question_id", ""))
        for row in rows
        if isinstance(row, dict) and row.get("question_id")
    ]


def _fail18_ids(path: str | pathlib.Path) -> set[str]:
    path = pathlib.Path(path)
    if not path.exists():
        return set()
    return {
        str(row.get("question_id", ""))
        for row in ambiguity_fail18.load_manifest(path)
        if isinstance(row, dict) and row.get("question_id")
    }


def _row_has_answer_sessions(row: dict[str, typing.Any]) -> bool:
    return bool(row.get("answer_session_ids")) and bool(row.get("haystack_sessions"))


def _split(payload: dict[str, typing.Any]) -> str:
    return str(payload.get("case", {}).get("source_split", ""))


def _load_json(path: str | pathlib.Path) -> dict[str, typing.Any]:
    return typing.cast(
        "dict[str, typing.Any]",
        json.loads(pathlib.Path(path).read_text(encoding="utf-8")),
    )


def _write_json(path: str | pathlib.Path, artifact: typing.Any) -> None:
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"{json.dumps(artifact, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
