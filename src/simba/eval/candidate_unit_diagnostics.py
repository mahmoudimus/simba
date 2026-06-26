"""Diagnostics for live candidate-unit compiler failures."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import typing

from simba.eval import ambiguity_fail18, interpretation_runner

DEFAULT_CANDIDATE_UNIT_REPORT_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_report.json"
)
DEFAULT_CANDIDATE_UNIT_OUTPUTS_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_outputs.jsonl"
)
DEFAULT_PAYLOAD_PROVENANCE_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_payloads_provenance.json"
)
DEFAULT_DIAGNOSTICS_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_diagnostics.json"
)


def build_candidate_unit_diagnostics(
    *,
    report_path: str | pathlib.Path = DEFAULT_CANDIDATE_UNIT_REPORT_PATH,
    outputs_path: str | pathlib.Path = DEFAULT_CANDIDATE_UNIT_OUTPUTS_PATH,
    payload_provenance_path: str | pathlib.Path = DEFAULT_PAYLOAD_PROVENANCE_PATH,
    manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
) -> dict[str, typing.Any]:
    """Explain candidate-unit failures using private benchmark provenance."""
    report = _load_json(report_path)
    rows = interpretation_runner.load_jsonl(outputs_path)
    rows_by_id = {str(row.get("case_id", "")): row for row in rows}
    provenance = _load_json(payload_provenance_path).get("evidence_provenance", {})
    manifest_by_id = {
        str(row.get("question_id", "")): row
        for row in ambiguity_fail18.load_manifest(manifest_path)
    }
    target_report_cases = [
        case
        for case in report.get("cases", [])
        if isinstance(case, dict)
        and (
            bool(case.get("quality_issues"))
            or not bool(case.get("useful_for_candidate_compilation", False))
        )
    ]
    cases = [
        diagnose_case(
            report_case=case,
            output_row=rows_by_id.get(str(case.get("case_id", "")), {}),
            case_provenance=typing.cast(
                "dict[str, dict[str, typing.Any]]",
                provenance.get(str(case.get("case_id", "")), {}),
            ),
            manifest_row=manifest_by_id.get(str(case.get("case_id", "")), {}),
        )
        for case in target_report_cases
    ]
    issue_counts: dict[str, int] = {}
    for case in cases:
        for issue in case["diagnostic_issues"]:
            issue_counts[str(issue)] = issue_counts.get(str(issue), 0) + 1
    return {
        "name": "fail18-candidate-unit-diagnostics",
        "artifact_kind": "candidate_unit_diagnostics",
        "gate": "candidate_unit_compiler",
        "gate_status": "candidate_unit_diagnostics_complete",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_candidate_unit_report": str(report_path),
        "source_candidate_unit_outputs": str(outputs_path),
        "source_payload_provenance": str(payload_provenance_path),
        "source_manifest": str(manifest_path),
        "summary": {
            "target_rows": len(cases),
            "issue_counts": dict(sorted(issue_counts.items())),
            "rows_with_missing_answer_sessions_from_payload": sum(
                1 for case in cases if case["missing_answer_session_ids_from_payload"]
            ),
            "rows_with_present_answer_session_without_included_unit": sum(
                1
                for case in cases
                if case["present_answer_session_ids_without_included_unit"]
            ),
            "rows_with_excluded_units_from_answer_sessions": sum(
                1 for case in cases if case["excluded_units_from_answer_sessions"]
            ),
            "rows_with_overmerge_risk": sum(
                1
                for case in cases
                if "overmerged_distinct_unit_possible"
                in case["diagnostic_issues"]
            ),
        },
        "cases": cases,
        "decision": {
            "next_slice": "candidate_unit_omission_verifier",
            "reason": (
                "The provider now emits parseable candidate units. The next "
                "gate is deterministic detection of missing answer-session "
                "coverage, weak exclusions, and over-aggressive merges."
            ),
        },
    }


def diagnose_case(
    *,
    report_case: dict[str, typing.Any],
    output_row: dict[str, typing.Any],
    case_provenance: dict[str, dict[str, typing.Any]],
    manifest_row: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    case_id = str(report_case.get("case_id", ""))
    answer_session_ids = tuple(
        str(item) for item in manifest_row.get("answer_session_ids", [])
    )
    answer_session_set = set(answer_session_ids)
    present_answer_evidence = _present_answer_evidence(
        case_provenance=case_provenance,
        answer_session_ids=answer_session_set,
    )
    present_answer_session_ids = {
        str(item["raw_session_id"]) for item in present_answer_evidence
    }
    units = [
        unit
        for unit in output_row.get("candidate_units", [])
        if isinstance(unit, dict)
    ]
    units_by_raw_session = _units_by_raw_session(
        units=units,
        case_provenance=case_provenance,
    )
    answer_sessions_with_any_unit = {
        raw_session_id
        for raw_session_id, raw_units in units_by_raw_session.items()
        if raw_session_id in answer_session_set and raw_units
    }
    answer_sessions_with_included_unit = {
        raw_session_id
        for raw_session_id, raw_units in units_by_raw_session.items()
        if raw_session_id in answer_session_set
        and any(str(unit.get("status", "")) == "included" for unit in raw_units)
    }
    missing_from_payload = tuple(
        session_id
        for session_id in answer_session_ids
        if session_id not in present_answer_session_ids
    )
    present_without_any_unit = tuple(
        session_id
        for session_id in answer_session_ids
        if session_id in present_answer_session_ids
        and session_id not in answer_sessions_with_any_unit
    )
    present_without_included_unit = tuple(
        session_id
        for session_id in answer_session_ids
        if session_id in present_answer_session_ids
        and session_id not in answer_sessions_with_included_unit
    )
    excluded_units_from_answer_sessions = _units_with_status_from_answer_sessions(
        status="excluded",
        units_by_raw_session=units_by_raw_session,
        answer_session_ids=answer_session_set,
    )
    merged_required_units = _units_with_status_from_answer_sessions(
        status="merged",
        units_by_raw_session=units_by_raw_session,
        answer_session_ids=answer_session_set,
    )
    included_units_from_non_answer_sessions = _included_units_from_non_answer_sessions(
        units=units,
        case_provenance=case_provenance,
        answer_session_ids=answer_session_set,
    )
    diagnostic_issues = _diagnostic_issues(
        report_case=report_case,
        missing_from_payload=missing_from_payload,
        present_without_any_unit=present_without_any_unit,
        present_without_included_unit=present_without_included_unit,
        excluded_units_from_answer_sessions=excluded_units_from_answer_sessions,
        merged_required_units=merged_required_units,
        included_units_from_non_answer_sessions=included_units_from_non_answer_sessions,
    )
    return {
        "case_id": case_id,
        "question": str(report_case.get("question", "")),
        "failure_mode": str(report_case.get("failure_mode", "")),
        "gold_value": report_case.get("gold_value"),
        "recomputed_answer": report_case.get("recomputed_answer"),
        "answer_session_ids": list(answer_session_ids),
        "present_answer_session_ids_in_payload": sorted(present_answer_session_ids),
        "missing_answer_session_ids_from_payload": list(missing_from_payload),
        "answer_session_ids_with_any_candidate_unit": sorted(
            answer_sessions_with_any_unit
        ),
        "answer_session_ids_with_included_candidate_unit": sorted(
            answer_sessions_with_included_unit
        ),
        "present_answer_session_ids_without_any_unit": list(
            present_without_any_unit
        ),
        "present_answer_session_ids_without_included_unit": list(
            present_without_included_unit
        ),
        "included_units_from_answer_sessions": _units_with_status_from_answer_sessions(
            status="included",
            units_by_raw_session=units_by_raw_session,
            answer_session_ids=answer_session_set,
        ),
        "excluded_units_from_answer_sessions": excluded_units_from_answer_sessions,
        "merged_required_units": merged_required_units,
        "included_units_from_non_answer_sessions": (
            included_units_from_non_answer_sessions
        ),
        "present_answer_evidence": present_answer_evidence,
        "diagnostic_issues": diagnostic_issues,
        "recommended_next_intervention": _recommended_next_intervention(
            diagnostic_issues
        ),
    }


def _present_answer_evidence(
    *,
    case_provenance: dict[str, dict[str, typing.Any]],
    answer_session_ids: set[str],
) -> list[dict[str, typing.Any]]:
    rows = []
    for evidence_id, meta in case_provenance.items():
        raw_session_id = str(meta.get("raw_session_id", ""))
        if raw_session_id not in answer_session_ids:
            continue
        rows.append(
            {
                "evidence_session_id": evidence_id,
                "raw_session_id": raw_session_id,
                "selection_rank": meta.get("selection_rank"),
                "selection_score": meta.get("selection_score"),
                "truncated": bool(meta.get("truncated", False)),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            int(row["selection_rank"])
            if isinstance(row.get("selection_rank"), int)
            else 9999
        ),
    )


def _units_by_raw_session(
    *,
    units: list[dict[str, typing.Any]],
    case_provenance: dict[str, dict[str, typing.Any]],
) -> dict[str, list[dict[str, typing.Any]]]:
    by_raw_session: dict[str, list[dict[str, typing.Any]]] = {}
    for unit in units:
        for evidence_id in unit.get("evidence_session_ids", []):
            if not isinstance(evidence_id, str):
                continue
            raw_session_id = str(
                case_provenance.get(evidence_id, {}).get("raw_session_id", "")
            )
            if not raw_session_id:
                continue
            by_raw_session.setdefault(raw_session_id, []).append(unit)
    return by_raw_session


def _units_with_status_from_answer_sessions(
    *,
    status: str,
    units_by_raw_session: dict[str, list[dict[str, typing.Any]]],
    answer_session_ids: set[str],
) -> list[dict[str, typing.Any]]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for raw_session_id in sorted(answer_session_ids):
        for unit in units_by_raw_session.get(raw_session_id, []):
            if str(unit.get("status", "")) != status:
                continue
            key = (raw_session_id, str(unit.get("unit_id", "")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "raw_session_id": raw_session_id,
                    "unit_id": str(unit.get("unit_id", "")),
                    "label": str(unit.get("label", "")),
                    "status": str(unit.get("status", "")),
                    "merge_target": unit.get("merge_target"),
                    "reason_code": str(unit.get("reason_code", "")),
                    "reason": str(unit.get("reason", "")),
                    "evidence_session_ids": list(unit.get("evidence_session_ids", [])),
                    "evidence_spans": list(unit.get("evidence_spans", [])),
                }
            )
    return rows


def _included_units_from_non_answer_sessions(
    *,
    units: list[dict[str, typing.Any]],
    case_provenance: dict[str, dict[str, typing.Any]],
    answer_session_ids: set[str],
) -> list[dict[str, typing.Any]]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for unit in units:
        if str(unit.get("status", "")) != "included":
            continue
        for evidence_id in unit.get("evidence_session_ids", []):
            if not isinstance(evidence_id, str):
                continue
            raw_session_id = str(
                case_provenance.get(evidence_id, {}).get("raw_session_id", "")
            )
            if not raw_session_id or raw_session_id in answer_session_ids:
                continue
            key = (raw_session_id, str(unit.get("unit_id", "")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "raw_session_id": raw_session_id,
                    "unit_id": str(unit.get("unit_id", "")),
                    "label": str(unit.get("label", "")),
                    "status": str(unit.get("status", "")),
                    "merge_target": unit.get("merge_target"),
                    "reason_code": str(unit.get("reason_code", "")),
                    "reason": str(unit.get("reason", "")),
                    "evidence_session_ids": list(unit.get("evidence_session_ids", [])),
                    "evidence_spans": list(unit.get("evidence_spans", [])),
                }
            )
    return rows


def _diagnostic_issues(
    *,
    report_case: dict[str, typing.Any],
    missing_from_payload: tuple[str, ...],
    present_without_any_unit: tuple[str, ...],
    present_without_included_unit: tuple[str, ...],
    excluded_units_from_answer_sessions: list[dict[str, typing.Any]],
    merged_required_units: list[dict[str, typing.Any]],
    included_units_from_non_answer_sessions: list[dict[str, typing.Any]],
) -> list[str]:
    issues: list[str] = []
    gold_value = _optional_float(report_case.get("gold_value"))
    recomputed_answer = _optional_float(report_case.get("recomputed_answer"))
    if (
        gold_value is not None
        and recomputed_answer is not None
        and recomputed_answer < gold_value
    ):
        issues.append("included_unit_under_count")
    if (
        gold_value is not None
        and recomputed_answer is not None
        and recomputed_answer > gold_value
    ):
        issues.append("included_unit_over_count")
    if missing_from_payload:
        issues.append("missing_answer_session_from_payload")
    if present_without_any_unit:
        issues.append("present_answer_session_without_candidate_unit")
    if present_without_included_unit:
        issues.append("present_answer_session_without_included_unit")
    if excluded_units_from_answer_sessions:
        issues.append("excluded_answer_session_unit")
    if merged_required_units and (
        gold_value is not None
        and recomputed_answer is not None
        and recomputed_answer < gold_value
    ):
        issues.append("overmerged_distinct_unit_possible")
    if included_units_from_non_answer_sessions and (
        gold_value is not None
        and recomputed_answer is not None
        and recomputed_answer > gold_value
    ):
        issues.append("included_non_answer_session_unit")
    return sorted(set(issues))


def _recommended_next_intervention(issues: list[str]) -> str:
    issue_set = set(issues)
    if "missing_answer_session_from_payload" in issue_set:
        return "retrieval_or_payload_budget_fix"
    if "present_answer_session_without_candidate_unit" in issue_set:
        return "evidence_to_unit_coverage_check"
    if "excluded_answer_session_unit" in issue_set:
        return "exclusion_reason_verifier"
    if "included_non_answer_session_unit" in issue_set:
        return "inclusion_scope_verifier"
    if "overmerged_distinct_unit_possible" in issue_set:
        return "merge_defensibility_verifier"
    if "included_unit_under_count" in issue_set:
        return "answer_unit_enumeration_prompt"
    if "included_unit_over_count" in issue_set:
        return "aggregation_scope_check"
    return "none"


def _optional_float(value: typing.Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _load_json(path: str | pathlib.Path) -> dict[str, typing.Any]:
    return typing.cast(
        "dict[str, typing.Any]",
        json.loads(pathlib.Path(path).read_text(encoding="utf-8")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        default=DEFAULT_CANDIDATE_UNIT_REPORT_PATH,
    )
    parser.add_argument(
        "--outputs",
        type=pathlib.Path,
        default=DEFAULT_CANDIDATE_UNIT_OUTPUTS_PATH,
    )
    parser.add_argument(
        "--payload-provenance",
        type=pathlib.Path,
        default=DEFAULT_PAYLOAD_PROVENANCE_PATH,
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=ambiguity_fail18.DEFAULT_MANIFEST,
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_DIAGNOSTICS_PATH)
    args = parser.parse_args(argv)

    artifact = build_candidate_unit_diagnostics(
        report_path=args.report,
        outputs_path=args.outputs,
        payload_provenance_path=args.payload_provenance,
        manifest_path=args.manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"{json.dumps(artifact, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
