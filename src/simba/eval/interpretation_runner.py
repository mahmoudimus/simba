"""Run and report ambiguous NLIDB interpretation provider outputs."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import pathlib
import re
import shlex
import subprocess
import time
import typing

from simba.eval import interpretation_parser
from simba.eval.interpretation_prompts import PROMPT_VERSION

DEFAULT_PAYLOADS_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_payloads.json"
)
DEFAULT_PROVENANCE_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_payloads_provenance.json"
)
DEFAULT_OUTPUTS_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_outputs.jsonl"
)
DEFAULT_REPORT_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_report.json"
)
DEFAULT_PROVIDER_COMMAND = (
    "claude -p --no-session-persistence --safe-mode --tools '' "
    "--system-prompt 'Return exactly one strict JSON object matching the "
    "requested schema. Do not include markdown or prose.'"
)
DEFAULT_TIMEOUT_SECONDS = 240

_PROMPT_NOUN_LEAKAGE_SEED_TERMS = (
    "zara",
    "boots",
    "blazer",
    "tame impala",
    "wedding",
    "hawaii",
    "new york",
)
_PROMPT_NOUN_LEAKAGE_STOPWORDS = {
    "about",
    "after",
    "all",
    "also",
    "and",
    "any",
    "are",
    "ambiguity",
    "before",
    "count",
    "did",
    "does",
    "for",
    "from",
    "have",
    "how",
    "into",
    "interpretation",
    "interpretations",
    "json",
    "many",
    "much",
    "natural",
    "need",
    "number",
    "object",
    "only",
    "question",
    "range",
    "return",
    "schema",
    "set",
    "should",
    "shape",
    "strict",
    "string",
    "sum",
    "the",
    "this",
    "total",
    "type",
    "types",
    "what",
    "when",
    "where",
    "which",
    "with",
}


@dataclasses.dataclass(frozen=True)
class ProviderResult:
    raw_output: str
    stderr: str
    exit_code: int
    latency_seconds: float
    timed_out: bool = False


def load_payload_artifact(path: str | pathlib.Path) -> dict[str, typing.Any]:
    return typing.cast(
        "dict[str, typing.Any]",
        json.loads(pathlib.Path(path).read_text(encoding="utf-8")),
    )


def load_jsonl(path: str | pathlib.Path) -> list[dict[str, typing.Any]]:
    rows: list[dict[str, typing.Any]] = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(
    path: str | pathlib.Path,
    rows: typing.Iterable[dict[str, typing.Any]],
) -> None:
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{json.dumps(row, sort_keys=True)}\n")


def build_provider_prompt(payload: dict[str, typing.Any]) -> str:
    return (
        "Complete this ambiguous NLIDB interpretation-generation task.\n"
        "Return exactly one strict JSON object matching output_schema. "
        "No markdown, prose, comments, or trailing text.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def run_provider(
    *,
    command: str,
    prompt: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> ProviderResult:
    argv = shlex.split(command)
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            input=prompt,
            capture_output=True,
            check=False,
            encoding="utf-8",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        latency = time.perf_counter() - start
        return ProviderResult(
            raw_output=_decode_timeout_value(exc.stdout),
            stderr=_decode_timeout_value(exc.stderr),
            exit_code=124,
            latency_seconds=latency,
            timed_out=True,
        )
    latency = time.perf_counter() - start
    return ProviderResult(
        raw_output=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
        latency_seconds=latency,
    )


def run_payloads(
    *,
    payload_artifact: dict[str, typing.Any],
    provider_command: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    limit: int = 0,
) -> list[dict[str, typing.Any]]:
    payloads = list(payload_artifact.get("payloads", []))
    if limit > 0:
        payloads = payloads[:limit]
    rows: list[dict[str, typing.Any]] = []
    prompt_version = str(payload_artifact.get("prompt_version", PROMPT_VERSION))
    for payload in payloads:
        case_id = str(payload.get("case", {}).get("id", ""))
        provider_result = run_provider(
            command=provider_command,
            prompt=build_provider_prompt(payload),
            timeout_seconds=timeout_seconds,
        )
        parsed = interpretation_parser.parse_interpretation_response(
            provider_result.raw_output,
            expected_case_id=case_id,
        )
        row = parsed.to_output_dict(
            provider=provider_command,
            prompt_version=prompt_version,
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
    return rows


def build_gate1_report(
    *,
    rows: list[dict[str, typing.Any]],
    payload_artifact: dict[str, typing.Any],
    outputs_path: str | pathlib.Path = DEFAULT_OUTPUTS_PATH,
    payloads_path: str | pathlib.Path = DEFAULT_PAYLOADS_PATH,
    provenance_path: str | pathlib.Path = DEFAULT_PROVENANCE_PATH,
) -> dict[str, typing.Any]:
    expected_case_ids = _expected_case_ids(payload_artifact)
    observed_case_ids = [str(row.get("case_id", "")) for row in rows]
    observed_counts = collections.Counter(observed_case_ids)
    missing_case_ids = sorted(set(expected_case_ids) - set(observed_case_ids))
    extra_case_ids = sorted(set(observed_case_ids) - set(expected_case_ids))
    duplicate_case_ids = sorted(
        case_id for case_id, count in observed_counts.items() if count > 1
    )

    parse_status_parsed_rows = [
        row
        for row in rows
        if row.get("parse_status") == interpretation_parser.PARSE_STATUS_PARSED
    ]
    provider_failed_rows = [row for row in rows if _provider_failed(row)]
    provider_success_rows = [
        row for row in rows if not _provider_failed(row)
    ]
    parsed_rows = [
        row
        for row in provider_success_rows
        if row.get("parse_status") == interpretation_parser.PARSE_STATUS_PARSED
    ]
    failed_rows = [
        row
        for row in rows
        if row.get("parse_status") != interpretation_parser.PARSE_STATUS_PARSED
        or _provider_failed(row)
    ]
    interpretation_counts = [
        len(row.get("interpretations", [])) for row in parsed_rows
    ]
    total_interpretations = sum(interpretation_counts)
    ambiguity_distribution: collections.Counter[str] = collections.Counter()
    shape_distribution: collections.Counter[str] = collections.Counter()
    for row in parsed_rows:
        for interpretation in row.get("interpretations", []):
            ambiguity_distribution.update(
                str(item) for item in interpretation.get("ambiguity_types", [])
            )
            shape = interpretation.get("expected_answer_shape")
            if shape:
                shape_distribution[str(shape)] += 1

    duplicate_count, duplicate_rows = _duplicate_interpretation_summary(
        parsed_rows
    )
    latencies = [
        float(row["latency_seconds"])
        for row in rows
        if isinstance(row.get("latency_seconds"), int | float)
    ]
    covers_exactly = (
        len(rows) == len(expected_case_ids)
        and not missing_case_ids
        and not extra_case_ids
        and not duplicate_case_ids
    )
    provider_rows_succeeded = not provider_failed_rows
    gate_status = (
        "slice1b_ready_for_review"
        if covers_exactly and provider_rows_succeeded
        else "slice1b_incomplete"
    )
    return {
        "name": "fail18-ambiguous-nlidb-gate1-report",
        "artifact_kind": "provider_output_report",
        "gate": "gate1",
        "gate_status": gate_status,
        "gate1_passed": False,
        "gate1_blocker": (
            "Gate 1 requires human review plus later verifier-backed "
            "performance evidence before candidate-unit compilation."
        ),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_payload_artifact": str(payloads_path),
        "source_provenance_artifact": str(provenance_path),
        "source_outputs_artifact": str(outputs_path),
        "prompt_version": str(payload_artifact.get("prompt_version", "")),
        "provider": _single_value(rows, "provider"),
        "rows_expected": len(expected_case_ids),
        "rows_total": len(rows),
        "rows_parse_status_parsed": len(parse_status_parsed_rows),
        "rows_parsed": len(parsed_rows),
        "rows_failed_parse": len(failed_rows),
        "rows_provider_succeeded": len(provider_success_rows),
        "rows_provider_failed": len(provider_failed_rows),
        "rows_provider_timed_out": sum(
            1 for row in provider_failed_rows if row.get("provider_timed_out")
        ),
        "provider_failure_case_ids": [
            str(row.get("case_id", "")) for row in provider_failed_rows
        ],
        "rows_with_zero_interpretations": sum(
            1 for count in interpretation_counts if count == 0
        ),
        "rows_with_one_interpretation": sum(
            1 for count in interpretation_counts if count == 1
        ),
        "rows_with_multiple_interpretations": sum(
            1 for count in interpretation_counts if count > 1
        ),
        "average_interpretations_per_row": (
            round(total_interpretations / len(rows), 3) if rows else 0.0
        ),
        "ambiguity_type_distribution": dict(sorted(ambiguity_distribution.items())),
        "expected_answer_shape_distribution": dict(
            sorted(shape_distribution.items())
        ),
        "duplicate_interpretation_count": duplicate_count,
        "duplicate_interpretation_rows": duplicate_rows,
        "case_coverage": {
            "covers_exactly_expected_cases": covers_exactly,
            "expected_case_ids": expected_case_ids,
            "observed_case_ids": observed_case_ids,
            "missing_case_ids": missing_case_ids,
            "extra_case_ids": extra_case_ids,
            "duplicate_case_ids": duplicate_case_ids,
        },
        "fail18_noun_leakage_check": _fail18_noun_leakage_check(
            payload_artifact
        ),
        "provider_cost_or_latency_if_available": {
            "cost": None,
            "latency_seconds_total": round(sum(latencies), 3),
            "latency_seconds_average": (
                round(sum(latencies) / len(latencies), 3) if latencies else None
            ),
        },
        "acceptance": {
            "outputs_cover_exactly_expected_cases": covers_exactly,
            "accepted_provider_outputs_cover_exactly_expected_cases": (
                covers_exactly and provider_rows_succeeded
            ),
            "outputs_cover_exactly_fail18_rows": (
                covers_exactly and len(expected_case_ids) == 18
            ),
            "accepted_provider_outputs_cover_exactly_fail18_rows": (
                covers_exactly
                and provider_rows_succeeded
                and len(expected_case_ids) == 18
            ),
            "provider_rows_succeeded": provider_rows_succeeded,
            "raw_provider_output_retained": all(
                isinstance(row.get("raw_output"), str) for row in rows
            ),
            "invalid_rows_reported": all(
                isinstance(row.get("parse_errors"), list) for row in rows
            ),
            "corpus_wide_run_performed": False,
            "candidate_unit_compilation_attempted": False,
            "review_required_before_candidate_unit_compilation": True,
        },
    }


def _decode_timeout_value(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _expected_case_ids(payload_artifact: dict[str, typing.Any]) -> list[str]:
    return [
        str(payload.get("case", {}).get("id", ""))
        for payload in payload_artifact.get("payloads", [])
    ]


def _single_value(rows: list[dict[str, typing.Any]], key: str) -> str:
    values = {str(row.get(key, "")) for row in rows if row.get(key)}
    if len(values) == 1:
        return next(iter(values))
    if not values:
        return ""
    return "mixed"


def _provider_failed(row: dict[str, typing.Any]) -> bool:
    try:
        exit_code = int(row.get("provider_exit_code", 0))
    except (TypeError, ValueError):
        exit_code = 1
    return exit_code != 0 or bool(row.get("provider_timed_out", False))


def _duplicate_interpretation_summary(
    rows: list[dict[str, typing.Any]],
) -> tuple[int, dict[str, list[str]]]:
    duplicate_count = 0
    duplicate_rows: dict[str, list[str]] = {}
    for row in rows:
        seen: dict[str, str] = {}
        case_duplicates: list[str] = []
        for interpretation in row.get("interpretations", []):
            key = _normalized_interpretation_key(interpretation)
            interpretation_id = str(interpretation.get("interpretation_id", ""))
            if key in seen:
                duplicate_count += 1
                case_duplicates.append(interpretation_id)
            else:
                seen[key] = interpretation_id
        if case_duplicates:
            duplicate_rows[str(row.get("case_id", ""))] = case_duplicates
    return duplicate_count, duplicate_rows


def _normalized_interpretation_key(
    interpretation: dict[str, typing.Any],
) -> str:
    text = str(interpretation.get("natural_language_interpretation", ""))
    normalized_text = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    shape = str(interpretation.get("expected_answer_shape", ""))
    ambiguity_types = ",".join(
        str(item) for item in interpretation.get("ambiguity_types", [])
    )
    return f"{normalized_text}|{shape}|{ambiguity_types}"


def _fail18_noun_leakage_check(
    payload_artifact: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    checked_values: list[str] = []
    for payload in payload_artifact.get("payloads", []):
        checked_values.append(str(payload.get("task", "")))
        checked_values.extend(
            str(item) for item in payload.get("generation_contract", [])
        )
        checked_values.append(json.dumps(payload.get("output_schema", {})))
    checked_text = "\n".join(checked_values).lower()
    checked_tokens = set(re.findall(r"[a-z0-9]+", checked_text))
    forbidden_terms = sorted(
        set(_PROMPT_NOUN_LEAKAGE_SEED_TERMS)
        | _payload_question_terms(payload_artifact)
    )
    found = [
        term
        for term in forbidden_terms
        if (
            term in checked_text
            if " " in term
            else term in checked_tokens
        )
    ]
    return {
        "check_kind": "derived_question_terms_against_prompt_contract",
        "confidence": (
            "Prompt-contract check only; evidence and generated "
            "interpretations are allowed to contain case nouns."
        ),
        "checked_fields": ["task", "generation_contract", "output_schema"],
        "forbidden_terms": forbidden_terms,
        "forbidden_terms_count": len(forbidden_terms),
        "found_terms": found,
        "passed": not found,
    }


def _payload_question_terms(
    payload_artifact: dict[str, typing.Any],
) -> set[str]:
    terms: set[str] = set()
    for payload in payload_artifact.get("payloads", []):
        question = str(payload.get("case", {}).get("question", ""))
        for token in re.findall(r"[a-z0-9]+", question.lower()):
            if (
                len(token) > 2
                and not token.isdigit()
                and token not in _PROMPT_NOUN_LEAKAGE_STOPWORDS
            ):
                terms.add(token)
    return terms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payloads", type=pathlib.Path, default=DEFAULT_PAYLOADS_PATH)
    parser.add_argument(
        "--provenance",
        type=pathlib.Path,
        default=DEFAULT_PROVENANCE_PATH,
    )
    parser.add_argument("--outputs", type=pathlib.Path, default=DEFAULT_OUTPUTS_PATH)
    parser.add_argument("--report", type=pathlib.Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER_COMMAND,
        help="Provider command that accepts the prompt on stdin.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-row provider timeout.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit payload rows.")
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Read an existing outputs JSONL and rebuild only the report.",
    )
    args = parser.parse_args(argv)

    payload_artifact = load_payload_artifact(args.payloads)
    if args.parse_only:
        rows = load_jsonl(args.outputs)
    else:
        rows = run_payloads(
            payload_artifact=payload_artifact,
            provider_command=args.provider,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
        )
        write_jsonl(args.outputs, rows)

    report = build_gate1_report(
        rows=rows,
        payload_artifact=payload_artifact,
        outputs_path=args.outputs,
        payloads_path=args.payloads,
        provenance_path=args.provenance,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
