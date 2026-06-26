"""Diagnostics and held-out guardrails for answer-unit witness evals."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import hashlib
import json
import pathlib
import re
import typing

from simba.eval import ambiguity_fail18, answer_unit_witness, interpretation_runner

DEFAULT_LONGMEMEVAL_S_PATH = pathlib.Path(".simba/benchmarks/longmemeval_s.json")
DEFAULT_HELDOUT_PAYLOADS_PATH = pathlib.Path(
    "_gitless/longmemeval_s_answer_unit_witness_heldout_payloads.json"
)
DEFAULT_HELDOUT_MANIFEST_PATH = pathlib.Path(
    "_gitless/longmemeval_s_answer_unit_witness_heldout_manifest.json"
)
DEFAULT_HELDOUT_PROVENANCE_PATH = pathlib.Path(
    "_gitless/longmemeval_s_answer_unit_witness_heldout_provenance.json"
)
DEFAULT_REASONING_MECHANISM_DIAGNOSTIC_PATH = pathlib.Path(
    "_gitless/fail18_answer_unit_witness_reasoning_mechanism_diagnostic.json"
)
DEFAULT_INCLUSION_POLICY_AB_REPORT_PATH = pathlib.Path(
    "_gitless/answer_unit_witness_inclusion_policy_v2_ab_report.json"
)
DEFAULT_PAYLOAD_BUDGET_AB_REPORT_PATH = pathlib.Path(
    "_gitless/answer_unit_witness_payload_budget_2500_vs_8k_ab_report.json"
)
DEFAULT_PAYLOAD_BUDGET_REGRESSION_DIAGNOSTIC_PATH = pathlib.Path(
    "_gitless/fail18_payload_budget_regression_diagnostic.json"
)
DEFAULT_FAIL18_GOLD_SPAN_NEEDLES_PATH = pathlib.Path(
    "_gitless/fail18_gold_span_needles.json"
)
DEFAULT_FAIL18_SPAN_SURVIVAL_REPORT_PATH = pathlib.Path(
    "_gitless/fail18_answer_unit_witness_span_survival_2500.json"
)
DEFAULT_STABLE_WRONG_DIAGNOSTIC_PATH = pathlib.Path(
    "_gitless/fail18_answer_unit_witness_stable_wrong_diagnostic.json"
)
DEFAULT_WITNESS_OUTPUTS_PATH = pathlib.Path(
    "_gitless/fail18_answer_unit_witness_outputs_opus_k3.jsonl"
)

DEFAULT_HELDOUT_SEED = "20260621"
DEFAULT_HELDOUT_LIMIT = 18
DEFAULT_HELDOUT_TOP_K = 12
DEFAULT_HELDOUT_SESSION_CHAR_LIMIT = 2500
INCLUSION_POLICY_PROMPT_VERSION = "answer_unit_witness_v2_inclusion_policy"
INCLUSION_POLICY_TARGET_CASE_IDS = (
    "0a995998",
    "3a704032",
    "9ee3ecd6",
    "edced276",
)
MEASUREMENT_DEFERRED_CASE_IDS = ("7024f17c",)

_INCLUSION_POLICY_CONTRACT = (
    (
        "Before deciding include/exclude, scan every evidence session and list "
        "all candidate units that plausibly satisfy the question; do not stop "
        "after the first matching unit."
    ),
    (
        "Use ordinary category membership and sortal bridges when enumerating: "
        "broad categories include common subtypes, and replacement/pickup/"
        "return obligations may be distinct units when the evidence says they "
        "are distinct pending actions."
    ),
    (
        "For acquired/current inventory questions, include items obtained by "
        "bought, got, received, gifted, adopted, or picked up unless the "
        "evidence clearly says planned, merely considered, borrowed, duplicate, "
        "or outside the requested time/status."
    ),
    (
        "For total duration or total amount questions, include every segment "
        "whose destination/activity/object matches the question, even if it was "
        "mentioned as background or as family-vs-solo context."
    ),
    (
        "For lookup questions with a stated target and current balance, compute "
        "the needed value as target minus current balance; otherwise use the "
        "explicit threshold/value asked for by the question."
    ),
)

_NUMERIC_ANSWER_RE = re.compile(
    r"\d|"
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|hundred)\b",
    re.IGNORECASE,
)
_ANSWER_QUESTION_PREFIXES = ("how many", "how much", "how long")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*")
_QUESTION_STOPWORDS = {
    "a",
    "about",
    "all",
    "am",
    "an",
    "and",
    "are",
    "at",
    "by",
    "did",
    "do",
    "for",
    "from",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "many",
    "much",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
}


@dataclasses.dataclass(frozen=True)
class RankedLmeSession:
    raw_session_index: int
    raw_session_id: str
    date: str
    selection_rank: int
    selection_score: int
    user_selection_score: int
    assistant_selection_score: int
    rendered_chars: int
    text: str


def build_longmemeval_s_heldout_artifacts(
    *,
    dataset_path: str | pathlib.Path = DEFAULT_LONGMEMEVAL_S_PATH,
    fail18_manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    seed: str = DEFAULT_HELDOUT_SEED,
    limit: int = DEFAULT_HELDOUT_LIMIT,
    top_k: int = DEFAULT_HELDOUT_TOP_K,
    session_char_limit: int = DEFAULT_HELDOUT_SESSION_CHAR_LIMIT,
) -> dict[str, typing.Any]:
    """Build answer-free held-out payload, private manifest, and provenance.

    The selected rows are deterministic numeric/count-style LongMemEval-S rows
    that are not in the fail18 fixture. Provider payloads get opaque evidence ids;
    gold answers and raw answer-session ids stay in private artifacts.
    """
    raw_rows = _load_json_list(dataset_path)
    fail18_ids = {
        str(row.get("question_id", ""))
        for row in ambiguity_fail18.load_manifest(fail18_manifest_path)
    }
    eligible = [
        row
        for row in raw_rows
        if _is_heldout_eligible(row, excluded_case_ids=fail18_ids)
    ]
    selected = sorted(
        eligible,
        key=lambda row: _stable_selection_key(str(row.get("question_id", "")), seed),
    )[:limit]
    payloads: list[dict[str, typing.Any]] = []
    manifest_rows: list[dict[str, typing.Any]] = []
    provenance: dict[str, typing.Any] = {}
    for row in selected:
        payload, case_provenance = _build_lme_heldout_payload(
            row,
            top_k=top_k,
            session_char_limit=session_char_limit,
        )
        payloads.append(payload)
        qid = str(row.get("question_id", ""))
        manifest_rows.append(
            {
                "question_id": qid,
                "question": str(row.get("question", "")),
                "question_type": str(row.get("question_type", "")),
                "gold_answer": str(row.get("answer", "")),
                "source_dataset": str(dataset_path),
            }
        )
        provenance[qid] = case_provenance
    generated_at = dt.datetime.now(dt.UTC).isoformat()
    payload_artifact = {
        "name": "longmemeval-s-answer-unit-witness-heldout-payloads",
        "artifact_kind": "provider_payloads",
        "prompt_version": answer_unit_witness.PROMPT_VERSION,
        "generated_at": generated_at,
        "source_dataset": str(dataset_path),
        "source_fail18_manifest": str(fail18_manifest_path),
        "selection": {
            "seed": seed,
            "limit": limit,
            "eligible_count": len(eligible),
            "selected_count": len(selected),
            "excluded_fail18_case_count": len(fail18_ids),
            "question_prefixes": list(_ANSWER_QUESTION_PREFIXES),
            "requires_numeric_answer": True,
        },
        "provider_visibility": {
            "gold_answer_visible": False,
            "answer_session_ids_visible": False,
            "raw_session_ids_visible": False,
        },
        "retrieval": {
            "method": "answer-free lexical top-k over LongMemEval-S haystack",
            "top_k": top_k,
            "chars_per_session": session_char_limit,
            "uses_answer_session_ids": False,
        },
        "total": len(payloads),
        "payloads": payloads,
    }
    provenance_artifact = {
        "name": "longmemeval-s-answer-unit-witness-heldout-provenance",
        "artifact_kind": "private_provenance",
        "generated_at": generated_at,
        "source_dataset": str(dataset_path),
        "payload_artifact": str(DEFAULT_HELDOUT_PAYLOADS_PATH),
        "manifest_artifact": str(DEFAULT_HELDOUT_MANIFEST_PATH),
        "cases": provenance,
    }
    return {
        "payloads": payload_artifact,
        "manifest": manifest_rows,
        "provenance": provenance_artifact,
    }


def build_fail18_reasoning_mechanism_diagnostic(
    *,
    stable_wrong_diagnostic_path: str | pathlib.Path = (
        DEFAULT_STABLE_WRONG_DIAGNOSTIC_PATH
    ),
    outputs_path: str | pathlib.Path = DEFAULT_WITNESS_OUTPUTS_PATH,
) -> dict[str, typing.Any]:
    """Classify the 5 reasoning/enumeration-capped fail18 rows by mechanism."""
    stable_artifact = _load_json(stable_wrong_diagnostic_path)
    rows = interpretation_runner.load_jsonl(outputs_path)
    rows_by_case: dict[str, list[dict[str, typing.Any]]] = collections.defaultdict(list)
    for row in rows:
        rows_by_case[str(row.get("case_id", ""))].append(row)
    cases = []
    mechanism_counts: collections.Counter[str] = collections.Counter()
    for case in stable_artifact.get("cases", []):
        if not isinstance(case, dict):
            continue
        if case.get("classification") != "reasoning_or_enumeration_capped":
            continue
        case_rows = sorted(
            rows_by_case.get(str(case.get("case_id", "")), []),
            key=lambda row: int(row.get("sample_index", 0) or 0),
        )
        classified = _classify_reasoning_case(case, case_rows)
        mechanism_counts[str(classified["mechanism"])] += 1
        cases.append(classified)
    return {
        "name": "fail18-answer-unit-witness-reasoning-mechanism-diagnostic",
        "artifact_kind": "answer_unit_witness_reasoning_mechanism_diagnostic",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_stable_wrong_diagnostic": str(stable_wrong_diagnostic_path),
        "source_outputs": str(outputs_path),
        "summary": {
            "total_reasoning_or_enumeration_capped": len(cases),
            "mechanism_counts": dict(sorted(mechanism_counts.items())),
            "dominant_mechanism": (
                mechanism_counts.most_common(1)[0][0] if mechanism_counts else ""
            ),
            "recommendation": _mechanism_recommendation(mechanism_counts),
        },
        "cases": cases,
    }


def build_inclusion_policy_payload_artifact(
    *,
    source_payloads_path: str | pathlib.Path,
    case_ids: tuple[str, ...] = (),
) -> dict[str, typing.Any]:
    """Clone witness payloads with the v2 inclusion-policy contract."""
    source = _load_json(source_payloads_path)
    selected_case_ids = {str(case_id) for case_id in case_ids if str(case_id)}
    payloads = []
    for payload in source.get("payloads", []):
        if not isinstance(payload, dict):
            continue
        case_id = str(payload.get("case", {}).get("id", ""))
        if selected_case_ids and case_id not in selected_case_ids:
            continue
        payloads.append(_with_inclusion_policy_contract(payload))
    return {
        **source,
        "name": f"{source.get('name', 'answer-unit-witness')}-inclusion-policy-v2",
        "prompt_version": INCLUSION_POLICY_PROMPT_VERSION,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_payload_artifact": str(source_payloads_path),
        "policy_variant": {
            "name": "inclusion_policy_v2",
            "prompt_version": INCLUSION_POLICY_PROMPT_VERSION,
            "target_case_ids": sorted(selected_case_ids),
            "measurement_deferred_case_ids": list(MEASUREMENT_DEFERRED_CASE_IDS),
            "contract_additions": list(_INCLUSION_POLICY_CONTRACT),
            "intent": (
                "Fix candidate enumeration and include/exclude policy without "
                "changing retrieval or provider-visible gold."
            ),
        },
        "total": len(payloads),
        "payloads": payloads,
    }


def build_inclusion_policy_ab_report(
    *,
    fail18_baseline_report_path: str | pathlib.Path,
    fail18_candidate_report_path: str | pathlib.Path,
    heldout_baseline_report_path: str | pathlib.Path,
    heldout_candidate_report_path: str | pathlib.Path,
    target_case_ids: tuple[str, ...] = INCLUSION_POLICY_TARGET_CASE_IDS,
) -> dict[str, typing.Any]:
    """Compare v1 vs v2 reports on fail18 target rows and held-out rows."""
    fail18_baseline = _load_json(fail18_baseline_report_path)
    fail18_candidate = _load_json(fail18_candidate_report_path)
    heldout_baseline = _load_json(heldout_baseline_report_path)
    heldout_candidate = _load_json(heldout_candidate_report_path)
    target_ids = tuple(str(case_id) for case_id in target_case_ids)
    fail18_cases = [
        _compare_case(case_id, fail18_baseline, fail18_candidate)
        for case_id in target_ids
    ]
    heldout_case_ids = tuple(
        str(case.get("case_id", ""))
        for case in heldout_baseline.get("cases", [])
        if isinstance(case, dict)
    )
    heldout_cases = [
        _compare_case(case_id, heldout_baseline, heldout_candidate)
        for case_id in heldout_case_ids
    ]
    fail18_summary = _comparison_summary(fail18_cases)
    heldout_summary = _comparison_summary(heldout_cases)
    return {
        "name": "answer-unit-witness-inclusion-policy-v2-ab-report",
        "artifact_kind": "answer_unit_witness_policy_ab_report",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "policy_variant": "inclusion_policy_v2",
        "source_reports": {
            "fail18_baseline": str(fail18_baseline_report_path),
            "fail18_candidate": str(fail18_candidate_report_path),
            "heldout_baseline": str(heldout_baseline_report_path),
            "heldout_candidate": str(heldout_candidate_report_path),
        },
        "targeting": {
            "fail18_target_case_ids": list(target_ids),
            "measurement_deferred_case_ids": list(MEASUREMENT_DEFERRED_CASE_IDS),
            "heldout_case_count": len(heldout_case_ids),
            "acceptance_rule": (
                "Do not accept a fail18 gain if held-out exact/support regresses "
                "without a mechanism-specific justification."
            ),
        },
        "summary": {
            "fail18": fail18_summary,
            "heldout": heldout_summary,
            "decision": _ab_decision(fail18_summary, heldout_summary),
        },
        "fail18_cases": fail18_cases,
        "heldout_cases": heldout_cases,
    }


def build_payload_budget_ab_report(
    *,
    fail18_baseline_report_path: str | pathlib.Path,
    fail18_candidate_report_path: str | pathlib.Path,
    heldout_baseline_report_path: str | pathlib.Path,
    heldout_candidate_report_path: str | pathlib.Path,
    baseline_label: str = "2500",
    candidate_label: str = "8000",
) -> dict[str, typing.Any]:
    """Compare answer-unit witness reports across payload char budgets."""
    fail18_baseline = _load_json(fail18_baseline_report_path)
    fail18_candidate = _load_json(fail18_candidate_report_path)
    heldout_baseline = _load_json(heldout_baseline_report_path)
    heldout_candidate = _load_json(heldout_candidate_report_path)
    fail18_case_ids = _report_case_ids(fail18_baseline)
    heldout_case_ids = _report_case_ids(heldout_baseline)
    fail18_cases = [
        _compare_case(case_id, fail18_baseline, fail18_candidate)
        for case_id in fail18_case_ids
    ]
    heldout_cases = [
        _compare_case(case_id, heldout_baseline, heldout_candidate)
        for case_id in heldout_case_ids
    ]
    fail18_summary = _comparison_summary(fail18_cases)
    heldout_summary = _comparison_summary(heldout_cases)
    return {
        "name": "answer-unit-witness-payload-budget-ab-report",
        "artifact_kind": "answer_unit_witness_payload_budget_ab_report",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "budget_variant": {
            "baseline_label": baseline_label,
            "candidate_label": candidate_label,
            "warning": (
                "Window size is a tuned context-construction parameter; gains "
                "must be interpreted net of held-out regressions."
            ),
        },
        "source_reports": {
            "fail18_baseline": str(fail18_baseline_report_path),
            "fail18_candidate": str(fail18_candidate_report_path),
            "heldout_baseline": str(heldout_baseline_report_path),
            "heldout_candidate": str(heldout_candidate_report_path),
        },
        "summary": {
            "fail18": fail18_summary,
            "heldout": heldout_summary,
            "decision": _ab_decision(fail18_summary, heldout_summary),
        },
        "fail18_cases": fail18_cases,
        "heldout_cases": heldout_cases,
    }


def build_payload_budget_regression_diagnostic(
    *,
    budget_ab_report_path: str | pathlib.Path = DEFAULT_PAYLOAD_BUDGET_AB_REPORT_PATH,
    baseline_payloads_path: str | pathlib.Path = (
        answer_unit_witness.DEFAULT_PAYLOADS_PATH
    ),
    candidate_payloads_path: str | pathlib.Path,
    baseline_outputs_path: str | pathlib.Path,
    candidate_outputs_path: str | pathlib.Path,
) -> dict[str, typing.Any]:
    """Classify rows that regressed when payload budget changed."""
    ab_report = _load_json(budget_ab_report_path)
    regressed_case_ids = [
        str(case_id)
        for case_id in ab_report.get("summary", {})
        .get("fail18", {})
        .get("regressed_case_ids", [])
    ]
    baseline_payloads = _payloads_by_case(_load_json(baseline_payloads_path))
    candidate_payloads = _payloads_by_case(_load_json(candidate_payloads_path))
    baseline_outputs = _outputs_by_case(baseline_outputs_path)
    candidate_outputs = _outputs_by_case(candidate_outputs_path)
    cases = []
    mechanism_counts: collections.Counter[str] = collections.Counter()
    for case_id in regressed_case_ids:
        case = _payload_budget_regression_case(
            case_id=case_id,
            baseline_payload=baseline_payloads.get(case_id),
            candidate_payload=candidate_payloads.get(case_id),
            baseline_output=baseline_outputs.get(case_id),
            candidate_output=candidate_outputs.get(case_id),
        )
        mechanism_counts[str(case["mechanism"])] += 1
        cases.append(case)
    return {
        "name": "fail18-payload-budget-regression-diagnostic",
        "artifact_kind": "answer_unit_witness_payload_budget_regression_diagnostic",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_budget_ab_report": str(budget_ab_report_path),
        "source_payloads": {
            "baseline": str(baseline_payloads_path),
            "candidate": str(candidate_payloads_path),
        },
        "source_outputs": {
            "baseline": str(baseline_outputs_path),
            "candidate": str(candidate_outputs_path),
        },
        "summary": {
            "case_count": len(cases),
            "mechanism_counts": dict(sorted(mechanism_counts.items())),
            "unified_mechanism": "salience_competition",
            "selector_implication": (
                "Optimize salience margin: answer-bearing user evidence must "
                "survive selection and dominate the strongest competing "
                "distractor under role-aware and operation-aware scoring."
            ),
        },
        "cases": cases,
    }


def build_fail18_span_survival_report(
    *,
    payloads_path: str | pathlib.Path = answer_unit_witness.DEFAULT_PAYLOADS_PATH,
    gold_span_needles_path: str | pathlib.Path = DEFAULT_FAIL18_GOLD_SPAN_NEEDLES_PATH,
    corpus_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_CORPUS,
    stable_wrong_diagnostic_path: str | pathlib.Path = (
        DEFAULT_STABLE_WRONG_DIAGNOSTIC_PATH
    ),
) -> dict[str, typing.Any]:
    """Check whether answer-bearing gold-span needles survive payload truncation."""
    payload_artifact = _load_json(payloads_path)
    needle_artifact = _load_json(gold_span_needles_path)
    corpus_rows = ambiguity_fail18.load_corpus(corpus_path)
    corpus_by_id = {str(row.get("question_id", "")): row for row in corpus_rows}
    stable = _load_json(stable_wrong_diagnostic_path)
    stable_wrong_ids = [
        str(case.get("case_id", ""))
        for case in stable.get("cases", [])
        if isinstance(case, dict)
    ]
    payload_by_id = {
        str(payload.get("case", {}).get("id", "")): payload
        for payload in payload_artifact.get("payloads", [])
        if isinstance(payload, dict)
    }
    needle_by_case = {
        str(case.get("case_id", "")): case
        for case in needle_artifact.get("cases", [])
        if isinstance(case, dict)
    }
    cases = []
    status_counts: collections.Counter[str] = collections.Counter()
    for case_id in stable_wrong_ids:
        case = _span_survival_case(
            case_id=case_id,
            payload=payload_by_id.get(case_id),
            corpus_row=corpus_by_id.get(case_id),
            needle_case=needle_by_case.get(case_id),
        )
        cases.append(case)
        status_counts.update(str(item["status"]) for item in case["needles"])
    return {
        "name": "fail18-answer-unit-witness-span-survival",
        "artifact_kind": "answer_unit_witness_span_survival_report",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_payloads": str(payloads_path),
        "source_gold_span_needles": str(gold_span_needles_path),
        "source_corpus": str(corpus_path),
        "source_stable_wrong_diagnostic": str(stable_wrong_diagnostic_path),
        "payload": {
            "prompt_version": str(
                payload_artifact.get(
                    "prompt_version",
                    answer_unit_witness.PROMPT_VERSION,
                )
            ),
            "chars_per_session": payload_artifact.get("retrieval", {}).get(
                "chars_per_session"
            ),
        },
        "summary": {
            "case_count": len(cases),
            "needle_count": sum(len(case["needles"]) for case in cases),
            "status_counts": dict(sorted(status_counts.items())),
            "cases_all_needles_present": [
                case["case_id"] for case in cases if case["all_needles_present"]
            ],
            "cases_with_truncation_loss": [
                case["case_id"] for case in cases if case["has_truncation_loss"]
            ],
            "cases_with_retrieval_loss": [
                case["case_id"] for case in cases if case["has_retrieval_loss"]
            ],
        },
        "cases": cases,
    }


def _payload_budget_regression_case(
    *,
    case_id: str,
    baseline_payload: dict[str, typing.Any] | None,
    candidate_payload: dict[str, typing.Any] | None,
    baseline_output: dict[str, typing.Any] | None,
    candidate_output: dict[str, typing.Any] | None,
) -> dict[str, typing.Any]:
    baseline_sessions = _payload_session_texts(baseline_payload)
    candidate_sessions = _payload_session_texts(candidate_payload)
    baseline_units = [
        unit
        for unit in (baseline_output or {}).get("units", [])
        if isinstance(unit, dict)
    ]
    candidate_units = [
        unit
        for unit in (candidate_output or {}).get("units", [])
        if isinstance(unit, dict)
    ]
    baseline_included = [
        _budget_unit_presence(unit, baseline_sessions, candidate_sessions)
        for unit in baseline_units
        if str(unit.get("decision", "")) == "include"
    ]
    candidate_added_included = []
    for unit in candidate_units:
        if str(unit.get("decision", "")) != "include":
            continue
        if not _unit_overlaps_any(unit, baseline_included):
            candidate_added_included.append(
                _budget_unit_presence(unit, baseline_sessions, candidate_sessions)
            )
    candidate_excluded_baseline_spans = [
        _budget_unit_presence(unit, baseline_sessions, candidate_sessions)
        for unit in candidate_units
        if str(unit.get("decision", "")) == "exclude"
        and _unit_overlaps_any(unit, baseline_included)
    ]
    introduced_by_candidate_window = [
        item
        for item in candidate_added_included
        if item["candidate_payload_contains_exact"]
        and not item["baseline_payload_contains_exact"]
    ]
    baseline_included_missing_in_candidate = [
        item
        for item in baseline_included
        if not item["candidate_payload_contains_exact"]
    ]
    if introduced_by_candidate_window:
        mechanism = "displacement_new_window_span"
        selector_implication = (
            "The larger window exposed a plausible wrong unit. A selector must "
            "suppress competing distractors, not merely maximize relevance."
        )
    elif candidate_excluded_baseline_spans:
        mechanism = "same_visible_span_decision_flip"
        selector_implication = (
            "The answer span was visible in both budgets. Prefer tight windows "
            "around the cited span and keep surrounding distractors minimal."
        )
    elif not baseline_included_missing_in_candidate:
        mechanism = "dilution_no_new_included_span"
        selector_implication = (
            "The gold-supporting spans survived, but extra context changed the "
            "answer. Keep selected windows few and compact."
        )
    else:
        mechanism = "answer_span_lost_between_budgets"
        selector_implication = (
            "The candidate payload lost a baseline answer span; treat this as "
            "a context-construction recall failure before scoring accuracy."
        )
    return {
        "case_id": case_id,
        "question": str(
            (candidate_payload or baseline_payload or {}).get("case", {}).get(
                "question", ""
            )
        ),
        "baseline_answer_number": (baseline_output or {}).get("answer_number"),
        "candidate_answer_number": (candidate_output or {}).get("answer_number"),
        "mechanism": mechanism,
        "selector_implication": selector_implication,
        "baseline_included_span_count": len(baseline_included),
        "baseline_included_missing_in_candidate_count": len(
            baseline_included_missing_in_candidate
        ),
        "candidate_added_included_count": len(candidate_added_included),
        "candidate_added_included_new_window_count": len(
            introduced_by_candidate_window
        ),
        "candidate_excluded_baseline_span_count": len(
            candidate_excluded_baseline_spans
        ),
        "baseline_included_spans": baseline_included,
        "candidate_added_included_spans": candidate_added_included,
        "candidate_excluded_baseline_spans": candidate_excluded_baseline_spans,
    }


def _budget_unit_presence(
    unit: dict[str, typing.Any],
    baseline_sessions: dict[str, str],
    candidate_sessions: dict[str, str],
) -> dict[str, typing.Any]:
    session_id = str(unit.get("evidence_session_id", ""))
    span = str(unit.get("evidence_span", ""))
    baseline_text = baseline_sessions.get(session_id)
    candidate_text = candidate_sessions.get(session_id)
    baseline_offset = baseline_text.find(span) if baseline_text is not None else -1
    candidate_offset = candidate_text.find(span) if candidate_text is not None else -1
    return {
        "unit": _unit_summary(unit),
        "span": span,
        "baseline_payload_has_session": baseline_text is not None,
        "candidate_payload_has_session": candidate_text is not None,
        "baseline_payload_contains_exact": baseline_offset >= 0,
        "candidate_payload_contains_exact": candidate_offset >= 0,
        "baseline_offset": baseline_offset if baseline_offset >= 0 else None,
        "candidate_offset": candidate_offset if candidate_offset >= 0 else None,
    }


def _unit_overlaps_any(
    unit: dict[str, typing.Any],
    existing: list[dict[str, typing.Any]],
) -> bool:
    session_id = str(unit.get("evidence_session_id", ""))
    span = str(unit.get("evidence_span", "")).casefold()
    if not span:
        return False
    for item in existing:
        other = item["unit"]
        if str(other.get("evidence_session_id", "")) != session_id:
            continue
        other_span = str(other.get("evidence_span", "")).casefold()
        if span in other_span or other_span in span:
            return True
    return False


def _payloads_by_case(
    payload_artifact: dict[str, typing.Any],
) -> dict[str, dict[str, typing.Any]]:
    return {
        str(payload.get("case", {}).get("id", "")): payload
        for payload in payload_artifact.get("payloads", [])
        if isinstance(payload, dict)
    }


def _payload_session_texts(
    payload: dict[str, typing.Any] | None,
) -> dict[str, str]:
    return {
        str(session.get("session_id", "")): str(session.get("text", ""))
        for session in (payload or {}).get("case", {}).get("evidence_sessions", [])
        if isinstance(session, dict)
    }


def _outputs_by_case(
    outputs_path: str | pathlib.Path,
) -> dict[str, dict[str, typing.Any]]:
    rows = interpretation_runner.load_jsonl(outputs_path)
    return {
        str(row.get("case_id", "")): row
        for row in rows
        if isinstance(row, dict)
    }


def _span_survival_case(
    *,
    case_id: str,
    payload: dict[str, typing.Any] | None,
    corpus_row: dict[str, typing.Any] | None,
    needle_case: dict[str, typing.Any] | None,
) -> dict[str, typing.Any]:
    payload_sessions = {
        str(session.get("session_id", "")): str(session.get("text", ""))
        for session in (payload or {}).get("case", {}).get("evidence_sessions", [])
        if isinstance(session, dict)
    }
    full_sessions = _corpus_session_texts(corpus_row)
    needles = []
    for raw in (needle_case or {}).get("needles", []):
        if not isinstance(raw, dict):
            continue
        session_id = str(raw.get("session_id", ""))
        needle = str(raw.get("needle", ""))
        payload_text = payload_sessions.get(session_id)
        full_text = full_sessions.get(session_id, "")
        payload_offset = payload_text.find(needle) if payload_text is not None else -1
        full_offset = full_text.find(needle) if full_text else -1
        if payload_text is None:
            status = "session_not_retrieved"
        elif payload_offset >= 0:
            status = "present"
        elif "...[truncated]" in payload_text:
            status = "absent_payload_truncated"
        else:
            status = "absent_payload_untruncated"
        needles.append(
            {
                "label": str(raw.get("label", "")),
                "session_id": session_id,
                "needle": needle,
                "status": status,
                "payload_contains_exact": payload_offset >= 0,
                "payload_offset": payload_offset if payload_offset >= 0 else None,
                "full_session_contains_exact": full_offset >= 0,
                "full_session_offset": full_offset if full_offset >= 0 else None,
                "payload_session_retrieved": payload_text is not None,
                "payload_session_truncated": bool(
                    payload_text and "...[truncated]" in payload_text
                ),
            }
        )
    return {
        "case_id": case_id,
        "question": str((needle_case or {}).get("question", "")),
        "needle_count": len(needles),
        "all_needles_present": all(
            item["status"] == "present" for item in needles
        )
        if needles
        else False,
        "has_truncation_loss": any(
            item["status"] == "absent_payload_truncated" for item in needles
        ),
        "has_retrieval_loss": any(
            item["status"] == "session_not_retrieved" for item in needles
        ),
        "needles": needles,
    }


def _corpus_session_texts(
    corpus_row: dict[str, typing.Any] | None,
) -> dict[str, str]:
    if not corpus_row:
        return {}
    return {
        str(session_id): _render_lme_session(session)[0]
        for session_id, session in zip(
            corpus_row.get("haystack_session_ids", []),
            corpus_row.get("haystack_sessions", []),
            strict=False,
        )
        if isinstance(session, list)
    }


def _with_inclusion_policy_contract(
    payload: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    cloned = json.loads(json.dumps(payload))
    existing_contract = list(cloned.get("contract", []))
    cloned["prompt_version"] = INCLUSION_POLICY_PROMPT_VERSION
    cloned["contract"] = [*existing_contract, *_INCLUSION_POLICY_CONTRACT]
    cloned["task"] = (
        str(cloned.get("task", "")).rstrip()
        + " Apply the inclusion_policy_v2 contract additions before returning."
    )
    return typing.cast("dict[str, typing.Any]", cloned)


def _build_lme_heldout_payload(
    row: dict[str, typing.Any],
    *,
    top_k: int,
    session_char_limit: int,
) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
    ranked = _rank_lme_sessions(row, top_k=top_k)
    evidence_sessions: list[dict[str, typing.Any]] = []
    evidence_map: dict[str, dict[str, typing.Any]] = {}
    for index, session in enumerate(ranked[:top_k], start=1):
        provider_id = f"evidence_{index:03d}"
        evidence_sessions.append(
            {
                "session_id": provider_id,
                "date": session.date,
                "selection_rank": session.selection_rank,
                "selection_score": session.selection_score,
                "text": _trim_text(session.text, session_char_limit),
            }
        )
        evidence_map[provider_id] = {
            "raw_session_id": session.raw_session_id,
            "raw_session_index": session.raw_session_index,
            "date": session.date,
            "selection_rank": session.selection_rank,
            "selection_score": session.selection_score,
            "user_selection_score": session.user_selection_score,
            "assistant_selection_score": session.assistant_selection_score,
            "rendered_chars": session.rendered_chars,
            "raw_session_is_answer_session": session.raw_session_id
            in {str(item) for item in row.get("answer_session_ids", [])},
        }
    payload = {
        "task": (
            "Answer the question by emitting a checkable answer-unit witness. "
            "Do not emit neutral facts or Datalog. The answer must be the simple "
            "aggregation over the units you list."
        ),
        "prompt_version": answer_unit_witness.PROMPT_VERSION,
        "contract": _heldout_contract(),
        "output_schema": {
            "case_id": str(row.get("question_id", "")),
            "answer_variable": "short noun phrase for what is counted/summed/looked up",
            "aggregation": "count_included|sum_included|lookup_value",
            "units": [
                {
                    "unit_id": "stable string unique within this response",
                    "label": "short unit label",
                    "decision": "include|exclude",
                    "borderline": False,
                    "value": "number for sum/lookup units, otherwise null",
                    "unit": "unit label such as days/dollars/hours/points or null",
                    "evidence_session_id": "one provided session_id",
                    "evidence_span": "short exact span copied from that session",
                    "reason_code": "stable snake_case string",
                    "reason": "one short reason for include/exclude",
                }
            ],
            "answer_number": 0,
            "rationale": "one terse sentence explaining the aggregation",
        },
        "case": {
            "id": str(row.get("question_id", "")),
            "question": str(row.get("question", "")),
            "question_type": str(row.get("question_type", "")),
            "question_date": str(row.get("question_date", "")),
            "evidence_sessions": evidence_sessions,
        },
    }
    provenance = {
        "question": str(row.get("question", "")),
        "question_type": str(row.get("question_type", "")),
        "gold_answer": str(row.get("answer", "")),
        "answer_session_ids": [str(item) for item in row.get("answer_session_ids", [])],
        "selected_raw_session_ids": [
            item["raw_session_id"] for item in evidence_map.values()
        ],
        "answer_sessions_retrieved": [
            item
            for item in row.get("answer_session_ids", [])
            if str(item) in {entry["raw_session_id"] for entry in evidence_map.values()}
        ],
        "evidence_id_map": evidence_map,
    }
    return payload, provenance


def _heldout_contract() -> list[str]:
    return [
        "Use only the evidence sessions in this payload.",
        "List every plausible answer-bearing unit needed for the numeric answer.",
        "Each unit decision must be include or exclude.",
        "Set borderline=true when the unit is a plausible swing vote.",
        (
            "Every unit must cite exactly one evidence_session_id from the "
            "payload and one short exact evidence_span from that session."
        ),
        (
            "For count questions, aggregation=count_included and "
            "answer_number is the number of include units."
        ),
        (
            "For sum questions, aggregation=sum_included and each included "
            "unit must have a numeric value."
        ),
        (
            "For lookup/scalar questions, aggregation=lookup_value and "
            "exactly one included unit must carry the numeric answer value."
        ),
        "Do not include hidden labels, hidden answer ids, or final prose outside JSON.",
    ]


def _rank_lme_sessions(
    row: dict[str, typing.Any],
    *,
    top_k: int,
) -> list[RankedLmeSession]:
    terms = _question_terms(str(row.get("question", "")))
    session_ids = [str(item) for item in row.get("haystack_session_ids", [])]
    dates = [str(item) for item in row.get("haystack_dates", [])]
    sessions = row.get("haystack_sessions", [])
    ranked: list[RankedLmeSession] = []
    for index, turns in enumerate(sessions):
        if not isinstance(turns, list):
            continue
        raw_session_id = session_ids[index] if index < len(session_ids) else str(index)
        date = dates[index] if index < len(dates) else ""
        text, user_text, assistant_text = _render_lme_session(turns)
        user_score = _term_score(user_text, terms)
        assistant_score = _term_score(assistant_text, terms)
        ranked.append(
            RankedLmeSession(
                raw_session_index=index,
                raw_session_id=raw_session_id,
                date=date,
                selection_rank=0,
                selection_score=(2 * user_score) + assistant_score,
                user_selection_score=user_score,
                assistant_selection_score=assistant_score,
                rendered_chars=len(text),
                text=text,
            )
        )
    ranked = sorted(
        ranked,
        key=lambda item: (-item.selection_score, item.raw_session_index),
    )
    if top_k > 0:
        ranked = ranked[:top_k]
    return [
        dataclasses.replace(item, selection_rank=index)
        for index, item in enumerate(ranked, start=1)
    ]


def _render_lme_session(
    turns: list[dict[str, typing.Any]],
) -> tuple[str, str, str]:
    chunks: list[str] = []
    user_chunks: list[str] = []
    assistant_chunks: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).strip()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        chunks.append(f"{role.upper()}: {content}" if role else content)
        if role.casefold() == "user":
            user_chunks.append(content)
        elif role.casefold() == "assistant":
            assistant_chunks.append(content)
    return "\n".join(chunks), "\n".join(user_chunks), "\n".join(assistant_chunks)


def _question_terms(question: str) -> tuple[str, ...]:
    terms = []
    for token in _TOKEN_RE.findall(question.casefold()):
        if token in _QUESTION_STOPWORDS or len(token) < 2:
            continue
        terms.append(token)
    return tuple(dict.fromkeys(terms))


def _term_score(text: str, terms: tuple[str, ...]) -> int:
    lowered = text.casefold()
    return sum(1 for term in terms if term in lowered)


def _trim_text(text: str, char_limit: int) -> str:
    if len(text) <= char_limit:
        return text
    return f"{text[:char_limit]}\n...[truncated]"


def _is_heldout_eligible(
    row: dict[str, typing.Any],
    *,
    excluded_case_ids: set[str],
) -> bool:
    qid = str(row.get("question_id", ""))
    question = str(row.get("question", "")).casefold().strip()
    answer = str(row.get("answer", ""))
    return (
        bool(qid)
        and qid not in excluded_case_ids
        and not qid.endswith("_abs")
        and bool(row.get("answer_session_ids"))
        and question.startswith(_ANSWER_QUESTION_PREFIXES)
        and bool(_NUMERIC_ANSWER_RE.search(answer))
    )


def _stable_selection_key(case_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()


def _classify_reasoning_case(
    case: dict[str, typing.Any],
    rows: list[dict[str, typing.Any]],
) -> dict[str, typing.Any]:
    answer_session_ids = {str(item) for item in case.get("answer_session_ids", [])}
    units_by_sample = [
        [unit for unit in row.get("units", []) if isinstance(unit, dict)]
        for row in rows
    ]
    covered_answer_sessions = {
        str(unit.get("evidence_session_id", ""))
        for units in units_by_sample
        for unit in units
        if str(unit.get("evidence_session_id", "")) in answer_session_ids
    }
    never_covered_answer_sessions = sorted(answer_session_ids - covered_answer_sessions)
    excluded_units = [
        unit
        for units in units_by_sample
        for unit in units
        if str(unit.get("decision", "")) == "exclude"
    ]
    included_units = [
        unit
        for units in units_by_sample
        for unit in units
        if str(unit.get("decision", "")) == "include"
    ]
    gold_value = _optional_float(case.get("gold_value"))
    excluded_gold_values = [
        unit
        for unit in excluded_units
        if _numbers_match(_optional_float(unit.get("value")), gold_value)
    ]
    aggregations = {str(row.get("aggregation", "")) for row in rows}
    if never_covered_answer_sessions:
        mechanism = "missing_answer_session_unit"
        recommendation = (
            "Fix enumeration coverage before inclusion policy: at least one "
            "gold answer session has no listed unit in any witness sample."
        )
    elif excluded_gold_values and "lookup_value" in aggregations:
        mechanism = "wrong_lookup_value_choice"
        recommendation = (
            "Fix scalar/lookup policy: the gold-compatible value is present "
            "as an excluded unit while a lower or different value is included."
        )
    elif excluded_units and not included_units:
        mechanism = "all_candidate_units_excluded"
        recommendation = (
            "Fix include/exclude semantics: the model found candidate units "
            "but excluded every one."
        )
    elif excluded_units:
        mechanism = "listed_candidate_excluded_or_misvalued"
        recommendation = (
            "Fix include/exclude or aggregation policy: candidate units are "
            "listed but excluded or assigned the wrong answer value."
        )
    else:
        mechanism = "missing_gold_relevant_unit"
        recommendation = (
            "Fix candidate enumeration: the witness does not list enough "
            "answer-bearing units."
        )
    return {
        "case_id": str(case.get("case_id", "")),
        "question": str(case.get("question", "")),
        "gold_value": case.get("gold_value"),
        "answer_support": case.get("answer_support", []),
        "classification": str(case.get("classification", "")),
        "prior_subtype": str(case.get("subtype", "")),
        "mechanism": mechanism,
        "recommendation": recommendation,
        "answer_session_ids": sorted(answer_session_ids),
        "covered_answer_sessions_by_any_unit": sorted(covered_answer_sessions),
        "never_covered_answer_sessions": never_covered_answer_sessions,
        "included_label_histogram": case.get("included_label_histogram", {}),
        "excluded_label_histogram": case.get("excluded_label_histogram", {}),
        "excluded_gold_value_units": [
            _unit_summary(unit) for unit in excluded_gold_values
        ],
        "sample_summaries": case.get("sample_summaries", []),
    }


def _mechanism_recommendation(counts: collections.Counter[str]) -> str:
    if not counts:
        return "No reasoning/enumeration-capped rows found."
    dominant = counts.most_common(1)[0][0]
    if dominant == "missing_answer_session_unit":
        return "Start with unit enumeration over retrieved answer sessions."
    if dominant == "wrong_lookup_value_choice":
        return "Start with scalar lookup intent policy."
    if dominant in {
        "all_candidate_units_excluded",
        "listed_candidate_excluded_or_misvalued",
    }:
        return "Start with include/exclude and aggregation semantics."
    return "Start with candidate enumeration."


def _compare_case(
    case_id: str,
    baseline_report: dict[str, typing.Any],
    candidate_report: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    baseline = _case_by_id(baseline_report, case_id)
    candidate = _case_by_id(candidate_report, case_id)
    baseline_support = bool(baseline.get("gold_in_answer_support", False))
    candidate_support = bool(candidate.get("gold_in_answer_support", False))
    return {
        "case_id": case_id,
        "question": str(candidate.get("question") or baseline.get("question") or ""),
        "gold_value": candidate.get("gold_value", baseline.get("gold_value")),
        "baseline_answer_support": baseline.get("answer_support", []),
        "candidate_answer_support": candidate.get("answer_support", []),
        "baseline_gold_in_support": baseline_support,
        "candidate_gold_in_support": candidate_support,
        "delta": (
            "improved"
            if candidate_support and not baseline_support
            else "regressed"
            if baseline_support and not candidate_support
            else "unchanged"
        ),
        "candidate_unstable_included_labels": candidate.get(
            "unstable_included_labels", []
        ),
    }


def _case_by_id(
    report: dict[str, typing.Any],
    case_id: str,
) -> dict[str, typing.Any]:
    for case in report.get("cases", []):
        if isinstance(case, dict) and str(case.get("case_id", "")) == case_id:
            return case
    return {"case_id": case_id}


def _report_case_ids(report: dict[str, typing.Any]) -> tuple[str, ...]:
    return tuple(
        str(case.get("case_id", ""))
        for case in report.get("cases", [])
        if isinstance(case, dict) and str(case.get("case_id", ""))
    )


def _comparison_summary(cases: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
    baseline_hits = sum(1 for case in cases if case["baseline_gold_in_support"])
    candidate_hits = sum(1 for case in cases if case["candidate_gold_in_support"])
    deltas = collections.Counter(str(case["delta"]) for case in cases)
    return {
        "case_count": len(cases),
        "baseline_gold_in_support": baseline_hits,
        "candidate_gold_in_support": candidate_hits,
        "support_delta": candidate_hits - baseline_hits,
        "delta_counts": dict(sorted(deltas.items())),
        "improved_case_ids": [
            str(case["case_id"]) for case in cases if case["delta"] == "improved"
        ],
        "regressed_case_ids": [
            str(case["case_id"]) for case in cases if case["delta"] == "regressed"
        ],
    }


def _ab_decision(
    fail18_summary: dict[str, typing.Any],
    heldout_summary: dict[str, typing.Any],
) -> str:
    fail18_delta = int(fail18_summary.get("support_delta", 0))
    heldout_delta = int(heldout_summary.get("support_delta", 0))
    if fail18_delta > 0 and heldout_delta >= 0:
        return "candidate_supported_for_next_k3_probe"
    if fail18_delta > 0 and heldout_delta < 0:
        return "reject_or_rework_possible_overfit"
    if fail18_delta == 0 and heldout_delta >= 0:
        return "no_fail18_gain_keep_diagnostic_only"
    return "reject_regression"


def _unit_summary(unit: dict[str, typing.Any]) -> dict[str, typing.Any]:
    return {
        "label": str(unit.get("label", "")),
        "decision": str(unit.get("decision", "")),
        "value": unit.get("value"),
        "unit": unit.get("unit"),
        "evidence_session_id": str(unit.get("evidence_session_id", "")),
        "evidence_span": str(unit.get("evidence_span", "")),
        "reason_code": str(unit.get("reason_code", "")),
    }


def _optional_float(value: typing.Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _numbers_match(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return abs(left - right) < 0.000001


def _load_json(path: str | pathlib.Path) -> dict[str, typing.Any]:
    return typing.cast(
        "dict[str, typing.Any]",
        json.loads(pathlib.Path(path).read_text(encoding="utf-8")),
    )


def _load_json_list(path: str | pathlib.Path) -> list[dict[str, typing.Any]]:
    return typing.cast(
        "list[dict[str, typing.Any]]",
        json.loads(pathlib.Path(path).read_text(encoding="utf-8")),
    )


def _write_json(path: pathlib.Path, artifact: typing.Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(artifact, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _parse_case_ids(raw: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in re.split(r"[,\s]+", raw)
        if item.strip()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-heldout", action="store_true")
    parser.add_argument("--build-reasoning-diagnostic", action="store_true")
    parser.add_argument("--build-policy-payloads", action="store_true")
    parser.add_argument("--build-policy-ab-report", action="store_true")
    parser.add_argument("--build-budget-ab-report", action="store_true")
    parser.add_argument("--build-budget-regression-diagnostic", action="store_true")
    parser.add_argument("--build-span-survival", action="store_true")
    parser.add_argument(
        "--longmemeval-s",
        type=pathlib.Path,
        default=DEFAULT_LONGMEMEVAL_S_PATH,
    )
    parser.add_argument(
        "--fail18-manifest",
        type=pathlib.Path,
        default=ambiguity_fail18.DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--heldout-payloads",
        type=pathlib.Path,
        default=DEFAULT_HELDOUT_PAYLOADS_PATH,
    )
    parser.add_argument(
        "--heldout-manifest",
        type=pathlib.Path,
        default=DEFAULT_HELDOUT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--heldout-provenance",
        type=pathlib.Path,
        default=DEFAULT_HELDOUT_PROVENANCE_PATH,
    )
    parser.add_argument("--heldout-seed", default=DEFAULT_HELDOUT_SEED)
    parser.add_argument("--heldout-limit", type=int, default=DEFAULT_HELDOUT_LIMIT)
    parser.add_argument("--heldout-top-k", type=int, default=DEFAULT_HELDOUT_TOP_K)
    parser.add_argument(
        "--heldout-session-char-limit",
        type=int,
        default=DEFAULT_HELDOUT_SESSION_CHAR_LIMIT,
    )
    parser.add_argument(
        "--stable-wrong-diagnostic",
        type=pathlib.Path,
        default=DEFAULT_STABLE_WRONG_DIAGNOSTIC_PATH,
    )
    parser.add_argument(
        "--outputs",
        type=pathlib.Path,
        default=DEFAULT_WITNESS_OUTPUTS_PATH,
    )
    parser.add_argument(
        "--reasoning-diagnostic",
        type=pathlib.Path,
        default=DEFAULT_REASONING_MECHANISM_DIAGNOSTIC_PATH,
    )
    parser.add_argument("--source-payloads", type=pathlib.Path)
    parser.add_argument(
        "--payloads",
        type=pathlib.Path,
        default=answer_unit_witness.DEFAULT_PAYLOADS_PATH,
    )
    parser.add_argument("--candidate-payloads", type=pathlib.Path)
    parser.add_argument("--baseline-outputs", type=pathlib.Path)
    parser.add_argument("--candidate-outputs", type=pathlib.Path)
    parser.add_argument("--policy-payloads", type=pathlib.Path)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--fail18-baseline-report", type=pathlib.Path)
    parser.add_argument("--fail18-candidate-report", type=pathlib.Path)
    parser.add_argument("--heldout-baseline-report", type=pathlib.Path)
    parser.add_argument("--heldout-candidate-report", type=pathlib.Path)
    parser.add_argument(
        "--policy-ab-report",
        type=pathlib.Path,
        default=DEFAULT_INCLUSION_POLICY_AB_REPORT_PATH,
    )
    parser.add_argument(
        "--budget-ab-report",
        type=pathlib.Path,
        default=DEFAULT_PAYLOAD_BUDGET_AB_REPORT_PATH,
    )
    parser.add_argument(
        "--budget-regression-diagnostic",
        type=pathlib.Path,
        default=DEFAULT_PAYLOAD_BUDGET_REGRESSION_DIAGNOSTIC_PATH,
    )
    parser.add_argument("--baseline-label", default="2500")
    parser.add_argument("--candidate-label", default="8000")
    parser.add_argument(
        "--gold-span-needles",
        type=pathlib.Path,
        default=DEFAULT_FAIL18_GOLD_SPAN_NEEDLES_PATH,
    )
    parser.add_argument(
        "--span-survival-report",
        type=pathlib.Path,
        default=DEFAULT_FAIL18_SPAN_SURVIVAL_REPORT_PATH,
    )
    parser.add_argument(
        "--corpus",
        type=pathlib.Path,
        default=ambiguity_fail18.DEFAULT_CORPUS,
    )
    args = parser.parse_args(argv)
    requested = any(
        (
            args.build_heldout,
            args.build_reasoning_diagnostic,
            args.build_policy_payloads,
            args.build_policy_ab_report,
            args.build_budget_ab_report,
            args.build_budget_regression_diagnostic,
            args.build_span_survival,
        )
    )
    if not requested:
        parser.print_help()
        return 2
    if args.build_heldout:
        artifacts = build_longmemeval_s_heldout_artifacts(
            dataset_path=args.longmemeval_s,
            fail18_manifest_path=args.fail18_manifest,
            seed=args.heldout_seed,
            limit=args.heldout_limit,
            top_k=args.heldout_top_k,
            session_char_limit=args.heldout_session_char_limit,
        )
        _write_json(args.heldout_payloads, artifacts["payloads"])
        _write_json(args.heldout_manifest, artifacts["manifest"])
        _write_json(args.heldout_provenance, artifacts["provenance"])
    if args.build_reasoning_diagnostic:
        artifact = build_fail18_reasoning_mechanism_diagnostic(
            stable_wrong_diagnostic_path=args.stable_wrong_diagnostic,
            outputs_path=args.outputs,
        )
        _write_json(args.reasoning_diagnostic, artifact)
    if args.build_policy_payloads:
        if args.source_payloads is None or args.policy_payloads is None:
            parser.error(
                "--build-policy-payloads requires --source-payloads and "
                "--policy-payloads"
            )
        artifact = build_inclusion_policy_payload_artifact(
            source_payloads_path=args.source_payloads,
            case_ids=_parse_case_ids(args.case_ids),
        )
        _write_json(args.policy_payloads, artifact)
    if args.build_policy_ab_report:
        required_reports = (
            args.fail18_baseline_report,
            args.fail18_candidate_report,
            args.heldout_baseline_report,
            args.heldout_candidate_report,
        )
        if any(path is None for path in required_reports):
            parser.error(
                "--build-policy-ab-report requires fail18 and heldout baseline/"
                "candidate report paths"
            )
        artifact = build_inclusion_policy_ab_report(
            fail18_baseline_report_path=typing.cast(
                "pathlib.Path", args.fail18_baseline_report
            ),
            fail18_candidate_report_path=typing.cast(
                "pathlib.Path", args.fail18_candidate_report
            ),
            heldout_baseline_report_path=typing.cast(
                "pathlib.Path", args.heldout_baseline_report
            ),
            heldout_candidate_report_path=typing.cast(
                "pathlib.Path", args.heldout_candidate_report
            ),
            target_case_ids=_parse_case_ids(args.case_ids)
            or INCLUSION_POLICY_TARGET_CASE_IDS,
        )
        _write_json(args.policy_ab_report, artifact)
    if args.build_budget_ab_report:
        required_reports = (
            args.fail18_baseline_report,
            args.fail18_candidate_report,
            args.heldout_baseline_report,
            args.heldout_candidate_report,
        )
        if any(path is None for path in required_reports):
            parser.error(
                "--build-budget-ab-report requires fail18 and heldout baseline/"
                "candidate report paths"
            )
        artifact = build_payload_budget_ab_report(
            fail18_baseline_report_path=typing.cast(
                "pathlib.Path", args.fail18_baseline_report
            ),
            fail18_candidate_report_path=typing.cast(
                "pathlib.Path", args.fail18_candidate_report
            ),
            heldout_baseline_report_path=typing.cast(
                "pathlib.Path", args.heldout_baseline_report
            ),
            heldout_candidate_report_path=typing.cast(
                "pathlib.Path", args.heldout_candidate_report
            ),
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
        )
        _write_json(args.budget_ab_report, artifact)
    if args.build_budget_regression_diagnostic:
        if args.candidate_payloads is None:
            parser.error(
                "--build-budget-regression-diagnostic requires "
                "--candidate-payloads"
            )
        if args.baseline_outputs is None:
            parser.error(
                "--build-budget-regression-diagnostic requires "
                "--baseline-outputs"
            )
        if args.candidate_outputs is None:
            parser.error(
                "--build-budget-regression-diagnostic requires "
                "--candidate-outputs"
            )
        artifact = build_payload_budget_regression_diagnostic(
            budget_ab_report_path=args.budget_ab_report,
            baseline_payloads_path=args.payloads,
            candidate_payloads_path=args.candidate_payloads,
            baseline_outputs_path=args.baseline_outputs,
            candidate_outputs_path=args.candidate_outputs,
        )
        _write_json(args.budget_regression_diagnostic, artifact)
    if args.build_span_survival:
        artifact = build_fail18_span_survival_report(
            payloads_path=args.payloads,
            gold_span_needles_path=args.gold_span_needles,
            corpus_path=args.corpus,
            stable_wrong_diagnostic_path=args.stable_wrong_diagnostic,
        )
        _write_json(args.span_survival_report, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
