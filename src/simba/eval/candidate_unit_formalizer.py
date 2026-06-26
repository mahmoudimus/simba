"""D&C formalizer for candidate-unit evidence sessions."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import pathlib
import re
import typing

from simba.eval import interpretation_runner

FORMALIZER_PROMPT_VERSION = "candidate_unit_formalizer_recursive_v2"
DEFAULT_CANDIDATE_PAYLOADS_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_payloads_candidate_unit_coverage_v1.json"
)
DEFAULT_FORMALIZER_PAYLOADS_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_formalizer_payloads_candidate_unit_coverage_v1.json"
)
DEFAULT_FORMALIZER_OUTPUTS_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_formalizer_outputs_candidate_unit_coverage_v1.jsonl"
)
DEFAULT_CANDIDATE_OUTPUTS_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_outputs_candidate_unit_coverage_v1.jsonl"
)
DEFAULT_PAYLOAD_PROVENANCE_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_payloads_candidate_unit_coverage_v1_provenance.json"
)
DEFAULT_REPORT_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_formalizer_report_candidate_unit_coverage_v1.json"
)

PARSE_STATUS_PARSED = "parsed"
PARSE_STATUS_INVALID_JSON = "invalid_json"
PARSE_STATUS_INVALID_SCHEMA = "invalid_schema"
PARSE_STATUS_EMPTY = "empty"

_ALLOWED_PREDICATES = {
    "action",
    "coreference",
    "distinct",
    "entity",
    "event",
    "object_type",
    "other",
    "property",
    "quantity",
    "relation",
    "sortal",
    "status",
    "time",
    "value",
    # Legacy v1 predicates are accepted so old provider outputs remain readable.
    "acquisition",
    "baking_event",
    "clothing_obligation",
    "fundraiser",
    "wedding_event",
}
_ALLOWED_POLARITIES = {
    "supports_answer",
    "contradicts_answer",
    "context_only",
    "irrelevant",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MATCH_STOPWORDS = {
    "already",
    "album",
    "answer",
    "bake",
    "baked",
    "baking",
    "benefit",
    "bread",
    "cake",
    "candidate",
    "charity",
    "clothing",
    "cookies",
    "event",
    "evidence",
    "from",
    "fundraiser",
    "included",
    "item",
    "obligation",
    "pick",
    "pickup",
    "reason",
    "return",
    "same",
    "session",
    "status",
    "unit",
    "user",
    "vinyl",
    "wedding",
    "weddings",
    "with",
}

_PREDICATE_ROLE_ARGUMENTS = {
    "action": ("subject", "object", "verb", "location", "status"),
    "coreference": ("entity", "same_as", "reason"),
    "distinct": ("a", "b", "reason"),
    "entity": ("entity", "name", "type"),
    "event": ("event", "type", "participants", "location"),
    "object_type": ("entity", "type"),
    "property": ("entity", "property", "value"),
    "quantity": ("entity", "attribute", "value", "unit"),
    "relation": ("source", "relation", "target"),
    "sortal": ("entity", "type", "source", "antecedent", "licensed_by"),
    "status": ("entity", "status"),
    "time": ("entity", "date", "time_window"),
    "value": ("entity", "attribute", "value", "unit"),
    "acquisition": ("item",),
    "baking_event": ("item",),
    "clothing_obligation": ("item", "location"),
    "fundraiser": ("amount", "beneficiary", "event"),
    "wedding_event": ("event", "participants"),
}

RECURSIVE_FACT_SCHEMA = (
    {
        "predicate": "action",
        "meaning": (
            "A grounded action or obligation. Use action(user, object, verb) "
            "style arguments; do not decide whether it answers the question."
        ),
        "arguments": ["subject", "object", "verb", "location", "status"],
    },
    {
        "predicate": "event",
        "meaning": "A grounded event mention with type, participants, place, or date.",
        "arguments": ["event", "type", "participants", "location", "date", "status"],
    },
    {
        "predicate": "object_type",
        "meaning": "A type assertion for an entity or object.",
        "arguments": ["entity", "type"],
    },
    {
        "predicate": "sortal",
        "meaning": (
            "A sortal/type inherited by bridging or ellipsis without claiming "
            "same-token identity. Example: 'new pair' may inherit type 'boots' "
            "from an antecedent pair while remaining a distinct entity."
        ),
        "arguments": ["entity", "type", "source", "antecedent", "licensed_by"],
    },
    {
        "predicate": "relation",
        "meaning": "A grounded binary relation between two entities.",
        "arguments": ["source", "relation", "target"],
    },
    {
        "predicate": "value",
        "meaning": "A scalar value or amount attached to an entity or event.",
        "arguments": ["entity", "attribute", "value", "unit"],
    },
    {
        "predicate": "time",
        "meaning": "A date, interval, or temporal qualifier attached to an entity.",
        "arguments": ["entity", "date", "time_window"],
    },
    {
        "predicate": "status",
        "meaning": "A state such as pending, completed, cancelled, owned, or borrowed.",
        "arguments": ["entity", "status"],
    },
    {
        "predicate": "coreference",
        "meaning": (
            "A local same-entity identity claim inside this evidence session "
            "only. Do not use this for bridging, type inheritance, replacement, "
            "or contrastive 'new/another/old' mentions."
        ),
        "arguments": ["entity", "same_as", "reason"],
    },
    {
        "predicate": "distinct",
        "meaning": (
            "A grounded non-identity claim between two entities, especially "
            "licensed by contrastive mentions such as new, another, different, "
            "replacement, or old."
        ),
        "arguments": ["a", "b", "reason"],
    },
    {
        "predicate": "other",
        "meaning": "A grounded fact that does not fit another predicate.",
        "arguments": ["description"],
    },
)

OFFLINE_LEXICON_CONTEXT = {
    "wordnet": ".simba/lexicon/nltk_data/corpora/wordnet.zip",
    "omw": ".simba/lexicon/nltk_data/corpora/omw-1.4.zip",
    "framenet": ".simba/lexicon/nltk_data/corpora/framenet_v17",
    "normalized_jsonl": ".simba/lexicon/nltk_lexicon.jsonl",
    "lancedb": ".simba/lexicon/lexicon.lance",
    "note": (
        "Offline lexicon resources may ratify terms later. Provider outputs must "
        "remain local evidence facts, not ontology lookups or final answers."
    ),
}


@dataclasses.dataclass(frozen=True)
class FormalFact:
    fact_id: str
    predicate: str
    arguments: dict[str, typing.Any]
    evidence_span: str
    confidence: float
    polarity: str = ""

    def to_dict(self) -> dict[str, typing.Any]:
        payload = {
            "fact_id": self.fact_id,
            "predicate": self.predicate,
            "arguments": dict(self.arguments),
            "evidence_span": self.evidence_span,
            "confidence": self.confidence,
        }
        if self.polarity:
            payload["polarity"] = self.polarity
        return payload


@dataclasses.dataclass(frozen=True)
class FormalizerParseResult:
    formalizer_id: str
    case_id: str
    evidence_session_id: str
    parse_status: str
    facts: tuple[FormalFact, ...]
    notes: str
    parse_errors: tuple[str, ...]

    def to_output_dict(
        self,
        *,
        provider: str,
        prompt_version: str,
        raw_output: str,
    ) -> dict[str, typing.Any]:
        return {
            "formalizer_id": self.formalizer_id,
            "case_id": self.case_id,
            "evidence_session_id": self.evidence_session_id,
            "provider": provider,
            "prompt_version": prompt_version,
            "raw_output": raw_output,
            "parse_status": self.parse_status,
            "facts": [fact.to_dict() for fact in self.facts],
            "notes": self.notes,
            "parse_errors": list(self.parse_errors),
        }


def build_formalizer_payload_artifact(
    *,
    candidate_payloads_path: str | pathlib.Path = DEFAULT_CANDIDATE_PAYLOADS_PATH,
) -> dict[str, typing.Any]:
    candidate_payloads = _load_json(candidate_payloads_path)
    payloads = []
    for candidate_payload in candidate_payloads.get("payloads", []):
        if not isinstance(candidate_payload, dict):
            continue
        case = candidate_payload.get("case", {})
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id", ""))
        question = str(case.get("question", ""))
        for evidence in case.get("evidence_sessions", []):
            if not isinstance(evidence, dict):
                continue
            payloads.append(
                build_formalizer_payload(
                    case_id=case_id,
                    question=question,
                    evidence_session=evidence,
                    source_prompt_version=str(
                        candidate_payloads.get("prompt_version", "")
                    ),
                )
            )
    return {
        "name": "fail18-candidate-unit-dnc-formalizer-payloads",
        "artifact_kind": "provider_payloads",
        "gate": "candidate_unit_formalizer",
        "gate_status": "formalizer_payloads_only_not_run",
        "prompt_version": FORMALIZER_PROMPT_VERSION,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_candidate_payloads": str(candidate_payloads_path),
        "provider_visibility": {
            "gold_answer_visible": False,
            "gold_value_visible": False,
            "failure_mode_visible": False,
            "answer_session_ids_visible": False,
            "raw_session_ids_visible": False,
            "final_answer_requested": False,
            "answer_support_labels_requested": False,
        },
        "total": len(payloads),
        "case_ids": sorted({str(payload["case_id"]) for payload in payloads}),
        "payloads": payloads,
    }


def build_formalizer_payload(
    *,
    case_id: str,
    question: str,
    evidence_session: dict[str, typing.Any],
    source_prompt_version: str,
) -> dict[str, typing.Any]:
    evidence_session_id = str(evidence_session.get("session_id", ""))
    return {
        "task": (
            "Formalize this one evidence session into neutral recursive facts "
            "only. Do not compute the answer and do not reason across sessions."
        ),
        "prompt_version": FORMALIZER_PROMPT_VERSION,
        "source_prompt_version": source_prompt_version,
        "formalizer_id": f"{case_id}::{evidence_session_id}",
        "case_id": case_id,
        "question": question,
        "fact_schema": list(RECURSIVE_FACT_SCHEMA),
        "offline_lexicon_context": OFFLINE_LEXICON_CONTEXT,
        "contract": [
            "Emit facts grounded only in this evidence_session text.",
            (
                "Use recursive fact shapes such as action(subject, object, verb), "
                "event(event, type), object_type(entity, type), "
                "sortal(entity, type, source, antecedent, licensed_by), "
                "relation(source, relation, target), value(entity, attribute, value)."
            ),
            (
                "Use coreference only for true same-entity identity. Do not use "
                "coreference for bridging, sortal inheritance, replacement, or "
                "contrastive mentions such as new, another, different, or old."
            ),
            (
                "For incomplete relational nouns such as 'new pair', recover the "
                "missing sortal with sortal(..., source='bridging', ...). If the "
                "mention is a replacement or another token, also emit distinct "
                "and, when grounded, relation(source, relation='replaces', target)."
            ),
            (
                "Keep entity symbols and type symbols disjoint. Relation endpoints, "
                "distinct arguments, sortal.entity, and sortal.antecedent are "
                "entity-sorted; sortal.type and object_type.type are type-sorted. "
                "Do not use a type word such as 'boots' as the entity endpoint for "
                "the old pair. Use a local handle such as pair_1, then type it with "
                "sortal(entity='pair_1', type='boots')."
            ),
            (
                "For unknown optional argument values, omit the argument or use "
                "JSON null. Do not use an empty string as an unknown placeholder."
            ),
            (
                "Do not emit final answers, counts, sums, inclusion decisions, "
                "exclusion decisions, or cross-session merge decisions."
            ),
            (
                "Do not emit answer-path labels, inclusion judgments, exclusion "
                "judgments, correctness judgments, or relevance grades."
            ),
            "Keep evidence_span short and copied from the evidence text.",
        ],
        "output_schema": {
            "formalizer_id": f"{case_id}::{evidence_session_id}",
            "case_id": case_id,
            "evidence_session_id": evidence_session_id,
            "facts": [
                {
                    "fact_id": "stable string",
                    "predicate": (
                        "action|event|object_type|sortal|relation|value|time|"
                        "status|coreference|distinct|entity|property|quantity|"
                        "other"
                    ),
                    "arguments": {"key": "value or null"},
                    "evidence_span": "short exact span",
                    "confidence": 0.0,
                }
            ],
            "notes": "short note or empty string",
        },
        "evidence_session": {
            "session_id": evidence_session_id,
            "date": str(evidence_session.get("date", "")),
            "selection_rank": evidence_session.get("selection_rank"),
            "selection_score": evidence_session.get("selection_score"),
            "text": str(evidence_session.get("text", "")),
        },
    }


def parse_formalizer_response(
    raw_output: str,
    *,
    expected_formalizer_id: str | None = None,
) -> FormalizerParseResult:
    fallback_id = expected_formalizer_id or ""
    if not raw_output.strip():
        return _invalid_result(
            formalizer_id=fallback_id,
            status=PARSE_STATUS_EMPTY,
            errors=("empty provider output",),
        )
    try:
        decoded = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return _invalid_result(
            formalizer_id=fallback_id,
            status=PARSE_STATUS_INVALID_JSON,
            errors=(f"invalid JSON: {exc.msg} at char {exc.pos}",),
        )
    if not isinstance(decoded, dict):
        return _invalid_result(
            formalizer_id=fallback_id,
            status=PARSE_STATUS_INVALID_SCHEMA,
            errors=("root output must be a JSON object",),
        )
    return parse_formalizer_object(
        decoded,
        expected_formalizer_id=expected_formalizer_id,
    )


def parse_formalizer_object(
    raw: dict[str, typing.Any],
    *,
    expected_formalizer_id: str | None = None,
) -> FormalizerParseResult:
    errors: list[str] = []
    formalizer_id = _string_field(raw, "formalizer_id", errors)
    if (
        expected_formalizer_id is not None
        and formalizer_id
        and formalizer_id != expected_formalizer_id
    ):
        errors.append(
            f"formalizer_id {formalizer_id!r} does not match expected "
            f"{expected_formalizer_id!r}"
        )
    case_id = _string_field(raw, "case_id", errors)
    evidence_session_id = _string_field(raw, "evidence_session_id", errors)
    notes = _string_field(raw, "notes", errors, allow_empty=True)
    facts = _facts(raw.get("facts"), errors)
    if errors:
        return _invalid_result(
            formalizer_id=formalizer_id or expected_formalizer_id or "",
            case_id=case_id,
            evidence_session_id=evidence_session_id,
            status=PARSE_STATUS_INVALID_SCHEMA,
            errors=tuple(errors),
        )
    return FormalizerParseResult(
        formalizer_id=formalizer_id,
        case_id=case_id,
        evidence_session_id=evidence_session_id,
        parse_status=PARSE_STATUS_PARSED,
        facts=tuple(facts),
        notes=notes,
        parse_errors=(),
    )


def build_provider_prompt(payload: dict[str, typing.Any]) -> str:
    return (
        "Complete this candidate-unit formalizer task.\n"
        "Return exactly one strict JSON object matching output_schema. "
        "No markdown, prose, comments, or trailing text.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def run_payloads(
    *,
    payload_artifact: dict[str, typing.Any],
    provider_command: str = interpretation_runner.DEFAULT_PROVIDER_COMMAND,
    timeout_seconds: int = interpretation_runner.DEFAULT_TIMEOUT_SECONDS,
    limit: int = 0,
) -> list[dict[str, typing.Any]]:
    payloads = list(payload_artifact.get("payloads", []))
    if limit > 0:
        payloads = payloads[:limit]
    rows: list[dict[str, typing.Any]] = []
    prompt_version = str(
        payload_artifact.get("prompt_version", FORMALIZER_PROMPT_VERSION)
    )
    for payload in payloads:
        formalizer_id = str(payload.get("formalizer_id", ""))
        provider_result = interpretation_runner.run_provider(
            command=provider_command,
            prompt=build_provider_prompt(payload),
            timeout_seconds=timeout_seconds,
        )
        parsed = parse_formalizer_response(
            provider_result.raw_output,
            expected_formalizer_id=formalizer_id,
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


def build_formalizer_report(
    *,
    formalizer_payload_artifact: dict[str, typing.Any],
    formalizer_rows: list[dict[str, typing.Any]],
    candidate_rows: list[dict[str, typing.Any]],
    payload_provenance_path: str | pathlib.Path = DEFAULT_PAYLOAD_PROVENANCE_PATH,
    outputs_path: str | pathlib.Path = DEFAULT_FORMALIZER_OUTPUTS_PATH,
    candidate_outputs_path: str | pathlib.Path = DEFAULT_CANDIDATE_OUTPUTS_PATH,
) -> dict[str, typing.Any]:
    expected_ids = [
        str(payload.get("formalizer_id", ""))
        for payload in formalizer_payload_artifact.get("payloads", [])
        if isinstance(payload, dict)
    ]
    parsed_rows = [
        row
        for row in formalizer_rows
        if row.get("parse_status") == PARSE_STATUS_PARSED
        and not _provider_failed(row)
    ]
    rows_by_session = {
        (
            str(row.get("case_id", "")),
            str(row.get("evidence_session_id", "")),
        ): row
        for row in parsed_rows
    }
    candidate_rows_by_id = {
        str(row.get("case_id", "")): row
        for row in candidate_rows
        if isinstance(row, dict)
    }
    provenance = _load_json(payload_provenance_path).get("evidence_provenance", {})
    case_ids = sorted(
        {
            str(payload.get("case_id", ""))
            for payload in formalizer_payload_artifact.get("payloads", [])
            if isinstance(payload, dict)
        }
    )
    cases = [
        _case_report(
            case_id=case_id,
            candidate_row=candidate_rows_by_id.get(case_id, {}),
            rows_by_session=rows_by_session,
            case_provenance=typing.cast(
                "dict[str, dict[str, typing.Any]]",
                provenance.get(case_id, {}),
            ),
        )
        for case_id in case_ids
    ]
    fact_predicate_counts: collections.Counter[str] = collections.Counter()
    fact_role_counts: collections.Counter[str] = collections.Counter()
    for row in parsed_rows:
        for fact in row.get("facts", []):
            if not isinstance(fact, dict):
                continue
            fact_predicate_counts[str(fact.get("predicate", ""))] += 1
            fact_role_counts[_fact_role(fact)] += 1
    observed_ids = [str(row.get("formalizer_id", "")) for row in formalizer_rows]
    observed_counts = collections.Counter(observed_ids)
    return {
        "name": "fail18-candidate-unit-dnc-formalizer-report",
        "artifact_kind": "candidate_unit_formalizer_report",
        "gate": "candidate_unit_formalizer",
        "gate_status": "formalizer_report_complete",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "prompt_version": str(
            formalizer_payload_artifact.get(
                "prompt_version",
                FORMALIZER_PROMPT_VERSION,
            )
        ),
        "source_formalizer_outputs": str(outputs_path),
        "source_candidate_outputs": str(candidate_outputs_path),
        "source_payload_provenance": str(payload_provenance_path),
        "summary": {
            "rows_total": len(cases),
            "evidence_payloads_total": len(expected_ids),
            "provider_rows_total": len(formalizer_rows),
            "provider_rows_parsed": len(parsed_rows),
            "provider_rows_failed": sum(
                1 for row in formalizer_rows if _provider_failed(row)
            ),
            "provider_rows_timed_out": sum(
                1 for row in formalizer_rows if bool(row.get("provider_timed_out"))
            ),
            "missing_formalizer_ids": sorted(set(expected_ids) - set(observed_ids)),
            "extra_formalizer_ids": sorted(set(observed_ids) - set(expected_ids)),
            "duplicate_formalizer_ids": sorted(
                row_id for row_id, count in observed_counts.items() if count > 1
            ),
            "rows_with_provider_facts": sum(
                1 for case in cases if case["provider_fact_count"] > 0
            ),
            "rows_with_missing_provider_facts": sum(
                1 for case in cases if case["missing_provider_fact_sessions"]
            ),
            "fact_predicate_counts": dict(sorted(fact_predicate_counts.items())),
            "fact_role_counts": dict(sorted(fact_role_counts.items())),
            "candidate_units_without_supporting_facts": sum(
                len(case["candidate_units_without_supporting_facts"])
                for case in cases
            ),
            "candidate_units_with_contradicting_facts": sum(
                len(case["candidate_units_with_contradicting_facts"])
                for case in cases
            ),
            "excluded_units_with_supporting_facts": sum(
                len(case["excluded_units_with_supporting_facts"])
                for case in cases
            ),
            "merged_units_with_supporting_facts": sum(
                len(case["merged_units_with_supporting_facts"])
                for case in cases
            ),
        },
        "cases": cases,
        "decision": {
            "next_slice": "provider_independent_scope_verifier",
            "reason": (
                "This report is non-oracle formalizer diagnostics only. Build "
                "a provider-independent scope verifier before using facts to "
                "reject candidate units or recompute answers."
            ),
        },
    }


def _case_report(
    *,
    case_id: str,
    candidate_row: dict[str, typing.Any],
    rows_by_session: dict[tuple[str, str], dict[str, typing.Any]],
    case_provenance: dict[str, dict[str, typing.Any]],
) -> dict[str, typing.Any]:
    candidate_units = [
        unit
        for unit in candidate_row.get("candidate_units", [])
        if isinstance(unit, dict)
    ]
    evidence_session_ids = sorted(case_provenance)
    facts_by_session = {
        evidence_id: list(
            rows_by_session.get((case_id, evidence_id), {}).get("facts", [])
        )
        for evidence_id in evidence_session_ids
    }
    missing_provider_fact_sessions = [
        evidence_id
        for evidence_id in evidence_session_ids
        if (case_id, evidence_id) not in rows_by_session
        or not facts_by_session[evidence_id]
    ]
    unit_reports = [
        _unit_report(
            unit=unit,
            facts_by_session=facts_by_session,
            case_provenance=case_provenance,
        )
        for unit in candidate_units
    ]
    return {
        "case_id": case_id,
        "provider_fact_count": sum(len(facts) for facts in facts_by_session.values()),
        "evidence_session_count": len(evidence_session_ids),
        "missing_provider_fact_sessions": missing_provider_fact_sessions,
        "candidate_units_without_supporting_facts": [
            unit
            for unit in unit_reports
            if unit["status"] == "included"
            and not unit["supporting_fact_ids"]
        ],
        "candidate_units_with_contradicting_facts": [
            unit
            for unit in unit_reports
            if unit["status"] == "included"
            and unit["contradicting_fact_ids"]
        ],
        "excluded_units_with_supporting_facts": [
            unit
            for unit in unit_reports
            if unit["status"] == "excluded"
            and unit["supporting_fact_ids"]
        ],
        "merged_units_with_supporting_facts": [
            unit
            for unit in unit_reports
            if unit["status"] == "merged"
            and unit["supporting_fact_ids"]
        ],
        "candidate_units": unit_reports,
    }


def _unit_report(
    *,
    unit: dict[str, typing.Any],
    facts_by_session: dict[str, list[typing.Any]],
    case_provenance: dict[str, dict[str, typing.Any]],
) -> dict[str, typing.Any]:
    evidence_session_ids = [
        str(item) for item in unit.get("evidence_session_ids", [])
    ]
    linked_facts = [
        fact
        for evidence_id in evidence_session_ids
        for fact in facts_by_session.get(evidence_id, [])
        if isinstance(fact, dict)
        and _fact_matches_unit(unit=unit, fact=fact)
    ]
    linked_fact_ids = [str(fact.get("fact_id", "")) for fact in linked_facts]
    supporting_fact_ids = [
        str(fact.get("fact_id", ""))
        for fact in linked_facts
        if _fact_supports_unit(fact)
    ]
    contradicting_fact_ids = [
        str(fact.get("fact_id", ""))
        for fact in linked_facts
        if _fact_contradicts_unit(fact)
    ]
    raw_session_ids = [
        str(case_provenance.get(evidence_id, {}).get("raw_session_id", ""))
        for evidence_id in evidence_session_ids
    ]
    return {
        "unit_id": str(unit.get("unit_id", "")),
        "label": str(unit.get("label", "")),
        "status": str(unit.get("status", "")),
        "merge_target": unit.get("merge_target"),
        "reason_code": str(unit.get("reason_code", "")),
        "value": unit.get("value"),
        "unit": unit.get("unit"),
        "evidence_session_ids": evidence_session_ids,
        "raw_session_ids": sorted({item for item in raw_session_ids if item}),
        "linked_fact_ids": linked_fact_ids,
        "supporting_fact_ids": supporting_fact_ids,
        "contradicting_fact_ids": contradicting_fact_ids,
    }


def _fact_matches_unit(
    *,
    unit: dict[str, typing.Any],
    fact: dict[str, typing.Any],
) -> bool:
    unit_tokens = _text_tokens(
        " ".join(
            [
                str(unit.get("unit_id", "")),
                str(unit.get("label", "")),
                str(unit.get("unit", "")),
                " ".join(str(item) for item in unit.get("evidence_spans", [])),
            ]
        )
    )
    fact_tokens = _text_tokens(
        _role_fact_text(fact)
    )
    if not unit_tokens or not fact_tokens:
        return False
    overlap = unit_tokens & fact_tokens
    if len(overlap) >= 2:
        return True
    return bool(overlap) and any(len(token) >= 5 for token in overlap)


def _text_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) >= 3 and token not in _MATCH_STOPWORDS
    }


def _argument_value_text(arguments: typing.Any) -> str:
    if not isinstance(arguments, dict):
        return ""
    return " ".join(str(value) for value in arguments.values() if value is not None)


def _role_fact_text(fact: dict[str, typing.Any]) -> str:
    arguments = fact.get("arguments", {})
    predicate = str(fact.get("predicate", ""))
    if not isinstance(arguments, dict):
        return str(fact.get("evidence_span", ""))
    keys = _PREDICATE_ROLE_ARGUMENTS.get(predicate)
    if not keys:
        return " ".join(
            [
                str(fact.get("evidence_span", "")),
                _argument_value_text(arguments),
            ]
        )
    values = [str(arguments.get(key, "")) for key in keys]
    return " ".join(value for value in values if value)


def _fact_role(fact: dict[str, typing.Any]) -> str:
    polarity = str(fact.get("polarity", "")).strip()
    if polarity:
        return f"legacy_{polarity}"
    if _fact_contradicts_unit(fact):
        return "negative_fact"
    return "neutral_fact"


def _fact_supports_unit(fact: dict[str, typing.Any]) -> bool:
    polarity = str(fact.get("polarity", "")).strip()
    if polarity:
        return polarity == "supports_answer"
    return not _fact_contradicts_unit(fact)


def _fact_contradicts_unit(fact: dict[str, typing.Any]) -> bool:
    polarity = str(fact.get("polarity", "")).strip()
    if polarity:
        return polarity == "contradicts_answer"
    text = " ".join(
        [
            str(fact.get("predicate", "")),
            _argument_value_text(fact.get("arguments", {})),
            str(fact.get("evidence_span", "")),
        ]
    ).lower()
    return any(
        marker in text
        for marker in (
            "cancelled",
            "canceled",
            "not ",
            "never",
            "already returned",
            "borrowed",
            "rented",
            "mistaken",
        )
    )


def _facts(raw_facts: typing.Any, errors: list[str]) -> list[FormalFact]:
    if not isinstance(raw_facts, list):
        errors.append("facts must be a list")
        return []
    facts: list[FormalFact] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_facts):
        if not isinstance(raw, dict):
            errors.append(f"facts[{index}] must be a JSON object")
            continue
        fact_errors: list[str] = []
        fact_id = _string_field(raw, "fact_id", fact_errors)
        if fact_id in seen:
            fact_errors.append(f"duplicate fact_id {fact_id!r}")
        seen.add(fact_id)
        predicate = _string_field(raw, "predicate", fact_errors)
        if predicate and predicate not in _ALLOWED_PREDICATES:
            fact_errors.append(f"unknown predicate {predicate!r}")
        arguments = _arguments_field(raw, "arguments", fact_errors)
        polarity = _optional_string_field(raw, "polarity", fact_errors)
        if polarity and polarity not in _ALLOWED_POLARITIES:
            fact_errors.append(f"unknown legacy polarity {polarity!r}")
        evidence_span = _string_field(raw, "evidence_span", fact_errors)
        confidence = _number_field(raw, "confidence", fact_errors)
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            fact_errors.append("confidence must be between 0.0 and 1.0")
        if fact_errors:
            errors.extend(f"facts[{index}]: {error}" for error in fact_errors)
            continue
        facts.append(
            FormalFact(
                fact_id=fact_id,
                predicate=predicate,
                arguments=arguments,
                evidence_span=evidence_span,
                confidence=typing.cast("float", confidence),
                polarity=polarity,
            )
        )
    return facts


def _string_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
    *,
    allow_empty: bool = False,
) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        errors.append(f"{key} must be a string")
        return ""
    if not allow_empty and not value.strip():
        errors.append(f"{key} must be non-empty")
        return ""
    return value


def _optional_string_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
) -> str:
    if key not in raw:
        return ""
    value = raw.get(key)
    if not isinstance(value, str):
        errors.append(f"{key} must be a string when present")
        return ""
    return value.strip()


def _arguments_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
) -> dict[str, typing.Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be an object")
        return {}
    bad_keys = [item for item in value if not isinstance(item, str)]
    if bad_keys:
        errors.append(f"{key} keys must be strings")
        return {}
    bad_values = [
        item
        for item in value.values()
        if isinstance(item, list | dict)
    ]
    if bad_values:
        errors.append(f"{key} values must be scalar or null")
        return {}
    return {
        str(item_key): _normalize_argument_value(item_value)
        for item_key, item_value in value.items()
    }


def _normalize_argument_value(value: typing.Any) -> typing.Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _number_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
) -> float | None:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        errors.append(f"{key} must be a number")
        return None
    return float(value)


def _invalid_result(
    *,
    formalizer_id: str,
    status: str,
    errors: tuple[str, ...],
    case_id: str = "",
    evidence_session_id: str = "",
) -> FormalizerParseResult:
    if "::" in formalizer_id and (not case_id or not evidence_session_id):
        case_id, evidence_session_id = formalizer_id.split("::", 1)
    return FormalizerParseResult(
        formalizer_id=formalizer_id,
        case_id=case_id,
        evidence_session_id=evidence_session_id,
        parse_status=status,
        facts=(),
        notes="",
        parse_errors=errors,
    )


def _provider_failed(row: dict[str, typing.Any]) -> bool:
    return (
        int(row.get("provider_exit_code", 0) or 0) != 0
        or bool(row.get("provider_timed_out", False))
    )


def _load_json(path: str | pathlib.Path) -> dict[str, typing.Any]:
    return typing.cast(
        "dict[str, typing.Any]",
        json.loads(pathlib.Path(path).read_text(encoding="utf-8")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-payloads", action="store_true")
    parser.add_argument("--run-provider", action="store_true")
    parser.add_argument("--build-report", action="store_true")
    parser.add_argument(
        "--candidate-payloads",
        type=pathlib.Path,
        default=DEFAULT_CANDIDATE_PAYLOADS_PATH,
    )
    parser.add_argument(
        "--formalizer-payloads",
        type=pathlib.Path,
        default=DEFAULT_FORMALIZER_PAYLOADS_PATH,
    )
    parser.add_argument(
        "--formalizer-outputs",
        type=pathlib.Path,
        default=DEFAULT_FORMALIZER_OUTPUTS_PATH,
    )
    parser.add_argument(
        "--candidate-outputs",
        type=pathlib.Path,
        default=DEFAULT_CANDIDATE_OUTPUTS_PATH,
    )
    parser.add_argument(
        "--payload-provenance",
        type=pathlib.Path,
        default=DEFAULT_PAYLOAD_PROVENANCE_PATH,
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--provider",
        default=interpretation_runner.DEFAULT_PROVIDER_COMMAND,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=interpretation_runner.DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.build_payloads and not args.run_provider and not args.build_report:
        parser.print_help()
        return 2

    if args.build_payloads:
        artifact = build_formalizer_payload_artifact(
            candidate_payloads_path=args.candidate_payloads,
        )
        args.formalizer_payloads.parent.mkdir(parents=True, exist_ok=True)
        args.formalizer_payloads.write_text(
            f"{json.dumps(artifact, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        if not args.run_provider and not args.build_report:
            return 0

    formalizer_payload_artifact = interpretation_runner.load_payload_artifact(
        args.formalizer_payloads
    )
    if args.run_provider:
        rows = run_payloads(
            payload_artifact=formalizer_payload_artifact,
            provider_command=args.provider,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
        )
        interpretation_runner.write_jsonl(args.formalizer_outputs, rows)
    else:
        rows = interpretation_runner.load_jsonl(args.formalizer_outputs)

    if args.run_provider or args.build_report:
        candidate_rows = interpretation_runner.load_jsonl(args.candidate_outputs)
        report = build_formalizer_report(
            formalizer_payload_artifact=formalizer_payload_artifact,
            formalizer_rows=rows,
            candidate_rows=candidate_rows,
            payload_provenance_path=args.payload_provenance,
            outputs_path=args.formalizer_outputs,
            candidate_outputs_path=args.candidate_outputs,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            f"{json.dumps(report, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
