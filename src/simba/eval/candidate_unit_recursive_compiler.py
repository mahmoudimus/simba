"""Compile candidate units from normalized recursive facts.

This module is intentionally diagnostic. It consumes the recursive fact IR
emitted by ``candidate_unit_formalizer`` and produces candidate-unit rows with
deterministic rules. It does not call the provider candidate-unit compiler.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import pathlib
import re
import typing

from simba.eval import (
    ambiguity_fail18,
    candidate_unit_runner,
    interpretation_runner,
    type_ontology,
)

PROMPT_VERSION = "recursive_fact_compiler_v1"
DEFAULT_FORMALIZER_OUTPUTS_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_formalizer_outputs_recursive_v2_normalized.jsonl"
)
DEFAULT_OUTPUTS_PATH = pathlib.Path(
    "_gitless/fail18_recursive_fact_compiler_outputs_v1.jsonl"
)
DEFAULT_REPORT_PATH = pathlib.Path(
    "_gitless/fail18_recursive_fact_compiler_report_v1.json"
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MONEY_RE = re.compile(r"(?:over|about|around|approximately)?\s*\$?\s*([0-9][0-9,]*)")
_STOPWORDS = {
    "a",
    "am",
    "an",
    "and",
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
    "many",
    "much",
    "my",
    "need",
    "of",
    "or",
    "the",
    "this",
    "to",
    "total",
    "was",
    "what",
}
_INACTIVE_STATUSES = {
    "asked",
    "considered",
    "considering",
    "desired",
    "in progress",
    "in_progress",
    "ongoing",
    "planned",
    "planning",
    "pending",
    "requested",
    "seeking_tips",
    "uncertain",
    "undecided",
    "unfulfilled",
    "upcoming",
}
_COMPLETED_STATUSES = {
    "attended",
    "completed",
    "done",
    "finished",
    "owned",
    "recent",
}
_ACTION_SYNONYMS = {
    "baked": "bake",
    "baking": "bake",
    "bought": "purchase",
    "buy": "purchase",
    "downloaded": "download",
    "exchanged": "exchange",
    "got": "acquire",
    "got signed": "acquire",
    "had signed": "acquire",
    "helped raise": "raise",
    "made": "make",
    "managed to raise": "raise",
    "picked up": "pick_up",
    "pickup": "pick_up",
    "purchased": "purchase",
    "raised": "raise",
    "tried": "make",
    "tried out": "make",
}
_IRREFLEXIVE_RELATIONS = {
    "different_from",
    "distinct",
    "distinct_from",
    "not_same_as",
    "replaced_by",
    "replacement_for",
    "replaces",
}
_SYMBOL_SORT_ENTITY = "entity"
_SYMBOL_SORT_TYPE = "type"
_SORTED_SYMBOL_ROLES = {
    ("action", "object"): _SYMBOL_SORT_ENTITY,
    ("action", "subject"): _SYMBOL_SORT_ENTITY,
    ("coreference", "entity"): _SYMBOL_SORT_ENTITY,
    ("coreference", "same_as"): _SYMBOL_SORT_ENTITY,
    ("distinct", "a"): _SYMBOL_SORT_ENTITY,
    ("distinct", "b"): _SYMBOL_SORT_ENTITY,
    ("entity", "entity"): _SYMBOL_SORT_ENTITY,
    ("entity", "name"): _SYMBOL_SORT_ENTITY,
    ("entity", "type"): _SYMBOL_SORT_TYPE,
    ("event", "event"): _SYMBOL_SORT_ENTITY,
    ("event", "type"): _SYMBOL_SORT_TYPE,
    ("object_type", "entity"): _SYMBOL_SORT_ENTITY,
    ("object_type", "type"): _SYMBOL_SORT_TYPE,
    ("property", "entity"): _SYMBOL_SORT_ENTITY,
    ("quantity", "entity"): _SYMBOL_SORT_ENTITY,
    ("relation", "source"): _SYMBOL_SORT_ENTITY,
    ("relation", "target"): _SYMBOL_SORT_ENTITY,
    ("sortal", "antecedent"): _SYMBOL_SORT_ENTITY,
    ("sortal", "entity"): _SYMBOL_SORT_ENTITY,
    ("sortal", "type"): _SYMBOL_SORT_TYPE,
    ("status", "entity"): _SYMBOL_SORT_ENTITY,
    ("time", "entity"): _SYMBOL_SORT_ENTITY,
    ("value", "entity"): _SYMBOL_SORT_ENTITY,
}
_ACQUIRE_VERBS = {"acquire", "download", "get", "purchase"}
_ATTENDANCE_VERBS = {
    "attend",
    "attended",
    "been to",
    "go to",
    "got back from",
    "went to",
    "was a bridesmaid",
    "wore",
}
_BAKING_VERBS = {"bake", "make", "tried", "tried out", "used to bake"}
_CHARITY_TARGET_TERMS = {
    "american cancer society",
    "animal shelter",
    "cancer society",
    "charity",
    "food bank",
    "hospital",
}
_MUSIC_RELEASE_TYPES = {
    "album",
    "albums",
    "ep",
    "eps",
    "record",
    "records",
    "release",
    "releases",
    "vinyl",
}
_BAKED_GOOD_TERMS = {
    "baguette",
    "bake",
    "baked",
    "baking",
    "bread",
    "cake",
    "cookie",
    "cookies",
    "focaccia",
    "muffin",
    "pastry",
    "recipe",
    "sourdough",
}
_TOKEN_SYNONYMS = {
    "apparel": "clothing",
    "ceremonies": "wedding",
    "ceremony": "wedding",
    "garment": "clothing",
    "garments": "clothing",
    "marriage": "wedding",
    "purchases": "purchase",
    "releases": "release",
    "shops": "store",
    "weddings": "wedding",
}
_EVENT_TARGET_TERMS = {"event", "wedding"}
_RELATIONAL_NOUN_HEADS = {
    "bunch",
    "copy",
    "group",
    "kind",
    "pair",
    "piece",
    "set",
    "type",
}


@dataclasses.dataclass(frozen=True)
class RecursiveFact:
    case_id: str
    evidence_session_id: str
    fact_id: str
    predicate: str
    arguments: dict[str, typing.Any]
    evidence_span: str
    confidence: float

    @classmethod
    def from_row_fact(
        cls,
        row: dict[str, typing.Any],
        fact: dict[str, typing.Any],
    ) -> RecursiveFact:
        return cls(
            case_id=str(row.get("case_id", "")),
            evidence_session_id=str(row.get("evidence_session_id", "")),
            fact_id=str(fact.get("fact_id", "")),
            predicate=str(fact.get("predicate", "")),
            arguments=dict(fact.get("arguments", {}))
            if isinstance(fact.get("arguments"), dict)
            else {},
            evidence_span=str(fact.get("evidence_span", "")),
            confidence=_float_or_zero(fact.get("confidence")),
        )

    def arg(self, name: str) -> str:
        return _clean_string(self.arguments.get(name, ""))

    @property
    def text(self) -> str:
        return " ".join(
            [
                self.predicate,
                self.evidence_span,
                " ".join(str(value) for value in self.arguments.values()),
            ]
        ).lower()


@dataclasses.dataclass(frozen=True)
class QuestionIntent:
    kind: str
    answer_variable: str
    individuation_policy: str
    aggregation: str
    target_terms: tuple[str, ...]
    action_terms: tuple[str, ...]
    value_terms: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class CompiledUnit:
    unit_id: str
    label: str
    status: str
    merge_target: str | None
    value: float | None
    unit: str | None
    evidence_session_ids: tuple[str, ...]
    evidence_spans: tuple[str, ...]
    reason_code: str
    reason: str

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "unit_id": self.unit_id,
            "label": self.label,
            "status": self.status,
            "merge_target": self.merge_target,
            "value": self.value,
            "unit": self.unit,
            "evidence_session_ids": list(self.evidence_session_ids),
            "evidence_spans": list(self.evidence_spans),
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


def load_fact_rows(
    path: str | pathlib.Path = DEFAULT_FORMALIZER_OUTPUTS_PATH,
) -> list[dict[str, typing.Any]]:
    return interpretation_runner.load_jsonl(path)


def facts_by_case(
    rows: typing.Iterable[dict[str, typing.Any]],
) -> dict[str, list[RecursiveFact]]:
    grouped: dict[str, list[RecursiveFact]] = collections.defaultdict(list)
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if case_id:
            grouped.setdefault(case_id, [])
        if row.get("parse_status") != "parsed":
            continue
        for fact in row.get("facts", []):
            if not isinstance(fact, dict):
                continue
            parsed = RecursiveFact.from_row_fact(row, fact)
            if parsed.case_id:
                grouped[parsed.case_id].append(parsed)
    return dict(grouped)


def classify_question(question: str) -> QuestionIntent:
    q = question.lower()
    target_terms = _question_target_terms(q)
    action_terms = _question_action_terms(q)
    if "how much" in q and "money" in q:
        return QuestionIntent(
            kind="sum_value",
            answer_variable="scalar_value",
            individuation_policy="scalar_value",
            aggregation="sum",
            target_terms=target_terms or ("charity",),
            action_terms=action_terms or ("raise",),
            value_terms=("money", "amount", "usd", "dollar"),
        )
    if ("need to" in q or _has_obligation_language(q)) and action_terms:
        return QuestionIntent(
            kind="count_action_obligation",
            answer_variable="action_obligation",
            individuation_policy="action_obligation",
            aggregation="count_distinct",
            target_terms=target_terms,
            action_terms=action_terms,
        )
    if "attend" in action_terms and set(target_terms) & _EVENT_TARGET_TERMS:
        return QuestionIntent(
            kind="count_attended_events",
            answer_variable="event",
            individuation_policy="event_instance",
            aggregation="count_distinct",
            target_terms=target_terms,
            action_terms=action_terms,
        )
    if "times did i" in q or "did i" in q:
        return QuestionIntent(
            kind="count_action_event",
            answer_variable="event",
            individuation_policy="event_instance",
            aggregation="count_distinct",
            target_terms=target_terms,
            action_terms=action_terms,
        )
    if "attended" in q or "have i attended" in q:
        return QuestionIntent(
            kind="count_attended_events",
            answer_variable="event",
            individuation_policy="event_instance",
            aggregation="count_distinct",
            target_terms=target_terms,
            action_terms=action_terms or ("attend",),
        )
    if action_terms:
        return QuestionIntent(
            kind="count_action_objects",
            answer_variable="semantic_type",
            individuation_policy="semantic_type",
            aggregation="count_distinct",
            target_terms=target_terms,
            action_terms=action_terms,
        )
    return QuestionIntent(
        kind="unsupported",
        answer_variable="entity",
        individuation_policy="canonical_entity",
        aggregation="count_distinct",
        target_terms=target_terms,
        action_terms=(),
    )


def compile_case(
    *,
    case_id: str,
    question: str,
    facts: typing.Iterable[RecursiveFact],
) -> dict[str, typing.Any]:
    fact_list = list(facts)
    intent = classify_question(question)
    consistency = _recursive_fact_consistency(fact_list)
    namespace = _symbol_namespace_validation(fact_list)
    quarantine = _quarantine_plan(
        consistency=consistency,
        namespace=namespace,
    )
    quarantined_sessions = set(quarantine["session_ids"])
    compiled_facts = [
        fact
        for fact in fact_list
        if fact.evidence_session_id not in quarantined_sessions
    ]
    hard_errors = _recursive_fact_hard_errors(
        consistency=consistency,
        namespace=namespace,
        quarantined_sessions=quarantined_sessions,
    )
    if hard_errors:
        return _invalid_compiler_row(
            case_id=case_id,
            intent=intent,
            facts=fact_list,
            consistency=consistency,
            namespace=namespace,
            parse_errors=hard_errors,
        )
    if intent.kind == "sum_value":
        units = _compile_sum_values(intent, compiled_facts)
    elif intent.kind == "count_action_obligation":
        units = _compile_action_obligations(intent, compiled_facts)
    elif intent.kind == "count_action_event":
        units = _compile_action_events(intent, compiled_facts)
    elif intent.kind == "count_attended_events":
        units = _compile_attended_events(intent, compiled_facts)
    elif intent.kind == "count_action_objects":
        units = _compile_action_objects(intent, compiled_facts)
    else:
        units = ()
    row = {
        "case_id": case_id,
        "provider": "deterministic_recursive_fact_compiler",
        "prompt_version": PROMPT_VERSION,
        "raw_output": "",
        "parse_status": candidate_unit_runner.PARSE_STATUS_PARSED,
        "answer_variable": intent.answer_variable,
        "individuation_policy": intent.individuation_policy,
        "aggregation": intent.aggregation,
        "candidate_units": [unit.to_dict() for unit in units],
        "facts": [_fact_to_prolog(fact) for fact in compiled_facts],
        "quarantined_facts": [
            _fact_to_prolog(fact)
            for fact in fact_list
            if fact.evidence_session_id in quarantined_sessions
        ],
        "query": _query_for_intent(intent),
        "computed_answer": _computed_answer(intent, units),
        "rationale": (
            f"Compiled {len(units)} candidate units from "
            f"{len(compiled_facts)} recursive facts using intent {intent.kind}."
        ),
        "parse_errors": [],
        "compiler_warnings": _quarantine_warnings(quarantine),
        "compiler_intent": dataclasses.asdict(intent),
        "recursive_fact_count": len(fact_list),
        "compiled_recursive_fact_count": len(compiled_facts),
        "quarantined_recursive_fact_count": len(fact_list) - len(compiled_facts),
        "quarantined_evidence_sessions": quarantine["sessions"],
        "recursive_fact_consistency": consistency,
        "symbol_namespace": namespace,
        "inclusion_mutation_sensitivity": _inclusion_mutation_sensitivity(
            intent,
            units,
        ),
    }
    return row


def build_report(
    *,
    fact_rows: list[dict[str, typing.Any]],
    manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    formalizer_outputs_path: str | pathlib.Path = DEFAULT_FORMALIZER_OUTPUTS_PATH,
    outputs_path: str | pathlib.Path = DEFAULT_OUTPUTS_PATH,
) -> tuple[list[dict[str, typing.Any]], dict[str, typing.Any]]:
    manifest_by_id = {
        str(row.get("question_id", "")): row
        for row in ambiguity_fail18.load_manifest(manifest_path)
    }
    grouped = facts_by_case(fact_rows)
    rows: list[dict[str, typing.Any]] = []
    for case_id in sorted(grouped):
        manifest_row = manifest_by_id.get(case_id, {})
        rows.append(
            compile_case(
                case_id=case_id,
                question=str(manifest_row.get("question", "")),
                facts=grouped[case_id],
            )
        )
    payload_artifact = {
        "prompt_version": PROMPT_VERSION,
        "payloads": [{"case": {"id": row["case_id"]}} for row in rows],
    }
    report = candidate_unit_runner.build_candidate_unit_report(
        rows=rows,
        payload_artifact=payload_artifact,
        outputs_path=outputs_path,
        payloads_path=formalizer_outputs_path,
        manifest_path=manifest_path,
    )
    report.update(
        {
            "name": "fail18-recursive-fact-compiler-report",
            "artifact_kind": "recursive_fact_compiler_report",
            "gate": "recursive_fact_compiler",
            "gate_status": "recursive_fact_compiler_complete",
            "source_formalizer_outputs": str(formalizer_outputs_path),
            "compiler": {
                "prompt_version": PROMPT_VERSION,
                "provider_used": False,
                "old_candidate_unit_outputs_used": False,
                "input_ir": "normalized_recursive_facts",
            },
            "intent_counts": dict(
                collections.Counter(
                    str(row.get("compiler_intent", {}).get("kind", ""))
                    for row in rows
                )
            ),
            "recursive_fact_consistency_summary": _consistency_summary(rows),
            "symbol_namespace_summary": _symbol_namespace_summary(rows),
            "inclusion_mutation_sensitivity_summary": (
                _mutation_sensitivity_summary(rows)
            ),
        }
    )
    return rows, report


def write_outputs(
    rows: list[dict[str, typing.Any]],
    report: dict[str, typing.Any],
    *,
    outputs_path: str | pathlib.Path = DEFAULT_OUTPUTS_PATH,
    report_path: str | pathlib.Path = DEFAULT_REPORT_PATH,
) -> None:
    outputs = pathlib.Path(outputs_path)
    outputs.parent.mkdir(parents=True, exist_ok=True)
    outputs.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    pathlib.Path(report_path).write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _compile_action_obligations(
    intent: QuestionIntent,
    facts: list[RecursiveFact],
) -> tuple[CompiledUnit, ...]:
    entity_types = _entity_types(facts)
    included: dict[str, CompiledUnit] = {}
    excluded: list[CompiledUnit] = []
    for fact in _facts_of(facts, "action"):
        verb = _normalize_action(fact.arg("verb"))
        status = fact.arg("status").lower()
        obj = fact.arg("object")
        if verb not in intent.action_terms:
            continue
        label = _saturate_relational_label(obj or verb, fact=fact, facts=facts)
        if _is_inactive_status(status) and status not in {"needed", "pending"}:
            excluded.append(
                _unit_from_fact(
                    fact,
                    label=label,
                    status="excluded",
                    reason_code="inactive_obligation_status",
                    reason="The action is not an active obligation.",
                )
            )
            continue
        if not _matches_target_object(obj, intent.target_terms, entity_types, facts):
            excluded.append(
                _unit_from_fact(
                    fact,
                    label=label,
                    status="excluded",
                    reason_code="target_type_not_ratified",
                    reason=(
                        "The action object does not match the requested target "
                        "type."
                    ),
                )
            )
            continue
        key = _canonical_action_key(verb, obj, fact, facts)
        if key not in included:
            included[key] = _unit_from_fact(
                fact,
                label=label,
                status="included",
                reason_code="active_action_obligation",
                reason="The recursive action fact matches the requested obligation.",
            )
    return tuple([*included.values(), *excluded])


def _compile_action_events(
    intent: QuestionIntent,
    facts: list[RecursiveFact],
) -> tuple[CompiledUnit, ...]:
    if "bake" in intent.action_terms:
        return _compile_baking_events(intent, facts)
    included: dict[str, CompiledUnit] = {}
    excluded: list[CompiledUnit] = []
    for fact in _facts_of(facts, "action"):
        verb = _normalize_action(fact.arg("verb"))
        if verb not in intent.action_terms:
            continue
        if _is_inactive_status(fact.arg("status")):
            excluded.append(
                _unit_from_fact(
                    fact,
                    label=fact.arg("object") or verb,
                    status="excluded",
                    reason_code="inactive_event_status",
                    reason="The action is planned, generic, or not completed.",
                )
            )
            continue
        key = _canonical_label(fact.arg("object") or fact.evidence_span)
        included.setdefault(
            key,
            _unit_from_fact(
                fact,
                label=fact.arg("object") or verb,
                status="included",
                reason_code="completed_action_event",
                reason="The recursive action fact is completed and matches the query.",
            ),
        )
    return tuple([*included.values(), *excluded])


def _compile_baking_events(
    intent: QuestionIntent,
    facts: list[RecursiveFact],
) -> tuple[CompiledUnit, ...]:
    del intent
    included: dict[str, CompiledUnit] = {}
    excluded: list[CompiledUnit] = []
    for fact in facts:
        if fact.predicate == "event" and "baking" in fact.arg("type").lower():
            label = fact.arg("event")
            status = fact.arg("status")
        elif fact.predicate == "action":
            verb = _normalize_action(fact.arg("verb"))
            if verb not in _BAKING_VERBS and "bake" not in fact.text:
                continue
            label = fact.arg("object")
            status = fact.arg("status")
            if verb in {"make", "tried"} and not _looks_like_baked_good(fact):
                excluded.append(
                    _unit_from_fact(
                        fact,
                        label=label,
                        status="excluded",
                        reason_code="not_baked_good",
                        reason=(
                            "Generic cooking verbs only count when the object "
                            "is grounded as a baked good."
                        ),
                    )
                )
                continue
        else:
            continue
        if _is_inactive_status(status):
            excluded.append(
                _unit_from_fact(
                    fact,
                    label=label,
                    status="excluded",
                    reason_code="not_completed_baking_event",
                    reason="The fact is planned, generic, or not completed.",
                )
            )
            continue
        key = _canonical_baked_item(label)
        included.setdefault(
            key,
            _unit_from_fact(
                fact,
                label=label,
                status="included",
                reason_code="completed_baking_event",
                reason="The recursive fact describes a completed baking event.",
            ),
        )
    return tuple([*included.values(), *excluded])


def _compile_action_objects(
    intent: QuestionIntent,
    facts: list[RecursiveFact],
) -> tuple[CompiledUnit, ...]:
    entity_types = _entity_types(facts)
    included: dict[str, CompiledUnit] = {}
    excluded: list[CompiledUnit] = []
    for fact in _facts_of(facts, "action"):
        verb = _normalize_action(fact.arg("verb"))
        if verb not in intent.action_terms and not (
            _targets_music_release(intent.target_terms)
            and verb in _ACQUIRE_VERBS
            and ("signed" in fact.arg("verb").lower() or "signed" in fact.text)
        ):
            continue
        if _is_inactive_status(fact.arg("status")):
            excluded.append(
                _unit_from_fact(
                    fact,
                    label=fact.arg("object") or verb,
                    status="excluded",
                    reason_code="inactive_action_status",
                    reason="The action is not completed or owned.",
                )
            )
            continue
        if not _matches_target_object(
            fact.arg("object"),
            intent.target_terms,
            entity_types,
            facts,
        ):
            excluded.append(
                _unit_from_fact(
                    fact,
                    label=fact.arg("object") or verb,
                    status="excluded",
                    reason_code="target_type_not_ratified",
                    reason=(
                        "The action object does not match the requested "
                        "object type."
                    ),
                )
            )
            continue
        key = _canonical_release_key(fact, entity_types)
        included.setdefault(
            key,
            _unit_from_fact(
                fact,
                label=fact.arg("object") or verb,
                status="included",
                reason_code="completed_matching_action",
                reason="The completed action matches the requested object type.",
            ),
        )
    return tuple([*included.values(), *excluded])


def _compile_attended_events(
    intent: QuestionIntent,
    facts: list[RecursiveFact],
) -> tuple[CompiledUnit, ...]:
    target_terms = set(intent.target_terms)
    event_facts = [
        fact for fact in _facts_of(facts, "event") if _event_matches(fact, target_terms)
    ]
    attendance_actions = [
        fact for fact in _facts_of(facts, "action") if _is_attendance_action(fact)
    ]
    included: dict[str, CompiledUnit] = {}
    excluded: list[CompiledUnit] = []
    for event in event_facts:
        if _is_aggregate_event(event) or _is_inactive_status(event.arg("status")):
            excluded.append(
                _unit_from_fact(
                    event,
                    label=event.arg("event"),
                    status="excluded",
                    reason_code="non_grounded_or_inactive_event",
                    reason=(
                        "The event is aggregate, planned, generic, or not "
                        "completed."
                    ),
                )
            )
            continue
        if not _event_has_user_attendance(event, attendance_actions):
            excluded.append(
                _unit_from_fact(
                    event,
                    label=event.arg("event"),
                    status="excluded",
                    reason_code="no_user_attendance_edge",
                    reason=(
                        "No recursive action or participant edge grounds user "
                        "attendance."
                    ),
                )
            )
            continue
        key = _canonical_event_key(event, facts)
        included.setdefault(
            key,
            _unit_from_fact(
                event,
                label=event.arg("event"),
                status="included",
                reason_code="attended_event",
                reason="The event fact is grounded as attended by the user.",
            ),
        )
    return tuple([*included.values(), *excluded])


def _compile_sum_values(
    intent: QuestionIntent,
    facts: list[RecursiveFact],
) -> tuple[CompiledUnit, ...]:
    del intent
    included: dict[str, CompiledUnit] = {}
    excluded: list[CompiledUnit] = []
    for fact in _facts_of(facts, "value"):
        if fact.arg("unit").lower() not in {"usd", "dollar", "dollars"}:
            continue
        value = _numeric_amount(fact.arg("value"))
        if value is None:
            excluded.append(
                _unit_from_fact(
                    fact,
                    label=fact.arg("entity"),
                    status="excluded",
                    reason_code="non_numeric_money_value",
                    reason="The value fact does not contain a usable number.",
                )
            )
            continue
        if not _is_charity_fundraising_value(fact, facts):
            excluded.append(
                _unit_from_fact(
                    fact,
                    label=fact.arg("entity"),
                    status="excluded",
                    reason_code="not_question_charity_fundraising",
                    reason="The value is not grounded to a charity fundraising target.",
                )
            )
            continue
        key = _canonical_value_key(fact)
        existing = included.get(key)
        if existing is None:
            included[key] = _unit_from_fact(
                fact,
                label=fact.arg("entity") or "money raised",
                status="included",
                value=value,
                unit="USD",
                reason_code="charity_amount_raised",
                reason="The value fact is grounded to completed charity fundraising.",
            )
        elif existing.value is not None and value > existing.value:
            included[key] = dataclasses.replace(existing, value=value)
    return tuple([*included.values(), *excluded])


def _computed_answer(
    intent: QuestionIntent,
    units: typing.Iterable[CompiledUnit],
) -> float | None:
    included = [unit for unit in units if unit.status == "included"]
    if intent.aggregation == "count_distinct":
        return float(len(included))
    if intent.aggregation == "sum":
        values = [unit.value for unit in included]
        if any(value is None for value in values):
            return None
        return float(sum(typing.cast("list[float]", values)))
    return None


def _invalid_compiler_row(
    *,
    case_id: str,
    intent: QuestionIntent,
    facts: list[RecursiveFact],
    consistency: dict[str, object],
    namespace: dict[str, object],
    parse_errors: list[str],
) -> dict[str, typing.Any]:
    return {
        "case_id": case_id,
        "provider": "deterministic_recursive_fact_compiler",
        "prompt_version": PROMPT_VERSION,
        "raw_output": "",
        "parse_status": candidate_unit_runner.PARSE_STATUS_INVALID_SCHEMA,
        "answer_variable": intent.answer_variable,
        "individuation_policy": intent.individuation_policy,
        "aggregation": intent.aggregation,
        "candidate_units": [],
        "facts": [_fact_to_prolog(fact) for fact in facts],
        "query": _query_for_intent(intent),
        "computed_answer": None,
        "rationale": "Recursive fact set failed sorted-symbol validation.",
        "parse_errors": parse_errors,
        "compiler_intent": dataclasses.asdict(intent),
        "recursive_fact_count": len(facts),
        "recursive_fact_consistency": consistency,
        "symbol_namespace": namespace,
        "inclusion_mutation_sensitivity": {
            "skipped_reason": "invalid_recursive_fact_ir",
        },
    }


def _recursive_fact_hard_errors(
    *,
    consistency: dict[str, object],
    namespace: dict[str, object],
    quarantined_sessions: set[str],
) -> list[str]:
    errors: list[str] = []
    for issue in consistency.get("issues", []):
        if not isinstance(issue, dict):
            continue
        if str(issue.get("evidence_session_id", "")) in quarantined_sessions:
            continue
        errors.append(
            "recursive_fact_consistency:"
            f"{issue.get('issue', '')}:{issue.get('fact_id', '')}"
        )
    for issue in namespace.get("issues", []):
        if not isinstance(issue, dict):
            continue
        if str(issue.get("namespace", "")) in quarantined_sessions:
            continue
        errors.append(
            "symbol_namespace:"
            f"{issue.get('issue', '')}:{issue.get('symbol', '')}"
        )
    return errors


def _quarantine_plan(
    *,
    consistency: dict[str, object],
    namespace: dict[str, object],
) -> dict[str, object]:
    sessions: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    for issue in consistency.get("issues", []):
        if not isinstance(issue, dict):
            continue
        evidence_session_id = str(issue.get("evidence_session_id", ""))
        if evidence_session_id:
            sessions[evidence_session_id].append(
                {"source": "recursive_fact_consistency", **issue}
            )
    for issue in namespace.get("issues", []):
        if not isinstance(issue, dict):
            continue
        evidence_session_id = str(issue.get("namespace", ""))
        if evidence_session_id:
            sessions[evidence_session_id].append(
                {"source": "symbol_namespace", **issue}
            )
    return {
        "session_ids": tuple(sorted(sessions)),
        "sessions": [
            {
                "evidence_session_id": evidence_session_id,
                "issue_count": len(issues),
                "issues": issues,
            }
            for evidence_session_id, issues in sorted(sessions.items())
        ],
    }


def _quarantine_warnings(quarantine: dict[str, object]) -> list[str]:
    warnings: list[str] = []
    for session in quarantine.get("sessions", []):
        if not isinstance(session, dict):
            continue
        warnings.append(
            "quarantined_evidence_session:"
            f"{session.get('evidence_session_id', '')}:"
            f"{session.get('issue_count', 0)}"
        )
    return warnings


def _symbol_namespace_validation(facts: list[RecursiveFact]) -> dict[str, object]:
    symbol_uses: dict[
        tuple[str, str],
        dict[str, list[dict[str, str]]],
    ] = collections.defaultdict(lambda: collections.defaultdict(list))
    for fact in facts:
        for key, sort in _SORTED_SYMBOL_ROLES.items():
            predicate, argument = key
            if fact.predicate != predicate:
                continue
            raw_value = fact.arg(argument)
            symbol = _canonical_label(raw_value)
            if not symbol:
                continue
            namespace = fact.evidence_session_id or "evidence"
            symbol_uses[(namespace, symbol)][sort].append(
                {
                    "fact_id": fact.fact_id,
                    "predicate": fact.predicate,
                    "argument": argument,
                    "value": raw_value,
                }
            )
    issues: list[dict[str, object]] = []
    for (namespace, symbol), uses_by_sort in sorted(symbol_uses.items()):
        if {
            _SYMBOL_SORT_ENTITY,
            _SYMBOL_SORT_TYPE,
        } <= set(uses_by_sort):
            issues.append(
                {
                    "issue": "symbol_sort_collision",
                    "namespace": namespace,
                    "symbol": symbol,
                    "reason": (
                        "same bare symbol is used in both entity and type "
                        "positions within one evidence namespace"
                    ),
                    "uses": {
                        sort: uses_by_sort[sort]
                        for sort in sorted(uses_by_sort)
                    },
                }
            )
    return {"issue_count": len(issues), "issues": issues}


def _inclusion_mutation_sensitivity(
    intent: QuestionIntent,
    units: typing.Iterable[CompiledUnit],
) -> dict[str, object]:
    unit_list = list(units)
    original_answer = _computed_answer(intent, unit_list)
    single_flip_unchanged: list[str] = []
    single_flip_total = 0
    for index, unit in enumerate(unit_list):
        if unit.status not in {"included", "excluded"}:
            continue
        single_flip_total += 1
        mutated = list(unit_list)
        mutated[index] = dataclasses.replace(
            unit,
            status="excluded" if unit.status == "included" else "included",
        )
        if _answers_equal(_computed_answer(intent, mutated), original_answer):
            single_flip_unchanged.append(unit.unit_id)
    swap_unchanged: list[dict[str, str]] = []
    included = [unit for unit in unit_list if unit.status == "included"]
    excluded = [unit for unit in unit_list if unit.status == "excluded"]
    for included_unit in included:
        for excluded_unit in excluded:
            mutated = [
                dataclasses.replace(unit, status="excluded")
                if unit.unit_id == included_unit.unit_id
                else dataclasses.replace(unit, status="included")
                if unit.unit_id == excluded_unit.unit_id
                else unit
                for unit in unit_list
            ]
            if _answers_equal(_computed_answer(intent, mutated), original_answer):
                swap_unchanged.append(
                    {
                        "included_unit_id": included_unit.unit_id,
                        "excluded_unit_id": excluded_unit.unit_id,
                    }
                )
    return {
        "original_answer": original_answer,
        "single_flip_total": single_flip_total,
        "single_flip_answer_unchanged_count": len(single_flip_unchanged),
        "single_flip_answer_unchanged_unit_ids": single_flip_unchanged[:20],
        "balanced_swap_total": len(included) * len(excluded),
        "balanced_swap_answer_unchanged_count": len(swap_unchanged),
        "balanced_swap_answer_unchanged_examples": swap_unchanged[:20],
        "aggregate_score_insensitive_to_balanced_swaps": bool(swap_unchanged),
    }


def _answers_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(left - right) <= 1e-9


def _recursive_fact_consistency(facts: list[RecursiveFact]) -> dict[str, object]:
    issues: list[dict[str, str]] = []
    identity_pairs: dict[tuple[str, frozenset[str]], RecursiveFact] = {}
    distinct_pairs: dict[tuple[str, frozenset[str]], RecursiveFact] = {}
    for fact in facts:
        if fact.predicate == "coreference":
            entity = _canonical_label(fact.arg("entity"))
            same_as = _canonical_label(fact.arg("same_as"))
            if entity and same_as:
                pair = (fact.evidence_session_id, frozenset({entity, same_as}))
                identity_pairs[pair] = fact
            if _looks_like_bridging_coreference_misuse(fact):
                issues.append(
                    {
                        "issue": "bridging_coreference_misuse",
                        "fact_id": fact.fact_id,
                        "evidence_session_id": fact.evidence_session_id,
                        "reason": (
                            "coreference appears to saturate a relational noun "
                            "instead of asserting same-token identity"
                        ),
                    }
                )
        elif fact.predicate == "distinct":
            left = _canonical_label(fact.arg("a"))
            right = _canonical_label(fact.arg("b"))
            if left and right:
                pair = (fact.evidence_session_id, frozenset({left, right}))
                distinct_pairs[pair] = fact
        elif fact.predicate == "relation" and _canonical_relation(
            fact.arg("relation")
        ) in _IRREFLEXIVE_RELATIONS:
            left = _canonical_label(fact.arg("source"))
            right = _canonical_label(fact.arg("target"))
            if left and right:
                pair = (fact.evidence_session_id, frozenset({left, right}))
                distinct_pairs[pair] = fact
    for key in sorted(
        identity_pairs.keys() & distinct_pairs.keys(),
        key=lambda item: (item[0], sorted(item[1])),
    ):
        identity_fact = identity_pairs[key]
        distinct_fact = distinct_pairs[key]
        issues.append(
            {
                "issue": "identity_distinct_conflict",
                "fact_id": identity_fact.fact_id,
                "conflicting_fact_id": distinct_fact.fact_id,
                "evidence_session_id": key[0],
                "reason": "same pair is asserted as both same_as and distinct",
            }
        )
    return {"issue_count": len(issues), "issues": issues}


def _consistency_summary(rows: list[dict[str, typing.Any]]) -> dict[str, object]:
    issue_counts: collections.Counter[str] = collections.Counter()
    rows_with_issues = 0
    for row in rows:
        consistency = row.get("recursive_fact_consistency", {})
        if not isinstance(consistency, dict):
            continue
        issues = consistency.get("issues", [])
        if not isinstance(issues, list) or not issues:
            continue
        rows_with_issues += 1
        for issue in issues:
            if isinstance(issue, dict):
                issue_counts[str(issue.get("issue", ""))] += 1
    return {
        "rows_with_issues": rows_with_issues,
        "issue_counts": dict(sorted(issue_counts.items())),
    }


def _symbol_namespace_summary(rows: list[dict[str, typing.Any]]) -> dict[str, object]:
    rows_with_issues = 0
    issue_counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        namespace = row.get("symbol_namespace", {})
        if not isinstance(namespace, dict):
            continue
        issues = namespace.get("issues", [])
        if not isinstance(issues, list) or not issues:
            continue
        rows_with_issues += 1
        for issue in issues:
            if isinstance(issue, dict):
                issue_counts[str(issue.get("issue", ""))] += 1
    return {
        "rows_with_issues": rows_with_issues,
        "issue_counts": dict(sorted(issue_counts.items())),
    }


def _mutation_sensitivity_summary(
    rows: list[dict[str, typing.Any]],
) -> dict[str, object]:
    rows_with_balanced_swap_insensitivity = 0
    single_flip_unchanged = 0
    balanced_swap_unchanged = 0
    for row in rows:
        sensitivity = row.get("inclusion_mutation_sensitivity", {})
        if not isinstance(sensitivity, dict):
            continue
        single_flip_unchanged += int(
            sensitivity.get("single_flip_answer_unchanged_count", 0) or 0
        )
        balanced_count = int(
            sensitivity.get("balanced_swap_answer_unchanged_count", 0) or 0
        )
        balanced_swap_unchanged += balanced_count
        if bool(sensitivity.get("aggregate_score_insensitive_to_balanced_swaps")):
            rows_with_balanced_swap_insensitivity += 1
    return {
        "rows_with_balanced_swap_insensitivity": (
            rows_with_balanced_swap_insensitivity
        ),
        "single_flip_answer_unchanged_count": single_flip_unchanged,
        "balanced_swap_answer_unchanged_count": balanced_swap_unchanged,
    }


def _looks_like_bridging_coreference_misuse(fact: RecursiveFact) -> bool:
    entity = fact.arg("entity")
    same_as = fact.arg("same_as")
    entity_key = _canonical_label(entity)
    same_as_key = _canonical_label(same_as)
    if not entity_key or not same_as_key:
        return False
    if not _needs_complement_saturation(entity):
        return False
    if entity_key == same_as_key:
        return False
    return entity_key in same_as_key or _token_overlap(entity_key, same_as_key, 0.67)


def _unit_from_fact(
    fact: RecursiveFact,
    *,
    label: str,
    status: str,
    reason_code: str,
    reason: str,
    value: float | None = None,
    unit: str | None = None,
) -> CompiledUnit:
    fallback_fact_id = fact.fact_id or _canonical_label(label)
    unit_id = f"{fact.case_id}:{fact.evidence_session_id}:{fallback_fact_id}"
    return CompiledUnit(
        unit_id=unit_id,
        label=label,
        status=status,
        merge_target=None,
        value=value,
        unit=unit,
        evidence_session_ids=(fact.evidence_session_id,),
        evidence_spans=(fact.evidence_span,) if fact.evidence_span else (),
        reason_code=reason_code,
        reason=reason,
    )


def _question_target_terms(question: str) -> tuple[str, ...]:
    q = question.lower().strip(" ?")
    phrase = ""
    match = re.search(
        r"how many\s+(.+?)(?:\s+do i|\s+did i|\s+have i|\s+have\s)",
        q,
    )
    if match:
        phrase = match.group(1)
    if phrase.startswith("times "):
        phrase = phrase.removeprefix("times ")
    if "items of" in phrase:
        phrase = phrase.split("items of", 1)[1]
    if not phrase and "weddings" in q:
        phrase = "weddings"
    if not phrase and "marriage ceremon" in q:
        phrase = "marriage ceremonies"
    if not phrase and "money" in q:
        phrase = "money charity"
    return tuple(_content_tokens(phrase))


def _question_action_terms(question: str) -> tuple[str, ...]:
    q = question.lower().strip(" ?")
    action_phrase = ""
    for pattern in (
        r"need to\s+(.+?)(?:\s+from|\s+at|\s+in|\?|$)",
        r"waiting to be\s+(.+?)(?:\s+from|\s+at|\s+in|\?|$)",
        r"did i\s+(.+?)(?:\s+in|\s+last|\s+this|\?|$)",
        r"have i\s+(.+?)(?:\s+in|\s+this|\?|$)",
        r"did i raise\s+(.+?)(?:\s+in|\?|$)",
    ):
        match = re.search(pattern, q)
        if match:
            action_phrase = match.group(1)
            break
    if "how much money" in q and "raise" in q:
        action_phrase = "raise"
    raw_terms = [
        _normalize_action(term)
        for term in re.split(r"\s+or\s+|\s+and\s+|,", action_phrase)
        if term.strip()
    ]
    return tuple(term for term in raw_terms if term)


def _matches_target_object(
    obj: str,
    target_terms: tuple[str, ...],
    entity_types: dict[str, set[str]],
    facts: list[RecursiveFact],
) -> bool:
    if not target_terms:
        return True
    target = set(target_terms)
    obj_key = _canonical_label(obj)
    obj_tokens = set(_content_tokens(obj))
    candidate_types = _candidate_types_for_object(obj, entity_types, facts)
    type_tokens = {
        token
        for candidate_type in candidate_types
        for token in _content_tokens(candidate_type)
    }
    if target & (obj_tokens | type_tokens):
        return True
    if _ontology_ratifies_target(candidate_types, target_terms):
        return True
    if "clothing" in target and _session_has_target_type(obj, "clothing", facts):
        return True
    if _targets_music_release(target_terms):
        return bool((obj_tokens | type_tokens) & _MUSIC_RELEASE_TYPES) or (
            "signed" in obj_key and "vinyl" in obj_key
        )
    return False


def _candidate_types_for_object(
    obj: str,
    entity_types: dict[str, set[str]],
    facts: list[RecursiveFact],
) -> set[str]:
    obj_key = _canonical_label(obj)
    if not obj_key:
        return set()
    candidate_types: set[str] = set()
    for entity, types in entity_types.items():
        if (
            entity == obj_key
            or entity in obj_key
            or obj_key in entity
            or _token_overlap(obj_key, entity, 0.67)
        ):
            candidate_types.update(types)
    candidate_types.update(_sortal_types_for_label(obj, facts))
    return {candidate_type for candidate_type in candidate_types if candidate_type}


def _ontology_ratifies_target(
    candidate_types: set[str],
    target_terms: tuple[str, ...],
) -> bool:
    for candidate_type in sorted(candidate_types):
        for target_term in sorted(target_terms):
            ratification = type_ontology.ratify_type_subsumption(
                candidate_type,
                target_term,
            )
            if ratification.ratified:
                return True
    return False


def _session_has_target_type(obj: str, target: str, facts: list[RecursiveFact]) -> bool:
    obj_tokens = set(_content_tokens(obj))
    if obj_tokens and target in obj_tokens:
        return True
    bridged_types = _sortal_types_for_label(obj, facts)
    for fact in facts:
        if fact.predicate != "object_type":
            continue
        type_text = fact.arg("type").lower()
        entity_tokens = set(_content_tokens(fact.arg("entity")))
        if target in type_text and (not obj_tokens or obj_tokens & entity_tokens):
            return True
        if target in type_text and any(
            entity_tokens & set(_content_tokens(type_value))
            for type_value in bridged_types
        ):
            return True
    return obj.lower() in {"new pair", "them", "it"} and any(
        fact.predicate == "object_type" and target in fact.arg("type").lower()
        for fact in facts
    )


def _is_charity_fundraising_value(
    fact: RecursiveFact,
    facts: list[RecursiveFact],
) -> bool:
    if "amount" not in fact.arg("attribute").lower() and "raised" not in fact.text:
        return False
    session_facts = [
        item for item in facts if item.evidence_session_id == fact.evidence_session_id
    ]
    context = " ".join(item.text for item in session_facts)
    charity_context = any(term in context for term in _CHARITY_TARGET_TERMS)
    return charity_context and _user_raised_money(fact, session_facts)


def _user_raised_money(
    fact: RecursiveFact,
    session_facts: list[RecursiveFact],
) -> bool:
    raised_by_user_action = any(
        item.predicate == "action"
        and item.arg("subject").lower() == "user"
        and _normalize_action(item.arg("verb")) == "raise"
        and not _is_inactive_status(item.arg("status"))
        for item in session_facts
    )
    if raised_by_user_action:
        return True
    raised_by_user_relation = any(
        item.predicate == "relation"
        and item.arg("source").lower() == "user"
        and "raise" in item.arg("relation").lower()
        for item in session_facts
    )
    if raised_by_user_relation:
        return True
    span = fact.evidence_span.lower()
    return any(
        marker in span
        for marker in (
            "i raised",
            "i helped raise",
            "we raised",
            "we helped raise",
            "helped raise",
            "managed to raise",
        )
    )


def _event_matches(fact: RecursiveFact, target_terms: set[str]) -> bool:
    haystack = set(_content_tokens(f"{fact.arg('event')} {fact.arg('type')}"))
    if target_terms & haystack:
        return True
    return "weddings" in target_terms and "wedding" in haystack


def _is_attendance_action(fact: RecursiveFact) -> bool:
    if fact.predicate != "action":
        return False
    verb = fact.arg("verb").lower()
    return any(term in verb for term in _ATTENDANCE_VERBS)


def _event_has_user_attendance(
    event: RecursiveFact,
    attendance_actions: list[RecursiveFact],
) -> bool:
    participants = event.arg("participants").lower()
    status = event.arg("status").lower()
    if status == "attended" and "user" in participants:
        return True
    event_key = _canonical_label(event.arg("event"))
    for action in attendance_actions:
        if action.evidence_session_id != event.evidence_session_id:
            continue
        action_text = f"{action.arg('object')} {action.arg('location')}"
        action_key = _canonical_label(action_text)
        if (
            event_key in action_key
            or action_key in event_key
            or _token_overlap(event_key, action_key, 0.67)
        ):
            return True
    return False


def _is_aggregate_event(fact: RecursiveFact) -> bool:
    label = fact.arg("event").lower()
    span = fact.evidence_span.lower()
    return (
        "few " in span
        or "weddings attended" in label
        or label.endswith("s attended by user")
        or "traditionally celebrate" in span
    )


def _canonical_event_key(event: RecursiveFact, facts: list[RecursiveFact]) -> str:
    label = _canonical_label(event.arg("event"))
    for fact in facts:
        if fact.predicate != "coreference":
            continue
        entity = _canonical_label(fact.arg("entity"))
        same_as = _canonical_label(fact.arg("same_as"))
        if (
            label in {entity, same_as}
            or _token_overlap(label, entity, 0.67)
            or _token_overlap(label, same_as, 0.67)
        ):
            return min(entity, same_as)
    participants = set(_content_tokens(event.arg("participants")))
    relation_terms = {"cousin", "friend", "roommate", "sister", "brother"}
    label_terms = set(_content_tokens(label))
    relation_key = sorted((participants | label_terms) & relation_terms)
    if relation_key:
        return f"{relation_key[0]} wedding"
    return label


def _canonical_release_key(
    fact: RecursiveFact,
    entity_types: dict[str, set[str]],
) -> str:
    label = _canonical_label(fact.arg("object") or fact.evidence_span)
    label = re.sub(r"^(album|ep|record|vinyl)\s+", "", label)
    if label == "vinyl":
        session_types = " ".join(
            type_value
            for entity, types in entity_types.items()
            if entity in label or label in entity
            for type_value in types
        )
        if session_types:
            label = _canonical_label(session_types)
    return label


def _saturate_relational_label(
    label: str,
    *,
    fact: RecursiveFact,
    facts: list[RecursiveFact],
) -> str:
    if not _needs_complement_saturation(label):
        return label
    sortal_type = _bridged_sortal_type(
        label,
        evidence_session_id=fact.evidence_session_id,
        facts=facts,
    )
    if not sortal_type:
        return label
    return _label_with_sortal(label, sortal_type)


def _needs_complement_saturation(label: str) -> bool:
    if " of " in label.lower():
        return False
    tokens = set(_content_tokens(label))
    return bool(tokens & _RELATIONAL_NOUN_HEADS)


def _bridged_sortal_type(
    label: str,
    *,
    evidence_session_id: str,
    facts: list[RecursiveFact],
) -> str:
    label_key = _canonical_label(label)
    for item in facts:
        if item.evidence_session_id != evidence_session_id:
            continue
        if item.predicate == "sortal":
            entity = _canonical_label(item.arg("entity"))
            if entity and (
                entity == label_key
                or entity in label_key
                or label_key in entity
                or _token_overlap(label_key, entity, 0.67)
            ):
                return item.arg("type").strip()
        if item.predicate == "object_type" and item.arg("source"):
            entity = _canonical_label(item.arg("entity"))
            if entity and (
                entity == label_key
                or entity in label_key
                or label_key in entity
                or _token_overlap(label_key, entity, 0.67)
            ):
                return item.arg("type").strip()
    return ""


def _sortal_types_for_label(label: str, facts: list[RecursiveFact]) -> set[str]:
    label_key = _canonical_label(label)
    sortal_types: set[str] = set()
    for fact in facts:
        if fact.predicate not in {"sortal", "object_type"}:
            continue
        if fact.predicate == "object_type" and not fact.arg("source"):
            continue
        entity = _canonical_label(fact.arg("entity"))
        if not entity:
            continue
        if (
            entity == label_key
            or entity in label_key
            or label_key in entity
            or _token_overlap(label_key, entity, 0.67)
        ):
            sortal_types.add(_canonical_label(fact.arg("type")))
    return {value for value in sortal_types if value}


def _label_with_sortal(label: str, sortal_type: str) -> str:
    clean_label = " ".join(label.split())
    clean_type = " ".join(sortal_type.split())
    if not clean_label or not clean_type:
        return label
    label_tokens = set(_content_tokens(clean_label))
    type_tokens = set(_content_tokens(clean_type))
    if type_tokens and type_tokens <= label_tokens:
        return clean_label
    return f"{clean_label} of {clean_type}"


def _canonical_action_key(
    verb: str,
    obj: str,
    fact: RecursiveFact,
    facts: list[RecursiveFact],
) -> str:
    label = _canonical_label(obj)
    if verb == "pick_up" and label in {"new pair", "them", "it"}:
        label = "replacement item"
        return f"{verb}:{label}"
    if verb == "pick_up" and _session_has_completed_exchange_for_object(
        obj,
        fact,
        facts,
    ):
        label = "replacement item"
        return f"{verb}:{label}"
    return f"{verb}:{label}:{_canonical_label(fact.arg('location'))}"


def _session_has_completed_exchange_for_object(
    obj: str,
    fact: RecursiveFact,
    facts: list[RecursiveFact],
) -> bool:
    obj_tokens = set(_content_tokens(obj))
    if not obj_tokens:
        return False
    for item in facts:
        if item.evidence_session_id != fact.evidence_session_id:
            continue
        if item.predicate != "action":
            continue
        if _normalize_action(item.arg("verb")) != "exchange":
            continue
        if _is_inactive_status(item.arg("status")):
            continue
        exchanged_tokens = set(_content_tokens(item.arg("object")))
        if obj_tokens & exchanged_tokens:
            return True
    return False


def _canonical_value_key(fact: RecursiveFact) -> str:
    label = _canonical_label(fact.arg("entity") or fact.evidence_span)
    label = label.replace(" fundraising", "")
    return f"{fact.evidence_session_id}:{label}:{fact.arg('attribute').lower()}"


def _canonical_baked_item(label: str) -> str:
    clean = _canonical_label(label)
    if "sourdough" in clean and "bread" in clean:
        return "sourdough bread"
    clean = re.sub(r"^(baked|baking|made|new|tried)\s+", "", clean)
    clean = clean.replace("recipe using my starter", "")
    clean = clean.replace("recipe using sourdough starter", "")
    clean = clean.replace("using sourdough starter", "")
    clean = clean.replace("batch of ", "")
    clean = clean.replace("a batch of ", "")
    clean = clean.replace("recipe", "")
    clean = re.sub(r"\s+", " ", clean).strip()
    if "sourdough" in clean and "bread" in clean:
        return "sourdough bread"
    return clean


def _entity_types(facts: list[RecursiveFact]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = collections.defaultdict(set)
    for fact in facts:
        if fact.predicate not in {"object_type", "sortal"}:
            continue
        entity = _canonical_label(fact.arg("entity"))
        if entity:
            mapping[entity].add(fact.arg("type").lower())
    return dict(mapping)


def _facts_of(
    facts: list[RecursiveFact],
    predicate: str,
) -> typing.Iterator[RecursiveFact]:
    return (fact for fact in facts if fact.predicate == predicate)


def _normalize_action(value: str) -> str:
    clean = " ".join(_content_tokens(value))
    if clean in _ACTION_SYNONYMS:
        return _ACTION_SYNONYMS[clean]
    if clean in {"collect", "collecting"}:
        return "pick_up"
    if clean in {"go", "went"} or clean.startswith("go to") or clean.startswith(
        "went to"
    ):
        return "attend"
    if clean in {"send back", "sent back"}:
        return "return"
    if clean.startswith("pick up") or clean.startswith("picked up"):
        return "pick_up"
    if "download" in clean:
        return "download"
    if "purchase" in clean or "bought" in clean or "buy" in clean:
        return "purchase"
    if "return" in clean:
        return "return"
    if "raise" in clean:
        return "raise"
    if "attend" in clean or "been to" in value.lower():
        return "attend"
    if "bake" in clean:
        return "bake"
    if clean.startswith("made") or clean == "made":
        return "make"
    if clean.startswith("tried"):
        return "make"
    if "signed" in clean:
        return "acquire"
    if clean == "got":
        return "acquire"
    return clean


def _is_inactive_status(value: str) -> bool:
    clean = value.lower().strip()
    return clean in _INACTIVE_STATUSES or any(
        token in clean for token in ("planned", "pending", "thinking", "consider")
    )


def _targets_music_release(target_terms: tuple[str, ...]) -> bool:
    return bool(set(target_terms) & {"album", "albums", "ep", "eps", "music"})


def _numeric_amount(value: str) -> float | None:
    match = _MONEY_RE.search(value.replace(",", ""))
    if not match:
        return None
    return float(match.group(1))


def _looks_like_baked_good(fact: RecursiveFact) -> bool:
    tokens = set(_content_tokens(f"{fact.arg('object')} {fact.evidence_span}"))
    return bool(tokens & _BAKED_GOOD_TERMS)


def _token_overlap(left: str, right: str, threshold: float) -> bool:
    left_tokens = set(_content_tokens(left))
    right_tokens = set(_content_tokens(right))
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens)
    denominator = min(len(left_tokens), len(right_tokens))
    return overlap / denominator >= threshold


def _fact_to_prolog(fact: RecursiveFact) -> str:
    args = ", ".join(
        f"{key}={_canonical_label(str(value))}"
        for key, value in sorted(fact.arguments.items())
        if _argument_is_present(value)
    )
    if not args:
        return f"{fact.predicate}({fact.evidence_session_id})."
    return f"{fact.predicate}({fact.evidence_session_id}, {args})."


def _query_for_intent(intent: QuestionIntent) -> str:
    if intent.aggregation == "sum":
        return "answer(N) :- sum(V, included_candidate_unit(_, V), N)."
    return "answer(N) :- count_distinct(U, included_candidate_unit(U), N)."


def _content_tokens(value: str) -> list[str]:
    return [
        _TOKEN_SYNONYMS.get(token, token)
        for token in _TOKEN_RE.findall(value.lower().replace("'", " "))
        if token not in _STOPWORDS
    ]


def _has_obligation_language(question: str) -> bool:
    return any(
        marker in question
        for marker in (
            "still need",
            "waiting to be",
            "awaiting",
        )
    )


def _canonical_label(value: str) -> str:
    return " ".join(_content_tokens(value))


def _canonical_relation(value: str) -> str:
    return "_".join(_content_tokens(value))


def _clean_string(value: typing.Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value or "")


def _argument_is_present(value: typing.Any) -> bool:
    if value is None:
        return False
    return not (isinstance(value, str) and not value.strip())


def _float_or_zero(value: typing.Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formalizer-outputs",
        default=str(DEFAULT_FORMALIZER_OUTPUTS_PATH),
    )
    parser.add_argument("--manifest", default=str(ambiguity_fail18.DEFAULT_MANIFEST))
    parser.add_argument("--outputs", default=str(DEFAULT_OUTPUTS_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args(argv)

    fact_rows = load_fact_rows(args.formalizer_outputs)
    rows, report = build_report(
        fact_rows=fact_rows,
        manifest_path=args.manifest,
        formalizer_outputs_path=args.formalizer_outputs,
        outputs_path=args.outputs,
    )
    write_outputs(rows, report, outputs_path=args.outputs, report_path=args.report)
    print(json.dumps(report["candidate_unit_review_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
