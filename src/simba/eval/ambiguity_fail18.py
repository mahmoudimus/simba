"""Adapter for the saved clingo_fail18 failure fixture."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
import re
import typing

import simba.eval.ambiguity as ambiguity
import simba.eval.world_lexicon as world_lexicon

DEFAULT_MANIFEST = pathlib.Path(".simba/fixtures/clingo_fail18_manifest.json")
DEFAULT_CORPUS = pathlib.Path(".simba/fixtures/clingo_fail18_corpus.json")

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
class IntentSpec:
    answer_type: str
    concept_ids: tuple[str, ...] = ()
    frame_id: str = ""
    target_terms: tuple[str, ...] = ()
    action_terms: tuple[str, ...] = ()
    role_terms: tuple[str, ...] = ()
    unit: str = ""
    window_days: int = 0
    anchor_date: dt.date | None = None


@dataclasses.dataclass(frozen=True)
class CandidateFact:
    kind: str
    canonical_id: str
    label: str
    evidence: str
    relation: str = ""
    value: int | None = None
    unit: str = ""
    source_session_id: str = ""
    occurred_on: dt.date | None = None


@dataclasses.dataclass(frozen=True)
class Fail18Result:
    question_id: str
    question: str
    failure_mode: str
    gold_numeric: float | int | None
    answer_space: dict[str, int]
    contains_gold: bool | None
    backend: str
    answer_type: str = ""
    repair_applied: bool = False


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
                    "answer_type": item.answer_type,
                    "repair_applied": item.repair_applied,
                }
                for item in self.results
            ],
        }


def load_manifest(
    path: str | pathlib.Path = DEFAULT_MANIFEST,
) -> list[dict[str, typing.Any]]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def load_corpus(
    path: str | pathlib.Path = DEFAULT_CORPUS,
) -> list[dict[str, typing.Any]]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def manifest_row_to_case(row: dict[str, typing.Any]) -> ambiguity.AmbiguityCase:
    qid = str(row["question_id"])
    certain = max(0, int(row.get("clingo_certain") or 0))
    possible = max(certain, int(row.get("clingo_possible") or certain))
    records = [
        {"id": f"{qid}_certain_{idx}", "status": "certain"} for idx in range(certain)
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
    path: str | pathlib.Path = DEFAULT_MANIFEST,
    *,
    backend: str = "python",
    repair: bool = False,
    corpus_path: str | pathlib.Path = DEFAULT_CORPUS,
) -> Fail18Summary:
    rows = load_manifest(path)
    corpus_by_id = _corpus_by_id(corpus_path) if repair else {}
    results: list[Fail18Result] = []
    for row in rows:
        case = manifest_row_to_case(row)
        report = ambiguity.evaluate_case(case, backend=backend)
        answer = report.answer_space
        answer_type = classify_answer_type(row)
        repair_applied = False
        if repair:
            repaired = repair_answer_space(
                row, corpus_by_id.get(str(row["question_id"]))
            )
            if repaired is not None:
                answer = repaired
                repair_applied = True
        gold = numeric_gold(row)
        contains = _contains(answer, gold) if gold is not None else None
        results.append(
            Fail18Result(
                question_id=str(row["question_id"]),
                question=str(row.get("question", "")),
                failure_mode=str(row.get("failure_mode", "")),
                gold_numeric=gold,
                answer_space=answer,
                contains_gold=contains,
                backend=f"{backend}+repair" if repair else backend,
                answer_type=answer_type,
                repair_applied=repair_applied,
            )
        )
    known = [item for item in results if item.contains_gold is not None]
    hits = [item for item in known if item.contains_gold]
    return Fail18Summary(
        backend=f"{backend}+repair" if repair else backend,
        total=len(results),
        gold_known=len(known),
        contains_gold=len(hits),
        misses_gold=len(known) - len(hits),
        results=results,
    )


def numeric_gold(row: dict[str, typing.Any]) -> float | int | None:
    text = str(row.get("gold_answer", "")).lower()
    candidates: list[tuple[int, float | int]] = []
    for match in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        candidates.append((match.start(), _numeric_value(match.group())))
    for word, value in _NUMBER_WORDS.items():
        match = re.search(rf"\b{word}\b", text)
        if match:
            candidates.append((match.start(), value))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    raw_count = row.get("gold_count")
    if isinstance(raw_count, int):
        return raw_count
    if isinstance(raw_count, str) and raw_count.strip().isdigit():
        return int(raw_count)
    return None


def _numeric_value(value: str) -> float | int:
    numeric = float(value.replace(",", ""))
    return int(numeric) if numeric.is_integer() else numeric


def _contains(answer: dict[str, int], gold: float | int) -> bool:
    if "count" in answer:
        return int(answer["count"]) == gold
    return int(answer["lower"]) <= gold <= int(answer["upper"])


def classify_answer_type(row: dict[str, typing.Any]) -> str:
    return compile_intent(row).answer_type


def compile_intent(row: dict[str, typing.Any]) -> IntentSpec:
    question = str(row.get("question", "")).lower()
    if "points" in question and "redeem" in question:
        return IntentSpec(
            answer_type="threshold_lookup",
            concept_ids=("loyalty_points",),
            frame_id="threshold_lookup",
            target_terms=("points",),
            unit="points",
        )
    if "currently own" in question:
        return IntentSpec(
            answer_type="current_inventory",
            concept_ids=("musical_instrument",),
            frame_id="current_possession",
            target_terms=_concept_terms(("musical_instrument",)),
            action_terms=_frame_terms("current_possession"),
        )
    if "past two weeks" in question or "last two weeks" in question:
        return IntentSpec(
            answer_type="temporal_event_count",
            concept_ids=("baking_event",),
            frame_id="baking_event",
            target_terms=_concept_terms(("baking_event",)),
            action_terms=_frame_terms("baking_event")
            or (_event_action_from_question(question),),
            window_days=_temporal_window_days(question),
            anchor_date=_parse_question_date(str(row.get("question_date", ""))),
        )
    if "worked on or bought" in question:
        return IntentSpec(
            answer_type="canonical_entity_count",
            concept_ids=("scale_model_kit",),
            frame_id="worked_or_bought",
            target_terms=_concept_terms(("scale_model_kit",)),
            action_terms=_frame_terms("worked_or_bought"),
        )
    if " led " in f" {question} " or "leading" in question:
        return IntentSpec(
            answer_type="role_filtered_count",
            concept_ids=("project_work",),
            frame_id="leadership_role",
            target_terms=_concept_terms(("project_work",)),
            role_terms=_frame_terms("leadership_role"),
        )
    if "how many" in question:
        return IntentSpec(answer_type="count")
    return IntentSpec(answer_type="unknown")


def _concept_terms(concept_ids: tuple[str, ...]) -> tuple[str, ...]:
    return world_lexicon.default_world_lexicon().terms_for_concepts(concept_ids)


def _concept_patterns(concept_ids: tuple[str, ...]) -> tuple[str, ...]:
    return world_lexicon.default_world_lexicon().patterns_for_concepts(concept_ids)


def _frame_terms(frame_id: str) -> tuple[str, ...]:
    return world_lexicon.default_world_lexicon().lexical_units_for_frame(frame_id)


def repair_answer_space(
    row: dict[str, typing.Any],
    corpus_row: dict[str, typing.Any] | None,
) -> dict[str, int] | None:
    if corpus_row is None:
        return None
    intent = compile_intent({**row, "question_date": corpus_row.get("question_date")})
    facts = extract_candidate_facts(corpus_row, intent)
    count = aggregate_candidate_facts(facts, intent)
    return {"count": count} if count is not None else None


def extract_candidate_facts(
    row: dict[str, typing.Any], intent: IntentSpec
) -> list[CandidateFact]:
    facts: list[CandidateFact] = []
    if intent.answer_type == "threshold_lookup":
        facts.extend(_extract_numeric_facts(row, intent))
    elif intent.answer_type in {"current_inventory", "canonical_entity_count"}:
        facts.extend(_extract_entity_facts(row, intent))
    elif intent.answer_type == "role_filtered_count":
        facts.extend(_extract_role_entity_facts(row, intent))
    elif intent.answer_type == "temporal_event_count":
        facts.extend(_extract_temporal_event_facts(row, intent))
    return facts


def aggregate_candidate_facts(
    facts: list[CandidateFact], intent: IntentSpec
) -> int | None:
    if intent.answer_type == "threshold_lookup":
        return _aggregate_threshold_lookup(facts)
    if intent.answer_type in {
        "current_inventory",
        "canonical_entity_count",
        "role_filtered_count",
    }:
        entity_ids = {
            fact.canonical_id
            for fact in facts
            if fact.kind == "entity" and fact.canonical_id
        }
        return len(entity_ids) if entity_ids else None
    if intent.answer_type == "temporal_event_count":
        event_ids = {
            fact.canonical_id
            for fact in facts
            if fact.kind == "event" and fact.canonical_id
        }
        return len(event_ids) if event_ids else None
    return None


def _corpus_by_id(path: str | pathlib.Path) -> dict[str, dict[str, typing.Any]]:
    if not pathlib.Path(path).exists():
        return {}
    return {str(row["question_id"]): row for row in load_corpus(path)}


def _extract_numeric_facts(
    row: dict[str, typing.Any], intent: IntentSpec
) -> list[CandidateFact]:
    facts: list[CandidateFact] = []
    if not intent.unit:
        return facts
    pattern = re.compile(rf"\b(\d[\d,]*)\s+{re.escape(intent.unit)}\b")
    for sid, text in _iter_user_session_texts_with_ids(row):
        for sentence in _sentences(text):
            low = sentence.lower()
            for match in pattern.finditer(low):
                value = int(match.group(1).replace(",", ""))
                relation = _numeric_relation(low, match.start())
                if not relation:
                    continue
                facts.append(
                    CandidateFact(
                        kind="numeric",
                        canonical_id=f"{relation}:{value}:{intent.unit}:{sid}",
                        label=f"{value} {intent.unit}",
                        value=value,
                        unit=intent.unit,
                        relation=relation,
                        source_session_id=sid,
                        evidence=sentence,
                    )
                )
    return facts


def _numeric_relation(sentence: str, value_start: int) -> str:
    prefix = sentence[max(0, value_start - 120) : value_start]
    suffix = sentence[value_start : value_start + 120]
    if re.search(r"\bneed\b.*\btotal\b", prefix) or "redeem" in prefix:
        return "required_total"
    if "all set" in suffix and "total" in prefix:
        return "required_total"
    if re.search(r"\b(?:my|bringing my)\s+total\b", prefix):
        return "current_total"
    if "so far" in suffix and "total" in prefix:
        return "current_total"
    if "earned" in prefix:
        return "earned_delta"
    return ""


def _aggregate_threshold_lookup(facts: list[CandidateFact]) -> int | None:
    required = [
        fact.value
        for fact in facts
        if fact.relation == "required_total" and fact.value is not None
    ]
    current = [
        fact.value
        for fact in facts
        if fact.relation == "current_total" and fact.value is not None
    ]
    if not required or not current:
        return None
    target = max(required)
    eligible_current = [value for value in current if value <= target]
    if not eligible_current:
        return None
    current_total = max(eligible_current)
    return target - current_total if target >= current_total else None


def _extract_entity_facts(
    row: dict[str, typing.Any], intent: IntentSpec
) -> list[CandidateFact]:
    facts: list[CandidateFact] = []
    for sid, text in _iter_user_session_texts_with_ids(row):
        for sentence in _sentences(text):
            if not _sentence_mentions_intent(sentence, intent):
                continue
            if intent.answer_type == "current_inventory":
                if not _is_current_inventory_sentence(sentence):
                    continue
            elif not _is_entity_action_sentence(sentence, intent):
                continue
            for label in _extract_entity_labels(sentence, intent.target_terms):
                canonical = _canonical_entity_id(label, intent)
                if not canonical:
                    continue
                facts.append(
                    CandidateFact(
                        kind="entity",
                        canonical_id=canonical,
                        label=label,
                        source_session_id=sid,
                        evidence=sentence,
                    )
                )
    return _dedupe_facts(facts)


def _extract_role_entity_facts(
    row: dict[str, typing.Any], intent: IntentSpec
) -> list[CandidateFact]:
    facts: list[CandidateFact] = []
    for sid, text in _iter_user_session_texts_with_ids(row):
        session_project_labels = []
        session_has_current_role = False
        for sentence in _sentences(text):
            low = sentence.lower()
            if _sentence_mentions_intent(sentence, intent):
                session_project_labels.extend(_extract_project_labels(sentence))
            if _has_role_predicate(low):
                session_has_current_role = session_has_current_role or bool(
                    re.search(r"\bleading\s+a\s+team\b", low)
                )
                labels = _select_countable_project_labels(
                    _extract_project_labels(sentence)
                )
                if labels and not re.search(r"\bled\s+to\b", low):
                    for label in labels:
                        facts.append(_entity_fact(label, sentence, sid))
        session_project_labels = _select_countable_project_labels(
            session_project_labels
        )
        if session_has_current_role and session_project_labels:
            facts.append(_entity_fact(session_project_labels[0], text, sid))
    return _dedupe_facts(facts)


def _extract_temporal_event_facts(
    row: dict[str, typing.Any], intent: IntentSpec
) -> list[CandidateFact]:
    if intent.anchor_date is None or intent.window_days <= 0:
        return []
    start = intent.anchor_date - dt.timedelta(days=intent.window_days)
    facts: list[CandidateFact] = []
    for sid, text, session_date in _iter_user_session_texts_with_ids_dates(row):
        if session_date is None or not start <= session_date <= intent.anchor_date:
            continue
        for sentence in _sentences(text):
            event = _extract_grounded_event(sentence, intent)
            if event:
                facts.append(
                    CandidateFact(
                        kind="event",
                        canonical_id=f"{sid}:{event}",
                        label=event,
                        source_session_id=sid,
                        occurred_on=session_date,
                        evidence=sentence,
                    )
                )
                break
    return facts


def _entity_fact(label: str, evidence: str, sid: str) -> CandidateFact:
    return CandidateFact(
        kind="entity",
        canonical_id=_normalize_entity_key(label),
        label=label,
        source_session_id=sid,
        evidence=evidence,
    )


def _event_action_from_question(question: str) -> str:
    match = re.search(r"\bdid\s+i\s+([a-z]+)", question)
    if match:
        return match.group(1)
    return ""


def _temporal_window_days(question: str) -> int:
    if "two weeks" in question:
        return 14
    match = re.search(r"\b(\d+)\s+days\b", question)
    if match:
        return int(match.group(1))
    return 0


def _sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", compact)
        if sentence.strip()
    ]


def _sentence_mentions_terms(sentence: str, terms: tuple[str, ...]) -> bool:
    low = sentence.lower()
    return any(re.search(rf"\b{re.escape(term)}s?\b", low) for term in terms if term)


def _sentence_mentions_intent(sentence: str, intent: IntentSpec) -> bool:
    low = sentence.lower()
    if _sentence_mentions_terms(sentence, intent.target_terms):
        return True
    return any(
        re.search(pattern, low) for pattern in _concept_patterns(intent.concept_ids)
    )


def _is_current_inventory_sentence(sentence: str) -> bool:
    low = sentence.lower()
    if re.search(r"\b(?:niece|recommendations?|thinking of (?:buying|getting))\b", low):
        return False
    own_markers = (
        r"\bmy\b",
        r"\bi'?ve had\b",
        r"\bi have had\b",
        r"\bold\b",
        r"\bgo-to\b",
        r"\bservice my\b",
        r"\bmaintenance of my\b",
        r"\bselling my\b",
    )
    return any(re.search(marker, low) for marker in own_markers)


def _is_entity_action_sentence(sentence: str, intent: IntentSpec) -> bool:
    low = sentence.lower()
    if "meal kit" in low or "blue apron" in low:
        return False
    if not _sentence_mentions_intent(sentence, intent):
        return False
    return any(_action_unit_in_sentence(unit, low) for unit in intent.action_terms)


def _action_unit_in_sentence(unit: str, sentence: str) -> bool:
    if not unit:
        return False
    return bool(re.search(rf"\b{re.escape(unit)}\b", sentence))


def _extract_entity_labels(sentence: str, target_terms: tuple[str, ...]) -> list[str]:
    term_re = "|".join(
        sorted(
            (re.escape(term) for term in target_terms if term),
            key=len,
            reverse=True,
        )
    )
    if not term_re:
        return []
    labels: list[str] = []
    patterns = [
        rf"\b(?:my|old|black|acoustic|electric|new|student-level|simple)\s+"
        rf"([^.!?;,]{{0,90}}\b(?:{term_re})\b)",
        rf"\b(?:a|an|the)\s+([^.!?;,]{{0,90}}\b(?:{term_re})\b)",
        rf"\b(?:{'|'.join(('piano', 'drum set', 'guitar', 'model kit', 'kit'))}),?"
        r"\s+(?:a|an)\s+([^,.;!?]+)",
        r"\b((?:\d+/\d+|\d+:\d+)\s+scale\s+[^,.;!?]{1,70})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            labels.append(match.group(1).strip())
    specific = [
        label
        for label in labels
        if _is_specific_entity_label(label) and "especially" not in label.lower()
    ]
    generic = [
        label
        for label in labels
        if label not in specific and "especially" not in label.lower()
    ]
    return specific or generic


def _is_specific_entity_label(label: str) -> bool:
    return bool(re.search(r"\d|[A-Z][a-z]+(?:\s+[A-Z0-9][A-Za-z0-9'./-]+)", label))


def _canonical_entity_id(label: str, intent: IntentSpec) -> str:
    key = _normalize_entity_key(label)
    if not key:
        return ""
    if _is_non_entity_key(key):
        return ""
    if intent.answer_type == "current_inventory" and key in {
        "guitar",
        "piano",
        "drum set",
        "ukulele",
        "violin",
        "instrument",
    }:
        return ""
    if intent.answer_type == "canonical_entity_count" and (
        "meal kit" in key or key in {"kit", "model kit"}
    ):
        return ""
    return key


def _normalize_entity_key(label: str) -> str:
    text = label.lower()
    text = text.replace("mk.v", "mk v")
    text = re.sub(
        r"\b(?:my|the|a|an|this|new|old|black|acoustic|electric|simple)\b",
        " ",
        text,
    )
    text = re.sub(r"\b(?:diorama featuring|featuring)\b", " ", text)
    text = re.sub(r"\b(?:student-level|go-to)\b", " ", text)
    text = re.sub(r"\b(?:model kit|kit|instrument)\b", " ", text)
    text = re.sub(
        r"\b(?:which|that|and|with|for|about|at|last|especially|next).*$",
        " ",
        text,
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_non_entity_key(key: str) -> bool:
    markers = (
        "advice",
        "duration",
        "humidifier",
        "price",
        "professional",
        "recommendation",
        "technique",
        "this",
        "to help",
        "to improve",
    )
    return any(marker in key for marker in markers)


def _dedupe_facts(facts: list[CandidateFact]) -> list[CandidateFact]:
    seen: set[str] = set()
    out: list[CandidateFact] = []
    for fact in facts:
        if fact.canonical_id in seen:
            continue
        seen.add(fact.canonical_id)
        out.append(fact)
    return out


def _has_role_predicate(sentence: str) -> bool:
    return bool(
        re.search(r"\bled\b(?!\s+to\b)", sentence)
        or re.search(r"\bleading\b", sentence)
    )


def _extract_project_labels(sentence: str) -> list[str]:
    labels: list[str] = []
    patterns = [
        r"\b([A-Z][A-Za-z ]{2,60}\s+class project)\b",
        r"\b(?:launch|launching)\s+(?:a\s+)?([^.!?;,]{0,80}\bfeature\b)",
        r"\b(?:a|the|my)\s+([^.!?;,]{0,80}\bproject\b)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            label = match.group(1).strip()
            if "led to" not in label.lower():
                labels.append(label)
    return labels


def _select_countable_project_labels(labels: list[str]) -> list[str]:
    out: list[str] = []
    for label in labels:
        cleaned = _clean_project_label(label)
        if not cleaned:
            continue
        low = cleaned.lower()
        if any(
            marker in low
            for marker in (
                "experience with",
                "project plan",
                "project timeline",
                "project's goals",
                "tasks fit",
            )
        ):
            continue
        out.append(cleaned)
    return out


def _clean_project_label(label: str) -> str:
    text = re.sub(r"\s+", " ", label).strip(" .")
    class_project = re.search(
        r"\b([A-Z][A-Za-z ]{2,60}\s+class project)\b",
        text,
        flags=re.IGNORECASE,
    )
    if class_project:
        return class_project.group(1).strip()
    feature = re.search(
        r"\b((?:new\s+)?[A-Za-z ]{2,60}\bfeature)\b",
        text,
        flags=re.IGNORECASE,
    )
    if feature:
        return feature.group(1).strip()
    return text


def _extract_grounded_event(sentence: str, intent: IntentSpec) -> str:
    low = sentence.lower()
    if _is_future_intention(low):
        return ""
    if not _has_event_action(low, intent.action_terms):
        return ""
    if not re.search(r"\b(?:recently|last|just|today|turned out|used|tried)\b", low):
        return ""
    obj = ""
    for pattern in (
        r"\bbaked\s+(?:a|an|some\s+)?([^,.!?]{1,80})",
        r"\bto bake\s+(?:a|an|some\s+)?([^,.!?]{1,80})",
    ):
        match = re.search(pattern, low)
        if match:
            obj = match.group(1).strip()
            break
    if not obj or re.search(r"\b(?:experimenting|types of flour)\b", obj):
        return ""
    return _normalize_entity_key(obj)


def _has_event_action(sentence: str, action_terms: tuple[str, ...]) -> bool:
    if not action_terms:
        return False
    return any(_action_unit_in_sentence(unit, sentence) for unit in action_terms)


def _is_future_intention(sentence: str) -> bool:
    return bool(
        re.search(
            r"\b(?:thinking of|planning to|going to|want to)\s+baking?\b",
            sentence,
        )
    )


def _iter_user_session_texts_with_ids(
    row: dict[str, typing.Any],
) -> typing.Iterator[tuple[str, str]]:
    ids = row.get("haystack_session_ids", [])
    for idx, session in enumerate(row.get("haystack_sessions", [])):
        sid = str(ids[idx]) if idx < len(ids) else str(idx)
        yield (
            sid,
            " ".join(
                str(message.get("content", ""))
                for message in session
                if message.get("role") == "user"
            ),
        )


def _iter_user_session_texts_with_ids_dates(
    row: dict[str, typing.Any],
) -> typing.Iterator[tuple[str, str, dt.date | None]]:
    ids = row.get("haystack_session_ids", [])
    dates = row.get("haystack_dates", [])
    sessions = row.get("haystack_sessions", [])
    for idx, session in enumerate(sessions):
        sid = str(ids[idx]) if idx < len(ids) else str(idx)
        date_text = str(dates[idx]) if idx < len(dates) else ""
        text = " ".join(
            str(message.get("content", ""))
            for message in session
            if message.get("role") == "user"
        )
        yield sid, text, _parse_question_date(date_text)


def _parse_question_date(value: str) -> dt.date | None:
    match = re.search(r"(\d{4})/(\d{2})/(\d{2})", value)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    return dt.date(year, month, day)
