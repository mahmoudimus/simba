"""Relevant-window selector for answer-unit witness payloads."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import functools
import json
import math
import pathlib
import re
import typing

from simba.eval import answer_unit_witness, type_ontology

DEFAULT_SOURCE_PAYLOADS_PATH = pathlib.Path(
    "_gitless/fail18_answer_unit_witness_payloads_8k.json"
)
DEFAULT_SELECTED_PAYLOADS_PATH = pathlib.Path(
    "_gitless/fail18_answer_unit_witness_payloads_selector_a5_augment.json"
)
DEFAULT_GOLD_SPAN_NEEDLES_PATH = pathlib.Path("_gitless/fail18_gold_span_needles.json")
DEFAULT_PRE_METRICS_PATH = pathlib.Path(
    "_gitless/fail18_answer_unit_witness_selector_a5_augment_pre_metrics.json"
)
DEFAULT_MARGIN_GATE_EXCLUDED_CASE_IDS = (
    "88432d0a",
    "b5ef892d",
    "gpt4_15e38248",
    "gpt4_194be4b3",
)
DEFAULT_SELECTOR_NAME = "selector_a5_augment"
DEFAULT_PREFIX_FLOOR_CHARS = 2500
DEFAULT_TYPE_CUE_WEIGHT = 0.25
DEFAULT_OPERATION_CUE_WEIGHT = 0.5
DEFAULT_OPERATION_CUE_RADIUS_CHARS = 120

SELECTOR_VERSION = "answer_window_selector_a5"
SELECTOR_GAP = "\n...[selector_gap]...\n"

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*")
_ROLE_RE = re.compile(r"(?m)(USER|ASSISTANT):")
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
    "does",
    "for",
    "from",
    "had",
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
class WindowSelectorConfig:
    """Ablatable selector configuration.

    Bare construction stays component-neutral for tests and ablations. The module
    DEFAULT_CONFIG below is the promoted A5 prefix-floor augment selector.
    """

    name: str = "selector_a2"
    prefix_floor_chars: int = 0
    window_radius_chars: int = 800
    max_windows_per_session: int = 2
    max_chars_per_session: int = 4000
    role_weight: float = 1.5
    compactness_weight: float = 0.75
    compactness_cap: float = 8.0
    type_cue_weight: float = 0.0
    type_cue_roles: tuple[str, ...] = ("USER",)
    operation_cue_weight: float = 0.0
    operation_cue_radius_chars: int = 120
    operation_cue_roles: tuple[str, ...] = ("USER",)
    distractor_penalty_weight: float = 0.0
    fallback_chars_per_session: int = 1000
    lexicon_path: str | None = None

    def to_dict(self) -> dict[str, typing.Any]:
        return dataclasses.asdict(self)


DEFAULT_CONFIG = WindowSelectorConfig(
    name=DEFAULT_SELECTOR_NAME,
    prefix_floor_chars=DEFAULT_PREFIX_FLOOR_CHARS,
    type_cue_weight=DEFAULT_TYPE_CUE_WEIGHT,
    operation_cue_weight=DEFAULT_OPERATION_CUE_WEIGHT,
    operation_cue_radius_chars=DEFAULT_OPERATION_CUE_RADIUS_CHARS,
)


@dataclasses.dataclass(frozen=True)
class RoleSpan:
    role: str
    label_start: int
    content_start: int
    end: int


@dataclasses.dataclass(frozen=True)
class TypeCueHit:
    start: int
    end: int
    matches: tuple[dict[str, typing.Any], ...]


@dataclasses.dataclass(frozen=True)
class OperationCueHit:
    start: int
    end: int
    verb: str
    source_type: str


@dataclasses.dataclass(frozen=True)
class CandidateWindow:
    session_id: str
    role: str
    start: int
    end: int
    score: float
    score_components: dict[str, float]
    type_cue_matches: tuple[dict[str, typing.Any], ...] = ()
    type_cue_hits: tuple[TypeCueHit, ...] = ()
    operation_cue_matches: tuple[dict[str, typing.Any], ...] = ()
    operation_cue_hits: tuple[OperationCueHit, ...] = ()

    def to_metadata(self) -> dict[str, typing.Any]:
        metadata = {
            "session_id": self.session_id,
            "role": self.role,
            "start": self.start,
            "end": self.end,
            "chars": self.end - self.start,
            "score": round(self.score, 6),
            "score_components": {
                key: round(value, 6)
                for key, value in sorted(self.score_components.items())
            },
        }
        if self.type_cue_matches:
            metadata["type_cue_matches"] = list(self.type_cue_matches)
        if self.operation_cue_matches:
            metadata["operation_cue_matches"] = list(self.operation_cue_matches)
        return metadata


def build_selected_payload_artifact(
    *,
    source_payloads_path: str | pathlib.Path = DEFAULT_SOURCE_PAYLOADS_PATH,
    config: WindowSelectorConfig = DEFAULT_CONFIG,
) -> dict[str, typing.Any]:
    """Build provider-compatible payloads with selected evidence windows."""
    source = _load_json(source_payloads_path)
    selected_payloads = []
    metadata_cases: dict[str, typing.Any] = {}
    for payload in source.get("payloads", []):
        if not isinstance(payload, dict):
            continue
        selected, metadata = select_payload_windows(payload, config=config)
        selected_payloads.append(selected)
        metadata_cases[str(selected.get("case", {}).get("id", ""))] = metadata
    return {
        **source,
        "name": f"{source.get('name', 'answer-unit-witness')}-{config.name}",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_payload_artifact": str(source_payloads_path),
        "selector": {
            "version": SELECTOR_VERSION,
            "config": config.to_dict(),
            "provider_payloads_contain_gold": False,
            "gold_spans_visible_to_provider": False,
            "scoring_components": [
                "base_question_overlap",
                "role_weight",
                "compactness",
                "type_cue",
                "operation_cue",
                "distractor_penalty",
            ],
        },
        "retrieval": {
            **source.get("retrieval", {}),
            "selector": SELECTOR_VERSION,
            "source_chars_per_session": source.get("retrieval", {}).get(
                "chars_per_session"
            ),
            "selected_max_chars_per_session": config.max_chars_per_session,
        },
        "total": len(selected_payloads),
        "payloads": selected_payloads,
        "selection_metadata": {
            "artifact_kind": "answer_window_selector_metadata",
            "cases": metadata_cases,
        },
    }


def select_payload_windows(
    payload: dict[str, typing.Any],
    *,
    config: WindowSelectorConfig = DEFAULT_CONFIG,
) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
    """Return a cloned payload with evidence sessions reduced to selected text."""
    cloned = json.loads(json.dumps(payload))
    question = str(cloned.get("case", {}).get("question", ""))
    sessions = cloned.get("case", {}).get("evidence_sessions", [])
    source_texts = {
        str(session.get("session_id", "")): str(session.get("text", ""))
        for session in sessions
        if isinstance(session, dict)
    }
    question_terms = question_terms_from_text(question)
    question_type_targets = question_type_targets_from_text(question)
    term_weights = _term_weights(question_terms, source_texts.values())
    metadata_sessions: dict[str, typing.Any] = {}
    selected_sessions = []
    for session in sessions:
        if not isinstance(session, dict):
            continue
        session_id = str(session.get("session_id", ""))
        source_text = str(session.get("text", ""))
        windows = select_session_windows(
            source_text,
            session_id=session_id,
            question_terms=question_terms,
            question_type_targets=question_type_targets,
            term_weights=term_weights,
            config=config,
        )
        selected_text = _render_selected_text(source_text, windows)
        selected_session = {**session, "text": selected_text}
        selected_sessions.append(selected_session)
        metadata_sessions[session_id] = {
            "source_chars": len(source_text),
            "selected_chars": len(selected_text),
            "selected_window_count": len(windows),
            "windows": [window.to_metadata() for window in windows],
        }
    cloned["case"]["evidence_sessions"] = selected_sessions
    metadata = {
        "question_terms": list(question_terms),
        "question_type_targets": list(question_type_targets),
        "sessions": metadata_sessions,
    }
    return typing.cast("dict[str, typing.Any]", cloned), metadata


def select_session_windows(
    text: str,
    *,
    session_id: str,
    question_terms: tuple[str, ...],
    question_type_targets: tuple[str, ...] = (),
    term_weights: dict[str, float],
    config: WindowSelectorConfig = DEFAULT_CONFIG,
) -> list[CandidateWindow]:
    """Select scored windows from a single rendered session."""
    candidates: list[CandidateWindow] = []
    for role_span in _role_spans(text):
        candidates.extend(
            _candidate_windows_for_span(
                text,
                role_span=role_span,
                session_id=session_id,
                question_terms=question_terms,
                question_type_targets=question_type_targets,
                term_weights=term_weights,
                config=config,
            )
        )
    if config.prefix_floor_chars > 0:
        return _select_augmented_session_windows(
            text,
            session_id=session_id,
            candidates=candidates,
            term_weights=term_weights,
            config=config,
        )
    if not candidates:
        return [
            CandidateWindow(
                session_id=session_id,
                role="UNKNOWN",
                start=0,
                end=min(len(text), config.fallback_chars_per_session),
                score=0.0,
                score_components={
                    "base_question_overlap": 0.0,
                    "role_weight": 0.0,
                    "compactness": 0.0,
                    "type_cue": 0.0,
                    "operation_cue": 0.0,
                    "distractor_penalty": 0.0,
                },
            )
        ]
    selected: list[CandidateWindow] = []
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.start)):
        if _overlaps_mergeable(candidate, selected):
            selected = _merge_overlapping(
                candidate,
                selected,
                text,
                term_weights,
                config,
            )
        elif len(selected) < config.max_windows_per_session:
            selected.append(candidate)
        selected = _trim_to_char_budget(selected, config.max_chars_per_session)
        if (
            len(selected) >= config.max_windows_per_session
            and _selected_chars(selected) >= config.max_chars_per_session
        ):
            break
    return sorted(selected, key=lambda item: item.start)


def _select_augmented_session_windows(
    text: str,
    *,
    session_id: str,
    candidates: list[CandidateWindow],
    term_weights: dict[str, float],
    config: WindowSelectorConfig,
) -> list[CandidateWindow]:
    prefix_end = min(len(text), max(config.prefix_floor_chars, 0))
    prefix_window = _prefix_floor_window(
        text,
        session_id=session_id,
        prefix_end=prefix_end,
        candidates=candidates,
    )
    extra_candidates = [
        _clamp_candidate_after_prefix(
            candidate,
            text=text,
            prefix_end=prefix_end,
            term_weights=term_weights,
            config=config,
        )
        for candidate in candidates
        if candidate.end > prefix_end
    ]
    extra_candidates = [
        candidate
        for candidate in extra_candidates
        if candidate is not None and candidate.end > candidate.start
    ]
    selected: list[CandidateWindow] = []
    for candidate in sorted(
        extra_candidates,
        key=lambda item: (-item.score, item.start),
    ):
        if _overlaps_mergeable(candidate, selected):
            selected = _merge_overlapping(
                candidate,
                selected,
                text,
                term_weights,
                config,
            )
        elif len(selected) < config.max_windows_per_session:
            selected.append(candidate)
        selected = _trim_augmented_to_char_budget(
            prefix_window,
            selected,
            config.max_chars_per_session,
        )
        if (
            len(selected) >= config.max_windows_per_session
            and _selected_chars([prefix_window, *selected])
            >= config.max_chars_per_session
        ):
            break
    return [prefix_window, *sorted(selected, key=lambda item: item.start)]


def _prefix_floor_window(
    text: str,
    *,
    session_id: str,
    prefix_end: int,
    candidates: list[CandidateWindow],
) -> CandidateWindow:
    best_prefix_candidate = max(
        (
            candidate
            for candidate in candidates
            if candidate.start < prefix_end and candidate.end > 0
        ),
        key=lambda item: item.score,
        default=None,
    )
    score = best_prefix_candidate.score if best_prefix_candidate else 0.0
    score_components = (
        dict(best_prefix_candidate.score_components)
        if best_prefix_candidate
        else {
            "base_question_overlap": 0.0,
            "role_weight": 0.0,
            "compactness": 0.0,
            "type_cue": 0.0,
            "operation_cue": 0.0,
            "distractor_penalty": 0.0,
        }
    )
    score_components["prefix_floor"] = 1.0
    return CandidateWindow(
        session_id=session_id,
        role="PREFIX",
        start=0,
        end=prefix_end,
        score=score,
        score_components=score_components,
        type_cue_matches=(
            best_prefix_candidate.type_cue_matches if best_prefix_candidate else ()
        ),
        type_cue_hits=best_prefix_candidate.type_cue_hits
        if best_prefix_candidate
        else (),
        operation_cue_matches=(
            best_prefix_candidate.operation_cue_matches if best_prefix_candidate else ()
        ),
        operation_cue_hits=best_prefix_candidate.operation_cue_hits
        if best_prefix_candidate
        else (),
    )


def _clamp_candidate_after_prefix(
    candidate: CandidateWindow,
    *,
    text: str,
    prefix_end: int,
    term_weights: dict[str, float],
    config: WindowSelectorConfig,
) -> CandidateWindow | None:
    start = max(candidate.start, prefix_end)
    end = candidate.end
    if end <= start:
        return None
    return _scored_window(
        text,
        session_id=candidate.session_id,
        role=candidate.role,
        start=start,
        end=end,
        term_weights=term_weights,
        type_cue_hits=candidate.type_cue_hits,
        operation_cue_hits=candidate.operation_cue_hits,
        config=config,
    )


def build_fail18_selector_pre_metrics(
    *,
    selected_payloads_path: str | pathlib.Path = DEFAULT_SELECTED_PAYLOADS_PATH,
    source_payloads_path: str | pathlib.Path = DEFAULT_SOURCE_PAYLOADS_PATH,
    gold_span_needles_path: str | pathlib.Path = DEFAULT_GOLD_SPAN_NEEDLES_PATH,
    baseline_pre_metrics_path: str | pathlib.Path | None = None,
    margin_gate_excluded_case_ids: typing.Iterable[str] = (
        DEFAULT_MARGIN_GATE_EXCLUDED_CASE_IDS
    ),
) -> dict[str, typing.Any]:
    """Build fail18-only span survival and salience-margin diagnostics."""
    selected_artifact = _load_json(selected_payloads_path)
    source_artifact = _load_json(source_payloads_path)
    needles_artifact = _load_json(gold_span_needles_path)
    baseline_cases = _pre_metric_cases_by_id(
        _load_json(baseline_pre_metrics_path) if baseline_pre_metrics_path else None
    )
    margin_gate_excluded = set(margin_gate_excluded_case_ids)
    selected_payloads = _payloads_by_case(selected_artifact)
    source_payloads = _payloads_by_case(source_artifact)
    metadata_by_case = (
        selected_artifact.get("selection_metadata", {}).get("cases", {})
        if isinstance(selected_artifact.get("selection_metadata"), dict)
        else {}
    )
    cases = []
    total_needles = 0
    source_present = 0
    selected_present = 0
    selector_dropped = 0
    for needle_case in needles_artifact.get("cases", []):
        if not isinstance(needle_case, dict):
            continue
        case = _pre_metric_case(
            needle_case=needle_case,
            selected_payload=selected_payloads.get(str(needle_case.get("case_id", ""))),
            source_payload=source_payloads.get(str(needle_case.get("case_id", ""))),
            metadata=metadata_by_case.get(str(needle_case.get("case_id", "")), {}),
            baseline_case=baseline_cases.get(str(needle_case.get("case_id", ""))),
            margin_gate_excluded=(
                str(needle_case.get("case_id", "")) in margin_gate_excluded
            ),
        )
        cases.append(case)
        total_needles += len(case["needles"])
        source_present += sum(
            1 for needle in case["needles"] if needle["source_contains_exact"]
        )
        selected_present += sum(
            1 for needle in case["needles"] if needle["selected_contains_exact"]
        )
        selector_dropped += sum(
            1 for needle in case["needles"] if needle["selector_dropped_source_span"]
        )
    cases_with_selector_drops = [
        case["case_id"] for case in cases if case["selector_dropped_count"] > 0
    ]
    cases_with_margin_regressions = [
        case["case_id"] for case in cases if case["salience_margin_regressed"]
    ]
    cases_excluded_from_margin_gate = [
        case["case_id"] for case in cases if case["margin_gate_excluded"]
    ]
    margin_values = [
        float(case["salience_margin_min"])
        for case in cases
        if case["salience_margin_min"] is not None
    ]
    span_survival_gate_passed = selector_dropped == 0
    margin_gate_passed = not cases_with_margin_regressions
    return {
        "name": "fail18-answer-window-selector-pre-metrics",
        "artifact_kind": "answer_window_selector_pre_metrics",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_payloads": str(source_payloads_path),
        "selected_payloads": str(selected_payloads_path),
        "source_gold_span_needles": str(gold_span_needles_path),
        "baseline_pre_metrics": (
            str(baseline_pre_metrics_path) if baseline_pre_metrics_path else None
        ),
        "margin_gate_excluded_case_ids": sorted(margin_gate_excluded),
        "selector": selected_artifact.get("selector", {}),
        "summary": {
            "case_count": len(cases),
            "needle_count": total_needles,
            "source_present_count": source_present,
            "selected_present_count": selected_present,
            "selector_dropped_count": selector_dropped,
            "cases_with_selector_drops": cases_with_selector_drops,
            "span_survival_gate_passed": span_survival_gate_passed,
            "salience_margin_min": min(margin_values) if margin_values else None,
            "salience_margin_nonpositive_cases": [
                case["case_id"]
                for case in cases
                if case["salience_margin_min"] is not None
                and float(case["salience_margin_min"]) <= 0
            ],
            "margin_regression_count": len(cases_with_margin_regressions),
            "cases_with_margin_regressions": cases_with_margin_regressions,
            "margin_gate_exclusion_count": len(cases_excluded_from_margin_gate),
            "cases_excluded_from_margin_gate": cases_excluded_from_margin_gate,
            "margin_gate_passed": margin_gate_passed,
            "kill_gate_passed": (span_survival_gate_passed and margin_gate_passed),
            "provider_run_allowed": (span_survival_gate_passed and margin_gate_passed),
        },
        "cases": cases,
    }


def question_terms_from_text(question: str) -> tuple[str, ...]:
    terms: list[str] = []
    for token in _TOKEN_RE.findall(question.casefold()):
        if len(token) < 2 or token in _QUESTION_STOPWORDS:
            continue
        terms.append(token)
        terms.extend(_term_variants(token))
    return tuple(dict.fromkeys(terms))


def question_type_targets_from_text(question: str) -> tuple[str, ...]:
    """Return candidate type targets from the question for ontology cueing."""
    tokens = [
        token
        for token in _TOKEN_RE.findall(question.casefold())
        if len(token) > 1 and token not in _QUESTION_STOPWORDS
    ]
    targets: list[str] = []
    for ngram_size in (3, 2, 1):
        for index in range(0, len(tokens) - ngram_size + 1):
            phrase = " ".join(tokens[index : index + ngram_size])
            targets.extend(_phrase_type_variants(phrase))
    return tuple(dict.fromkeys(targets))


def _term_variants(token: str) -> tuple[str, ...]:
    variants = []
    if token.endswith("s") and len(token) > 3:
        variants.append(token[:-1])
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        variants.append(stem)
    if token.endswith("ed") and len(token) > 4:
        variants.append(token[:-2])
    return tuple(variant for variant in variants if variant and variant != token)


def _phrase_type_variants(phrase: str) -> tuple[str, ...]:
    words = phrase.split()
    if not words:
        return ()
    variants = {phrase}
    for singular in _singular_variants(words[-1]):
        variants.add(" ".join((*words[:-1], singular)))
    return tuple(sorted(variants))


def _singular_variants(word: str) -> tuple[str, ...]:
    variants = {word}
    if word.endswith("ies") and len(word) > 3:
        variants.add(f"{word[:-3]}y")
    if word.endswith("es") and len(word) > 3:
        variants.add(word[:-2])
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        variants.add(word[:-1])
    return tuple(sorted(variants))


def _candidate_windows_for_span(
    text: str,
    *,
    role_span: RoleSpan,
    session_id: str,
    question_terms: tuple[str, ...],
    question_type_targets: tuple[str, ...],
    term_weights: dict[str, float],
    config: WindowSelectorConfig,
) -> list[CandidateWindow]:
    content = text[role_span.content_start : role_span.end]
    lexical_hits = [
        (role_span.content_start + start, role_span.content_start + end)
        for start, end in _term_hits(content, question_terms)
    ]
    relative_type_hits = tuple(
        _type_cue_hits(
            content,
            role=role_span.role,
            question_type_targets=question_type_targets,
            config=config,
        )
    )
    type_hits = tuple(
        TypeCueHit(
            start=role_span.content_start + hit.start,
            end=role_span.content_start + hit.end,
            matches=hit.matches,
        )
        for hit in relative_type_hits
    )
    operation_hits = tuple(
        OperationCueHit(
            start=role_span.content_start + hit.start,
            end=role_span.content_start + hit.end,
            verb=hit.verb,
            source_type=hit.source_type,
        )
        for hit in _operation_cue_hits(
            content,
            role=role_span.role,
            type_cue_hits=relative_type_hits,
            config=config,
        )
    )
    regular_hits = [
        *lexical_hits,
        *((hit.start, hit.end) for hit in type_hits),
    ]
    hits = [*regular_hits, *((hit.start, hit.end) for hit in operation_hits)]
    if not hits:
        return []
    candidate_ranges: set[tuple[int, int]] = set()
    for start, end in regular_hits:
        candidate_ranges.add(
            (
                max(role_span.label_start, start - config.window_radius_chars),
                min(role_span.end, end + config.window_radius_chars),
            )
        )
    for hit in operation_hits:
        candidate_ranges.add(
            (
                max(
                    role_span.label_start,
                    hit.start - config.operation_cue_radius_chars,
                ),
                min(role_span.end, hit.end + config.operation_cue_radius_chars),
            )
        )
    if regular_hits:
        first_hit = min(start for start, _ in regular_hits)
        last_hit = max(end for _, end in regular_hits)
        combined_range = (
            max(role_span.label_start, first_hit - config.window_radius_chars),
            min(role_span.end, last_hit + config.window_radius_chars),
        )
        if combined_range[1] - combined_range[0] <= config.max_chars_per_session:
            candidate_ranges.add(combined_range)
    windows = []
    for start, end in candidate_ranges:
        if end <= start:
            continue
        windows.append(
            _scored_window(
                text,
                session_id=session_id,
                role=role_span.role,
                start=start,
                end=end,
                term_weights=term_weights,
                type_cue_hits=type_hits,
                operation_cue_hits=operation_hits,
                config=config,
            )
        )
    return windows


def _scored_window(
    text: str,
    *,
    session_id: str,
    role: str,
    start: int,
    end: int,
    term_weights: dict[str, float],
    type_cue_hits: tuple[TypeCueHit, ...],
    operation_cue_hits: tuple[OperationCueHit, ...],
    config: WindowSelectorConfig,
) -> CandidateWindow:
    window_text = text[start:end]
    lowered = window_text.casefold()
    base = sum(weight for term, weight in term_weights.items() if term in lowered)
    window_type_hits = tuple(
        hit for hit in type_cue_hits if start <= hit.start and hit.end <= end
    )
    type_cue_matches = _type_cue_matches_for_window(
        type_cue_hits=window_type_hits,
    )
    type_cue = config.type_cue_weight * min(len(type_cue_matches), 3)
    window_operation_hits = tuple(
        hit for hit in operation_cue_hits if start <= hit.start and hit.end <= end
    )
    operation_cue_matches = _operation_cue_matches_for_window(
        operation_cue_hits=window_operation_hits,
    )
    operation_cue = config.operation_cue_weight * min(len(operation_cue_matches), 3)
    operation_type_unlock = operation_cue > 0 and type_cue > 0
    unlock_signal = base if base > 0 else type_cue + operation_cue
    role_component = (
        config.role_weight
        if role == "USER" and (base > 0 or operation_type_unlock)
        else 0.0
    )
    compactness = 0.0
    if base > 0 or operation_type_unlock:
        compactness = min(
            config.compactness_weight
            * (unlock_signal / max(len(window_text), 1))
            * 1000,
            config.compactness_cap,
        )
    distractor_penalty = 0.0
    score = (
        base
        + type_cue
        + role_component
        + compactness
        + operation_cue
        - distractor_penalty
    )
    return CandidateWindow(
        session_id=session_id,
        role=role,
        start=start,
        end=end,
        score=score,
        score_components={
            "base_question_overlap": base,
            "role_weight": role_component,
            "compactness": compactness,
            "type_cue": type_cue,
            "operation_cue": operation_cue,
            "distractor_penalty": distractor_penalty,
        },
        type_cue_matches=type_cue_matches,
        type_cue_hits=window_type_hits,
        operation_cue_matches=operation_cue_matches,
        operation_cue_hits=window_operation_hits,
    )


def _role_spans(text: str) -> list[RoleSpan]:
    matches = list(_ROLE_RE.finditer(text))
    if not matches:
        return [RoleSpan(role="UNKNOWN", label_start=0, content_start=0, end=len(text))]
    spans = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        spans.append(
            RoleSpan(
                role=match.group(1),
                label_start=match.start(),
                content_start=match.end(),
                end=end,
            )
        )
    return spans


def _operation_cue_hits(
    text: str,
    *,
    role: str,
    type_cue_hits: tuple[TypeCueHit, ...],
    config: WindowSelectorConfig,
) -> tuple[OperationCueHit, ...]:
    if (
        config.operation_cue_weight <= 0
        or role not in config.operation_cue_roles
        or not type_cue_hits
    ):
        return ()
    hits: list[OperationCueHit] = []
    seen: set[tuple[int, int, str, str]] = set()
    for type_hit in type_cue_hits:
        for source_type in _source_types_for_type_hit(type_hit):
            for start, end, verb in _acquisition_matches_for_source_type(
                text,
                source_type,
            ):
                item = (start, end, verb, source_type)
                if item in seen:
                    continue
                seen.add(item)
                hits.append(
                    OperationCueHit(
                        start=start,
                        end=end,
                        verb=verb,
                        source_type=source_type,
                    )
                )
    return tuple(sorted(hits, key=lambda hit: (hit.start, hit.end, hit.source_type)))


def _source_types_for_type_hit(type_hit: TypeCueHit) -> tuple[str, ...]:
    source_types: list[str] = []
    for match in type_hit.matches:
        source_type = str(match.get("source_type", "")).strip()
        if source_type:
            source_types.append(source_type)
    return tuple(dict.fromkeys(source_types))


def _acquisition_matches_for_source_type(
    text: str,
    source_type: str,
) -> tuple[tuple[int, int, str], ...]:
    source_type = re.sub(r"\s+", " ", source_type.casefold()).strip()
    if not source_type:
        return ()
    escaped_source_type = re.escape(source_type).replace(r"\ ", r"\s+")
    verb_pattern = "|".join(
        re.escape(verb).replace(r"\ ", r"\s+")
        for verb in answer_unit_witness.ACQUISITION_VERBS
    )
    patterns = (
        rf"\b{escaped_source_type}\b\s*,?\s*(?:which|that)\s+i\s+"
        rf"\b(?P<verb>{verb_pattern})\b",
        rf"\bi\s+\b(?P<verb>{verb_pattern})\b[\s\S]{{0,80}}"
        rf"\b{escaped_source_type}\b",
        rf"\b(?P<verb>{verb_pattern})\b[\s\S]{{0,80}}"
        rf"\b{escaped_source_type}\b",
    )
    matches: list[tuple[int, int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            context_start = max(0, match.start() - 80)
            context_end = min(len(text), match.end() + 120)
            if answer_unit_witness._acquisition_span_is_blocked(
                text[context_start:context_end]
            ):
                continue
            matches.append(
                (
                    match.start(),
                    match.end(),
                    re.sub(r"\s+", " ", match.group("verb").casefold()),
                )
            )
    return tuple(matches)


def _operation_cue_matches_for_window(
    *,
    operation_cue_hits: tuple[OperationCueHit, ...],
) -> tuple[dict[str, typing.Any], ...]:
    matches: list[dict[str, typing.Any]] = []
    seen: set[tuple[str, str]] = set()
    for hit in operation_cue_hits:
        item = (hit.source_type, hit.verb)
        if item in seen:
            continue
        seen.add(item)
        matches.append({"source_type": hit.source_type, "verb": hit.verb})
    return tuple(matches)


def _term_hits(text: str, question_terms: tuple[str, ...]) -> list[tuple[int, int]]:
    lowered = text.casefold()
    hits = []
    for term in question_terms:
        start = 0
        while True:
            index = lowered.find(term, start)
            if index < 0:
                break
            hits.append((index, index + len(term)))
            start = index + len(term)
    return sorted(hits)


def _term_weights(
    question_terms: tuple[str, ...],
    texts: typing.Iterable[str],
) -> dict[str, float]:
    text_list = [text.casefold() for text in texts]
    weights = {}
    for term in question_terms:
        df = sum(1 for text in text_list if term in text)
        weights[term] = 1.0 / (1.0 + math.log1p(df))
    return weights


def _type_cue_hits(
    text: str,
    *,
    role: str,
    question_type_targets: tuple[str, ...],
    config: WindowSelectorConfig,
) -> list[TypeCueHit]:
    if (
        config.type_cue_weight <= 0
        or not question_type_targets
        or role not in config.type_cue_roles
    ):
        return []
    known_targets = _known_question_type_targets(
        question_type_targets,
        config=config,
    )
    if not known_targets:
        return []
    hits: list[TypeCueHit] = []
    for source_type, start, end in _type_candidate_phrases_with_offsets(
        text,
        config=config,
    ):
        matches = _source_type_matches(
            source_type,
            question_type_targets=known_targets,
            config=config,
        )
        if matches:
            hits.append(TypeCueHit(start=start, end=end, matches=matches))
    return hits


def _type_cue_matches_for_window(
    *,
    type_cue_hits: tuple[TypeCueHit, ...],
) -> tuple[dict[str, typing.Any], ...]:
    matches: list[dict[str, typing.Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for hit in type_cue_hits:
        for match in hit.matches:
            source_type = str(match.get("source_type", ""))
            target_type = str(match.get("target_type", ""))
            pair = (source_type, target_type)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            matches.append(match)
            if len(matches) >= 8:
                return tuple(matches)
    return tuple(matches)


def _known_question_type_targets(
    question_type_targets: tuple[str, ...],
    *,
    config: WindowSelectorConfig,
) -> tuple[str, ...]:
    known: list[str] = []
    for target_type in question_type_targets:
        if not _type_known_cached(target_type, _lexicon_path(config)):
            continue
        if any(_type_terms_equivalent(target_type, existing) for existing in known):
            continue
        known.append(target_type)
    return tuple(
        target_type
        for target_type in known
        if not any(
            target_type != other and _type_phrase_contains(other, target_type)
            for other in known
        )
    )


def _source_type_matches(
    source_type: str,
    *,
    question_type_targets: tuple[str, ...],
    config: WindowSelectorConfig,
) -> tuple[dict[str, typing.Any], ...]:
    matches = []
    for target_type in question_type_targets:
        if _type_terms_equivalent(source_type, target_type):
            continue
        ratified, reason, path = _ratify_type_cached(
            source_type,
            target_type,
            _lexicon_path(config),
        )
        if ratified:
            matches.append(
                {
                    "source_type": source_type,
                    "target_type": target_type,
                    "reason": reason,
                    "path": path,
                }
            )
    return tuple(matches)


def _type_candidate_phrases_with_offsets(
    text: str,
    *,
    config: WindowSelectorConfig,
) -> tuple[tuple[str, int, int], ...]:
    token_spans = [
        (match.group(0).casefold(), match.start(), match.end())
        for match in _TOKEN_RE.finditer(text)
        if len(match.group(0)) > 1
        and match.group(0).casefold() not in _QUESTION_STOPWORDS
    ]
    candidates: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for ngram_size in (3, 2, 1):
        for index in range(0, len(token_spans) - ngram_size + 1):
            ngram = token_spans[index : index + ngram_size]
            phrase = " ".join(token for token, _, _ in ngram)
            start = ngram[0][1]
            end = ngram[-1][2]
            for variant in _phrase_type_variants(phrase):
                if not _type_known_cached(variant, _lexicon_path(config)):
                    continue
                item = (variant, start, end)
                if item in seen:
                    continue
                seen.add(item)
                candidates.append(item)
    return tuple(candidates)


def _type_terms_equivalent(left: str, right: str) -> bool:
    return bool(set(_phrase_type_variants(left)) & set(_phrase_type_variants(right)))


def _type_phrase_contains(longer: str, shorter: str) -> bool:
    longer_tokens = longer.split()
    shorter_tokens = shorter.split()
    if len(shorter_tokens) >= len(longer_tokens):
        return False
    return any(
        longer_tokens[index : index + len(shorter_tokens)] == shorter_tokens
        for index in range(0, len(longer_tokens) - len(shorter_tokens) + 1)
    )


@functools.cache
def _type_known_cached(type_phrase: str, lexicon_path: str) -> bool:
    graph = type_ontology._load_lexicon(lexicon_path)
    return bool(graph.lookup(type_phrase))


@functools.cache
def _ratify_type_cached(
    source_type: str,
    target_type: str,
    lexicon_path: str,
) -> tuple[bool, str, tuple[str, ...]]:
    ratification = type_ontology.ratify_type_subsumption(
        source_type,
        target_type,
        lexicon_path=lexicon_path,
    )
    return ratification.ratified, ratification.reason, ratification.path


def _lexicon_path(config: WindowSelectorConfig) -> str:
    return config.lexicon_path or str(type_ontology.DEFAULT_LEXICON_PATH)


def _overlaps_mergeable(
    candidate: CandidateWindow,
    selected: list[CandidateWindow],
) -> bool:
    return any(
        (
            candidate.role == item.role
            and candidate.start < item.end
            and item.start < candidate.end
        )
        for item in selected
    )


def _merge_overlapping(
    candidate: CandidateWindow,
    selected: list[CandidateWindow],
    text: str,
    term_weights: dict[str, float],
    config: WindowSelectorConfig,
) -> list[CandidateWindow]:
    merged: list[CandidateWindow] = []
    current = candidate
    for item in selected:
        if current.start < item.end and item.start < current.end:
            merged_start = min(current.start, item.start)
            merged_end = max(current.end, item.end)
            if merged_end - merged_start <= config.max_chars_per_session:
                merged_window = _scored_window(
                    text,
                    session_id=current.session_id,
                    role=current.role,
                    start=merged_start,
                    end=merged_end,
                    term_weights=term_weights,
                    type_cue_hits=tuple(
                        {
                            (hit.start, hit.end): hit
                            for hit in (*current.type_cue_hits, *item.type_cue_hits)
                        }.values()
                    ),
                    operation_cue_hits=tuple(
                        {
                            (hit.start, hit.end, hit.verb, hit.source_type): hit
                            for hit in (
                                *current.operation_cue_hits,
                                *item.operation_cue_hits,
                            )
                        }.values()
                    ),
                    config=config,
                )
                if current.operation_cue_hits or item.operation_cue_hits:
                    current = max(
                        (current, item, merged_window),
                        key=lambda window: (
                            window.score,
                            -(window.end - window.start),
                        ),
                    )
                else:
                    current = merged_window
            else:
                current = max(
                    (current, item),
                    key=lambda window: (window.score, -(window.end - window.start)),
                )
        else:
            merged.append(item)
    merged.append(current)
    return merged


def _trim_to_char_budget(
    windows: list[CandidateWindow],
    max_chars: int,
) -> list[CandidateWindow]:
    selected = list(windows)
    while len(selected) > 1 and _selected_chars(selected) > max_chars:
        selected = sorted(selected, key=lambda item: item.score, reverse=True)[:-1]
    return selected


def _trim_augmented_to_char_budget(
    prefix_window: CandidateWindow,
    windows: list[CandidateWindow],
    max_chars: int,
) -> list[CandidateWindow]:
    selected = list(windows)
    while selected and _selected_chars([prefix_window, *selected]) > max_chars:
        selected = sorted(selected, key=lambda item: item.score, reverse=True)[:-1]
    return selected


def _selected_chars(windows: list[CandidateWindow]) -> int:
    if not windows:
        return 0
    return sum(window.end - window.start for window in windows) + (
        len(windows) - 1
    ) * len(SELECTOR_GAP)


def _render_selected_text(
    source_text: str,
    windows: list[CandidateWindow],
) -> str:
    chunks = [source_text[window.start : window.end].strip() for window in windows]
    return SELECTOR_GAP.join(chunk for chunk in chunks if chunk)


def _pre_metric_case(
    *,
    needle_case: dict[str, typing.Any],
    selected_payload: dict[str, typing.Any] | None,
    source_payload: dict[str, typing.Any] | None,
    metadata: dict[str, typing.Any],
    baseline_case: dict[str, typing.Any] | None = None,
    margin_gate_excluded: bool = False,
) -> dict[str, typing.Any]:
    case_id = str(needle_case.get("case_id", ""))
    selected_sessions = _payload_session_texts(selected_payload)
    source_sessions = _payload_session_texts(source_payload)
    source_window_texts = _payload_session_texts(source_payload)
    needles = []
    case_needles = [
        raw for raw in needle_case.get("needles", []) if isinstance(raw, dict)
    ]
    for raw in case_needles:
        session_id = str(raw.get("session_id", ""))
        needle = str(raw.get("needle", ""))
        source_text = source_sessions.get(session_id)
        selected_text = selected_sessions.get(session_id)
        source_offset = source_text.find(needle) if source_text is not None else -1
        selected_offset = (
            selected_text.find(needle) if selected_text is not None else -1
        )
        gold_salience = _needle_salience(
            session_id=session_id,
            needle=needle,
            source_text=source_window_texts.get(session_id, ""),
            metadata=metadata,
        )
        distractor_salience = _max_distractor_salience(
            metadata=metadata,
            source_sessions=source_window_texts,
            case_needles=case_needles,
        )
        margin = (
            gold_salience - distractor_salience
            if gold_salience is not None and distractor_salience is not None
            else None
        )
        needles.append(
            {
                "label": str(raw.get("label", "")),
                "session_id": session_id,
                "needle": needle,
                "source_contains_exact": source_offset >= 0,
                "selected_contains_exact": selected_offset >= 0,
                "selector_dropped_source_span": (
                    source_offset >= 0 and selected_offset < 0
                ),
                "source_offset": source_offset if source_offset >= 0 else None,
                "selected_offset": selected_offset if selected_offset >= 0 else None,
                "gold_window_salience": gold_salience,
                "max_distractor_salience": distractor_salience,
                "salience_margin": margin,
            }
        )
    margins = [
        float(item["salience_margin"])
        for item in needles
        if item["salience_margin"] is not None
    ]
    salience_margin_min = min(margins) if margins else None
    baseline_margin = _number_or_none((baseline_case or {}).get("salience_margin_min"))
    margin_delta = (
        salience_margin_min - baseline_margin
        if salience_margin_min is not None and baseline_margin is not None
        else None
    )
    salience_margin_crossed_nonpositive = (
        baseline_margin is not None
        and baseline_margin > 0
        and salience_margin_min is not None
        and salience_margin_min <= 0
    )
    margin_regressed = salience_margin_crossed_nonpositive and not margin_gate_excluded
    selector_dropped_count = sum(
        1 for item in needles if item["selector_dropped_source_span"]
    )
    return {
        "case_id": case_id,
        "question": str(needle_case.get("question", "")),
        "needle_count": len(needles),
        "source_present_count": sum(
            1 for item in needles if item["source_contains_exact"]
        ),
        "selected_present_count": sum(
            1 for item in needles if item["selected_contains_exact"]
        ),
        "selector_dropped_count": selector_dropped_count,
        "kill_gate_passed": selector_dropped_count == 0 and not margin_regressed,
        "salience_margin_min": salience_margin_min,
        "baseline_salience_margin_min": baseline_margin,
        "salience_margin_delta_from_baseline": margin_delta,
        "salience_margin_crossed_nonpositive": salience_margin_crossed_nonpositive,
        "margin_gate_excluded": margin_gate_excluded,
        "margin_gate_exclusion_reason": (
            "retrieval_capped" if margin_gate_excluded else None
        ),
        "salience_margin_regressed": margin_regressed,
        "needles": needles,
    }


def _pre_metric_cases_by_id(
    artifact: dict[str, typing.Any] | None,
) -> dict[str, dict[str, typing.Any]]:
    if not artifact:
        return {}
    return {
        str(case.get("case_id", "")): case
        for case in artifact.get("cases", [])
        if isinstance(case, dict)
    }


def _number_or_none(value: typing.Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _needle_salience(
    *,
    session_id: str,
    needle: str,
    source_text: str,
    metadata: dict[str, typing.Any],
) -> float | None:
    offset = source_text.find(needle)
    if offset < 0:
        return None
    end = offset + len(needle)
    scores = []
    session_meta = metadata.get("sessions", {}).get(session_id, {})
    for window in session_meta.get("windows", []):
        if not isinstance(window, dict):
            continue
        if int(window.get("start", -1)) <= offset and end <= int(window.get("end", -1)):
            scores.append(float(window.get("score", 0.0) or 0.0))
    return max(scores) if scores else None


def _max_distractor_salience(
    *,
    metadata: dict[str, typing.Any],
    source_sessions: dict[str, str],
    case_needles: list[dict[str, typing.Any]],
) -> float | None:
    needle_ranges_by_session: dict[str, list[tuple[int, int]]] = {}
    for raw in case_needles:
        session_id = str(raw.get("session_id", ""))
        needle = str(raw.get("needle", ""))
        source_text = source_sessions.get(session_id, "")
        offset = source_text.find(needle)
        if offset >= 0:
            needle_ranges_by_session.setdefault(session_id, []).append(
                (offset, offset + len(needle))
            )
    scores = []
    for session_id, session_meta in metadata.get("sessions", {}).items():
        for window in session_meta.get("windows", []):
            if not isinstance(window, dict):
                continue
            start = int(window.get("start", -1))
            end = int(window.get("end", -1))
            if start < 0 or end < 0:
                continue
            if any(
                start <= needle_start and needle_end <= end
                for needle_start, needle_end in needle_ranges_by_session.get(
                    str(session_id), []
                )
            ):
                continue
            scores.append(float(window.get("score", 0.0) or 0.0))
    return max(scores) if scores else 0.0


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-payloads", action="store_true")
    parser.add_argument("--build-pre-metrics", action="store_true")
    parser.add_argument(
        "--source-payloads",
        type=pathlib.Path,
        default=DEFAULT_SOURCE_PAYLOADS_PATH,
    )
    parser.add_argument(
        "--selected-payloads",
        type=pathlib.Path,
        default=DEFAULT_SELECTED_PAYLOADS_PATH,
    )
    parser.add_argument(
        "--gold-span-needles",
        type=pathlib.Path,
        default=DEFAULT_GOLD_SPAN_NEEDLES_PATH,
    )
    parser.add_argument(
        "--pre-metrics",
        type=pathlib.Path,
        default=DEFAULT_PRE_METRICS_PATH,
    )
    parser.add_argument(
        "--baseline-pre-metrics",
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument(
        "--margin-gate-exclude-case-ids",
        default=",".join(DEFAULT_MARGIN_GATE_EXCLUDED_CASE_IDS),
    )
    parser.add_argument("--selector-name", default=DEFAULT_SELECTOR_NAME)
    parser.add_argument(
        "--prefix-floor-chars",
        type=int,
        default=DEFAULT_PREFIX_FLOOR_CHARS,
    )
    parser.add_argument("--window-radius-chars", type=int, default=800)
    parser.add_argument("--max-windows-per-session", type=int, default=2)
    parser.add_argument("--max-chars-per-session", type=int, default=4000)
    parser.add_argument("--role-weight", type=float, default=1.5)
    parser.add_argument("--compactness-weight", type=float, default=0.75)
    parser.add_argument("--compactness-cap", type=float, default=8.0)
    parser.add_argument(
        "--type-cue-weight",
        type=float,
        default=DEFAULT_TYPE_CUE_WEIGHT,
    )
    parser.add_argument("--type-cue-roles", default="USER")
    parser.add_argument(
        "--operation-cue-weight",
        type=float,
        default=DEFAULT_OPERATION_CUE_WEIGHT,
    )
    parser.add_argument(
        "--operation-cue-radius-chars",
        type=int,
        default=DEFAULT_OPERATION_CUE_RADIUS_CHARS,
    )
    parser.add_argument("--operation-cue-roles", default="USER")
    parser.add_argument("--distractor-penalty-weight", type=float, default=0.0)
    parser.add_argument(
        "--lexicon-path",
        type=pathlib.Path,
        default=None,
    )
    args = parser.parse_args(argv)
    if not args.build_payloads and not args.build_pre_metrics:
        parser.print_help()
        return 2
    config = WindowSelectorConfig(
        name=args.selector_name,
        prefix_floor_chars=args.prefix_floor_chars,
        window_radius_chars=args.window_radius_chars,
        max_windows_per_session=args.max_windows_per_session,
        max_chars_per_session=args.max_chars_per_session,
        role_weight=args.role_weight,
        compactness_weight=args.compactness_weight,
        compactness_cap=args.compactness_cap,
        type_cue_weight=args.type_cue_weight,
        type_cue_roles=tuple(
            role.strip().upper()
            for role in args.type_cue_roles.split(",")
            if role.strip()
        ),
        operation_cue_weight=args.operation_cue_weight,
        operation_cue_radius_chars=args.operation_cue_radius_chars,
        operation_cue_roles=tuple(
            role.strip().upper()
            for role in args.operation_cue_roles.split(",")
            if role.strip()
        ),
        distractor_penalty_weight=args.distractor_penalty_weight,
        lexicon_path=str(args.lexicon_path) if args.lexicon_path else None,
    )
    if args.build_payloads:
        artifact = build_selected_payload_artifact(
            source_payloads_path=args.source_payloads,
            config=config,
        )
        _write_json(args.selected_payloads, artifact)
    if args.build_pre_metrics:
        artifact = build_fail18_selector_pre_metrics(
            selected_payloads_path=args.selected_payloads,
            source_payloads_path=args.source_payloads,
            gold_span_needles_path=args.gold_span_needles,
            baseline_pre_metrics_path=args.baseline_pre_metrics,
            margin_gate_excluded_case_ids=tuple(
                case_id.strip()
                for case_id in args.margin_gate_exclude_case_ids.split(",")
                if case_id.strip()
            ),
        )
        _write_json(args.pre_metrics, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
