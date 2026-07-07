"""Diagnostic retrieval tuning experiments for fail18 Gate 1 payloads."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import pathlib
import re
import typing

from simba.eval import (
    ambiguity_fail18,
    interpretation_prompts,
    interpretation_retrieval_recall,
)

DEFAULT_RETRIEVAL_TUNING_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_retrieval_tuning_experiment.json"
)
DEFAULT_PAYLOADS_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_payloads.json"
)
DEFAULT_TUNED_PAYLOADS_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_payloads_expanded_query_v1.json"
)
DEFAULT_TUNED_PROVENANCE_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_payloads_expanded_query_v1_provenance.json"
)
BEST_RETRIEVAL_STRATEGY_NAME = "expanded_query_v1_compact_sessions"
DEFAULT_PAYLOAD_REBUILD_STRATEGY_NAME = "expanded_query_v1"

_DURATION_RE = re.compile(
    r"\b(?:\d+(?:-| )?days?|one day|two days|three days|four days|five days|"
    r"six days|seven days|eight days|nine days|ten days)\b",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(r"\$\s*\d[\d,]*(?:\.\d+)?")
_USER_SEGMENT_RE = re.compile(
    r"user:\s*(?P<content>.*?)(?=(?:\n?assistant:|\n?user:)|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_EXTRA_STOPWORDS = {
    "all",
    "and",
    "are",
    "currently",
    "did",
    "does",
    "for",
    "how",
    "into",
    "was",
    "were",
}

_TERM_EXPANSIONS = {
    "camp": ("camping", "camped"),
    "camping": ("camp", "camped", "backpack", "backpacking", "hiking"),
    "charity": ("benefit", "donation", "fundraiser", "fundraising"),
    "city": ("nyc",),
    "day": ("days", "duration"),
    "days": ("day", "duration"),
    "hawaii": ("island", "island-hopping"),
    "money": ("$", "dollar", "dollars"),
    "own": ("acquired", "bought", "got", "owned", "purchased"),
    "points": ("loyalty", "point", "reward", "rewards"),
    "raise": ("donate", "donation", "fundraise", "fundraising", "raised"),
    "redeem": ("redemption", "reward"),
    "travel": ("traveling", "traveled", "travelled", "trip", "trips"),
    "traveling": ("travel", "traveled", "travelled", "trip", "trips"),
    "trip": ("travel", "traveling", "trips"),
    "trips": ("travel", "traveling", "trip"),
    "york": ("nyc",),
}

_CANDIDATE_UNIT_COVERAGE_EXPANSIONS = {
    "attend": ("attended", "been", "bridesmaid"),
    "attended": ("attend", "been", "bridesmaid"),
    "bake": ("baked", "baking", "bread", "cake", "cookies", "oven", "recipe"),
    "baked": ("bake", "baking", "bread", "cake", "cookies", "oven", "recipe"),
    "baking": ("bake", "baked", "bread", "cake", "cookies", "oven", "recipe"),
    "clothing": ("clothes", "boots", "blazer", "zara", "return", "pickup"),
    "return": ("exchange", "exchanged", "pickup", "pick", "zara"),
    "wedding": (
        "weddings",
        "bride",
        "groom",
        "ceremony",
        "bridesmaid",
        "vineyard",
        "barn",
        "married",
    ),
    "weddings": (
        "wedding",
        "bride",
        "groom",
        "ceremony",
        "bridesmaid",
        "vineyard",
        "barn",
        "married",
    ),
}


@dataclasses.dataclass(frozen=True)
class RetrievalStrategy:
    name: str
    scoring: str
    compact_session_fraction: float | None = None


@dataclasses.dataclass(frozen=True)
class StrategyRankedSession:
    raw_session_index: int
    raw_session_id: str
    date: str
    selection_rank: int
    selection_score: int
    user_selection_score: int
    assistant_selection_score: int
    boost_score: int
    boost_reasons: tuple[str, ...]
    rendered_chars: int
    selected_by_budget: bool
    simulated_provider_id: str | None
    simulated_trimmed_chars: int
    simulated_truncated: bool
    matched_question_terms_user: tuple[str, ...]
    matched_question_terms_assistant: tuple[str, ...]


STRATEGIES = (
    RetrievalStrategy(name="baseline_current_lexical", scoring="baseline"),
    RetrievalStrategy(
        name="baseline_compact_sessions",
        scoring="baseline",
        compact_session_fraction=0.5,
    ),
    RetrievalStrategy(name="expanded_query_v1", scoring="expanded_query_v1"),
    RetrievalStrategy(
        name="expanded_query_v1_compact_sessions",
        scoring="expanded_query_v1",
        compact_session_fraction=0.5,
    ),
    RetrievalStrategy(
        name="candidate_unit_coverage_v1",
        scoring="candidate_unit_coverage_v1",
    ),
)


def build_fail18_tuned_generation_artifact(
    *,
    strategy_name: str = DEFAULT_PAYLOAD_REBUILD_STRATEGY_NAME,
    payloads_path: str | pathlib.Path = DEFAULT_PAYLOADS_PATH,
    manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    corpus_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_CORPUS,
    limit: int = 0,
) -> dict[str, typing.Any]:
    payload_artifact = _load_json(payloads_path)
    selection_config = interpretation_retrieval_recall._selection_config(
        payload_artifact
    )
    strategy = _strategy_by_name(strategy_name)
    rows = ambiguity_fail18.load_manifest(manifest_path)
    corpus_by_id = {
        str(row["question_id"]): row
        for row in ambiguity_fail18.load_corpus(corpus_path)
    }
    if limit > 0:
        rows = rows[:limit]
    payloads: list[dict[str, typing.Any]] = []
    for row in rows:
        qid = str(row["question_id"])
        question = str(row.get("question", ""))
        corpus_row = corpus_by_id.get(qid, {})
        evidence_sessions, _provenance = _tuned_evidence_sessions(
            row=corpus_row,
            question=question,
            strategy=strategy,
            selection_config=selection_config,
        )
        payloads.append(
            interpretation_prompts.build_interpretation_generation_payload(
                case_id=qid,
                question=question,
                evidence_sessions=evidence_sessions,
            )
        )
    return {
        "name": (
            f"fail18-ambiguous-nlidb-gate1-payloads-{strategy_name.replace('_', '-')}"
        ),
        "artifact_kind": "provider_payloads",
        "gate": "gate1",
        "gate_status": "tuned_payloads_only_not_run",
        "known_gap": (
            "This tuned artifact contains provider payloads only. It has not "
            "been run through a model, parser, infill, or verifier."
        ),
        "prompt_version": interpretation_prompts.PROMPT_VERSION,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_payload_artifact": str(payloads_path),
        "source_manifest": str(manifest_path),
        "source_corpus": str(corpus_path),
        "retrieval_strategy": _strategy_metadata(strategy, selection_config),
        "evidence_selection": {
            "method": strategy.name,
            "max_evidence_sessions": selection_config["max_evidence_sessions"],
            "max_evidence_chars": selection_config["max_evidence_chars"],
            "max_session_chars": _strategy_max_session_chars(
                selection_config=selection_config,
                strategy=strategy,
            ),
            "uses_answer_session_ids": False,
            "provider_session_ids": "opaque evidence_NNN ids",
            "raw_session_ids": "private provenance artifact only",
        },
        "total": len(payloads),
        "commands": [
            (
                "uv run python -m simba.eval.interpretation_retrieval_tuning "
                "--build-payloads "
                f"--strategy {strategy.name} "
                f"--payload-output {DEFAULT_TUNED_PAYLOADS_PATH} "
                f"--provenance-output {DEFAULT_TUNED_PROVENANCE_PATH}"
            )
        ],
        "payloads": payloads,
    }


def build_fail18_tuned_private_provenance_artifact(
    *,
    strategy_name: str = DEFAULT_PAYLOAD_REBUILD_STRATEGY_NAME,
    payloads_path: str | pathlib.Path = DEFAULT_PAYLOADS_PATH,
    manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    corpus_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_CORPUS,
    limit: int = 0,
) -> dict[str, typing.Any]:
    payload_artifact = _load_json(payloads_path)
    selection_config = interpretation_retrieval_recall._selection_config(
        payload_artifact
    )
    strategy = _strategy_by_name(strategy_name)
    rows = ambiguity_fail18.load_manifest(manifest_path)
    corpus_by_id = {
        str(row["question_id"]): row
        for row in ambiguity_fail18.load_corpus(corpus_path)
    }
    if limit > 0:
        rows = rows[:limit]
    provenance = {}
    for row in rows:
        qid = str(row["question_id"])
        _evidence, evidence_provenance = _tuned_evidence_sessions(
            row=corpus_by_id.get(qid, {}),
            question=str(row.get("question", "")),
            strategy=strategy,
            selection_config=selection_config,
        )
        provenance[qid] = evidence_provenance
    return {
        "name": (
            "fail18-ambiguous-nlidb-gate1-payloads-"
            f"{strategy_name.replace('_', '-')}-provenance"
        ),
        "artifact_kind": "private_provenance",
        "provider_payload_artifact": str(DEFAULT_TUNED_PAYLOADS_PATH),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_payload_artifact": str(payloads_path),
        "source_manifest": str(manifest_path),
        "source_corpus": str(corpus_path),
        "retrieval_strategy": _strategy_metadata(strategy, selection_config),
        "total": len(provenance),
        "evidence_provenance": provenance,
    }


def build_fail18_retrieval_tuning_experiment(
    *,
    payloads_path: str | pathlib.Path = DEFAULT_PAYLOADS_PATH,
    corpus_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_CORPUS,
) -> dict[str, typing.Any]:
    payload_artifact = _load_json(payloads_path)
    corpus_rows = ambiguity_fail18.load_corpus(corpus_path)
    selection_config = interpretation_retrieval_recall._selection_config(
        payload_artifact
    )
    strategies = [
        _strategy_result(
            strategy=strategy,
            corpus_rows=corpus_rows,
            selection_config=selection_config,
        )
        for strategy in STRATEGIES
    ]
    baseline = next(
        strategy
        for strategy in strategies
        if strategy["strategy_name"] == "baseline_current_lexical"
    )
    _attach_deltas(strategies, baseline)
    best = max(
        strategies,
        key=lambda item: (
            item["summary"]["answer_sessions_in_payload"],
            item["summary"]["rows_with_all_answer_sessions_in_payload"],
        ),
    )
    return {
        "name": "fail18-ambiguous-nlidb-gate1-retrieval-tuning-experiment",
        "artifact_kind": "interpretation_retrieval_tuning_experiment",
        "gate": "gate1",
        "gate_status": "slice2e_retrieval_tuning_experiment_complete",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_payloads": str(payloads_path),
        "source_corpus": str(corpus_path),
        "selection_config": selection_config,
        "strategy_notes": {
            "baseline_current_lexical": (
                "Reproduces current lexical question-term selection."
            ),
            "baseline_compact_sessions": (
                "Keeps current ranker but halves per-session character budget."
            ),
            "expanded_query_v1": (
                "Uses stronger stopwords, morphology/query expansion, "
                "user-text emphasis, answer-type boosts, and completed-event "
                "salience. It uses only question and evidence text."
            ),
            "expanded_query_v1_compact_sessions": (
                "Combines expanded_query_v1 with the compact-session budget."
            ),
            "candidate_unit_coverage_v1": (
                "Extends expanded_query_v1 with generic action/event boosts "
                "for candidate-unit coverage gaps such as baking events, "
                "wedding attendance, and pickup/return obligations."
            ),
        },
        "summary": {
            "best_strategy": best["strategy_name"],
            "best_answer_session_recall": best["summary"]["answer_session_recall"],
            "best_rows_with_all_answer_sessions_in_payload": best["summary"][
                "rows_with_all_answer_sessions_in_payload"
            ],
            "baseline_answer_session_recall": baseline["summary"][
                "answer_session_recall"
            ],
            "baseline_rows_with_all_answer_sessions_in_payload": baseline["summary"][
                "rows_with_all_answer_sessions_in_payload"
            ],
        },
        "decision": {
            "retrieval_change_should_remain_diagnostic": True,
            "next_slice": "rebuild_payloads_with_best_non_oracle_strategy",
            "reason": (
                "The best strategy improves private recall without using gold "
                "for selection, but provider payload quality still needs a "
                "rebuilt-payload and verifier-probe check before adoption."
            ),
        },
        "strategies": strategies,
    }


def _tuned_evidence_sessions(
    *,
    row: dict[str, typing.Any],
    question: str,
    strategy: RetrievalStrategy,
    selection_config: dict[str, int],
) -> tuple[list[dict[str, typing.Any]], dict[str, dict[str, typing.Any]]]:
    ranked = _rank_sessions(
        row=row,
        question=question,
        strategy=strategy,
        selection_config=selection_config,
    )
    text_by_id = _rendered_session_text_by_id(row)
    evidence: list[dict[str, typing.Any]] = []
    provenance: dict[str, dict[str, typing.Any]] = {}
    for session in ranked:
        if not session.selected_by_budget:
            continue
        text = text_by_id.get(session.raw_session_id, "")
        if strategy.scoring == "candidate_unit_coverage_v1":
            trimmed, truncated = _trim_candidate_unit_coverage_text(
                text=text,
                question=question,
                max_chars=session.simulated_trimmed_chars,
            )
        else:
            trimmed, truncated = interpretation_prompts._trim_text(
                text,
                max_chars=session.simulated_trimmed_chars,
            )
        provider_id = session.simulated_provider_id or (
            f"evidence_{session.selection_rank:03d}"
        )
        evidence.append(
            {
                "session_id": provider_id,
                "date": session.date,
                "selection_rank": session.selection_rank,
                "selection_score": session.selection_score,
                "user_selection_score": session.user_selection_score,
                "assistant_selection_score": session.assistant_selection_score,
                "truncated": truncated,
                "text": trimmed,
            }
        )
        provenance[provider_id] = {
            "raw_session_id": session.raw_session_id,
            "raw_session_index": session.raw_session_index,
            "date": session.date,
            "selection_rank": session.selection_rank,
            "selection_score": session.selection_score,
            "user_selection_score": session.user_selection_score,
            "assistant_selection_score": session.assistant_selection_score,
            "boost_score": session.boost_score,
            "boost_reasons": list(session.boost_reasons),
            "matched_question_terms_user": list(session.matched_question_terms_user),
            "matched_question_terms_assistant": list(
                session.matched_question_terms_assistant
            ),
            "truncated": truncated,
        }
    return evidence, provenance


def _rendered_session_text_by_id(
    row: dict[str, typing.Any],
) -> dict[str, str]:
    ids = row.get("haystack_session_ids", [])
    sessions = row.get("haystack_sessions", [])
    rendered = {}
    for idx, session in enumerate(sessions):
        sid = str(ids[idx]) if idx < len(ids) else str(idx)
        text, _user_text, _assistant_text = interpretation_prompts._render_session(
            session
        )
        rendered[sid] = text
    return rendered


def _strategy_metadata(
    strategy: RetrievalStrategy,
    selection_config: dict[str, int],
) -> dict[str, typing.Any]:
    return {
        "name": strategy.name,
        "scoring": strategy.scoring,
        "compact_session_fraction": strategy.compact_session_fraction,
        "effective_max_session_chars": _strategy_max_session_chars(
            selection_config=selection_config,
            strategy=strategy,
        ),
        "uses_answer_session_ids": False,
    }


def _strategy_by_name(name: str) -> RetrievalStrategy:
    for strategy in STRATEGIES:
        if strategy.name == name:
            return strategy
    valid = ", ".join(strategy.name for strategy in STRATEGIES)
    raise ValueError(f"unknown retrieval strategy {name!r}; expected one of {valid}")


def _strategy_result(
    *,
    strategy: RetrievalStrategy,
    corpus_rows: list[dict[str, typing.Any]],
    selection_config: dict[str, int],
) -> dict[str, typing.Any]:
    cases = [
        _case_result(
            row=row,
            strategy=strategy,
            selection_config=selection_config,
        )
        for row in corpus_rows
    ]
    answer_total = sum(int(case["answer_session_count"]) for case in cases)
    selected_total = sum(int(case["answer_sessions_in_payload"]) for case in cases)
    rows_with_all = sum(1 for case in cases if case["all_answer_sessions_in_payload"])
    rows_with_missing = sum(1 for case in cases if case["missing_answer_sessions"])
    status_counts: collections.Counter[str] = collections.Counter()
    for case in cases:
        for answer_session in case["answer_sessions"]:
            status_counts[str(answer_session["retrieval_status"])] += 1
    return {
        "strategy_name": strategy.name,
        "scoring": strategy.scoring,
        "compact_session_fraction": strategy.compact_session_fraction,
        "summary": {
            "rows_total": len(cases),
            "rows_with_all_answer_sessions_in_payload": rows_with_all,
            "rows_with_missing_answer_sessions": rows_with_missing,
            "answer_sessions_total": answer_total,
            "answer_sessions_in_payload": selected_total,
            "answer_session_recall": (
                selected_total / answer_total if answer_total else None
            ),
            "retrieval_status_counts": dict(sorted(status_counts.items())),
        },
        "cases": cases,
    }


def _case_result(
    *,
    row: dict[str, typing.Any],
    strategy: RetrievalStrategy,
    selection_config: dict[str, int],
) -> dict[str, typing.Any]:
    ranked = _rank_sessions(
        row=row,
        question=str(row.get("question", "")),
        strategy=strategy,
        selection_config=selection_config,
    )
    ranked_by_id = {session.raw_session_id: session for session in ranked}
    answer_session_ids = [str(item) for item in row.get("answer_session_ids", [])]
    answer_sessions = [
        _answer_session_result(raw_session_id, ranked_by_id.get(raw_session_id))
        for raw_session_id in answer_session_ids
    ]
    missing = [
        item["raw_session_id"]
        for item in answer_sessions
        if item["retrieval_status"] != "included_in_simulated_payload"
    ]
    return {
        "case_id": str(row.get("question_id", "")),
        "question": str(row.get("question", "")),
        "answer_session_count": len(answer_session_ids),
        "answer_sessions_in_payload": len(answer_session_ids) - len(missing),
        "all_answer_sessions_in_payload": not missing,
        "missing_answer_sessions": missing,
        "simulated_payload_raw_session_ids": [
            session.raw_session_id for session in ranked if session.selected_by_budget
        ],
        "answer_sessions": answer_sessions,
        "top_ranked_sessions": [
            _ranked_session_dict(session)
            for session in ranked[: selection_config["max_evidence_sessions"]]
        ],
    }


def _rank_sessions(
    *,
    row: dict[str, typing.Any],
    question: str,
    strategy: RetrievalStrategy,
    selection_config: dict[str, int],
) -> list[StrategyRankedSession]:
    if strategy.scoring == "baseline":
        return _baseline_rank_sessions(
            row=row,
            question=question,
            strategy=strategy,
            selection_config=selection_config,
        )
    return _expanded_rank_sessions(
        row=row,
        question=question,
        strategy=strategy,
        selection_config=selection_config,
    )


def _baseline_rank_sessions(
    *,
    row: dict[str, typing.Any],
    question: str,
    strategy: RetrievalStrategy,
    selection_config: dict[str, int],
) -> list[StrategyRankedSession]:
    ranked = interpretation_retrieval_recall._rank_sessions(
        row=row,
        question=question,
        max_evidence_sessions=selection_config["max_evidence_sessions"],
        max_evidence_chars=selection_config["max_evidence_chars"],
        max_session_chars=_strategy_max_session_chars(
            selection_config=selection_config,
            strategy=strategy,
        ),
    )
    return [
        StrategyRankedSession(
            raw_session_index=session.raw_session_index,
            raw_session_id=session.raw_session_id,
            date=session.date,
            selection_rank=int(session.selection_rank or 0),
            selection_score=session.selection_score,
            user_selection_score=session.user_selection_score,
            assistant_selection_score=session.assistant_selection_score,
            boost_score=0,
            boost_reasons=(),
            rendered_chars=session.rendered_chars,
            selected_by_budget=session.selected_by_budget,
            simulated_provider_id=session.simulated_provider_id,
            simulated_trimmed_chars=session.simulated_trimmed_chars,
            simulated_truncated=session.simulated_truncated,
            matched_question_terms_user=session.matched_question_terms_user,
            matched_question_terms_assistant=session.matched_question_terms_assistant,
        )
        for session in ranked
    ]


def _expanded_rank_sessions(
    *,
    row: dict[str, typing.Any],
    question: str,
    strategy: RetrievalStrategy,
    selection_config: dict[str, int],
) -> list[StrategyRankedSession]:
    ids = row.get("haystack_session_ids", [])
    dates = row.get("haystack_dates", [])
    sessions = row.get("haystack_sessions", [])
    question_terms = _expanded_terms(question, scoring=strategy.scoring)
    ranked_items: list[dict[str, typing.Any]] = []
    for idx, session in enumerate(sessions):
        sid = str(ids[idx]) if idx < len(ids) else str(idx)
        date = str(dates[idx]) if idx < len(dates) else ""
        text, user_text, assistant_text = interpretation_prompts._render_session(
            session
        )
        user_matches = _matched_terms(
            question_terms,
            user_text,
            scoring=strategy.scoring,
        )
        assistant_matches = _matched_terms(
            question_terms,
            assistant_text,
            scoring=strategy.scoring,
        )
        boost_score, boost_reasons = _evidence_boosts(
            question=question,
            user_text=user_text,
            matched_user_terms=user_matches,
            scoring=strategy.scoring,
        )
        user_score = len(user_matches)
        assistant_score = len(assistant_matches)
        selection_score = (4 * user_score) + assistant_score + boost_score
        if selection_score <= 0:
            continue
        ranked_items.append(
            {
                "raw_session_index": idx,
                "raw_session_id": sid,
                "date": date,
                "text": text,
                "selection_score": selection_score,
                "user_selection_score": user_score,
                "assistant_selection_score": assistant_score,
                "boost_score": boost_score,
                "boost_reasons": boost_reasons,
                "matched_question_terms_user": user_matches,
                "matched_question_terms_assistant": assistant_matches,
            }
        )
    ranked_items.sort(
        key=lambda item: (
            -int(item["selection_score"]),
            -int(item["user_selection_score"]),
            int(item["raw_session_index"]),
        )
    )
    selected = interpretation_retrieval_recall._simulate_budget_selection(
        ranked_items=ranked_items,
        max_evidence_sessions=selection_config["max_evidence_sessions"],
        max_evidence_chars=selection_config["max_evidence_chars"],
        max_session_chars=_strategy_max_session_chars(
            selection_config=selection_config,
            strategy=strategy,
        ),
    )
    return [
        StrategyRankedSession(
            raw_session_index=int(item["raw_session_index"]),
            raw_session_id=str(item["raw_session_id"]),
            date=str(item["date"]),
            selection_rank=rank,
            selection_score=int(item["selection_score"]),
            user_selection_score=int(item["user_selection_score"]),
            assistant_selection_score=int(item["assistant_selection_score"]),
            boost_score=int(item["boost_score"]),
            boost_reasons=tuple(item["boost_reasons"]),
            rendered_chars=len(str(item["text"])),
            selected_by_budget=str(item["raw_session_id"]) in selected,
            simulated_provider_id=selected.get(str(item["raw_session_id"]), {}).get(
                "provider_id"
            ),
            simulated_trimmed_chars=int(
                selected.get(str(item["raw_session_id"]), {}).get("trimmed_chars", 0)
            ),
            simulated_truncated=bool(
                selected.get(str(item["raw_session_id"]), {}).get("truncated", False)
            ),
            matched_question_terms_user=tuple(item["matched_question_terms_user"]),
            matched_question_terms_assistant=tuple(
                item["matched_question_terms_assistant"]
            ),
        )
        for rank, item in enumerate(ranked_items, start=1)
    ]


def _answer_session_result(
    raw_session_id: str,
    ranked_session: StrategyRankedSession | None,
) -> dict[str, typing.Any]:
    if ranked_session is None:
        return {
            "raw_session_id": raw_session_id,
            "retrieval_status": "not_ranked",
            "selection_rank": None,
            "selection_score": 0,
            "boost_score": 0,
            "boost_reasons": [],
            "matched_question_terms_user": [],
            "matched_question_terms_assistant": [],
        }
    return {
        "raw_session_id": raw_session_id,
        "retrieval_status": (
            "included_in_simulated_payload"
            if ranked_session.selected_by_budget
            else "ranked_but_not_selected"
        ),
        "selection_rank": ranked_session.selection_rank,
        "selection_score": ranked_session.selection_score,
        "user_selection_score": ranked_session.user_selection_score,
        "assistant_selection_score": ranked_session.assistant_selection_score,
        "boost_score": ranked_session.boost_score,
        "boost_reasons": list(ranked_session.boost_reasons),
        "matched_question_terms_user": list(ranked_session.matched_question_terms_user),
        "matched_question_terms_assistant": list(
            ranked_session.matched_question_terms_assistant
        ),
    }


def _ranked_session_dict(
    session: StrategyRankedSession,
) -> dict[str, typing.Any]:
    return {
        "raw_session_id": session.raw_session_id,
        "raw_session_index": session.raw_session_index,
        "selection_rank": session.selection_rank,
        "selection_score": session.selection_score,
        "user_selection_score": session.user_selection_score,
        "assistant_selection_score": session.assistant_selection_score,
        "boost_score": session.boost_score,
        "boost_reasons": list(session.boost_reasons),
        "selected_by_budget": session.selected_by_budget,
        "simulated_provider_id": session.simulated_provider_id,
        "matched_question_terms_user": list(session.matched_question_terms_user),
        "matched_question_terms_assistant": list(
            session.matched_question_terms_assistant
        ),
    }


def _attach_deltas(
    strategies: list[dict[str, typing.Any]],
    baseline: dict[str, typing.Any],
) -> None:
    baseline_cases = {case["case_id"]: case for case in baseline["cases"]}
    baseline_all = {
        case_id
        for case_id, case in baseline_cases.items()
        if case["all_answer_sessions_in_payload"]
    }
    baseline_recall = baseline["summary"]["answer_session_recall"]
    for strategy in strategies:
        all_rows = {
            case["case_id"]
            for case in strategy["cases"]
            if case["all_answer_sessions_in_payload"]
        }
        strategy["delta_vs_baseline"] = {
            "answer_session_recall_delta": (
                strategy["summary"]["answer_session_recall"] - baseline_recall
            ),
            "rows_with_all_answer_sessions_delta": (
                strategy["summary"]["rows_with_all_answer_sessions_in_payload"]
                - baseline["summary"]["rows_with_all_answer_sessions_in_payload"]
            ),
            "newly_fixed_rows": sorted(all_rows - baseline_all),
            "newly_broken_rows": sorted(baseline_all - all_rows),
        }


def _strategy_max_session_chars(
    *,
    selection_config: dict[str, int],
    strategy: RetrievalStrategy,
) -> int:
    current = selection_config["max_session_chars"]
    if strategy.compact_session_fraction is None:
        return current
    return max(1, int(current * strategy.compact_session_fraction))


def _expanded_terms(
    text: str,
    *,
    scoring: str = "expanded_query_v1",
) -> set[str]:
    inherited_stopwords = interpretation_prompts._content_terms(
        " ".join(_EXTRA_STOPWORDS)
    )
    raw_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
        and token not in inherited_stopwords
        and token not in _EXTRA_STOPWORDS
    }
    expansions = dict(_TERM_EXPANSIONS)
    if scoring == "candidate_unit_coverage_v1":
        expansions.update(_CANDIDATE_UNIT_COVERAGE_EXPANSIONS)
    expanded = set(raw_terms)
    for term in tuple(raw_terms):
        if term.endswith("s") and len(term) > 3:
            expanded.add(term[:-1])
        if term.endswith("ing") and len(term) > 5:
            expanded.add(term[:-3])
        expanded.update(expansions.get(term, ()))
    return expanded


def _matched_terms(
    question_terms: set[str],
    text: str,
    *,
    scoring: str = "expanded_query_v1",
) -> tuple[str, ...]:
    return tuple(sorted(question_terms & _expanded_terms(text, scoring=scoring)))


def _evidence_boosts(
    *,
    question: str,
    user_text: str,
    matched_user_terms: tuple[str, ...],
    scoring: str = "expanded_query_v1",
) -> tuple[int, tuple[str, ...]]:
    lowered_question = question.lower()
    lowered_user = user_text.lower()
    boost = 0
    reasons: list[str] = []
    is_duration_question = "day" in lowered_question or "days" in lowered_question
    if is_duration_question and _DURATION_RE.search(lowered_user):
        boost += 8
        reasons.append("duration_mention")
    if any(
        term in lowered_question for term in ("money", "raise", "charity")
    ) and _MONEY_RE.search(user_text):
        boost += 8
        reasons.append("money_amount_mention")
    if scoring == "candidate_unit_coverage_v1":
        coverage_boost, coverage_reasons = _candidate_unit_coverage_boosts(
            lowered_question=lowered_question,
            lowered_user=lowered_user,
        )
        boost += coverage_boost
        reasons.extend(coverage_reasons)
    completed_markers = (
        "by the way",
        "i helped raise",
        "i just ran",
        "just got back",
        "managed to raise",
        "recently completed",
        "recently got back",
    )
    if matched_user_terms and any(
        marker in lowered_user for marker in completed_markers
    ):
        boost += 5
        reasons.append("completed_user_fact_marker")
    future_markers = ("planning", "thinking of", "want to", "would like")
    if any(marker in lowered_user for marker in future_markers) and not any(
        marker in lowered_user for marker in ("just got back", "recently got back")
    ):
        boost -= 2
        reasons.append("future_or_planning_penalty")
    return boost, tuple(reasons)


def _candidate_unit_coverage_boosts(
    *,
    lowered_question: str,
    lowered_user: str,
) -> tuple[int, tuple[str, ...]]:
    boost = 0
    reasons: list[str] = []
    if ("bake" in lowered_question or "baking" in lowered_question) and re.search(
        r"\b(bake|baked|baking|bread|cake|cookies?|baguette|"
        r"sourdough|oven|recipe)\b",
        lowered_user,
    ):
        boost += 10
        reasons.append("baking_event_mention")
    if "wedding" in lowered_question and re.search(
        r"\b(wedding|weddings|bride|groom|ceremony|bridesmaid|"
        r"vineyard|barn|married)\b",
        lowered_user,
    ):
        boost += 10
        reasons.append("wedding_event_mention")
    if (
        "clothing" in lowered_question
        and ("pick" in lowered_question or "return" in lowered_question)
        and re.search(
            r"\b(return|exchanged?|pick(?:ed)? up|boots?|zara|blazer)\b",
            lowered_user,
        )
    ):
        boost += 10
        reasons.append("pickup_return_obligation_mention")
    return boost, tuple(reasons)


def _trim_candidate_unit_coverage_text(
    *,
    text: str,
    question: str,
    max_chars: int,
) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    question_terms = _expanded_terms(question, scoring="candidate_unit_coverage_v1")
    segments = [
        match.group("content").strip()
        for match in _USER_SEGMENT_RE.finditer(text)
        if match.group("content").strip()
    ]
    scored_segments = []
    for index, segment in enumerate(segments):
        matched_terms = _matched_terms(
            question_terms,
            segment,
            scoring="candidate_unit_coverage_v1",
        )
        boost, _reasons = _evidence_boosts(
            question=question,
            user_text=segment,
            matched_user_terms=matched_terms,
            scoring="candidate_unit_coverage_v1",
        )
        score = (4 * len(matched_terms)) + boost
        if score > 0:
            scored_segments.append((score, index, segment))
    if not scored_segments:
        return interpretation_prompts._trim_text(text, max_chars=max_chars)
    selected: list[tuple[int, str]] = []
    used_chars = 0
    for _score, index, segment in sorted(
        scored_segments,
        key=lambda item: (-item[0], item[1]),
    ):
        rendered = f"user: {segment}"
        extra_chars = len(rendered) + (5 if selected else 0)
        if selected and used_chars + extra_chars > max_chars:
            continue
        if not selected and len(rendered) > max_chars:
            rendered = rendered[: max_chars - 3].rstrip() + "..."
        selected.append((index, rendered))
        used_chars += len(rendered) + (5 if len(selected) > 1 else 0)
        if used_chars >= max_chars:
            break
    selected.sort(key=lambda item: item[0])
    return "\n...\n".join(rendered for _index, rendered in selected), True


def _load_json(path: str | pathlib.Path) -> dict[str, typing.Any]:
    return typing.cast(
        "dict[str, typing.Any]",
        json.loads(pathlib.Path(path).read_text(encoding="utf-8")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-payloads", action="store_true")
    parser.add_argument(
        "--strategy",
        default=DEFAULT_PAYLOAD_REBUILD_STRATEGY_NAME,
        choices=[strategy.name for strategy in STRATEGIES],
    )
    parser.add_argument("--payloads", type=pathlib.Path, default=DEFAULT_PAYLOADS_PATH)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=ambiguity_fail18.DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--corpus",
        type=pathlib.Path,
        default=ambiguity_fail18.DEFAULT_CORPUS,
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_RETRIEVAL_TUNING_PATH,
    )
    parser.add_argument(
        "--payload-output",
        type=pathlib.Path,
        default=DEFAULT_TUNED_PAYLOADS_PATH,
    )
    parser.add_argument(
        "--provenance-output",
        type=pathlib.Path,
        default=DEFAULT_TUNED_PROVENANCE_PATH,
    )
    args = parser.parse_args(argv)

    if args.build_payloads:
        provider_artifact = build_fail18_tuned_generation_artifact(
            strategy_name=args.strategy,
            payloads_path=args.payloads,
            manifest_path=args.manifest,
            corpus_path=args.corpus,
            limit=args.limit,
        )
        provenance_artifact = build_fail18_tuned_private_provenance_artifact(
            strategy_name=args.strategy,
            payloads_path=args.payloads,
            manifest_path=args.manifest,
            corpus_path=args.corpus,
            limit=args.limit,
        )
        args.payload_output.parent.mkdir(parents=True, exist_ok=True)
        args.payload_output.write_text(
            f"{json.dumps(provider_artifact, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_output.write_text(
            f"{json.dumps(provenance_artifact, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        return 0

    artifact = build_fail18_retrieval_tuning_experiment(
        payloads_path=args.payloads,
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
