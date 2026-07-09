"""Measure fail18 retrieval recall for Gate 1 interpretation payloads."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import pathlib
import typing

from simba.eval import ambiguity_fail18, interpretation_prompts

DEFAULT_RETRIEVAL_RECALL_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_retrieval_recall_probe.json"
)
DEFAULT_PAYLOADS_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_payloads.json"
)
DEFAULT_PAYLOAD_PROVENANCE_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_payloads_provenance.json"
)


@dataclasses.dataclass(frozen=True)
class RankedSession:
    raw_session_index: int
    raw_session_id: str
    date: str
    selection_rank: int | None
    selection_score: int
    user_selection_score: int
    assistant_selection_score: int
    rendered_chars: int
    selected_by_budget: bool
    simulated_provider_id: str | None
    simulated_trimmed_chars: int
    simulated_truncated: bool
    matched_question_terms_user: tuple[str, ...]
    matched_question_terms_assistant: tuple[str, ...]


def build_fail18_retrieval_recall_probe(
    *,
    payloads_path: str | pathlib.Path = DEFAULT_PAYLOADS_PATH,
    payload_provenance_path: str | pathlib.Path = DEFAULT_PAYLOAD_PROVENANCE_PATH,
    corpus_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_CORPUS,
) -> dict[str, typing.Any]:
    payload_artifact = _load_json(payloads_path)
    provenance_artifact = _load_json(payload_provenance_path)
    corpus_rows = ambiguity_fail18.load_corpus(corpus_path)
    selection_config = _selection_config(payload_artifact)
    payloads_by_id = _payloads_by_id(payload_artifact)
    provenance_by_id = typing.cast(
        "dict[str, dict[str, dict[str, typing.Any]]]",
        provenance_artifact.get("evidence_provenance", {}),
    )
    cases = [
        _probe_case(
            row=row,
            payload=payloads_by_id.get(str(row.get("question_id", "")), {}),
            provenance=provenance_by_id.get(str(row.get("question_id", "")), {}),
            selection_config=selection_config,
        )
        for row in corpus_rows
    ]
    reason_counts: collections.Counter[str] = collections.Counter()
    for case in cases:
        for answer_session in case["answer_sessions"]:
            reason_counts[str(answer_session["retrieval_status"])] += 1
    answer_total = sum(int(case["answer_session_count"]) for case in cases)
    selected_total = sum(int(case["answer_sessions_in_payload"]) for case in cases)
    rows_with_all = sum(1 for case in cases if case["all_answer_sessions_in_payload"])
    rows_with_any_missing = sum(1 for case in cases if case["missing_answer_sessions"])
    return {
        "name": "fail18-ambiguous-nlidb-gate1-retrieval-recall-probe",
        "artifact_kind": "interpretation_retrieval_recall_probe",
        "gate": "gate1",
        "gate_status": "slice2d_retrieval_recall_probe_complete",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_payloads": str(payloads_path),
        "source_payload_provenance": str(payload_provenance_path),
        "source_corpus": str(corpus_path),
        "selection_config": selection_config,
        "summary": {
            "rows_total": len(cases),
            "rows_with_all_answer_sessions_in_payload": rows_with_all,
            "rows_with_missing_answer_sessions": rows_with_any_missing,
            "answer_sessions_total": answer_total,
            "answer_sessions_in_payload": selected_total,
            "answer_session_recall": (
                selected_total / answer_total if answer_total else None
            ),
            "retrieval_status_counts": dict(sorted(reason_counts.items())),
        },
        "decision": {
            "retrieval_should_be_tuned_before_candidate_compilation": (
                rows_with_any_missing > 0
            ),
            "next_slice": "retrieval_tuning_experiment",
            "reason": (
                "The probe uses private answer_session_ids only for eval. Any "
                "retrieval change must be expressible from question and evidence "
                "text before rebuilt payloads can be credited."
            ),
        },
        "cases": cases,
    }


def _probe_case(
    *,
    row: dict[str, typing.Any],
    payload: dict[str, typing.Any],
    provenance: dict[str, dict[str, typing.Any]],
    selection_config: dict[str, int],
) -> dict[str, typing.Any]:
    case_id = str(row.get("question_id", ""))
    question = str(row.get("question", ""))
    ranked_sessions = _rank_sessions(row=row, question=question, **selection_config)
    ranked_by_id = {session.raw_session_id: session for session in ranked_sessions}
    selected_by_id = _selected_payload_sessions(provenance)
    actual_payload_ids = sorted(selected_by_id)
    simulated_payload_ids = sorted(
        session.raw_session_id
        for session in ranked_sessions
        if session.selected_by_budget
    )
    answer_session_ids = [str(item) for item in row.get("answer_session_ids", [])]
    answer_sessions = [
        _answer_session_result(
            raw_session_id=raw_session_id,
            ranked_session=ranked_by_id.get(raw_session_id),
            selected_by_id=selected_by_id,
            selection_config=selection_config,
        )
        for raw_session_id in answer_session_ids
    ]
    missing = [
        item["raw_session_id"]
        for item in answer_sessions
        if item["retrieval_status"] != "included_in_payload"
    ]
    return {
        "case_id": case_id,
        "question": question,
        "question_date": row.get("question_date"),
        "answer_session_count": len(answer_session_ids),
        "answer_sessions_in_payload": len(answer_session_ids) - len(missing),
        "all_answer_sessions_in_payload": not missing,
        "missing_answer_sessions": missing,
        "selected_payload_raw_session_ids": actual_payload_ids,
        "simulated_payload_raw_session_ids": simulated_payload_ids,
        "simulated_payload_matches_artifact": actual_payload_ids
        == simulated_payload_ids,
        "answer_sessions": answer_sessions,
        "top_ranked_sessions": [
            _ranked_session_dict(session)
            for session in ranked_sessions[: selection_config["max_evidence_sessions"]]
        ],
        "recommendation": _case_recommendation(answer_sessions),
    }


def _rank_sessions(
    *,
    row: dict[str, typing.Any],
    question: str,
    max_evidence_sessions: int,
    max_evidence_chars: int,
    max_session_chars: int,
) -> list[RankedSession]:
    ids = row.get("haystack_session_ids", [])
    dates = row.get("haystack_dates", [])
    sessions = row.get("haystack_sessions", [])
    question_terms = interpretation_prompts._content_terms(question)
    scored: list[dict[str, typing.Any]] = []
    for idx, session in enumerate(sessions):
        sid = str(ids[idx]) if idx < len(ids) else str(idx)
        date = str(dates[idx]) if idx < len(dates) else ""
        text, user_text, assistant_text = interpretation_prompts._render_session(
            session
        )
        matched_user = _matched_terms(question_terms, user_text)
        matched_assistant = _matched_terms(question_terms, assistant_text)
        user_score = len(matched_user)
        assistant_score = len(matched_assistant)
        selection_score = (
            user_score * interpretation_prompts.USER_SCORE_WEIGHT
            + assistant_score * interpretation_prompts.ASSISTANT_SCORE_WEIGHT
        )
        scored.append(
            {
                "raw_session_index": idx,
                "raw_session_id": sid,
                "date": date,
                "text": text,
                "user_selection_score": user_score,
                "assistant_selection_score": assistant_score,
                "selection_score": selection_score,
                "matched_question_terms_user": matched_user,
                "matched_question_terms_assistant": matched_assistant,
            }
        )

    positives = [item for item in scored if int(item["selection_score"]) > 0]
    ranked_items = sorted(
        positives,
        key=lambda item: (
            -int(item["selection_score"]),
            -int(item["user_selection_score"]),
            int(item["raw_session_index"]),
        ),
    )
    if not ranked_items:
        ranked_items = scored
    selected_by_budget = _simulate_budget_selection(
        ranked_items=ranked_items,
        max_evidence_sessions=max_evidence_sessions,
        max_evidence_chars=max_evidence_chars,
        max_session_chars=max_session_chars,
    )
    return [
        RankedSession(
            raw_session_index=int(item["raw_session_index"]),
            raw_session_id=str(item["raw_session_id"]),
            date=str(item["date"]),
            selection_rank=rank,
            selection_score=int(item["selection_score"]),
            user_selection_score=int(item["user_selection_score"]),
            assistant_selection_score=int(item["assistant_selection_score"]),
            rendered_chars=len(str(item["text"])),
            selected_by_budget=str(item["raw_session_id"]) in selected_by_budget,
            simulated_provider_id=selected_by_budget.get(
                str(item["raw_session_id"]), {}
            ).get("provider_id"),
            simulated_trimmed_chars=int(
                selected_by_budget.get(str(item["raw_session_id"]), {}).get(
                    "trimmed_chars", 0
                )
            ),
            simulated_truncated=bool(
                selected_by_budget.get(str(item["raw_session_id"]), {}).get(
                    "truncated", False
                )
            ),
            matched_question_terms_user=tuple(item["matched_question_terms_user"]),
            matched_question_terms_assistant=tuple(
                item["matched_question_terms_assistant"]
            ),
        )
        for rank, item in enumerate(ranked_items, start=1)
    ]


def _simulate_budget_selection(
    *,
    ranked_items: list[dict[str, typing.Any]],
    max_evidence_sessions: int,
    max_evidence_chars: int,
    max_session_chars: int,
) -> dict[str, dict[str, typing.Any]]:
    selected: dict[str, dict[str, typing.Any]] = {}
    remaining_chars = max(0, max_evidence_chars)
    for rank, item in enumerate(ranked_items[: max(0, max_evidence_sessions)], start=1):
        if remaining_chars <= 0:
            break
        trimmed, truncated = interpretation_prompts._trim_text(
            str(item["text"]),
            max_chars=min(max_session_chars, remaining_chars),
        )
        remaining_chars -= len(trimmed)
        selected[str(item["raw_session_id"])] = {
            "provider_id": f"evidence_{rank:03d}",
            "trimmed_chars": len(trimmed),
            "truncated": truncated,
        }
    return selected


def _answer_session_result(
    *,
    raw_session_id: str,
    ranked_session: RankedSession | None,
    selected_by_id: dict[str, dict[str, typing.Any]],
    selection_config: dict[str, int],
) -> dict[str, typing.Any]:
    if ranked_session is None:
        return {
            "raw_session_id": raw_session_id,
            "retrieval_status": "not_ranked_zero_score_filtered",
            "selection_rank": None,
            "selection_score": 0,
            "user_selection_score": 0,
            "assistant_selection_score": 0,
            "matched_question_terms_user": [],
            "matched_question_terms_assistant": [],
            "payload_evidence_id": None,
            "reason": (
                "The answer session had zero lexical overlap while other "
                "positive-scoring sessions existed."
            ),
        }
    payload_hit = selected_by_id.get(raw_session_id)
    status = _retrieval_status(
        ranked_session=ranked_session,
        selected=payload_hit is not None,
        max_evidence_sessions=selection_config["max_evidence_sessions"],
    )
    return {
        "raw_session_id": raw_session_id,
        "retrieval_status": status,
        "selection_rank": ranked_session.selection_rank,
        "selection_score": ranked_session.selection_score,
        "user_selection_score": ranked_session.user_selection_score,
        "assistant_selection_score": ranked_session.assistant_selection_score,
        "rendered_chars": ranked_session.rendered_chars,
        "matched_question_terms_user": list(ranked_session.matched_question_terms_user),
        "matched_question_terms_assistant": list(
            ranked_session.matched_question_terms_assistant
        ),
        "payload_evidence_id": (
            str(payload_hit["provider_id"]) if payload_hit is not None else None
        ),
        "reason": _retrieval_reason(status),
    }


def _retrieval_status(
    *,
    ranked_session: RankedSession,
    selected: bool,
    max_evidence_sessions: int,
) -> str:
    if selected:
        return "included_in_payload"
    if (
        ranked_session.selection_rank is not None
        and ranked_session.selection_rank > max_evidence_sessions
    ):
        return "outside_max_evidence_sessions"
    if (
        ranked_session.selection_rank is not None
        and not ranked_session.selected_by_budget
    ):
        return "omitted_by_total_char_budget"
    return "selected_by_simulation_but_missing_from_artifact"


def _retrieval_reason(status: str) -> str:
    reasons = {
        "included_in_payload": (
            "The bounded provider payload included this answer session."
        ),
        "outside_max_evidence_sessions": (
            "The answer session had lexical overlap but ranked below the "
            "configured evidence-session cutoff."
        ),
        "omitted_by_total_char_budget": (
            "The answer session ranked within the session cutoff but was not "
            "selected before the total evidence character budget was exhausted."
        ),
        "selected_by_simulation_but_missing_from_artifact": (
            "The reproduced selector would include this session, but the "
            "stored payload artifact does not."
        ),
    }
    return reasons.get(status, "Unknown retrieval status.")


def _case_recommendation(
    answer_sessions: list[dict[str, typing.Any]],
) -> dict[str, typing.Any]:
    statuses = {str(item["retrieval_status"]) for item in answer_sessions}
    if statuses == {"included_in_payload"} or not answer_sessions:
        return {
            "action": "no_retrieval_change_needed",
            "reason": "All answer sessions are present in the bounded payload.",
        }
    if "outside_max_evidence_sessions" in statuses:
        return {
            "action": "improve_query_or_rerank_before_expanding_budget",
            "reason": (
                "At least one answer session has positive lexical overlap but "
                "loses to distractors before the top-k cutoff."
            ),
        }
    if "omitted_by_total_char_budget" in statuses:
        return {
            "action": "compress_or_chunk_sessions_before_increasing_budget",
            "reason": (
                "At least one answer session is within top-k but the selected "
                "sessions consume the evidence character budget."
            ),
        }
    return {
        "action": "add_query_expansion_or_semantic_retrieval",
        "reason": (
            "At least one answer session is invisible to lexical question-term "
            "matching."
        ),
    }


def _ranked_session_dict(session: RankedSession) -> dict[str, typing.Any]:
    return {
        "raw_session_id": session.raw_session_id,
        "raw_session_index": session.raw_session_index,
        "date": session.date,
        "selection_rank": session.selection_rank,
        "selection_score": session.selection_score,
        "user_selection_score": session.user_selection_score,
        "assistant_selection_score": session.assistant_selection_score,
        "rendered_chars": session.rendered_chars,
        "selected_by_budget": session.selected_by_budget,
        "simulated_provider_id": session.simulated_provider_id,
        "matched_question_terms_user": list(session.matched_question_terms_user),
        "matched_question_terms_assistant": list(
            session.matched_question_terms_assistant
        ),
    }


def _selected_payload_sessions(
    provenance: dict[str, dict[str, typing.Any]],
) -> dict[str, dict[str, typing.Any]]:
    selected = {}
    for provider_id, metadata in provenance.items():
        raw_session_id = str(metadata.get("raw_session_id", ""))
        if raw_session_id:
            selected[raw_session_id] = {
                "provider_id": provider_id,
                "selection_rank": metadata.get("selection_rank"),
                "selection_score": metadata.get("selection_score"),
            }
    return selected


def _matched_terms(question_terms: set[str], text: str) -> tuple[str, ...]:
    text_terms = interpretation_prompts._content_terms(text)
    return tuple(sorted(question_terms & text_terms))


def _selection_config(payload_artifact: dict[str, typing.Any]) -> dict[str, int]:
    selection = payload_artifact.get("evidence_selection", {})
    if not isinstance(selection, dict):
        selection = {}
    return {
        "max_evidence_sessions": int(
            selection.get(
                "max_evidence_sessions",
                interpretation_prompts.DEFAULT_MAX_EVIDENCE_SESSIONS,
            )
        ),
        "max_evidence_chars": int(
            selection.get(
                "max_evidence_chars",
                interpretation_prompts.DEFAULT_MAX_EVIDENCE_CHARS,
            )
        ),
        "max_session_chars": int(
            selection.get(
                "max_session_chars",
                interpretation_prompts.DEFAULT_MAX_SESSION_CHARS,
            )
        ),
    }


def _payloads_by_id(
    payload_artifact: dict[str, typing.Any],
) -> dict[str, dict[str, typing.Any]]:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=DEFAULT_RETRIEVAL_RECALL_PATH,
    )
    args = parser.parse_args(argv)

    artifact = build_fail18_retrieval_recall_probe(
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
