"""Prompt payload builders for ambiguous NLIDB interpretation generation."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import pathlib
import re
import typing

import simba.eval.ambiguity_fail18 as ambiguity_fail18
from simba.eval.ambiguity_taxonomy import ambiguity_type_values

PROMPT_VERSION = "interpretation_generator_v1"
DEFAULT_MAX_EVIDENCE_SESSIONS = 8
DEFAULT_MAX_EVIDENCE_CHARS = 24_000
DEFAULT_MAX_SESSION_CHARS = 4_000
USER_SCORE_WEIGHT = 2
ASSISTANT_SCORE_WEIGHT = 1

INTERPRETATION_GENERATOR_CONTRACT = (
    "Return exactly one strict JSON object. No markdown.",
    "Generate natural-language interpretations before compiling facts or "
    "candidate units.",
    "List semantically distinct readings of the question, including reasonable "
    "alternatives the first reading may hide.",
    "Use only the question and supplied evidence. Do not use web knowledge.",
    "Do not compute the final answer and do not choose one interpretation as "
    "the winner.",
    "Keep interpretations domain-general: label ambiguity families, "
    "assumptions, and expected answer shape.",
    "Prefer answer spaces or ranges when the evidence supports multiple "
    "reasonable readings.",
)


@dataclasses.dataclass(frozen=True)
class _ScoredSession:
    raw_session_index: int
    raw_session_id: str
    date: str
    text: str
    user_selection_score: int
    assistant_selection_score: int

    @property
    def selection_score(self) -> int:
        return (
            self.user_selection_score * USER_SCORE_WEIGHT
            + self.assistant_selection_score * ASSISTANT_SCORE_WEIGHT
        )


def build_interpretation_generation_payload(
    *,
    case_id: str,
    question: str,
    evidence_sessions: list[dict[str, typing.Any]],
) -> dict[str, typing.Any]:
    """Build the provider-facing payload for first-pass interpretations."""
    return {
        "task": (
            "Generate candidate natural-language interpretations for an "
            "ambiguous memory question before any executable compilation."
        ),
        "prompt_version": PROMPT_VERSION,
        "generation_contract": list(INTERPRETATION_GENERATOR_CONTRACT),
        "allowed_ambiguity_types": list(ambiguity_type_values()),
        "output_schema": {
            "case_id": case_id,
            "interpretations": [
                {
                    "interpretation_id": "stable string",
                    "natural_language_interpretation": "string",
                    "ambiguity_types": ["one or more allowed_ambiguity_types"],
                    "assumptions": ["string"],
                    "expected_answer_shape": "count|sum|lookup|range|set",
                }
            ],
        },
        "case": {
            "id": case_id,
            "question": question,
            "evidence_sessions": evidence_sessions,
        },
    }


def fail18_interpretation_generation_payloads(
    *,
    manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    corpus_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_CORPUS,
    limit: int = 0,
    max_evidence_sessions: int = DEFAULT_MAX_EVIDENCE_SESSIONS,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
    max_session_chars: int = DEFAULT_MAX_SESSION_CHARS,
) -> list[dict[str, typing.Any]]:
    payloads, _provenance = _fail18_payloads_and_provenance(
        manifest_path=manifest_path,
        corpus_path=corpus_path,
        limit=limit,
        max_evidence_sessions=max_evidence_sessions,
        max_evidence_chars=max_evidence_chars,
        max_session_chars=max_session_chars,
    )
    return payloads


def _fail18_payloads_and_provenance(
    *,
    manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    corpus_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_CORPUS,
    limit: int = 0,
    max_evidence_sessions: int = DEFAULT_MAX_EVIDENCE_SESSIONS,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
    max_session_chars: int = DEFAULT_MAX_SESSION_CHARS,
) -> tuple[
    list[dict[str, typing.Any]],
    dict[str, dict[str, dict[str, typing.Any]]],
]:
    rows = ambiguity_fail18.load_manifest(manifest_path)
    corpus_by_id = {
        str(row["question_id"]): row
        for row in ambiguity_fail18.load_corpus(corpus_path)
    }
    if limit > 0:
        rows = rows[:limit]
    payloads: list[dict[str, typing.Any]] = []
    provenance: dict[str, dict[str, dict[str, typing.Any]]] = {}
    for row in rows:
        qid = str(row["question_id"])
        question = str(row.get("question", ""))
        corpus_row = corpus_by_id.get(qid)
        evidence_sessions: list[dict[str, typing.Any]] = []
        evidence_provenance: dict[str, dict[str, typing.Any]] = {}
        if corpus_row is not None:
            evidence_sessions, evidence_provenance = _evidence_sessions(
                corpus_row,
                question=question,
                max_sessions=max_evidence_sessions,
                max_total_chars=max_evidence_chars,
                max_session_chars=max_session_chars,
            )
        provenance[qid] = evidence_provenance
        payloads.append(
            build_interpretation_generation_payload(
                case_id=qid,
                question=question,
                evidence_sessions=evidence_sessions,
            )
        )
    return payloads, provenance


def build_fail18_generation_artifact(
    *,
    manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    corpus_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_CORPUS,
    limit: int = 0,
    max_evidence_sessions: int = DEFAULT_MAX_EVIDENCE_SESSIONS,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
    max_session_chars: int = DEFAULT_MAX_SESSION_CHARS,
) -> dict[str, typing.Any]:
    payloads, _provenance = _fail18_payloads_and_provenance(
        manifest_path=manifest_path,
        corpus_path=corpus_path,
        limit=limit,
        max_evidence_sessions=max_evidence_sessions,
        max_evidence_chars=max_evidence_chars,
        max_session_chars=max_session_chars,
    )
    return {
        "name": "fail18-ambiguous-nlidb-gate1-payloads",
        "artifact_kind": "provider_payloads",
        "gate": "gate1",
        "gate_status": "payloads_only_not_run",
        "known_gap": (
            "This artifact contains provider payloads only. It does not contain "
            "model outputs, parsed InterpretationRecords, duplicate filtering, "
            "or execution coverage metrics."
        ),
        "prompt_version": PROMPT_VERSION,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_manifest": str(manifest_path),
        "source_corpus": str(corpus_path),
        "evidence_selection": {
            "method": "lexical_question_topk",
            "max_evidence_sessions": max_evidence_sessions,
            "max_evidence_chars": max_evidence_chars,
            "max_session_chars": max_session_chars,
            "score_formula": (
                f"{USER_SCORE_WEIGHT} * user_term_overlap + "
                f"{ASSISTANT_SCORE_WEIGHT} * assistant_term_overlap"
            ),
            "uses_answer_session_ids": False,
            "provider_session_ids": "opaque evidence_NNN ids",
            "raw_session_ids": "private provenance artifact only",
        },
        "total": len(payloads),
        "commands": [
            (
                "uv run python -m simba.eval.interpretation_prompts "
                "--fail18 "
                "--output _gitless/fail18_ambiguous_nlidb_gate1_payloads.json "
                "--provenance-output "
                "_gitless/fail18_ambiguous_nlidb_gate1_payloads_provenance.json"
            )
        ],
        "payloads": payloads,
    }


def build_fail18_private_provenance_artifact(
    *,
    manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    corpus_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_CORPUS,
    limit: int = 0,
    max_evidence_sessions: int = DEFAULT_MAX_EVIDENCE_SESSIONS,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
    max_session_chars: int = DEFAULT_MAX_SESSION_CHARS,
) -> dict[str, typing.Any]:
    _payloads, provenance = _fail18_payloads_and_provenance(
        manifest_path=manifest_path,
        corpus_path=corpus_path,
        limit=limit,
        max_evidence_sessions=max_evidence_sessions,
        max_evidence_chars=max_evidence_chars,
        max_session_chars=max_session_chars,
    )
    return {
        "name": "fail18-ambiguous-nlidb-gate1-payloads-provenance",
        "artifact_kind": "private_provenance",
        "provider_payload_artifact": (
            "_gitless/fail18_ambiguous_nlidb_gate1_payloads.json"
        ),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_manifest": str(manifest_path),
        "source_corpus": str(corpus_path),
        "total": len(provenance),
        "evidence_provenance": provenance,
    }


def _evidence_sessions(
    row: dict[str, typing.Any],
    *,
    question: str,
    max_sessions: int,
    max_total_chars: int,
    max_session_chars: int,
) -> tuple[list[dict[str, typing.Any]], dict[str, dict[str, typing.Any]]]:
    ids = row.get("haystack_session_ids", [])
    dates = row.get("haystack_dates", [])
    sessions = row.get("haystack_sessions", [])
    scored: list[_ScoredSession] = []
    question_terms = _content_terms(question)
    for idx, session in enumerate(sessions):
        sid = str(ids[idx]) if idx < len(ids) else str(idx)
        date = str(dates[idx]) if idx < len(dates) else ""
        text, user_text, assistant_text = _render_session(session)
        scored.append(
            _ScoredSession(
                raw_session_index=idx,
                raw_session_id=sid,
                date=date,
                text=text,
                user_selection_score=_lexical_score(question_terms, user_text),
                assistant_selection_score=_lexical_score(
                    question_terms, assistant_text
                ),
            )
        )

    positives = [item for item in scored if item.selection_score > 0]
    ranked = sorted(
        positives,
        key=lambda item: (
            -item.selection_score,
            -item.user_selection_score,
            item.raw_session_index,
        ),
    )
    if not ranked:
        ranked = scored

    evidence: list[dict[str, typing.Any]] = []
    provenance: dict[str, dict[str, typing.Any]] = {}
    remaining_chars = max(0, max_total_chars)
    for rank, item in enumerate(
        ranked[: max(0, max_sessions)],
        start=1,
    ):
        if remaining_chars <= 0:
            break
        trimmed, truncated = _trim_text(
            item.text,
            max_chars=min(max_session_chars, remaining_chars),
        )
        remaining_chars -= len(trimmed)
        provider_id = f"evidence_{rank:03d}"
        evidence.append(
            {
                "session_id": provider_id,
                "date": item.date,
                "selection_rank": rank,
                "selection_score": item.selection_score,
                "user_selection_score": item.user_selection_score,
                "assistant_selection_score": item.assistant_selection_score,
                "truncated": truncated,
                "text": trimmed,
            }
        )
        provenance[provider_id] = {
            "raw_session_id": item.raw_session_id,
            "raw_session_index": item.raw_session_index,
            "date": item.date,
            "selection_rank": rank,
            "selection_score": item.selection_score,
            "user_selection_score": item.user_selection_score,
            "assistant_selection_score": item.assistant_selection_score,
            "truncated": truncated,
        }
    return evidence, provenance


def _render_session(session: typing.Any) -> tuple[str, str, str]:
    if not isinstance(session, list):
        return "", "", ""
    lines: list[str] = []
    user_lines: list[str] = []
    assistant_lines: list[str] = []
    for message in session:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip() or "unknown"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
            if role == "user":
                user_lines.append(content)
            elif role == "assistant":
                assistant_lines.append(content)
    return "\n".join(lines), "\n".join(user_lines), "\n".join(assistant_lines)


def _content_terms(text: str) -> set[str]:
    stopwords = {
        "about",
        "after",
        "also",
        "from",
        "have",
        "many",
        "much",
        "need",
        "that",
        "the",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _lexical_score(question_terms: set[str], text: str) -> int:
    if not question_terms:
        return 0
    text_terms = _content_terms(text)
    return len(question_terms & text_terms)


def _trim_text(text: str, *, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    marker = "\n...[truncated]"
    if max_chars <= len(marker):
        return text[:max_chars], True
    return f"{text[: max_chars - len(marker)].rstrip()}{marker}", True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail18", action="store_true", help="Build fail18 payloads.")
    parser.add_argument("--path", default="", help="fail18 manifest path.")
    parser.add_argument("--corpus", default="", help="fail18 corpus path.")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows.")
    parser.add_argument(
        "--max-evidence-sessions",
        type=int,
        default=DEFAULT_MAX_EVIDENCE_SESSIONS,
        help="Maximum evidence sessions per payload.",
    )
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=DEFAULT_MAX_EVIDENCE_CHARS,
        help="Maximum rendered evidence characters per payload.",
    )
    parser.add_argument(
        "--max-session-chars",
        type=int,
        default=DEFAULT_MAX_SESSION_CHARS,
        help="Maximum rendered characters per evidence session.",
    )
    parser.add_argument("--output", type=pathlib.Path, help="Write JSON artifact.")
    parser.add_argument(
        "--provenance-output",
        type=pathlib.Path,
        help="Write private raw-session provenance JSON artifact.",
    )
    args = parser.parse_args(argv)

    if not args.fail18:
        raise SystemExit("pass --fail18")
    manifest_path = (
        pathlib.Path(args.path) if args.path else ambiguity_fail18.DEFAULT_MANIFEST
    )
    corpus_path = (
        pathlib.Path(args.corpus)
        if args.corpus
        else ambiguity_fail18.DEFAULT_CORPUS
    )
    artifact = build_fail18_generation_artifact(
        manifest_path=manifest_path,
        corpus_path=corpus_path,
        limit=args.limit,
        max_evidence_sessions=args.max_evidence_sessions,
        max_evidence_chars=args.max_evidence_chars,
        max_session_chars=args.max_session_chars,
    )
    encoded = json.dumps(artifact, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(f"{encoded}\n", encoding="utf-8")
    else:
        print(encoded)
    if args.provenance_output:
        provenance = build_fail18_private_provenance_artifact(
            manifest_path=manifest_path,
            corpus_path=corpus_path,
            limit=args.limit,
            max_evidence_sessions=args.max_evidence_sessions,
            max_evidence_chars=args.max_evidence_chars,
            max_session_chars=args.max_session_chars,
        )
        args.provenance_output.write_text(
            f"{json.dumps(provenance, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
