"""LongMemEval loader: xiaowu0162 `longmemeval_*` JSON -> simba eval Datasets.

One Dataset per question (each question carries its own multi-session haystack).
Corpus units are individual chat turns keyed ``{session_id}#{turn_index}``;
gold evidence is the set of turns flagged ``has_answer``, so the existing recall
harness scores turn-level recall@k of evidence directly.

Abstention questions (``question_id`` ending ``_abs``) are excluded by default:
they test refusal, so "evidence recall" is ill-defined and their ``has_answer``
marking is inconsistent. Pass ``include_abstention=True`` to keep the resolvable
ones. Questions whose evidence can't be resolved (no ``has_answer`` turn) are
always dropped.

Note on granularity: against the *oracle* haystack (only the evidence sessions),
recall is an **upper bound** — the real test is the full ``longmemeval_s``
haystack (~hundreds of distractor sessions per question).
"""

from __future__ import annotations

import json
import pathlib
import typing

from simba.eval.dataset import Dataset, EvalCase, Memory


def is_abstention(question_id: str) -> bool:
    """Return True when the question_id ends with '_abs'."""
    return str(question_id).endswith("_abs")


def _session_turns(
    session_id: str, turns: list[dict[str, typing.Any]], date: str = ""
) -> list[Memory]:
    out: list[Memory] = []
    date = date.strip()
    for i, turn in enumerate(turns):
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        role = turn.get("role", "")
        body = f"{role}: {content}" if role else content
        # prefix the session date so relative time ("last week") is groundable
        text = f"[{date}] {body}" if date else body
        out.append(Memory(id=f"{session_id}#{i}", content=text, type="PATTERN"))
    return out


def load_longmemeval_data(
    raw: list[dict[str, typing.Any]], *, include_abstention: bool = False
) -> list[Dataset]:
    """Convert parsed LongMemEval questions into one Dataset per question."""
    datasets: list[Dataset] = []
    for q in raw:
        qid = str(q.get("question_id", f"q-{len(datasets)}"))
        if qid.endswith("_abs") and not include_abstention:
            continue

        session_ids = q.get("haystack_session_ids", [])
        sessions = q.get("haystack_sessions", [])
        dates = q.get("haystack_dates", [])

        corpus: list[Memory] = []
        gold: list[str] = []
        for idx, (sid, turns) in enumerate(zip(session_ids, sessions, strict=False)):
            sid = str(sid)
            date = str(dates[idx]) if idx < len(dates) else ""
            corpus.extend(_session_turns(sid, turns, date))
            for i, turn in enumerate(turns):
                if turn.get("has_answer") and (turn.get("content") or "").strip():
                    gold.append(f"{sid}#{i}")

        if not corpus or not gold:  # unresolvable -> drop
            continue
        cases = [
            EvalCase(
                id=qid,
                query=str(q.get("question", "")),
                relevant_ids=gold,
                intent=str(q.get("question_type", "")),
                answer=str(q.get("answer") or ""),
                question_date=str(q.get("question_date", "")),
            )
        ]
        datasets.append(Dataset(name=qid, corpus=corpus, cases=cases))
    return datasets


def load_longmemeval(
    path: str | pathlib.Path, *, include_abstention: bool = False
) -> list[Dataset]:
    """Load + parse a longmemeval_*.json file into per-question Datasets."""
    raw = json.loads(pathlib.Path(path).read_text())
    return load_longmemeval_data(raw, include_abstention=include_abstention)
