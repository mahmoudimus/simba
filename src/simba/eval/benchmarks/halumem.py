"""HaluMem loader + QA aggregation (IAAR-Shanghai/HaluMem).

HaluMem (arXiv 2511.03506, MemTensor) is an *operation-level memory-hallucination*
benchmark — NOT a recall@k set. Each JSONL line is one user: ``persona_info`` plus
``sessions``; each session carries:

- ``memory_points``: ground-truth memories. A point with ``is_update=True``
  supersedes the ids in ``original_memories`` (the contradiction/staleness signal).
- ``questions``: ``question`` / ``answer`` / ``evidence`` (the memory points needed)
  / ``question_type`` / ``difficulty``. ``question_type == "Memory Boundary"`` are
  **abstention** cases — the gold answer is "not in memory", so a confident answer
  is a *hallucination*.

Its metrics reward *not* surfacing wrong/stale memories (the inverse of recall@k),
so simba's Phase-6 dormant tier + Phase-7 contradiction-resolution can finally show
measurable value. This module is the loader + the pure (LLM-free) QA aggregation;
the judge-backed evaluator that produces per-question outcomes lives in the QA
runner (see ``halumem_qa``).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import typing

BOUNDARY_TYPE = "Memory Boundary"

# QA outcome labels (assigned by the LLM judge in the evaluator).
QA_CORRECT = "correct"
QA_HALLUCINATION = "hallucination"
QA_OMISSION = "omission"


@dataclasses.dataclass
class MemoryPoint:
    index: int
    content: str
    memory_type: str = ""
    is_update: bool = False
    original_memories: list[typing.Any] = dataclasses.field(default_factory=list)
    timestamp: str = ""
    importance: typing.Any = None

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> MemoryPoint:
        return cls(
            index=int(raw.get("index", 0)),
            content=str(raw.get("memory_content", "")),
            memory_type=str(raw.get("memory_type", "")),
            is_update=bool(raw.get("is_update", False)),
            original_memories=list(raw.get("original_memories", []) or []),
            timestamp=str(raw.get("timestamp", "")),
            importance=raw.get("importance"),
        )


@dataclasses.dataclass
class HaluQuestion:
    question: str
    answer: str
    evidence: list[str]  # the memory_content strings needed to answer
    question_type: str = ""
    difficulty: str = ""

    @property
    def is_boundary(self) -> bool:
        """Abstention case (gold = "not in memory"); answering it is a hallucination."""
        return self.question_type == BOUNDARY_TYPE or not self.evidence

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> HaluQuestion:
        ev = [
            str(e.get("memory_content", ""))
            for e in (raw.get("evidence") or [])
            if isinstance(e, dict)
        ]
        return cls(
            question=str(raw.get("question", "")),
            answer=str(raw.get("answer", "")),
            evidence=[e for e in ev if e],
            question_type=str(raw.get("question_type", "")),
            difficulty=str(raw.get("difficulty", "")),
        )


@dataclasses.dataclass
class HaluSession:
    memory_points: list[MemoryPoint]
    questions: list[HaluQuestion]
    start_time: str = ""
    end_time: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> HaluSession:
        return cls(
            memory_points=[
                MemoryPoint.from_dict(m) for m in (raw.get("memory_points") or [])
            ],
            questions=[
                HaluQuestion.from_dict(q) for q in (raw.get("questions") or [])
            ],
            start_time=str(raw.get("start_time", "")),
            end_time=str(raw.get("end_time", "")),
        )


@dataclasses.dataclass
class HaluUser:
    uuid: str
    persona: str
    sessions: list[HaluSession]

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> HaluUser:
        return cls(
            uuid=str(raw.get("uuid", "")),
            persona=str(raw.get("persona_info", "")),
            sessions=[HaluSession.from_dict(s) for s in (raw.get("sessions") or [])],
        )


def load_halumem(
    path: str | pathlib.Path, *, user_limit: int = 0
) -> list[HaluUser]:
    """Load a HaluMem ``*.jsonl`` (one user/line). ``user_limit>0`` subsamples
    the first N users (the corpus is >1M tokens/user — see docs/plans/10)."""
    users: list[HaluUser] = []
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        users.append(HaluUser.from_dict(json.loads(line)))
        if user_limit and len(users) >= user_limit:
            break
    return users


def _rate(rows: list[tuple[str, str]], label: str) -> float:
    return (sum(1 for _, o in rows if o == label) / len(rows)) if rows else 0.0


def aggregate_qa(outcomes: list[tuple[str, str]]) -> dict[str, typing.Any]:
    """Aggregate per-question ``(question_type, outcome)`` into HaluMem-style rates.

    Returns overall + per-type ``accuracy`` / ``hallucination_rate`` /
    ``omission_rate``, plus a dedicated ``boundary`` block (abstention accuracy on
    "Memory Boundary" questions — the cleanest hallucination-resistance signal).
    Higher accuracy + lower hallucination_rate is the goal; unlike recall@k this
    rewards *not* surfacing wrong/stale memories.
    """

    def block(rows: list[tuple[str, str]]) -> dict[str, typing.Any]:
        return {
            "n": len(rows),
            "accuracy": _rate(rows, QA_CORRECT),
            "hallucination_rate": _rate(rows, QA_HALLUCINATION),
            "omission_rate": _rate(rows, QA_OMISSION),
        }

    by_type: dict[str, list[tuple[str, str]]] = {}
    for qtype, outcome in outcomes:
        by_type.setdefault(qtype or "?", []).append((qtype, outcome))

    boundary = [r for r in outcomes if r[0] == BOUNDARY_TYPE]
    return {
        "overall": block(outcomes),
        "by_type": {t: block(rows) for t, rows in sorted(by_type.items())},
        "boundary": block(boundary),
    }
