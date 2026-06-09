"""SubtleMemory loader + per-relation-slice aggregation.

SubtleMemory is a *relational / contradiction* memory eval: the hard signal is
not raw recall but whether the system surfaces the right combination of memories
when they **relate** — complementary (combine / any-one), nuanced (a time /
context boundary decides which applies), or **contradictory** (an unresolved
conflict that should be surfaced, not silently collapsed). ``contradictory`` is
the headline slice — the primary differentiator between memory systems.

Data layout (one dir per persona)::

    data/subtlememory/persona_{0..9}/
        bench_instances.json   # cases: relation labels + facts + QA pairs
        history_sessions.json  # the conversation corpus (multi-turn, timestamped)

This loader maps one persona -> one simba ``Dataset`` (mirroring
``halumem.py`` / ``locomo.py``): every dialogue turn becomes a ``Memory`` keyed
``{session_id}_{turn_index}``; every QA pair becomes an ``EvalCase`` whose gold
(``relevant_ids``) is all turns of the case's target ``session_ids`` and whose
``intent`` is the ``relation_type`` (so the existing recall / QA harness groups
by relation slice automatically) and ``note`` the ``relation_subtype``.

``aggregate_by_relation`` mirrors ``halumem.aggregate_qa``: overall + per-slice
accuracy with a dedicated ``contradictory`` block flagged ``is_headline``.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import typing

from simba.eval.dataset import Dataset, EvalCase, Memory

# The headline relation slice: an unresolved conflict between memories.
CONTRADICTORY = "contradictory"
COMPLEMENTARY = "complementary"
NUANCED = "nuanced"

_MAX_PERSONAS = 10  # persona_0 .. persona_9


@dataclasses.dataclass
class QAPair:
    query: str
    correct_answers: list[str]
    incorrect_answers: list[str] = dataclasses.field(default_factory=list)

    @property
    def gold(self) -> str:
        """Canonical gold answer for the LLM-judge QA layer (first correct one)."""
        return self.correct_answers[0] if self.correct_answers else ""

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> QAPair:
        return cls(
            query=str(raw.get("query", "")),
            correct_answers=[str(a) for a in (raw.get("correct_answers") or [])],
            incorrect_answers=[str(a) for a in (raw.get("incorrect_answers") or [])],
        )


@dataclasses.dataclass
class BenchInstance:
    instance_id: str
    case_id: str
    persona_id: str
    relation_type: str
    relation_subtype: str
    session_ids: list[str]
    qas: list[QAPair]
    topic: str = ""
    source: str = ""
    case: str = ""
    facts: list[str] = dataclasses.field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> BenchInstance:
        return cls(
            instance_id=str(raw.get("instance_id", "")),
            case_id=str(raw.get("case_id", "")),
            persona_id=str(raw.get("persona_id", "")),
            relation_type=str(raw.get("relation_type", "")),
            relation_subtype=str(raw.get("relation_subtype", "")),
            session_ids=[str(s) for s in (raw.get("session_ids") or [])],
            qas=[QAPair.from_dict(q) for q in (raw.get("qas") or [])],
            topic=str(raw.get("topic", "")),
            source=str(raw.get("source", "")),
            case=str(raw.get("case", "")),
            facts=[str(f) for f in (raw.get("facts") or [])],
        )


@dataclasses.dataclass
class Turn:
    role: str
    content: str

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> Turn:
        return cls(role=str(raw.get("role", "")), content=str(raw.get("content", "")))


@dataclasses.dataclass
class HistorySession:
    session_id: str
    persona_id: str
    case_id: str
    source: str
    timestamp: str
    history: list[Turn]
    conversation_type: str = ""
    order: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> HistorySession:
        return cls(
            session_id=str(raw.get("session_id", "")),
            persona_id=str(raw.get("persona_id", "")),
            case_id=str(raw.get("case_id", "")),
            source=str(raw.get("source", "")),
            timestamp=str(raw.get("timestamp", "")),
            history=[Turn.from_dict(t) for t in (raw.get("history") or [])],
            conversation_type=str(raw.get("conversation_type", "")),
            order=int(raw.get("order", 0) or 0),
        )


def _persona_corpus(
    sessions: list[HistorySession],
) -> tuple[list[Memory], dict[str, list[str]]]:
    """Build the retrievable corpus + a session_id -> [turn memory ids] index.

    Each turn is one ``Memory`` id'd ``{session_id}_{turn_index}`` (globally
    unique within the persona). The timestamp goes to ``created_at`` (the recency
    signal ``format_memories`` injects) and is also prefixed into the content so
    a bare answerer still sees it; ``session_source`` carries the session id.
    """
    corpus: list[Memory] = []
    by_session: dict[str, list[str]] = {}
    for session in sessions:
        for ti, turn in enumerate(session.history):
            text = turn.content.strip()
            if not text:
                continue
            mid = f"{session.session_id}_{ti}"
            body = f"{turn.role}: {text}" if turn.role else text
            content = f"[{session.timestamp}] {body}" if session.timestamp else body
            corpus.append(
                Memory(
                    id=mid,
                    content=content,
                    type="HISTORY",
                    session_source=session.session_id,
                    created_at=session.timestamp,
                    confidence=0.9,
                )
            )
            by_session.setdefault(session.session_id, []).append(mid)
    return corpus, by_session


_RawJson = dict[str, typing.Any]
_PersonaTriple = tuple[str, list[_RawJson], list[_RawJson]]


def load_subtlememory_data(personas: list[_PersonaTriple]) -> list[Dataset]:
    """Convert ``(persona_id, bench_instances, history_sessions)`` tuples into
    one ``Dataset`` per persona.

    Each QA pair becomes one ``EvalCase``; its gold is every turn of the case's
    target ``session_ids`` (unresolvable session ids are dropped; a case with no
    resolvable gold is dropped — it isn't scoreable). ``intent`` =
    ``relation_type``, ``note`` = ``relation_subtype``.
    """
    datasets: list[Dataset] = []
    for persona_id, raw_bench, raw_history in personas:
        sessions = [HistorySession.from_dict(s) for s in raw_history]
        corpus, by_session = _persona_corpus(sessions)

        cases: list[EvalCase] = []
        for raw_inst in raw_bench:
            inst = BenchInstance.from_dict(raw_inst)
            gold: list[str] = []
            for sid in inst.session_ids:
                gold.extend(by_session.get(sid, []))
            if not gold:  # nothing resolves -> not scoreable
                continue
            for qi, qa in enumerate(inst.qas):
                if not qa.query.strip():
                    continue
                cases.append(
                    EvalCase(
                        id=f"{persona_id}_{inst.instance_id}_{qi}",
                        query=qa.query,
                        relevant_ids=list(gold),
                        intent=inst.relation_type,
                        note=inst.relation_subtype,
                        answer=qa.gold,
                    )
                )
        datasets.append(
            Dataset(
                name=f"subtlememory_persona_{persona_id}",
                corpus=corpus,
                cases=cases,
            )
        )
    return datasets


def load_persona(path: str | pathlib.Path, persona_id: int | str) -> Dataset:
    """Load a single ``persona_{N}/`` dir into one ``Dataset``."""
    pdir = pathlib.Path(path) / f"persona_{persona_id}"
    bench = json.loads((pdir / "bench_instances.json").read_text())
    history = json.loads((pdir / "history_sessions.json").read_text())
    return load_subtlememory_data([(str(persona_id), bench, history)])[0]


def load_subtlememory(
    path: str | pathlib.Path, *, persona_limit: int = 0
) -> list[Dataset]:
    """Load ``persona_{0..N}/`` dirs under ``path`` into per-persona Datasets.

    ``persona_limit > 0`` loads only the first N personas (cheap smoke runs;
    1 persona ~= 100 cases / ~2.5k turns). ``0`` loads all available personas.
    """
    root = pathlib.Path(path)
    limit = persona_limit if persona_limit > 0 else _MAX_PERSONAS
    datasets: list[Dataset] = []
    for pid in range(limit):
        pdir = root / f"persona_{pid}"
        if not (pdir / "bench_instances.json").exists():
            continue
        datasets.append(load_persona(root, pid))
    return datasets


def _acc(rows: list[tuple[str, bool]]) -> float:
    return (sum(1 for _, c in rows if c) / len(rows)) if rows else 0.0


def aggregate_by_relation(rows: list[tuple[str, bool]]) -> dict[str, typing.Any]:
    """Aggregate per-case ``(relation_type, correct)`` into per-slice accuracy.

    Returns overall + ``by_relation`` (one block per relation_type) + a dedicated
    ``contradictory`` block — the headline differentiator (unresolved conflict
    that should be surfaced, not collapsed). The contradictory block inside
    ``by_relation`` is flagged ``is_headline`` so report consumers can highlight
    it. Mirrors ``halumem.aggregate_qa``.
    """

    def block(slice_rows: list[tuple[str, bool]]) -> dict[str, typing.Any]:
        return {"n": len(slice_rows), "accuracy": _acc(slice_rows)}

    by_relation: dict[str, list[tuple[str, bool]]] = {}
    for relation, correct in rows:
        by_relation.setdefault(relation or "?", []).append((relation, correct))

    by_relation_out: dict[str, typing.Any] = {}
    for relation, slice_rows in sorted(by_relation.items()):
        blk = block(slice_rows)
        if relation == CONTRADICTORY:
            blk["is_headline"] = True
        by_relation_out[relation] = blk

    contradictory = [r for r in rows if r[0] == CONTRADICTORY]
    return {
        "overall": block(rows),
        "by_relation": by_relation_out,
        "contradictory": block(contradictory),
    }
