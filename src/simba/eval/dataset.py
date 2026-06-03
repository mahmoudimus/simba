"""Eval dataset model + loader.

A dataset is a single JSON file::

    {
      "name": "...",
      "corpus": [{"id", "content", "type"?, "context"?, ...}, ...],
      "cases":  [{"id", "query", "relevant_ids": [...], "intent"?, "note"?}, ...]
    }

``relevant_ids`` reference corpus ids; loading validates that they exist and
that corpus ids are unique, so a typo fails loudly instead of silently scoring
zero recall.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any


@dataclasses.dataclass
class Memory:
    id: str
    content: str
    type: str = "PATTERN"
    context: str = ""
    project_path: str = ""
    session_source: str = ""
    created_at: str = ""
    confidence: float = 0.85

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Memory:
        return cls(
            id=raw["id"],
            content=raw["content"],
            type=raw.get("type") or "PATTERN",
            context=raw.get("context", ""),
            project_path=raw.get("project_path", ""),
            session_source=raw.get("session_source", ""),
            created_at=raw.get("created_at", ""),
            confidence=float(raw.get("confidence", 0.85)),
        )


@dataclasses.dataclass
class EvalCase:
    id: str
    query: str
    relevant_ids: list[str]
    intent: str = ""
    note: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalCase:
        return cls(
            id=raw["id"],
            query=raw["query"],
            relevant_ids=list(raw.get("relevant_ids", [])),
            intent=raw.get("intent", ""),
            note=raw.get("note", ""),
        )


@dataclasses.dataclass
class Dataset:
    name: str
    corpus: list[Memory]
    cases: list[EvalCase]

    def corpus_ids(self) -> set[str]:
        return {m.id for m in self.corpus}


def load_dataset(path: str | pathlib.Path) -> Dataset:
    """Load + validate a dataset JSON file."""
    raw = json.loads(pathlib.Path(path).read_text())
    corpus = [Memory.from_dict(m) for m in raw.get("corpus", [])]

    seen: set[str] = set()
    for mem in corpus:
        if mem.id in seen:
            raise ValueError(f"duplicate corpus id: {mem.id}")
        seen.add(mem.id)

    cases = [EvalCase.from_dict(c) for c in raw.get("cases", [])]
    for case in cases:
        for rid in case.relevant_ids:
            if rid not in seen:
                raise ValueError(
                    f"case {case.id!r} references unknown memory id: {rid!r}"
                )

    return Dataset(name=raw.get("name", "unnamed"), corpus=corpus, cases=cases)
