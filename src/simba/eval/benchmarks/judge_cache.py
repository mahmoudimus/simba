"""Persistent judge-verdict cache.

Re-judging an identical (judge_model, question, gold, predicted) tuple is
deterministic enough to cache: it lets a benchmark re-run skip the judge LLM call
whenever the answerer produced the same prediction as before. Append-only-safe
(INSERT OR REPLACE keyed by hash; never deletes).
"""

from __future__ import annotations

import hashlib
import pathlib
import sqlite3

_SEP = "\x00"


class JudgeCache:
    def __init__(self, path: str | pathlib.Path) -> None:
        self._path = pathlib.Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS verdicts "
            "(key TEXT PRIMARY KEY, correct INTEGER NOT NULL)"
        )
        self._conn.commit()

    @staticmethod
    def key(judge_model: str, question: str, gold: str, predicted: str) -> str:
        h = hashlib.sha1()
        h.update(
            f"{judge_model}{_SEP}{question}{_SEP}{gold}{_SEP}{predicted}".encode()
        )
        return h.hexdigest()

    def get(
        self, judge_model: str, question: str, gold: str, predicted: str
    ) -> bool | None:
        row = self._conn.execute(
            "SELECT correct FROM verdicts WHERE key = ?",
            (self.key(judge_model, question, gold, predicted),),
        ).fetchone()
        return bool(row[0]) if row else None

    def put(
        self,
        judge_model: str,
        question: str,
        gold: str,
        predicted: str,
        correct: bool,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO verdicts (key, correct) VALUES (?, ?)",
            (self.key(judge_model, question, gold, predicted), 1 if correct else 0),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
