"""Tiny eval fixture for UserPromptSubmit retrieval triage."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import simba.hooks.recall_triage

if TYPE_CHECKING:
    import pathlib


@dataclasses.dataclass(frozen=True)
class TriageCase:
    prompt: str
    expected: str
    category: str = "general"


DEFAULT_CASES: tuple[TriageCase, ...] = (
    TriageCase("thanks!", "skip", "ack"),
    TriageCase("what is the current time?", "skip", "self_contained"),
    TriageCase("rewrite this sentence: the build failed", "skip", "text_task"),
    TriageCase("what is next from the borrow roadmap?", "retrieve", "memory"),
    TriageCase("continue the implementation", "retrieve", "memory"),
    TriageCase("run the focused hook tests", "retrieve", "repo_action"),
    TriageCase("why did the daemon restart fail earlier?", "retrieve", "memory"),
    TriageCase("do it", "retrieve", "ambiguous_action"),
)


def load_cases(path: pathlib.Path | None = None) -> list[TriageCase]:
    if path is None:
        return list(DEFAULT_CASES)
    cases: list[TriageCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(
            TriageCase(
                prompt=str(row["prompt"]),
                expected=str(row["expected"]),
                category=str(row.get("category", "general")),
            )
        )
    return cases


def evaluate(cases: list[TriageCase]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    false_negatives = 0
    false_positives = 0
    correct = 0
    for case in cases:
        triage = simba.hooks.recall_triage.classify(case.prompt)
        actual = "retrieve" if triage.should_retrieve else "skip"
        ok = actual == case.expected
        correct += int(ok)
        if case.expected == "retrieve" and actual == "skip":
            false_negatives += 1
        if case.expected == "skip" and actual == "retrieve":
            false_positives += 1
        rows.append(
            {
                "prompt": case.prompt,
                "category": case.category,
                "expected": case.expected,
                "actual": actual,
                "decision": triage.decision,
                "reason": triage.reason,
                "ok": ok,
            }
        )
    n = len(cases)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "gate": "pass" if false_negatives == 0 else "fail",
        "cases": rows,
    }
