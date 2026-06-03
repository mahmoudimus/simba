"""Tests for the per-memory extraction strategy (regex | llm | llm+regex)."""

from __future__ import annotations

import simba.sync.extractor as ex


class FakeClient:
    def __init__(self, triples_json, available=True):
        self._json = triples_json
        self._available = available

    def available(self):
        return self._available

    def complete_json(self, prompt):
        return self._json


# regex matches "use X for Y" -> (X, solves, Y)
_REGEX_CONTENT = "use ruff for linting"


def _llm(*triples):
    return FakeClient(
        [
            {"subject": s, "predicate": p, "object": o}
            for (s, p, o) in triples
        ]
    )


def _spo(triples):
    return {(s, p, o) for (s, p, o, _proof) in triples}


def test_regex_strategy_ignores_llm() -> None:
    out = ex._extract_for_memory(
        "WORKING_SOLUTION", _REGEX_CONTENT, "", "m1",
        strategy="regex", llm_client=_llm(("x", "y", "z")), llm_vocab=[], max_triples=8,
    )
    assert _spo(out) == {("ruff", "solves", "linting")}  # no llm triple


def test_llm_strategy_skips_regex() -> None:
    out = ex._extract_for_memory(
        "WORKING_SOLUTION", _REGEX_CONTENT, "", "m1",
        strategy="llm", llm_client=_llm(("gh", "causes", "401")), llm_vocab=[],
        max_triples=8,
    )
    assert _spo(out) == {("gh", "causes", "401")}  # regex not run


def test_llm_plus_regex_unions_both() -> None:
    out = ex._extract_for_memory(
        "WORKING_SOLUTION", _REGEX_CONTENT, "", "m1",
        strategy="llm+regex", llm_client=_llm(("gh", "causes", "401")), llm_vocab=[],
        max_triples=8,
    )
    assert _spo(out) == {("ruff", "solves", "linting"), ("gh", "causes", "401")}


def test_llm_strategy_falls_back_to_regex_when_no_provider() -> None:
    out = ex._extract_for_memory(
        "WORKING_SOLUTION", _REGEX_CONTENT, "", "m1",
        strategy="llm", llm_client=None, llm_vocab=[], max_triples=8,
    )
    assert _spo(out) == {("ruff", "solves", "linting")}


def test_union_dedups_identical_triples() -> None:
    out = ex._extract_for_memory(
        "WORKING_SOLUTION", _REGEX_CONTENT, "", "m1",
        strategy="llm+regex",
        llm_client=_llm(("ruff", "solves", "linting")),  # same as regex
        llm_vocab=[], max_triples=8,
    )
    assert len(out) == 1
