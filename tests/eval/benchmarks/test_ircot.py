"""Tests for IRCoT (interleaved retrieve-and-reason) answer-time QA.

Pure/injectable: a fake retriever (sub-query -> id list) + a fake LLM client
exercise the whole loop with no LanceDB and no live model.
"""

from __future__ import annotations

import typing

import simba.eval.benchmarks.ircot as ircot
from simba.eval.dataset import EvalCase

Retriever = typing.Callable[[str], list[str]]


class FakeLlm:
    """Tracks calls; returns configured step JSON then a final answer + verdict."""

    def __init__(
        self,
        step_responses: list[dict | None],
        final_answer: str = "The answer",
        verdict: dict | None = None,
    ) -> None:
        self.step_responses = list(step_responses)
        self.final_answer = final_answer
        self.verdict = verdict or {"correct": True}
        self.complete_calls: list[str] = []
        self.complete_json_calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.complete_calls.append(prompt)
        return self.final_answer

    def complete_json(self, prompt: str) -> object:
        self.complete_json_calls.append(prompt)
        if self.step_responses:
            return self.step_responses.pop(0)
        return self.verdict  # judge call


def _make_retriever(mapping: dict[str, list[str]]) -> Retriever:
    return lambda q: mapping.get(q, [])


def test_build_step_prompt_includes_question_and_evidence() -> None:
    prompt = ircot.build_step_prompt("Who?", ["Alice was there"], step=1)
    assert "Who?" in prompt
    assert "Alice was there" in prompt
    assert "sub_query" in prompt.lower() or "follow" in prompt.lower()


def test_build_step_prompt_step_zero_with_no_evidence() -> None:
    prompt = ircot.build_step_prompt("Where?", [], step=0)
    assert "Where?" in prompt
    assert "sub_query" in prompt.lower()


def test_ircot_answer_single_step_retrieves_and_answers() -> None:
    step_resp = [{"reasoning": "Alice was there", "sub_query": "where was Alice"}]
    llm = FakeLlm(step_resp, final_answer="Paris")
    mapping = {"where was Alice": ["m1"]}
    id2content = {"m1": "Alice was in Paris"}
    result = ircot.ircot_answer(
        "Where was Alice?", _make_retriever(mapping), id2content, llm, max_steps=1
    )
    assert result == "Paris"
    assert len(llm.complete_json_calls) == 1  # one step prompt
    assert len(llm.complete_calls) == 1  # one final prompt
    assert "Alice was in Paris" in llm.complete_calls[0]  # evidence fed to final


def test_ircot_answer_accumulates_evidence_across_steps() -> None:
    step_resp = [
        {"reasoning": "Need Alice", "sub_query": "Alice location"},
        {"reasoning": "Need Bob", "sub_query": "Bob location"},
    ]
    llm = FakeLlm(step_resp, final_answer="Both in Paris")
    mapping = {"Alice location": ["m1"], "Bob location": ["m2"]}
    id2content = {"m1": "Alice: Paris", "m2": "Bob: Lyon"}
    ircot.ircot_answer("Where?", _make_retriever(mapping), id2content, llm, max_steps=3)
    final_prompt = llm.complete_calls[0]
    assert "Alice: Paris" in final_prompt
    assert "Bob: Lyon" in final_prompt


def test_ircot_answer_stops_early_when_step_json_is_none() -> None:
    step_resp: list[dict | None] = [None]  # LLM returns None for step JSON
    llm = FakeLlm(step_resp, final_answer="fallback")
    result = ircot.ircot_answer("Q?", _make_retriever({}), {}, llm, max_steps=4)
    assert result == "fallback"
    assert len(llm.complete_json_calls) == 1  # only one step attempt


def test_ircot_answer_stops_early_when_sub_query_empty() -> None:
    step_resp = [{"reasoning": "hmm", "sub_query": ""}]
    llm = FakeLlm(step_resp, final_answer="done")
    ircot.ircot_answer("Q?", _make_retriever({}), {}, llm, max_steps=4)
    assert len(llm.complete_json_calls) == 1


def test_ircot_answer_deduplicates_evidence_ids() -> None:
    step_resp = [
        {"reasoning": "r1", "sub_query": "q1"},
        {"reasoning": "r2", "sub_query": "q2"},
    ]
    mapping = {"q1": ["m1", "m2"], "q2": ["m1", "m3"]}
    id2content = {"m1": "A", "m2": "B", "m3": "C"}
    llm = FakeLlm(step_resp, final_answer="X")
    ircot.ircot_answer("Q?", _make_retriever(mapping), id2content, llm, max_steps=3)
    final_prompt = llm.complete_calls[0]
    # m1 ("A") fed once despite being retrieved by both sub-queries. Count the
    # evidence bullet, not bare "A" (the answer-prompt template contains "Answer").
    assert final_prompt.count("- A") == 1  # m1 not duplicated


def test_ircot_answer_respects_k_final_cap() -> None:
    step_resp = [{"reasoning": f"r{i}", "sub_query": f"q{i}"} for i in range(6)]
    mapping = {f"q{i}": [f"m{i}"] for i in range(6)}
    id2content = {f"m{i}": f"content_{i}" for i in range(6)}
    llm = FakeLlm(step_resp, final_answer="Y")
    ircot.ircot_answer(
        "Q?",
        _make_retriever(mapping),
        id2content,
        llm,
        max_steps=6,
        k_per_step=1,
        k_final=3,
    )
    final_prompt = llm.complete_calls[0]
    assert final_prompt.count("content_") <= 3


def test_ircot_answer_fail_open_on_exception() -> None:
    class BoomLlm:
        def complete_json(self, p: str) -> object:
            raise RuntimeError("boom")

        def complete(self, p: str) -> str:
            raise RuntimeError("boom")

    result = ircot.ircot_answer("Q?", lambda q: [], {}, BoomLlm(), max_steps=2)
    assert result == ""


def test_score_case_ircot_returns_true_when_correct() -> None:
    case = EvalCase(id="q1", query="Who?", relevant_ids=["m1"], answer="Alice")
    llm = FakeLlm(
        [{"reasoning": "r", "sub_query": "who"}],
        final_answer="Alice",
        verdict={"correct": True},
    )
    mapping = {"who": ["m1"]}
    id2content = {"m1": "Alice"}
    result = ircot.score_case_ircot(case, _make_retriever(mapping), id2content, llm)
    assert result is True


def test_score_case_ircot_returns_false_when_incorrect() -> None:
    case = EvalCase(id="q1", query="Who?", relevant_ids=["m1"], answer="Alice")
    llm = FakeLlm(
        [{"reasoning": "r", "sub_query": "who"}],
        final_answer="Bob",
        verdict={"correct": False},
    )
    mapping = {"who": ["m1"]}
    id2content = {"m1": "Alice"}
    result = ircot.score_case_ircot(case, _make_retriever(mapping), id2content, llm)
    assert result is False


def test_score_case_ircot_returns_none_when_final_answer_empty() -> None:
    case = EvalCase(id="q1", query="Who?", relevant_ids=["m1"], answer="Alice")
    llm = FakeLlm(
        [{"reasoning": "r", "sub_query": "who"}],
        final_answer="   ",
        verdict={"correct": True},
    )
    mapping = {"who": ["m1"]}
    id2content = {"m1": "Alice"}
    result = ircot.score_case_ircot(case, _make_retriever(mapping), id2content, llm)
    assert result is None
