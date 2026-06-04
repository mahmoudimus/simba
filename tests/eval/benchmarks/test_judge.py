"""Tests for the LLM-judge QA layer: answer generation + grading + aggregation.

Pure/injectable: a fake retriever + fake LLM client exercise the flow with no
LanceDB and no live model.
"""

from __future__ import annotations

import simba.eval.benchmarks.judge as judge
from simba.eval.dataset import EvalCase


class FakeLlm:
    def __init__(self, answer: str = "7 May 2023", verdict: object = None) -> None:
        self.answer = answer
        self.verdict = {"correct": True} if verdict is None else verdict
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer

    def complete_json(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return self.verdict


def test_answer_prompt_includes_question_and_contexts() -> None:
    p = judge.build_answer_prompt("When was the trip?", ["A: went 7 May", "B: hi"])
    assert "When was the trip?" in p
    assert "went 7 May" in p and "B: hi" in p


def test_judge_prompt_includes_gold_predicted_and_asks_json() -> None:
    p = judge.build_judge_prompt("Q?", gold="7 May", predicted="May 7th")
    assert "7 May" in p and "May 7th" in p and "Q?" in p
    assert "json" in p.lower() and "correct" in p.lower()


def test_score_case_retrieves_topk_then_grades() -> None:
    case = EvalCase(id="q1", query="When?", relevant_ids=["c1"], answer="7 May")
    id2content = {"c1": "went 7 May", "c2": "noise", "c3": "more noise"}
    llm = FakeLlm(answer="7 May", verdict={"correct": True})
    retr = lambda q: ["c1", "c2", "c3"]  # noqa: E731
    correct = judge.score_case(case, retr, id2content, llm, k=2)
    assert correct is True
    # only top-2 contexts went into the answer prompt
    assert "went 7 May" in llm.prompts[0] and "more noise" not in llm.prompts[0]


def test_score_case_incorrect_verdict() -> None:
    case = EvalCase(id="q1", query="When?", relevant_ids=["c1"], answer="7 May")
    llm = FakeLlm(answer="never", verdict={"correct": False})
    assert judge.score_case(case, lambda q: ["c1"], {"c1": "x"}, llm, k=5) is False


def test_score_case_none_when_generation_empty() -> None:
    case = EvalCase(id="q1", query="When?", relevant_ids=["c1"], answer="7 May")
    llm = FakeLlm(answer="   ")
    assert judge.score_case(case, lambda q: ["c1"], {"c1": "x"}, llm, k=5) is None


def test_aggregate_overall_and_by_category() -> None:
    rows = [("multi-hop", True), ("multi-hop", False), ("single-hop", True)]
    rep = judge.aggregate(rows)
    assert rep["n_graded"] == 3
    assert abs(rep["overall"]["accuracy"] - 2 / 3) < 1e-9
    assert rep["by_category"]["multi-hop"] == {"n": 2, "accuracy": 0.5}
    assert rep["by_category"]["single-hop"] == {"n": 1, "accuracy": 1.0}


def test_evalcase_answer_round_trips() -> None:
    c = EvalCase(id="q1", query="When?", relevant_ids=["c1"], answer="7 May")
    assert c.answer == "7 May"
    raw = {"id": "q1", "query": "When?", "relevant_ids": ["c1"], "answer": "7 May"}
    assert EvalCase.from_dict(raw).answer == "7 May"
