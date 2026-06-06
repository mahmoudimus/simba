"""Tests for the LLM-judge QA layer: answer generation + grading + aggregation.

Pure/injectable: a fake retriever + fake LLM client exercise the flow with no
LanceDB and no live model.
"""

from __future__ import annotations

import simba.eval.benchmarks.judge as judge
from simba.eval.dataset import Dataset, EvalCase, Memory


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


def _ds(name: str, cases: list[EvalCase]) -> Dataset:
    corpus = [Memory(id=f"{name}-c", content="x")]
    return Dataset(name=name, corpus=corpus, cases=cases)


def _case(cid: str, intent: str, answer: str = "a") -> EvalCase:
    return EvalCase(id=cid, query="q", relevant_ids=["x"], intent=intent, answer=answer)


def test_sample_cases_first_n_skips_unanswerable() -> None:
    d = _ds("d1", [_case("a", "hop"), _case("b", "hop", answer=""), _case("c", "hop")])
    out = judge.sample_cases([d], n=2)
    assert [c.id for c in out[0].cases] == ["a", "c"]  # "b" (no answer) skipped


def test_sample_cases_stratifies_across_categories_and_datasets() -> None:
    d1 = _ds("d1", [_case("a", "single"), _case("b", "single"), _case("c", "multi")])
    d2 = _ds("d2", [_case("d", "multi"), _case("e", "single")])
    out = judge.sample_cases([d1, d2], per_category=1)
    picked = {c.intent: c.id for ds in out for c in ds.cases}
    # exactly one per category, datasets without picks are dropped
    assert sorted(picked) == ["multi", "single"]
    assert sum(len(ds.cases) for ds in out) == 2


def test_evalcase_answer_round_trips() -> None:
    c = EvalCase(id="q1", query="When?", relevant_ids=["c1"], answer="7 May")
    assert c.answer == "7 May"
    raw = {"id": "q1", "query": "When?", "relevant_ids": ["c1"], "answer": "7 May"}
    assert EvalCase.from_dict(raw).answer == "7 May"


# ── IRCoT routing in run_qa ────────────────────────────────────────────────


def _qa_ds() -> Dataset:
    corpus = [Memory(id="m1", content="x"), Memory(id="m2", content="y")]
    cases = [
        EvalCase(
            id="mh",
            query="multi q",
            relevant_ids=["m1"],
            intent="multi-hop",
            answer="A",
        ),
        EvalCase(
            id="sh",
            query="single q",
            relevant_ids=["m2"],
            intent="single-hop",
            answer="B",
        ),
    ]
    return Dataset(name="qa-routing", corpus=corpus, cases=cases)


def _patch_retriever(monkeypatch) -> None:
    # Avoid building a real LanceDB store: the routing test only cares which
    # scorer each case is sent to, not what the retriever returns.
    monkeypatch.setattr(
        "simba.eval.recall_adapter.build_retriever",
        lambda *a, **k: lambda q: [],
    )


def test_run_qa_uses_ircot_for_multi_hop_when_enabled(monkeypatch) -> None:
    import simba.eval.benchmarks.ircot as ircot
    import simba.eval.config as ec

    _patch_retriever(monkeypatch)
    ircot_cases: list[str] = []
    std_cases: list[str] = []

    def fake_ircot(case, *a, **k) -> bool:
        ircot_cases.append(case.id)
        return True

    def fake_std(case, *a, **k) -> bool:
        std_cases.append(case.id)
        return True

    monkeypatch.setattr(ircot, "score_case_ircot", fake_ircot)
    monkeypatch.setattr(judge, "score_case", fake_std)

    eval_cfg = ec.EvalConfig(
        ircot_enabled=True, ircot_max_steps=1, ircot_k_per_step=1, ircot_k_final=2
    )
    judge.run_qa(
        [_qa_ds()],
        embed_doc=lambda t: [0.0],
        embed_query=lambda t: [0.0],
        cfg=None,
        llm=FakeLlm(),
        eval_cfg=eval_cfg,
    )
    assert ircot_cases == ["mh"]  # multi-hop went to IRCoT
    assert std_cases == ["sh"]  # single-hop went to standard score_case


def test_run_qa_uses_standard_path_when_ircot_disabled(monkeypatch) -> None:
    import simba.eval.benchmarks.ircot as ircot
    import simba.eval.config as ec

    _patch_retriever(monkeypatch)
    ircot_calls = 0
    std_cases: list[str] = []

    def fake_ircot(case, *a, **k) -> bool:
        nonlocal ircot_calls
        ircot_calls += 1
        return True

    monkeypatch.setattr(ircot, "score_case_ircot", fake_ircot)
    monkeypatch.setattr(
        judge, "score_case", lambda case, *a, **k: std_cases.append(case.id) or True
    )

    judge.run_qa(
        [_qa_ds()],
        embed_doc=lambda t: [0.0],
        embed_query=lambda t: [0.0],
        cfg=None,
        llm=FakeLlm(),
        eval_cfg=ec.EvalConfig(ircot_enabled=False),
    )
    assert ircot_calls == 0
    assert sorted(std_cases) == ["mh", "sh"]


def test_run_qa_ircot_none_eval_cfg_uses_standard_path(monkeypatch) -> None:
    import simba.eval.benchmarks.ircot as ircot

    _patch_retriever(monkeypatch)
    ircot_calls = 0
    std_cases: list[str] = []

    def fake_ircot(case, *a, **k) -> bool:
        nonlocal ircot_calls
        ircot_calls += 1
        return True

    monkeypatch.setattr(ircot, "score_case_ircot", fake_ircot)
    monkeypatch.setattr(
        judge, "score_case", lambda case, *a, **k: std_cases.append(case.id) or True
    )

    judge.run_qa(
        [_qa_ds()],
        embed_doc=lambda t: [0.0],
        embed_query=lambda t: [0.0],
        cfg=None,
        llm=FakeLlm(),
        eval_cfg=None,
    )
    assert ircot_calls == 0
    assert sorted(std_cases) == ["mh", "sh"]
