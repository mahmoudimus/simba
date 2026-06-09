"""Tests for the LLM-judge QA layer: answer generation + grading + aggregation.

Pure/injectable: a fake retriever + fake LLM client exercise the flow with no
LanceDB and no live model.
"""

from __future__ import annotations

import pytest

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


def test_answer_prompt_no_dates_has_no_recency_language() -> None:
    # Backward-compatible: without dates the prompt stays the bare format.
    p = judge.build_answer_prompt("Q?", ["a", "b"])
    assert "most recent" not in p.lower()
    assert "- a" in p and "- b" in p


def test_answer_prompt_recency_labels_and_flags_newest() -> None:
    # Mirrors format_memories: date-label each memory and flag the newest, plus a
    # most-recent-wins instruction. This is what the product injects; the eval was
    # stripping it, understating temporal accuracy.
    p = judge.build_answer_prompt(
        "current income?",
        ["income 18000", "income 22000"],
        dates=["2025-01-01", "2025-06-01"],
    )
    assert "2025-01-01" in p and "2025-06-01" in p
    assert "most recent" in p.lower()
    bullet_lines = [ln for ln in p.splitlines() if ln.startswith("- ")]
    flagged = [ln for ln in bullet_lines if "most recent" in ln.lower()]
    assert len(flagged) == 1 and "22000" in flagged[0]


def test_answer_prompt_parses_halumem_date_format() -> None:
    p = judge.build_answer_prompt("Q?", ["x"], dates=["Sep 04, 2025, 21:12:18"])
    assert "2025-09-04" in p


def test_answer_prompt_mirrors_format_memories_no_resolution_instruction() -> None:
    # Faithful to the daemon's format_memories: annotate (date label + newest
    # flag) but DON'T inject a recency-resolution instruction — the product never
    # does, and an A/B showed it's a no-op for a capable consumer. The eval must
    # measure what ships, not hand the answerer an extra hint.
    p = judge.build_answer_prompt(
        "current income?",
        ["income 18000", "income 22000"],
        dates=["2025-01-01", "2025-06-01"],
    )
    assert "[2025-01-01]" in p and "[2025-06-01]" in p  # date labels kept
    assert "(most recent)" in p  # newest flag kept
    # no resolution instruction (the phrase the A/B showed was a no-op)
    assert "current truth" not in p.lower()


def test_score_case_threads_dates_from_id2date() -> None:
    case = EvalCase(id="q1", query="income?", relevant_ids=["c1"], answer="22000")
    llm = FakeLlm(answer="22000", verdict={"correct": True})
    judge.score_case(
        case,
        lambda q: ["c1", "c2"],
        {"c1": "income 18000", "c2": "income 22000"},
        llm,
        k=2,
        id2date={"c1": "2025-01-01", "c2": "2025-06-01"},
    )
    assert "2025-06-01" in llm.prompts[0] and "most recent" in llm.prompts[0].lower()


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


# --- B1: separate judge from answerer ---------------------------------------


def test_score_case_uses_judge_for_grading_not_answerer() -> None:
    """answerer.complete_json should NOT be called when a separate judge is given."""
    case = EvalCase(id="q1", query="When?", relevant_ids=["c1"], answer="7 May")
    answerer = FakeLlm(answer="7 May", verdict={"correct": True})
    judge_llm = FakeLlm(answer="ignored", verdict={"correct": True})
    correct = judge.score_case(
        case, lambda q: ["c1"], {"c1": "x"}, answerer, judge=judge_llm, k=5
    )
    assert correct is True
    # answerer called once (generate answer), judge called once (grade)
    assert len(answerer.prompts) == 1  # build_answer_prompt only
    assert len(judge_llm.prompts) == 1  # build_judge_prompt only


def test_score_case_judge_none_falls_back_to_answerer() -> None:
    """Legacy path: no judge kwarg -> answerer grades its own answer."""
    case = EvalCase(id="q1", query="When?", relevant_ids=["c1"], answer="7 May")
    llm = FakeLlm(answer="7 May", verdict={"correct": False})
    correct = judge.score_case(case, lambda q: ["c1"], {"c1": "x"}, llm, k=5)
    assert correct is False
    assert len(llm.prompts) == 2  # answer + grade


def test_run_qa_passes_judge_to_score_case() -> None:
    """run_qa with judge kwarg should route grading to the judge client."""
    import unittest.mock

    import simba.eval.benchmarks.judge as jmod
    import simba.memory.config as mc
    from simba.eval.dataset import Dataset, EvalCase, Memory

    called_with: list[dict] = []

    def patched_score(case, retriever, id2content, answerer, *, judge=None, **kw):
        called_with.append({"answerer": answerer, "judge": judge})
        return True

    with unittest.mock.patch.object(jmod, "score_case", patched_score):
        ds = Dataset(
            name="t",
            corpus=[Memory(id="c1", content="x")],
            cases=[EvalCase(id="q1", query="q", relevant_ids=["c1"], answer="a")],
        )
        answerer = FakeLlm(answer="a")
        judge_llm = FakeLlm(answer="x")
        cfg = mc.MemoryConfig(
            llm_rerank_enabled=False,
            scoring_enabled=False,
            expansion_enabled=False,
        )
        embed = lambda t: [0.0] * 384  # noqa: E731
        jmod.run_qa(
            [ds],
            embed_doc=embed,
            embed_query=embed,
            cfg=cfg,
            llm=answerer,
            judge=judge_llm,
        )
    assert called_with[0]["answerer"] is answerer
    assert called_with[0]["judge"] is judge_llm


# --- B3: abstention scoring --------------------------------------------------


def test_score_abstention_heuristic_match_returns_true_without_judge() -> None:
    case = EvalCase(
        id="q1_abs",
        query="When did I buy a boat?",
        relevant_ids=["c1"],
        answer="no information available",
    )
    answerer = FakeLlm(answer="I don't know, no information available.")
    judge_llm = FakeLlm(answer="x", verdict={"abstained": False})  # NOT called
    result = judge.score_abstention(
        case,
        lambda q: ["c1"],
        {"c1": "x"},
        answerer,
        judge=judge_llm,
        k=5,
        abstention_phrases=["don't know", "no information"],
    )
    assert result is True
    assert len(judge_llm.prompts) == 0  # heuristic short-circuited


def test_score_abstention_no_phrase_match_calls_judge() -> None:
    case = EvalCase(id="q1_abs", query="Q?", relevant_ids=["c1"], answer="n/a")
    answerer = FakeLlm(answer="The answer is 42.")
    judge_llm = FakeLlm(answer="x", verdict={"abstained": False})
    result = judge.score_abstention(
        case,
        lambda q: ["c1"],
        {"c1": "x"},
        answerer,
        judge=judge_llm,
        k=5,
        abstention_phrases=["don't know"],
    )
    assert result is False
    assert len(judge_llm.prompts) == 1


def test_score_abstention_returns_none_on_empty_answer() -> None:
    case = EvalCase(id="q1_abs", query="Q?", relevant_ids=["c1"], answer="n/a")
    answerer = FakeLlm(answer="   ")
    result = judge.score_abstention(
        case,
        lambda q: ["c1"],
        {"c1": "x"},
        answerer,
        k=5,
        abstention_phrases=["don't know"],
    )
    assert result is None


def test_aggregate_with_abstention_includes_abstention_block() -> None:
    rows = [("single-hop", True), ("multi-hop", False)]
    abs_rows = [("temporal", True), ("temporal", False)]
    rep = judge.aggregate_with_abstention(rows, abs_rows)
    assert rep["n_graded"] == 2
    assert rep["abstention"]["n"] == 2
    assert rep["abstention"]["accuracy"] == pytest.approx(0.5)


def test_run_qa_abstention_cases_scored_separately() -> None:
    """run_qa with include_abstention=True scores _abs cases via score_abstention."""
    import unittest.mock

    import simba.eval.benchmarks.judge as jmod
    import simba.memory.config as mc
    from simba.eval.dataset import Dataset, Memory

    normal_case = EvalCase(id="q1", query="q", relevant_ids=["c1"], answer="a")
    abs_case = EvalCase(id="q2_abs", query="q_abs", relevant_ids=["c1"], answer="n/a")
    ds = Dataset(
        name="t",
        corpus=[Memory(id="c1", content="x")],
        cases=[normal_case, abs_case],
    )
    answerer = FakeLlm(answer="a")
    cfg = mc.MemoryConfig(
        llm_rerank_enabled=False, scoring_enabled=False, expansion_enabled=False
    )
    embed = lambda t: [0.0] * 384  # noqa: E731
    with (
        unittest.mock.patch.object(jmod, "score_case", return_value=True) as m_sc,
        unittest.mock.patch.object(jmod, "score_abstention", return_value=True) as m_sa,
    ):
        jmod.run_qa(
            [ds],
            embed_doc=embed,
            embed_query=embed,
            cfg=cfg,
            llm=answerer,
            include_abstention=True,
            abstention_phrases=["don't know"],
        )
    assert m_sc.call_count == 1  # only normal case
    assert m_sa.call_count == 1  # only abs case


# --- B4: per-query latency in run_qa report ---------------------------------


def test_run_qa_report_has_latency_block() -> None:
    """run_qa report dict must include latency.p50_ms and latency.p95_ms."""
    import simba.eval.benchmarks.judge as jmod
    import simba.memory.config as mc
    from simba.eval.dataset import Dataset, EvalCase, Memory

    ds = Dataset(
        name="t",
        corpus=[Memory(id="c1", content="x")],
        cases=[EvalCase(id="q1", query="q", relevant_ids=["c1"], answer="a")],
    )
    cfg = mc.MemoryConfig(
        llm_rerank_enabled=False, scoring_enabled=False, expansion_enabled=False
    )
    embed = lambda t: [0.0] * 384  # noqa: E731
    llm = FakeLlm(answer="a", verdict={"correct": True})
    report = jmod.run_qa([ds], embed_doc=embed, embed_query=embed, cfg=cfg, llm=llm)
    assert "latency" in report
    assert "p50_ms" in report["latency"]
    assert "p95_ms" in report["latency"]
    assert report["latency"]["n"] == 1


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
