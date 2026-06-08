"""HaluMem QA evaluator — mocked retriever + judge (no real LLM/LanceDB)."""

from __future__ import annotations

import simba.eval.benchmarks.halumem as hm
import simba.eval.benchmarks.halumem_qa as qa
import simba.memory.config as mc


class _Answerer:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    def complete(self, prompt: str) -> str:
        return self._answer


class _Judge:
    """Returns preset outcomes in order from complete_json."""

    def __init__(self, outcomes: list[str]) -> None:
        self._outcomes = list(outcomes)

    def complete_json(self, prompt: str) -> dict:
        return {"outcome": self._outcomes.pop(0)}


def _user() -> hm.HaluUser:
    return hm.HaluUser(
        uuid="u1",
        persona="Name: Martin; Gender: M",
        sessions=[
            hm.HaluSession(
                memory_points=[hm.MemoryPoint(0, "Martin lives in Berlin")],
                questions=[
                    hm.HaluQuestion(
                        "Where does Martin live?",
                        "Berlin",
                        ["Martin lives in Berlin"],
                        "Basic Fact Recall",
                    ),
                    hm.HaluQuestion(
                        "What is Martin's middle name?",
                        "Unknown; not provided.",
                        [],
                        "Memory Boundary",
                    ),
                ],
            )
        ],
    )


def test_user_corpus_ids_unique_across_sessions():
    # HaluMem numbers memory_points per-session (index resets to 1 each session),
    # so f"{uuid}_mp_{index}" collides across sessions — masking earlier gold
    # content in id2content and breaking id->content resolution. Ids must be
    # globally unique within a user so every memory_point is retrievable.
    user = hm.HaluUser(
        uuid="u1",
        persona="",
        sessions=[
            hm.HaluSession(
                memory_points=[hm.MemoryPoint(3, "Martin's birth date is 1996-08-02")],
                questions=[],
            ),
            hm.HaluSession(
                memory_points=[hm.MemoryPoint(3, "Martin expressed gratitude")],
                questions=[],
            ),
        ],
    )
    corpus = qa._user_corpus(user)
    ids = [m.id for m in corpus]
    assert len(ids) == 2
    assert len(set(ids)) == 2, f"colliding ids: {ids}"
    # both distinct contents survive the id->content mapping
    id2content = {m.id: m.content for m in corpus}
    assert "1996-08-02" in " ".join(id2content.values())
    assert "gratitude" in " ".join(id2content.values())


def test_build_judge_prompt_boundary_vs_normal():
    b = qa.build_halumem_judge_prompt("q", "Unknown", "Berlin", is_boundary=True)
    n = qa.build_halumem_judge_prompt("q", "Berlin", "Berlin", is_boundary=False)
    assert "cannot be answered" in b.lower() and "omission" not in b.lower()
    assert "omission" in n.lower() and "gold" in n.lower()


def test_judge_outcome_failopen():
    class _Bad:
        def complete_json(self, prompt):
            return "not a dict"

    assert qa.judge_outcome("q", "g", "p", False, _Bad()) is None


def test_run_halumem_qa(monkeypatch):
    # Mock the retriever so no LanceDB/embedder is needed; it surfaces the gold mp.
    monkeypatch.setattr(
        "simba.eval.recall_adapter.build_retriever",
        lambda *a, **k: lambda q: ["u1_s0_mp_0"],
    )
    answerer = _Answerer("Berlin")
    # fact answered correctly; boundary question fabricated -> hallucination
    judge = _Judge([hm.QA_CORRECT, hm.QA_HALLUCINATION])
    rep = qa.run_halumem_qa(
        [_user()],
        embed_doc=lambda t: [0.0],
        embed_query=lambda t: [0.0],
        cfg=mc.MemoryConfig(),
        llm=answerer,
        judge=judge,
        k=5,
    )
    assert rep["n_graded"] == 2 and rep["n_skipped"] == 0
    assert rep["overall"]["accuracy"] == 0.5
    # the Memory-Boundary question was answered (fabricated) -> hallucination
    assert rep["boundary"]["n"] == 1
    assert rep["boundary"]["hallucination_rate"] == 1.0


def test_run_skips_empty_prediction(monkeypatch):
    monkeypatch.setattr(
        "simba.eval.recall_adapter.build_retriever",
        lambda *a, **k: lambda q: ["u1_s0_mp_0"],
    )
    rep = qa.run_halumem_qa(
        [_user()],
        embed_doc=lambda t: [0.0],
        embed_query=lambda t: [0.0],
        cfg=mc.MemoryConfig(),
        llm=_Answerer("   "),  # blank predictions -> skipped, not graded
        judge=_Judge([]),
        k=5,
    )
    assert rep["n_graded"] == 0 and rep["n_skipped"] == 2
