"""LLM-judge QA layer for the benchmarks.

The recall@k harness measures whether the gold *evidence* is retrieved. This
layer measures end-to-end **answer accuracy** (the metric Mem0/Zep/LongMemEval
publish): retrieve top-k context -> generate an answer with the LLM -> grade the
answer against the gold with the LLM (binary correct/incorrect).

Kept pure/injectable: ``score_case`` takes a retriever callable + an LLM client,
so the grading flow is unit-tested with fakes (no LanceDB, no live model).
``run_qa`` wires the real recall_adapter retriever and aggregates.
"""

from __future__ import annotations

import tempfile
import typing

import simba.eval.recall_adapter

if typing.TYPE_CHECKING:
    from simba.eval.dataset import Dataset, EvalCase

EmbedFn = typing.Callable[[str], list[float]]
Retriever = typing.Callable[[str], list[str]]


def build_answer_prompt(question: str, contexts: list[str]) -> str:
    """Prompt the model to answer strictly from the retrieved context."""
    joined = "\n".join(f"- {c}" for c in contexts) if contexts else "(no context)"
    return (
        "You are answering a question using ONLY the conversation memories below. "
        "If the answer is not present, say you don't know. Answer concisely.\n\n"
        f"Memories:\n{joined}\n\nQuestion: {question}\nAnswer:"
    )


def build_judge_prompt(question: str, gold: str, predicted: str) -> str:
    """Prompt the model to grade a predicted answer against the gold answer."""
    return (
        "You are grading an answer. Given the question, the gold answer, and a "
        "predicted answer, decide if the prediction is correct (same meaning as "
        "the gold; wording/format may differ). Reply with JSON only: "
        '{"correct": true} or {"correct": false}.\n\n'
        f"Question: {question}\nGold answer: {gold}\n"
        f"Predicted answer: {predicted}\nJSON:"
    )


def score_case(
    case: EvalCase,
    retriever: Retriever,
    id2content: dict[str, str],
    llm: typing.Any,
    *,
    k: int = 10,
) -> bool | None:
    """Retrieve top-k, generate an answer, grade it. None = couldn't grade."""
    ids = retriever(case.query)[:k]
    contexts = [id2content[i] for i in ids if i in id2content]
    predicted = llm.complete(build_answer_prompt(case.query, contexts))
    if not predicted or not predicted.strip():
        return None
    verdict = llm.complete_json(build_judge_prompt(case.query, case.answer, predicted))
    if not isinstance(verdict, dict) or "correct" not in verdict:
        return None
    return bool(verdict["correct"])


def aggregate(rows: list[tuple[str, bool]]) -> dict[str, typing.Any]:
    """Aggregate (intent, correct) pairs into overall + by-category accuracy."""
    by_cat: dict[str, list[bool]] = {}
    for intent, correct in rows:
        by_cat.setdefault(intent or "?", []).append(correct)
    all_correct = [c for _, c in rows]

    def _acc(xs: list[bool]) -> float:
        return sum(1 for x in xs if x) / len(xs) if xs else 0.0

    return {
        "n_graded": len(rows),
        "overall": {"accuracy": _acc(all_correct)},
        "by_category": {
            cat: {"n": len(xs), "accuracy": _acc(xs)}
            for cat, xs in sorted(by_cat.items())
        },
    }


def run_qa(
    datasets: list[Dataset],
    *,
    embed_doc: EmbedFn,
    embed_query: EmbedFn,
    cfg: typing.Any,
    llm: typing.Any,
    k: int = 10,
    answerable_only: bool = True,
) -> dict[str, typing.Any]:
    """Run the full retrieve -> answer -> grade loop over datasets, aggregate."""
    rows: list[tuple[str, bool]] = []
    skipped = 0
    for dset in datasets:
        id2content = {m.id: m.content for m in dset.corpus}
        with tempfile.TemporaryDirectory(prefix="simba-qa-") as td:
            retriever = simba.eval.recall_adapter.build_retriever(
                dset, cfg, embed_doc=embed_doc, embed_query=embed_query,
                data_dir=td, llm_client=None,
            )
            for case in dset.cases:
                if answerable_only and not case.answer.strip():
                    skipped += 1
                    continue
                correct = score_case(case, retriever, id2content, llm, k=k)
                if correct is None:
                    skipped += 1
                    continue
                rows.append((case.intent or "?", correct))
    report = aggregate(rows)
    report["n_skipped"] = skipped
    return report
