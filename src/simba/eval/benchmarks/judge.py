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

import dataclasses
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
    cache: typing.Any = None,
    judge_model: str = "",
) -> bool | None:
    """Retrieve top-k, generate an answer, grade it. None = couldn't grade.

    When ``cache`` is a ``JudgeCache``, an identical (judge_model, question, gold,
    predicted) verdict is served from disk instead of re-calling the judge LLM.
    """
    ids = retriever(case.query)[:k]
    contexts = [id2content[i] for i in ids if i in id2content]
    predicted = llm.complete(build_answer_prompt(case.query, contexts))
    if not predicted or not predicted.strip():
        return None
    if cache is not None:
        hit = cache.get(judge_model, case.query, case.answer, predicted)
        if hit is not None:
            return hit
    verdict = llm.complete_json(build_judge_prompt(case.query, case.answer, predicted))
    if not isinstance(verdict, dict) or "correct" not in verdict:
        return None
    correct = bool(verdict["correct"])
    if cache is not None:
        cache.put(judge_model, case.query, case.answer, predicted, correct)
    return correct


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


def sample_cases(
    datasets: list[Dataset],
    *,
    n: int | None = None,
    per_category: int | None = None,
) -> list[Dataset]:
    """Select answerable cases, returning datasets with only the picked cases.

    ``per_category``: balanced — up to that many cases per intent, pooled across
    all datasets (representative). ``n``: the first n answerable cases in order.
    Datasets with no picks are dropped; each kept dataset keeps its full corpus.
    """
    selected: set[str] = set()
    if per_category is not None:
        counts: dict[str, int] = {}
        for dset in datasets:
            for c in dset.cases:
                if not c.answer.strip():
                    continue
                intent = c.intent or "?"
                if counts.get(intent, 0) < per_category:
                    counts[intent] = counts.get(intent, 0) + 1
                    selected.add(c.id)
    else:
        limit = n if n is not None else 0
        count = 0
        for dset in datasets:
            for c in dset.cases:
                if count >= limit:
                    break
                if c.answer.strip():
                    selected.add(c.id)
                    count += 1
            if count >= limit:
                break

    out: list[Dataset] = []
    for dset in datasets:
        kept = [c for c in dset.cases if c.id in selected]
        if kept:
            out.append(dataclasses.replace(dset, cases=kept))
    return out


def run_qa(
    datasets: list[Dataset],
    *,
    embed_doc: EmbedFn,
    embed_query: EmbedFn,
    cfg: typing.Any,
    llm: typing.Any,
    k: int = 10,
    answerable_only: bool = True,
    cache: typing.Any = None,
    judge_model: str = "",
    eval_cfg: typing.Any = None,
) -> dict[str, typing.Any]:
    """Run the full retrieve -> answer -> grade loop over datasets, aggregate.

    When ``eval_cfg.ircot_enabled`` is True, cases with ``intent == "multi-hop"``
    are routed through the IRCoT interleaved retrieve-and-reason loop instead of
    the single-pass ``score_case``. ``eval_cfg`` None ⇒ current behavior.
    """
    rows: list[tuple[str, bool]] = []
    skipped = 0
    ircot_on = eval_cfg is not None and getattr(eval_cfg, "ircot_enabled", False)
    for dset in datasets:
        id2content = {m.id: m.content for m in dset.corpus}
        with tempfile.TemporaryDirectory(prefix="simba-qa-") as td:
            retriever = simba.eval.recall_adapter.build_retriever(
                dset,
                cfg,
                embed_doc=embed_doc,
                embed_query=embed_query,
                data_dir=td,
                llm_client=None,
            )
            for case in dset.cases:
                if answerable_only and not case.answer.strip():
                    skipped += 1
                    continue
                if ircot_on and (case.intent or "") == "multi-hop":
                    import simba.eval.benchmarks.ircot as ircot

                    correct = ircot.score_case_ircot(
                        case,
                        retriever,
                        id2content,
                        llm,
                        max_steps=eval_cfg.ircot_max_steps,
                        k_per_step=eval_cfg.ircot_k_per_step,
                        k_final=eval_cfg.ircot_k_final,
                        cache=cache,
                        judge_model=judge_model,
                    )
                else:
                    correct = score_case(
                        case,
                        retriever,
                        id2content,
                        llm,
                        k=k,
                        cache=cache,
                        judge_model=judge_model,
                    )
                if correct is None:
                    skipped += 1
                    continue
                rows.append((case.intent or "?", correct))
    report = aggregate(rows)
    report["n_skipped"] = skipped
    return report
