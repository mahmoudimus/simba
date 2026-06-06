"""IRCoT — interleaved retrieve-and-reason for answer-time multi-hop QA.

For a multi-hop question, IRCoT alternates between reasoning and retrieval: the
LLM emits one reasoning sentence plus a follow-up sub-query, the retriever pulls
evidence for it, the evidence is accumulated, and the loop repeats up to
``max_steps`` before a final answer is generated from everything gathered. This
runs at benchmark *judge* time, not at daemon recall time.

Pure/injectable by design — the retriever and LLM are always passed in, so the
whole loop is exercised with fakes in CI. ``ircot_answer`` is fully fail-open:
any exception collapses to ``""``.
"""

from __future__ import annotations

import typing

import simba.eval.benchmarks.judge as judge

if typing.TYPE_CHECKING:
    from simba.eval.dataset import EvalCase

Retriever = typing.Callable[[str], list[str]]


def build_step_prompt(question: str, evidence: list[str], step: int) -> str:
    """Prompt asking the LLM for one reasoning sentence and a follow-up sub-query.

    Expects a JSON reply ``{"reasoning": "...", "sub_query": "..."}``. ``evidence``
    is everything accumulated so far (may be empty on step 0).
    """
    joined = "\n".join(f"- {e}" for e in evidence) if evidence else "(none yet)"
    return (
        "You are answering a multi-hop question by gathering evidence one step at "
        "a time. Read the question and the evidence so far, write ONE short "
        "reasoning sentence, then propose the single most useful follow-up search "
        "query to find the next missing fact. If you already have enough to "
        'answer, return an empty sub_query. Reply with JSON only: {"reasoning": '
        '"...", "sub_query": "..."}.\n\n'
        f"Question: {question}\nStep: {step}\nEvidence so far:\n{joined}\nJSON:"
    )


def build_final_prompt(question: str, evidence: list[str]) -> str:
    """Prompt the LLM to answer from accumulated evidence (reuses the QA prompt)."""
    return judge.build_answer_prompt(question, evidence)


def ircot_answer(
    question: str,
    retriever: Retriever,
    id2content: dict[str, str],
    llm: typing.Any,
    *,
    max_steps: int = 4,
    k_per_step: int = 3,
    k_final: int = 10,
) -> str:
    """Run the IRCoT loop and return a final answer string (or "" on failure)."""
    try:
        evidence_ids: list[str] = []
        seen: set[str] = set()
        for step in range(max_steps):
            evidence_texts = [
                id2content[i] for i in evidence_ids[-k_final:] if i in id2content
            ]
            step_json = llm.complete_json(
                build_step_prompt(question, evidence_texts, step)
            )
            if not isinstance(step_json, dict) or not step_json.get("sub_query"):
                break
            sub_query = step_json["sub_query"]
            new_ids = retriever(sub_query)[:k_per_step]
            for i in new_ids:
                if i not in seen:
                    seen.add(i)
                    evidence_ids.append(i)
        final_evidence = [
            id2content[i] for i in evidence_ids[:k_final] if i in id2content
        ]
        return llm.complete(build_final_prompt(question, final_evidence))
    except Exception:
        return ""


def score_case_ircot(
    case: EvalCase,
    retriever: Retriever,
    id2content: dict[str, str],
    llm: typing.Any,
    *,
    max_steps: int = 4,
    k_per_step: int = 3,
    k_final: int = 10,
    cache: typing.Any = None,
    judge_model: str = "",
    judge_llm: typing.Any | None = None,
) -> bool | None:
    """IRCoT version of ``judge.score_case`` — same return contract (True/False/None).

    Runs ``ircot_answer`` to produce the predicted answer, then grades it with
    ``judge_llm`` (or ``llm`` when ``judge_llm`` is None) via
    ``judge.build_judge_prompt``. ``cache`` is the same ``JudgeCache`` as in
    ``score_case``.
    """
    predicted = ircot_answer(
        case.query,
        retriever,
        id2content,
        llm,
        max_steps=max_steps,
        k_per_step=k_per_step,
        k_final=k_final,
    )
    if not predicted or not predicted.strip():
        return None
    grader = judge_llm if judge_llm is not None else llm
    if cache is not None:
        hit = cache.get(judge_model, case.query, case.answer, predicted)
        if hit is not None:
            return hit
    verdict = grader.complete_json(
        judge.build_judge_prompt(case.query, case.answer, predicted)
    )
    if not isinstance(verdict, dict) or "correct" not in verdict:
        return None
    correct = bool(verdict["correct"])
    if cache is not None:
        cache.put(judge_model, case.query, case.answer, predicted, correct)
    return correct
