"""HaluMem QA evaluator — recall → answer → 3-way judge.

Outcomes: correct / hallucination / omission.

Ingests a user's ``memory_points`` into a fresh retriever (the same
``recall_adapter`` path the recall benchmarks use, so it measures what ships), then
for each question recalls top-k, answers with the ``llm``, and asks a separate
``judge`` to classify the prediction. Unlike the recall@k harness, the metric
(`halumem.aggregate_qa`) rewards *not* surfacing wrong/stale memories — see
docs/plans/10. Phase-6 dormancy / Phase-7 contradiction-resolution can be ablated
by toggling their config and re-running (they change what the retriever surfaces).
"""

from __future__ import annotations

import tempfile
import typing

import simba.eval.benchmarks.halumem as hm
import simba.eval.recall_adapter
from simba.eval.benchmarks.judge import build_answer_prompt
from simba.eval.dataset import Dataset, Memory

EmbedFn = typing.Callable[[str], list[float]]


def build_halumem_judge_prompt(
    question: str, gold: str, predicted: str, is_boundary: bool
) -> str:
    """Classify a predicted answer into correct / hallucination / omission."""
    if is_boundary:
        return (
            "This question CANNOT be answered from memory (the information was never "
            "provided). Classify the predicted answer. Reply JSON only: "
            '{"outcome": "correct"} if it correctly declines or says it is '
            'unknown/unavailable, or {"outcome": "hallucination"} if it states a '
            "specific answer anyway.\n\n"
            f"Question: {question}\nPredicted: {predicted}\nJSON:"
        )
    return (
        "Classify the predicted answer against the gold answer. Reply JSON only "
        'with exactly one of: {"outcome": "correct"} (same meaning as gold), '
        '{"outcome": "omission"} (declines / says unknown although the gold answer '
        'exists), or {"outcome": "hallucination"} (gives a different or fabricated '
        "specific answer).\n\n"
        f"Question: {question}\nGold: {gold}\nPredicted: {predicted}\nJSON:"
    )


_VALID = {hm.QA_CORRECT, hm.QA_HALLUCINATION, hm.QA_OMISSION}


def judge_outcome(
    question: str,
    gold: str,
    predicted: str,
    is_boundary: bool,
    judge: typing.Any,
) -> str | None:
    """LLM-judge a prediction into a HaluMem outcome; None if unjudgeable."""
    verdict = judge.complete_json(
        build_halumem_judge_prompt(question, gold, predicted, is_boundary)
    )
    if not isinstance(verdict, dict):
        return None
    outcome = str(verdict.get("outcome", "")).strip().lower()
    return outcome if outcome in _VALID else None


def _user_corpus(user: hm.HaluUser) -> list[Memory]:
    """All of a user's memory_points as a retrievable corpus (originals + updates).

    Keeping superseded originals in the store is deliberate: it's exactly the
    condition under which Phase-6 dormancy / Phase-7 supersession should *help* —
    demoting the stale point lowers hallucination on update questions.
    """
    mems: list[Memory] = []
    for si, s in enumerate(user.sessions):
        for mp in s.memory_points:
            if not mp.content.strip():
                continue
            # HaluMem numbers memory_points per-session (index resets each
            # session), so the session index must be in the id — otherwise
            # f"{uuid}_mp_{index}" collides across sessions, masking earlier
            # content and making the gold point unretrievable.
            mems.append(
                Memory(
                    id=f"{user.uuid}_s{si}_mp_{mp.index}",
                    content=mp.content,
                    type=mp.memory_type or "FACT",
                    created_at=mp.timestamp,
                )
            )
    return mems


def run_halumem_qa(
    users: list[hm.HaluUser],
    *,
    embed_doc: EmbedFn,
    embed_query: EmbedFn,
    cfg: typing.Any,
    llm: typing.Any,
    judge: typing.Any,
    k: int = 10,
) -> dict[str, typing.Any]:
    """Run the HaluMem QA loop over users; return ``aggregate_qa`` + ``n_skipped``."""
    outcomes: list[tuple[str, str]] = []
    skipped = 0
    for user in users:
        corpus = _user_corpus(user)
        if not corpus:
            continue
        id2content = {m.id: m.content for m in corpus}
        ds = Dataset(name=user.uuid, corpus=corpus, cases=[])
        with tempfile.TemporaryDirectory(prefix="simba-halu-") as td:
            retriever = simba.eval.recall_adapter.build_retriever(
                ds,
                cfg,
                embed_doc=embed_doc,
                embed_query=embed_query,
                data_dir=td,
                llm_client=None,
            )
            for session in user.sessions:
                for q in session.questions:
                    ids = retriever(q.question)[:k]
                    contexts = [id2content[i] for i in ids if i in id2content]
                    predicted = llm.complete(build_answer_prompt(q.question, contexts))
                    if not predicted or not predicted.strip():
                        skipped += 1
                        continue
                    outcome = judge_outcome(
                        q.question, q.answer, predicted, q.is_boundary, judge
                    )
                    if outcome is None:
                        skipped += 1
                        continue
                    outcomes.append((q.question_type or "?", outcome))

    report = hm.aggregate_qa(outcomes)
    report["n_graded"] = len(outcomes)
    report["n_skipped"] = skipped
    return report
