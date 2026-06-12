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
import datetime
import tempfile
import time
import typing

import simba.eval.recall_adapter
import simba.memory.conflict
from simba.eval.runner import _percentile

if typing.TYPE_CHECKING:
    from simba.eval.dataset import Dataset, EvalCase

EmbedFn = typing.Callable[[str], list[float]]
Retriever = typing.Callable[[str], list[str]]


def _parse_date(s: str) -> datetime.datetime | None:
    """Parse the corpus date formats we see (ISO-8601 and HaluMem's 'Mon DD, YYYY')."""
    if not s:
        return None
    s = s.strip()
    try:
        return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%b %d, %Y, %H:%M:%S", "%b %d, %Y"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _date_label(s: str) -> str:
    d = _parse_date(s)
    return d.strftime("%Y-%m-%d") if d else s.strip()


def build_answer_prompt(
    question: str,
    contexts: list[str],
    dates: list[str] | None = None,
    *,
    question_date: str = "",
) -> str:
    """Prompt the model to answer strictly from the retrieved context.

    When ``dates`` (parallel to ``contexts``) is given and non-empty, the prompt
    mirrors what the daemon injects via ``format_memories``: each memory is
    date-labelled and the most-recent one is flagged, with a most-recent-wins
    instruction. This is the recency signal the product ships; omitting it (the
    old bare format) understates temporal accuracy. Falls back to the bare format
    when no dates are available.

    ``question_date`` fills the official LongMemEval reader's "Current Date"
    slot — the anchor for relative-time resolution ("how long ago...").
    Measured (n=72 sweep): +0.111 overall, temporal-reasoning 0.417->0.833.
    Production parity: the host agent always knows today's date, so omitting
    it UNDERSTATES the shipping behavior.
    """
    date_line = f"Current date: {question_date}\n" if question_date else ""
    if not contexts:
        return (
            "You are answering a question using ONLY the conversation memories "
            "below. If the answer is not present, say you don't know. Answer "
            f"concisely.\n\nMemories:\n(no context)\n\n{date_line}Question: "
            f"{question}\nAnswer:"
        )
    if dates and any(d for d in dates):
        parsed = [
            (_parse_date(dates[i]) if i < len(dates) else None)
            for i in range(len(contexts))
        ]
        dated = [(i, p) for i, p in enumerate(parsed) if p is not None]
        newest_idx = max(dated, key=lambda t: t[1])[0] if dated else -1
        lines = []
        for i, c in enumerate(contexts):
            raw = dates[i] if i < len(dates) else ""
            tag = f"[{_date_label(raw)}] " if raw else ""
            flag = " (most recent)" if i == newest_idx else ""
            lines.append(f"- {tag}{c}{flag}")
        joined = "\n".join(lines)
        # Mirror the daemon's format_memories: annotate each memory with its date
        # and flag the most recent, but do NOT inject a recency-resolution
        # instruction — the product never does, and an A/B (deepseek) showed the
        # instruction is a no-op for a capable consumer (the date+newest
        # annotation already does the work). Keeps the benchmark measuring what
        # ships rather than handing the answerer an extra hint.
        return (
            "You are answering a question using ONLY the conversation memories "
            "below. Each memory is tagged with the date it was recorded. If the "
            "answer is not present, say you don't know. Answer concisely.\n\n"
            f"Memories:\n{joined}\n\n{date_line}Question: {question}\nAnswer:"
        )
    joined = "\n".join(f"- {c}" for c in contexts)
    return (
        "You are answering a question using ONLY the conversation memories below. "
        "If the answer is not present, say you don't know. Answer concisely.\n\n"
        f"Memories:\n{joined}\n\n{date_line}Question: {question}\nAnswer:"
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


# Official LongMemEval per-question-type judge prompts — verbatim port of
# xiaowu0162/LongMemEval ``get_anscheck_prompt`` (the templates hebb-mind used
# for its 0.79 and what the official benchmark ships). Keyed on case.intent.
# Measured on simba outputs: official-vs-generic judge = +3.6pp (p=5e-4);
# preference slice 0.167 -> 0.300 from the judge prompt alone. Grading is a
# plain completion checked for a "yes" substring (upstream's decision rule),
# NOT the generic JSON verdict.
_OFFICIAL_DEFAULT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also "
    "answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}"
    "\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
)
_OFFICIAL_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also "
    "answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no. In addition, do not penalize off-by-one "
    "errors for the number of days. If the question asks for the number of "
    "days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
    "predicting 19 days when the answer is 18), the model's response is still "
    "correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_OFFICIAL_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}"
    "\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
)
_OFFICIAL_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and "
    "a response from a model. Please answer yes if the response satisfies the "
    "desired response. Otherwise, answer no. The model does not need to reflect "
    "all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\nQuestion: {}\n\n"
    "Rubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer "
    "yes or no only."
)
_OFFICIAL_BY_INTENT = {
    "single-session-user": _OFFICIAL_DEFAULT,
    "single-session-assistant": _OFFICIAL_DEFAULT,
    "multi-session": _OFFICIAL_DEFAULT,
    "temporal-reasoning": _OFFICIAL_TEMPORAL,
    "knowledge-update": _OFFICIAL_KNOWLEDGE_UPDATE,
    "single-session-preference": _OFFICIAL_PREFERENCE,
}


def build_official_judge_prompt(
    intent: str, question: str, gold: str, predicted: str
) -> str:
    """Official LongMemEval judge prompt for a question type (``case.intent``).

    Unknown/empty intents fall back to the default template, mirroring upstream.
    """
    template = _OFFICIAL_BY_INTENT.get(intent, _OFFICIAL_DEFAULT)
    return template.format(question, gold, predicted)


def score_case(
    case: EvalCase,
    retriever: Retriever,
    id2content: dict[str, str],
    answerer: typing.Any,
    *,
    judge: typing.Any | None = None,
    k: int = 10,
    cache: typing.Any = None,
    judge_model: str = "",
    id2date: dict[str, str] | None = None,
    conflict_cfg: typing.Any = None,
    judge_style: str = "generic",
) -> bool | None:
    """Retrieve top-k, generate an answer, grade it. None = couldn't grade.

    The ``answerer`` generates the prediction; a separate ``judge`` client grades
    it. When ``judge`` is None the answerer grades its own answer (legacy path).

    When ``cache`` is a ``JudgeCache``, an identical (judge_model, question, gold,
    predicted) verdict is served from disk instead of re-calling the judge LLM.

    ``id2date`` (id -> created_at) adds the recency annotation the daemon injects.

    ``conflict_cfg`` (a memory config) mirrors the answer-time conflict note the
    daemon injects via ``format_memories``: ``conflict_note`` is gated inside —
    disabled config (the default) means zero LLM cost and an unchanged prompt,
    so the bench only pays for what production would.

    ``judge_style`` selects the grading protocol: "official" grades with the
    LongMemEval per-type prompt (``build_official_judge_prompt`` keyed on
    ``case.intent``) via a plain completion and upstream's yes-substring
    decision rule; "generic" (the function-level default, for back-compat)
    keeps the JSON {"correct": bool} verdict. The bench config defaults to
    "official" (measured +3.6pp on simba outputs, p=5e-4).
    """
    _judge = judge if judge is not None else answerer
    if not judge_model and judge is not None:
        judge_model = getattr(getattr(_judge, "_cfg", None), "model", "")
    # The JudgeCache key is (judge_model, question, gold, predicted) — it does
    # NOT include the prompt style. Namespace the model string for the official
    # path so cached generic verdicts can't contaminate official-judge runs
    # (and per-intent, since the official prompt differs by question type).
    # The generic key stays bare for back-compat with existing caches.
    if judge_style == "official":
        judge_model = f"{judge_model}|anscheck-{case.intent or 'default'}"
    ids = [i for i in retriever(case.query)[:k] if i in id2content]
    contexts = [id2content[i] for i in ids]
    dates = [(id2date or {}).get(i, "") for i in ids] if id2date else None
    prompt = build_answer_prompt(
        case.query,
        contexts,
        dates,
        question_date=getattr(case, "question_date", ""),
    )
    if conflict_cfg is not None:
        note = simba.memory.conflict.conflict_note(
            contexts, case.query, cfg=conflict_cfg, llm_client=answerer
        )
        if note:
            prompt = f"{prompt}\n\n{note}"
    predicted = answerer.complete(prompt)
    if not predicted or not predicted.strip():
        return None
    if cache is not None:
        hit = cache.get(judge_model, case.query, case.answer, predicted)
        if hit is not None:
            return hit
    if judge_style == "official":
        reply = _judge.complete(
            build_official_judge_prompt(
                case.intent or "", case.query, case.answer, predicted
            )
        )
        if not reply or not reply.strip():
            return None
        # Upstream's exact decision rule: any "yes" substring => correct.
        correct = "yes" in reply.lower()
    else:
        verdict = _judge.complete_json(
            build_judge_prompt(case.query, case.answer, predicted)
        )
        if not isinstance(verdict, dict) or "correct" not in verdict:
            return None
        correct = bool(verdict["correct"])
    if cache is not None:
        cache.put(judge_model, case.query, case.answer, predicted, correct)
    return correct


def build_abstention_judge_prompt(question: str, predicted: str) -> str:
    """Prompt the judge to decide if the predicted answer is a proper refusal."""
    return (
        "You are judging whether a predicted answer correctly declines to answer "
        "a question that cannot be answered from the available memories. A correct "
        "refusal says the information is unavailable. Reply JSON only: "
        '{"abstained": true} or {"abstained": false}.\n\n'
        f"Question: {question}\nPredicted answer: {predicted}\nJSON:"
    )


def score_abstention(
    case: EvalCase,
    retriever: Retriever,
    id2content: dict[str, str],
    answerer: typing.Any,
    *,
    judge: typing.Any | None = None,
    k: int = 10,
    abstention_phrases: list[str] | None = None,
    id2date: dict[str, str] | None = None,
) -> bool | None:
    """Retrieve, generate, then check for refusal.

    Strategy: heuristic-first (phrase match against ``abstention_phrases``), then
    judge-LLM confirmation only when the heuristic is ambiguous. Returns True
    (correctly abstained), False (wrongly answered), or None (unscored).
    """
    _judge = judge if judge is not None else answerer
    phrases = abstention_phrases if abstention_phrases is not None else []
    ids = [i for i in retriever(case.query)[:k] if i in id2content]
    contexts = [id2content[i] for i in ids]
    dates = [(id2date or {}).get(i, "") for i in ids] if id2date else None
    predicted = answerer.complete(
        build_answer_prompt(
            case.query,
            contexts,
            dates,
            question_date=getattr(case, "question_date", ""),
        )
    )
    if not predicted or not predicted.strip():
        return None
    lowered = predicted.lower()
    if any(phrase.lower() in lowered for phrase in phrases):
        return True
    verdict = _judge.complete_json(build_abstention_judge_prompt(case.query, predicted))
    if not isinstance(verdict, dict) or "abstained" not in verdict:
        return None
    return bool(verdict["abstained"])


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


def aggregate_with_abstention(
    rows: list[tuple[str, bool]],
    abstention_rows: list[tuple[str, bool]],
) -> dict[str, typing.Any]:
    """Extend ``aggregate()`` output with an abstention_accuracy block."""
    report = aggregate(rows)
    correct = [c for _, c in abstention_rows]
    accuracy = sum(1 for x in correct if x) / len(correct) if correct else 0.0
    report["abstention"] = {"n": len(abstention_rows), "accuracy": accuracy}
    return report


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


def _load_abstention_phrases() -> list[str]:
    """Load the configured abstention phrase list from the eval config."""
    import simba.config
    import simba.eval.config  # registers the "eval" section

    _ = simba.eval.config
    raw = simba.config.load("eval").abstention_phrases
    return [p.strip() for p in raw.split(",") if p.strip()]


def run_qa(
    datasets: list[Dataset],
    *,
    embed_doc: EmbedFn,
    embed_query: EmbedFn,
    cfg: typing.Any,
    llm: typing.Any,
    judge: typing.Any | None = None,
    k: int = 10,
    answerable_only: bool = True,
    include_abstention: bool = False,
    abstention_phrases: list[str] | None = None,
    cache: typing.Any = None,
    judge_model: str = "",
    eval_cfg: typing.Any = None,
    judge_style: str = "generic",
) -> dict[str, typing.Any]:
    """Run the full retrieve -> answer -> grade loop over datasets, aggregate.

    ``llm`` is the answerer; ``judge`` (when given) grades its answers. When
    ``judge`` is None the answerer grades itself (legacy behaviour).

    ``judge_style`` ("official" | "generic") selects the grading protocol per
    ``score_case``; the bench CLI passes ``bench.judge_style`` (default
    "official" — the canonical LongMemEval axis). The function-level default
    stays "generic" for back-compat with direct callers.

    When ``include_abstention`` is True, ``_abs`` cases are routed to
    ``score_abstention`` and reported in a separate ``abstention`` block. The
    report always carries an ``abstention`` key for a consistent shape.

    When ``eval_cfg.ircot_enabled`` is True, cases with ``intent == "multi-hop"``
    are routed through the IRCoT interleaved retrieve-and-reason loop instead of
    the single-pass ``score_case``. ``eval_cfg`` None ⇒ current behavior.
    """
    import simba.eval.benchmarks.longmemeval as longmemeval

    if include_abstention and abstention_phrases is None:
        abstention_phrases = _load_abstention_phrases()

    rows: list[tuple[str, bool]] = []
    abstention_rows: list[tuple[str, bool]] = []
    # End-to-end latency per question (retriever + answer + grade), not
    # retriever-only; split score_case later if retriever-only timing is needed.
    latencies: list[float] = []
    skipped = 0
    ircot_on = eval_cfg is not None and getattr(eval_cfg, "ircot_enabled", False)
    for dset in datasets:
        id2content = {m.id: m.content for m in dset.corpus}
        # Recency annotation the daemon injects (format_memories) — mirrored here
        # so the benchmark measures what ships, not a no-recency degradation.
        id2date = {m.id: (m.created_at or "") for m in dset.corpus}
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
                if include_abstention and longmemeval.is_abstention(case.id):
                    t0 = time.perf_counter()
                    result = score_abstention(
                        case,
                        retriever,
                        id2content,
                        llm,
                        judge=judge,
                        k=k,
                        abstention_phrases=abstention_phrases,
                        id2date=id2date,
                    )
                    latencies.append((time.perf_counter() - t0) * 1000)
                    if result is None:
                        skipped += 1
                        continue
                    abstention_rows.append((case.intent or "?", result))
                    continue
                if answerable_only and not case.answer.strip():
                    skipped += 1
                    continue
                t0 = time.perf_counter()
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
                        judge_llm=judge,
                    )
                else:
                    # Conflict surfacing the daemon injects (format_memories) —
                    # mirrored like the recency annotation above so the benchmark
                    # measures what ships. Gated inside conflict_note: zero cost
                    # unless cfg.conflict_surfacing_enabled.
                    correct = score_case(
                        case,
                        retriever,
                        id2content,
                        llm,
                        judge=judge,
                        k=k,
                        cache=cache,
                        judge_model=judge_model,
                        id2date=id2date,
                        conflict_cfg=cfg,
                        judge_style=judge_style,
                    )
                latencies.append((time.perf_counter() - t0) * 1000)
                if correct is None:
                    skipped += 1
                    continue
                rows.append((case.intent or "?", correct))
    if include_abstention:
        report = aggregate_with_abstention(rows, abstention_rows)
    else:
        report = aggregate(rows)
        report["abstention"] = {"n": 0, "accuracy": 0.0}
    report["n_skipped"] = skipped
    report["latency"] = {
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "n": len(latencies),
    }
    return report
