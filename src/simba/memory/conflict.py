"""Answer-time conflict surfacing: detect a *real* contradiction among the
retrieved memories and emit a directive that NAMES it and tells the consumer to
surface it (state what must be confirmed) rather than silently pick a side.

Motivation (measured): simba retrieves conflicting memories fine, but the
answerer collapses latent contradictions — it picks one side or abstains. A
probe showed the answerer is steerable by a directive, but a *generic* always-on
directive is too weak AND it harms non-conflict cases (over-hedging). So the
lever must DETECT a real conflict, NAME it specifically, and only THEN surface —
gated, not blanket.

One LLM call asks whether any two memories mutually exclude / contradict / compete
in a way that matters for the question, replying JSON only. Robust parse (mirrors
``eval/benchmarks/judge.py``). FAIL-OPEN throughout: empty input, parse failure,
or any exception returns ``None`` / ``""`` — never raises into the recall path.
``conflict_note`` is the gated entry point both eval and production call; it is a
no-op (returns ``""`` with zero LLM cost) whenever the feature is disabled, the
candidate set is below the minimum, or no llm_client is wired.
"""

from __future__ import annotations

import dataclasses
import typing


@dataclasses.dataclass
class ConflictResult:
    """The two conflicting memory texts + a short description of the conflict."""

    a: str
    b: str
    description: str


def build_detect_prompt(memories: list[str], query: str) -> str:
    """Prompt the model to flag a contradiction among numbered memories.

    Replies JSON only: ``{"conflict": bool, "a": <int>, "b": <int>,
    "description": "<short>"}`` where ``a``/``b`` index into ``memories``.
    """
    numbered = "\n".join(f"{i}. {m}" for i, m in enumerate(memories))
    return (
        "You are checking retrieved memories for a contradiction that matters for "
        "answering a question. Given the question and the numbered memories below, "
        "decide whether any TWO of them CONFLICT — mutually exclusive, "
        "contradictory, or competing claims that cannot both be true for this "
        "question. Ignore memories that merely differ in topic or that are "
        "compatible. Reply with JSON only: "
        '{"conflict": true, "a": <int index>, "b": <int index>, '
        '"description": "<short explanation>"} '
        'or {"conflict": false}.\n\n'
        f"Question: {query}\nMemories:\n{numbered}\nJSON:"
    )


def detect_conflict(
    memories: list[str],
    query: str,
    *,
    llm_client: typing.Any,
) -> ConflictResult | None:
    """One LLM call: do any two memories conflict for this question?

    Returns a ``ConflictResult`` (indices resolved to the memory texts) when the
    model reports a conflict, else ``None``. FAIL-OPEN: returns ``None`` on empty
    input, a missing client, a malformed/garbage reply, out-of-range or identical
    indices, or any exception — never raises into the recall path.
    """
    if not memories or len(memories) < 2 or llm_client is None:
        return None
    try:
        verdict = llm_client.complete_json(build_detect_prompt(memories, query))
    except Exception:
        return None
    if not isinstance(verdict, dict) or not verdict.get("conflict"):
        return None
    a = verdict.get("a")
    b = verdict.get("b")
    if not isinstance(a, int) or not isinstance(b, int):
        return None
    if a == b or not (0 <= a < len(memories)) or not (0 <= b < len(memories)):
        return None
    description = verdict.get("description")
    if not isinstance(description, str):
        description = ""
    return ConflictResult(a=memories[a], b=memories[b], description=description.strip())


def build_pair_detect_prompt(
    a: str, b: str, query: str, *, generous: bool = False
) -> str:
    """Prompt the model to judge whether exactly TWO memories conflict.

    A focused variant of :func:`build_detect_prompt`: isolating the pair removes
    the distractors that bury a subtle contradiction in the all-at-once prompt.
    Replies JSON only: ``{"conflict": bool, "description": "<short>"}``.

    ``generous`` (B2b) switches to a recall-biased prompt used at WRITE time: it
    flags the pair if they COULD plausibly conflict / be inconsistent / compete,
    maximizing RECALL. Precision is recovered later at recall via the query-aware
    re-check (:func:`conflict_note_from_store`), so over-flagging here is fine.
    Default ``False`` keeps the strict (mutual-exclusion) prompt unchanged.
    """
    if generous:
        return (
            "You are pre-screening two retrieved memories for a POSSIBLE "
            "contradiction. Decide whether the TWO memories below could plausibly "
            "conflict — be inconsistent, compete, or fail to both hold — for some "
            "question. Be GENEROUS: flag the pair if there is any plausible tension "
            "between them; a later step confirms whether it truly matters. Only "
            "skip pairs that are clearly about unrelated topics. Reply with JSON "
            'only: {"conflict": true, "description": "<short explanation>"} or '
            '{"conflict": false}.\n\n'
            f"Question: {query}\nMemory A: {a}\nMemory B: {b}\nJSON:"
        )
    return (
        "You are checking two retrieved memories for a contradiction that matters "
        "for answering a question. Decide whether the TWO memories below CONFLICT "
        "— mutually exclusive, contradictory, or competing claims that cannot both "
        "be true for this question. If they merely differ in topic or are "
        "compatible, that is NOT a conflict. Reply with JSON only: "
        '{"conflict": true, "description": "<short explanation>"} or '
        '{"conflict": false}.\n\n'
        f"Question: {query}\nMemory A: {a}\nMemory B: {b}\nJSON:"
    )


def detect_conflict_pairwise(
    memories: list[str],
    query: str,
    *,
    llm_client: typing.Any,
    max_pairs: int = 45,
) -> ConflictResult | None:
    """Check candidate PAIRS in isolation; return the first flagged conflict.

    For each unordered pair ``(i < j)`` — up to ``max_pairs`` pairs in order —
    one focused LLM call asks whether THOSE TWO memories conflict for the
    question. Isolating the pair lifts detection recall on subtle conflicts that
    a single all-at-once prompt buries among k distractors. Short-circuits on the
    first pair the model flags, returning a ``ConflictResult`` (``a=memories[i]``,
    ``b=memories[j]``). FAIL-OPEN: returns ``None`` on empty input, fewer than two
    memories, a missing client, or any exception — never raises into the recall
    path.
    """
    if not memories or len(memories) < 2 or llm_client is None:
        return None
    try:
        checked = 0
        n = len(memories)
        for i in range(n):
            for j in range(i + 1, n):
                if checked >= max_pairs:
                    return None
                checked += 1
                verdict = llm_client.complete_json(
                    build_pair_detect_prompt(memories[i], memories[j], query)
                )
                if not isinstance(verdict, dict) or not verdict.get("conflict"):
                    continue
                description = verdict.get("description")
                if not isinstance(description, str):
                    description = ""
                return ConflictResult(
                    a=memories[i], b=memories[j], description=description.strip()
                )
    except Exception:
        return None
    return None


def surface_directive(conflict: ConflictResult) -> str:
    """A directive that NAMES the specific conflict and tells the consumer to
    surface it (state what must be confirmed) rather than pick a side."""
    detail = f" ({conflict.description})" if conflict.description else ""
    return (
        f'NOTE: two retrieved memories conflict — "{conflict.a}" vs '
        f'"{conflict.b}"{detail}. Do not choose one or guess; surface this '
        "conflict and state what must be confirmed to resolve it."
    )


def surface_directive_from_description(description: str) -> str:
    """Directive built from a precomputed conflict's description alone.

    The recall-read path (:func:`conflict_note_from_store`) has the two memories'
    *ids*, not their texts — and the LLM-generated ``description`` already names
    both sides ("Memory A says X; Memory B says Y"). Leading with opaque ids is
    noise to the answerer (it underperformed the live B1 directive in the B2
    smoke), so build the directive from the description.
    """
    body = (description or "").strip() or "two retrieved memories conflict"
    return (
        f"NOTE: {body}. Do not choose one or guess; surface this conflict and "
        "state what must be confirmed to resolve it."
    )


def conflict_note(
    memories: list[str],
    query: str,
    *,
    cfg: typing.Any,
    llm_client: typing.Any,
) -> str:
    """Gated, fail-open entry point used by both eval and production.

    Returns ``""`` (with no LLM cost) when the feature is disabled, the candidate
    set is below ``cfg.conflict_surfacing_min_memories``, or no ``llm_client`` is
    wired. Otherwise runs the configured detector (``cfg.conflict_detect_strategy``
    — ``"single"`` one all-at-once call, ``"pairwise"`` focused pairs in isolation)
    and returns ``surface_directive`` for a real conflict, or ``""`` when none is
    found.
    """
    if not getattr(cfg, "conflict_surfacing_enabled", False):
        return ""
    if llm_client is None:
        return ""
    min_memories = getattr(cfg, "conflict_surfacing_min_memories", 2)
    if len(memories) < min_memories:
        return ""
    strategy = getattr(cfg, "conflict_detect_strategy", "single")
    if strategy == "pairwise":
        max_pairs = getattr(cfg, "conflict_detect_max_pairs", 45)
        conflict = detect_conflict_pairwise(
            memories, query, llm_client=llm_client, max_pairs=max_pairs
        )
    else:
        conflict = detect_conflict(memories, query, llm_client=llm_client)
    if conflict is None:
        return ""
    return surface_directive(conflict)


# ── Write-time conflict engine (B2) ──────────────────────────────────────────
# Move detection OFF the answer-time path: detect a NEW memory's conflicts
# against its nearest neighbors at WRITE time (one focused pairwise call per
# neighbor), persist them via ``simba.memory.conflict_store``, and at recall just
# READ the precomputed conflict among the recalled set + annotate. Both functions
# here stay pure of persistence/time: ``detect_conflicts_on_write`` returns the
# flagged neighbors (the caller records them); ``conflict_note_from_store`` reads
# the store and builds the directive. FAIL-OPEN throughout.


def detect_conflicts_on_write(
    new_id: str,
    new_text: str,
    neighbors: list[tuple[str, str]],
    *,
    llm_client: typing.Any,
    max_neighbors: int = 5,
    generous: bool = False,
) -> list[tuple[str, str]]:
    """Compare a NEW memory against its nearest neighbors; flag conflicts.

    For each ``(neighbor_id, neighbor_text)`` — up to ``max_neighbors`` in order
    — one focused 2-memory check (:func:`build_pair_detect_prompt`, neutral
    query) asks whether the new memory CONFLICTS with that neighbor. Returns the
    list of ``(neighbor_id, description)`` for the neighbors that conflict. Pure:
    no persistence, no time, no ``new_id`` mutation (it is accepted for the
    caller's bookkeeping). FAIL-OPEN: empty neighbors, a missing client, or any
    exception yields ``[]`` — never raises into the write path. One
    ``llm_client.complete_json`` call per checked neighbor.

    ``generous`` (B2b) uses the recall-biased pair prompt: this write-time pass
    becomes a cheap, high-recall/low-precision PRE-FILTER that stores candidate
    conflicts. Precision is recovered at recall by the query-aware re-check
    (:func:`conflict_note_from_store`). Default ``False`` = strict prompt
    (current behavior unchanged).
    """
    if not neighbors or llm_client is None:
        return []
    out: list[tuple[str, str]] = []
    try:
        for neighbor_id, neighbor_text in neighbors[:max_neighbors]:
            verdict = llm_client.complete_json(
                build_pair_detect_prompt(new_text, neighbor_text, "", generous=generous)
            )
            if not isinstance(verdict, dict) or not verdict.get("conflict"):
                continue
            description = verdict.get("description")
            if not isinstance(description, str):
                description = ""
            out.append((neighbor_id, description.strip()))
    except Exception:
        return []
    return out


def build_recall_recheck_prompt(query: str, descriptions: list[str]) -> str:
    """Prompt the model to confirm which stored candidate conflict matters HERE.

    The write-time pre-filter stores candidate conflicts query-INDEPENDENTLY
    (high recall). At recall we have the question, so one query-aware call asks
    which (if any) of the candidate conflict descriptions is a REAL conflict that
    matters for answering THIS question — recovering precision. Replies JSON only:
    ``{"relevant": true, "index": <int>}`` (index into ``descriptions``) or
    ``{"relevant": false}``.
    """
    numbered = "\n".join(f"{i}. {d}" for i, d in enumerate(descriptions))
    return (
        "You are deciding whether a previously-flagged candidate conflict between "
        "two memories actually MATTERS for answering the question below. Given the "
        "question and the numbered candidate conflict descriptions, pick the ONE "
        "that is a real conflict relevant to answering this specific question. If "
        "none is relevant to this question, say so. Reply with JSON only: "
        '{"relevant": true, "index": <int index>} or {"relevant": false}.\n\n'
        f"Question: {query}\nCandidate conflicts:\n{numbered}\nJSON:"
    )


def _recall_recheck_directive(
    query: str,
    descriptions: list[str],
    *,
    llm_client: typing.Any,
) -> str | None:
    """One query-aware confirm over the stored candidate descriptions.

    Returns the directive for the confirmed candidate, ``""`` when the model
    confirms NONE is relevant (the precision filter), or ``None`` on any
    failure/garbage (the caller then falls back to the non-recheck path). Mirrors
    the robust JSON parse of ``eval/benchmarks/judge.py``.
    """
    if not descriptions or llm_client is None or not query:
        return None
    verdict = llm_client.complete_json(build_recall_recheck_prompt(query, descriptions))
    if not isinstance(verdict, dict) or "relevant" not in verdict:
        return None
    if not verdict.get("relevant"):
        return ""  # the precision win: no candidate matters for this question
    idx = verdict.get("index", 0)
    if not isinstance(idx, int) or not (0 <= idx < len(descriptions)):
        return None
    return surface_directive_from_description(descriptions[idx])


def conflict_note_from_store(
    recalled_ids: list[str],
    *,
    project_path: str,
    cfg: typing.Any,
    query: str = "",
    llm_client: typing.Any = None,
) -> str:
    """Gated, fail-open recall-read: annotate a precomputed conflict.

    Reads ``conflict_store.conflicts_among(recalled_ids, project_path=...)`` and,
    if any conflict among the recalled set was precomputed at write time, builds a
    directive from the first recorded conflict's stored description (via
    :func:`surface_directive_from_description` — the store has memory *ids*, not
    texts, and the description already names both sides). Returns ``""`` — with
    zero LLM cost and no detection — when the feature is disabled, no conflict is
    recorded, or anything fails. Must run inside a ``simba.db.connect()`` context.

    B2b query-aware re-check: when ``cfg.conflict_recall_recheck`` is True AND an
    ``llm_client`` and ``query`` are wired AND there are stored candidate(s) among
    the recalled set, run ONE query-aware confirm over the candidate descriptions
    (:func:`build_recall_recheck_prompt`). A confirmed candidate yields its
    directive; a "none relevant" verdict yields ``""`` (the precision filter that
    drops a query-irrelevant stored candidate). FAIL-OPEN: any exception in the
    re-check falls back to the non-recheck path (the first stored candidate).
    Default (flag off) keeps the current behavior — first stored candidate, no
    LLM call — backward compatible with existing callers.
    """
    if not getattr(cfg, "conflict_surfacing_enabled", False):
        return ""
    try:
        import simba.memory.conflict_store as conflict_store

        rows = conflict_store.conflicts_among(recalled_ids, project_path=project_path)
    except Exception:
        return ""
    if not rows:
        return ""
    if getattr(cfg, "conflict_recall_recheck", False) and llm_client and query:
        try:
            directive = _recall_recheck_directive(
                query, [r.description for r in rows], llm_client=llm_client
            )
            if directive is not None:
                return directive
        except Exception:
            pass  # fall back to the non-recheck path below
    return surface_directive_from_description(rows[0].description)
