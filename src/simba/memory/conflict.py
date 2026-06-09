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


def surface_directive(conflict: ConflictResult) -> str:
    """A directive that NAMES the specific conflict and tells the consumer to
    surface it (state what must be confirmed) rather than pick a side."""
    detail = f" ({conflict.description})" if conflict.description else ""
    return (
        f'NOTE: two retrieved memories conflict — "{conflict.a}" vs '
        f'"{conflict.b}"{detail}. Do not choose one or guess; surface this '
        "conflict and state what must be confirmed to resolve it."
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
    wired. Otherwise runs ``detect_conflict`` and returns ``surface_directive``
    for a real conflict, or ``""`` when none is found.
    """
    if not getattr(cfg, "conflict_surfacing_enabled", False):
        return ""
    if llm_client is None:
        return ""
    min_memories = getattr(cfg, "conflict_surfacing_min_memories", 2)
    if len(memories) < min_memories:
        return ""
    conflict = detect_conflict(memories, query, llm_client=llm_client)
    if conflict is None:
        return ""
    return surface_directive(conflict)
