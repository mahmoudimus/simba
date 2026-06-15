"""PreToolUse pitfall/doctrine enforcement gate — surface a stored scar as a
STOP-and-confirm DIRECTIVE when the agent's pending move matches it.

The two-half memory-surfacing cure: the retrieval half (entropy-boost, shipped) makes
the right scar RANK well; this enforcement half fires it as a *directive* at the
decision point — the *"you're about to take the workaround you told me not to"* gate.
The gap it closes was measured: the agent CITED its own doctrine yet still proposed
the workaround it warns against — the scar surfaced as a passive fact, not a gate.

Pure + fail-open, like ``conflict.py``/``intent.py``. The recall + TYPE filtering
(FAILURE / PREFERENCE / GOTCHA — the doctrine/scar/trap types) happens in the daemon
via the hook; this module only (1) decides whether the TOP-RANKED candidate clears a
directive FLOOR and (2) frames it. The floor is STRICTER than recall's min_similarity
because a directive interrupts the agent — it must fire only on a strong, specific
match.

Why only the top-ranked candidate: the no-false-positive guarantee is measured on the
top candidate. On a real project memory store (probe, 2026-06-15) the 3 labeled moments
fire their top candidate at similarity >= 0.82 while 6 benign moves top out at <= 0.73 —
a clean gap; floor 0.78 sits in it (0/6 benign FP, 3/3 fire). A deep scan past the top
candidate would re-admit the benign matches that the top-only check excludes.
"""

from __future__ import annotations

import typing

# Type -> (lead-in framing, the verb of the warning). Doctrine/scar/trap types get a
# STOP-and-confirm framing so the model treats the memory as a gate, not passive info.
_FRAMING = {
    "PREFERENCE": "You previously set this as doctrine",
    "FAILURE": "This was already tried and failed",
    "GOTCHA": "Known trap on this",
}
_DEFAULT_FRAMING = "Relevant prior scar"


# ── Violation detection (the real gate) ──────────────────────────────────────
# Measured (sweep over real d810 moves, 2026-06-15): a similarity-only gate over-fires
# on 29-46% of moves because in a dense domain almost every move is TOPICALLY close to
# some doctrine — and topical closeness is NOT violation. The fix: for a topically-close
# candidate, ask the LLM whether the move would VIOLATE the doctrine (do the thing it
# warns against / contradict it / repeat the recorded failure) vs merely share its topic
# or follow it. Fire only on a violation. Same robust-parse + fail-open contract as
# conflict.py. When no llm_client is wired, the caller falls back to the conservative
# FAILURE-only similarity path (FAILURE is the one pitfall-shaped type).


def build_violation_prompt(move: str, doctrine: str) -> str:
    """Prompt the model to judge whether ``move`` VIOLATES ``doctrine``.

    Abstention-biased by construction (measured 2026-06-15: the permissive prompt
    confabulated violations on topically-related scars — ~3/5 of real-move fires were
    the LLM rationalizing a contradiction on read/search moves). So: DEFAULT to
    violates=false; exploratory moves are never violations; no guessing about what the
    agent will do next. Replies JSON only: ``{"violates": bool, "why": "<short>"}``.
    """
    return (
        "You are a STRICT guardrail for a coding agent about to take an ACTION. You "
        "are shown the agent's CURRENT MOVE (its latest reasoning) and a STORED "
        "DOCTRINE/SCAR from its past work. Flag a violation ONLY IF the move would "
        "UNAMBIGUOUSLY do the very thing the doctrine warns against, directly "
        "contradict it, or repeat the exact failure it records.\n"
        "DEFAULT TO violates=false. It is NOT a violation if ANY of these hold: the "
        "move merely shares the doctrine's topic; the move is consistent with or "
        "follows the doctrine; the move is EXPLORATORY (reading, searching, "
        "summarizing, planning, extracting) rather than a concrete state-changing "
        "action; or deciding would require GUESSING what the agent does next. When "
        "in doubt, violates=false.\n"
        "Reply JSON only: "
        '{"violates": true, "why": "<short reason>"} or {"violates": false}.\n\n'
        f"CURRENT MOVE:\n{move}\n\nDOCTRINE/SCAR:\n{doctrine}\n\nJSON:"
    )


def detect_violation(
    move: str, doctrine: str, *, llm_client: typing.Any
) -> tuple[bool, str] | None:
    """One LLM call: does ``move`` violate ``doctrine``? Returns ``(violates, why)``.

    FAIL-OPEN: missing client, empty input, a malformed/garbage reply, or any exception
    returns ``None`` (the caller treats None as "couldn't decide" → does not fire).
    """
    if not move or not doctrine or llm_client is None:
        return None
    try:
        verdict = llm_client.complete_json(build_violation_prompt(move, doctrine))
    except Exception:
        return None
    if not isinstance(verdict, dict) or "violates" not in verdict:
        return None
    violates = bool(verdict.get("violates"))
    why = verdict.get("why")
    if not isinstance(why, str):
        why = ""
    return (violates, why.strip())


def select_violation(
    memories: list[dict],
    move: str,
    *,
    llm_client: typing.Any,
    topical_floor: float,
    max_checks: int,
) -> tuple[dict, str] | None:
    """Return the highest-ranked candidate the LLM judges the move VIOLATES, else None.

    Gathers up to ``max_checks`` candidates whose similarity clears ``topical_floor``
    (a permissive gate — we want plausible candidates to LLM-check, then let the LLM
    supply precision), in rank order, and asks :func:`detect_violation` for each.
    Returns ``(memory, why)`` for the first violation; ``None`` if none violate or on
    any failure. FAIL-OPEN throughout.
    """
    if not memories or llm_client is None:
        return None
    try:
        checked = 0
        for m in memories:
            if not isinstance(m, dict):
                continue
            raw = m.get("similarity", 0)
            sim = float(raw) if raw is not None else 0.0
            if sim < topical_floor:
                continue
            if checked >= max_checks:
                break
            checked += 1
            verdict = detect_violation(
                move, m.get("content", ""), llm_client=llm_client
            )
            if verdict is not None and verdict[0]:
                return (m, verdict[1])
    except (TypeError, ValueError, AttributeError):
        return None
    return None


def select_failure_fallback(
    memories: list[dict], *, min_similarity: float
) -> dict | None:
    """Conservative no-LLM fallback: the highest-ranked FAILURE-type candidate whose
    similarity clears ``min_similarity`` — else ``None``.

    FAILURE is the one type that is naturally pitfall-shaped ("this exact thing broke"),
    so a topical match to a FAILURE is far likelier to be a real repeat than a match to
    a PREFERENCE/GOTCHA (which the sweep showed over-fire on workflow descriptions).
    """
    if not memories:
        return None
    try:
        for m in memories:
            if not isinstance(m, dict) or (m.get("type") or "").upper() != "FAILURE":
                continue
            raw = m.get("similarity", 0)
            sim = float(raw) if raw is not None else 0.0
            if sim >= min_similarity:
                return m
    except (TypeError, ValueError, AttributeError):
        return None
    return None


def select_pitfall(memories: list[dict], *, min_similarity: float) -> dict | None:
    """Return the TOP-RANKED recalled doctrine/scar memory iff its similarity clears
    ``min_similarity`` — else ``None``.

    ``memories`` is the daemon's final rank-ordered, already type-filtered recall
    (best first). We judge ONLY ``memories[0]`` against the directive floor: that is
    where the measured no-false-positive guarantee holds. FAIL-OPEN: empty input, a
    missing/garbage similarity, or any exception yields ``None`` — never raises into
    the hook path.
    """
    if not memories:
        return None
    try:
        top = memories[0]
        if not isinstance(top, dict):
            return None
        raw = top.get("similarity", 0)
        sim = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError, AttributeError, IndexError):
        return None
    if sim < min_similarity:
        return None
    return top


def surface_pitfall_directive(memory: dict, *, reason: str = "") -> str:
    """Frame a matched scar/doctrine as a STOP-and-confirm directive (not passive
    context). The framing is type-aware (doctrine vs already-failed vs known-trap) so
    the model treats the memory as a gate on its current move. When ``reason`` is given
    (the LLM's violation explanation), it is included so the warning names the specific
    conflict rather than a generic topical match."""
    mtype = (memory.get("type") or "").upper() if isinstance(memory, dict) else ""
    content = (
        (memory.get("content") if isinstance(memory, dict) else "") or ""
    ).strip()
    lead = _FRAMING.get(mtype, _DEFAULT_FRAMING)
    body = content or "(a prior scar matched this move)"
    why = f"  Why this fires: {reason.strip()}\n" if reason and reason.strip() else ""
    return (
        "<pitfall-warning>\n"
        f"  {lead}: {body}\n"
        f"{why}"
        "  Your pending move appears to violate it. Do NOT proceed on autopilot — "
        "confirm your current action honors this before continuing; if you are "
        "knowingly overriding it, say so explicitly and why.\n"
        "</pitfall-warning>"
    )


def pitfall_note(
    memories: typing.Any, move: str, *, cfg: typing.Any, llm_client: typing.Any = None
) -> str:
    """Gated, fail-open orchestrator the hook calls. Returns a directive or ``""``.

    Strategy from ``cfg.pitfall_gate_mode``:
      - ``"violation"`` (default): for candidates above ``pitfall_gate_topical_floor``,
        ask the LLM whether the move VIOLATES the doctrine; fire only on a violation.
        When no ``llm_client`` is wired, fall back per ``cfg.pitfall_gate_fallback``
        (``"failure_only"`` → conservative FAILURE-type similarity gate at
        ``pitfall_gate_min_similarity``; ``"off"`` → fire nothing).
      - ``"similarity"`` (ablation/legacy): top candidate ≥ the min_similarity floor
        (the measured over-firer — kept for comparison only).

    Any exception or malformed input yields ``""`` — never raises into the hook path.
    """
    try:
        if not isinstance(memories, list) or not memories:
            return ""
        mode = getattr(cfg, "pitfall_gate_mode", "violation")
        floor = float(getattr(cfg, "pitfall_gate_min_similarity", 0.78))

        if mode == "similarity":
            hit = select_pitfall(memories, min_similarity=floor)
            return surface_pitfall_directive(hit) if hit else ""

        # violation mode
        if llm_client is not None:
            topical = float(getattr(cfg, "pitfall_gate_topical_floor", 0.70))
            max_checks = int(getattr(cfg, "pitfall_gate_max_checks", 3))
            found = select_violation(
                memories,
                move,
                llm_client=llm_client,
                topical_floor=topical,
                max_checks=max_checks,
            )
            if found is None:
                return ""
            return surface_pitfall_directive(found[0], reason=found[1])

        # no LLM → fallback
        if getattr(cfg, "pitfall_gate_fallback", "failure_only") == "failure_only":
            hit = select_failure_fallback(memories, min_similarity=floor)
            return surface_pitfall_directive(hit) if hit else ""
        return ""
    except Exception:
        return ""


def pitfall_directive(memories: typing.Any, *, min_similarity: float) -> str:
    """Gated, fail-open entry point the hook calls: return a directive string when the
    top recalled candidate clears the floor, else ``""`` (silent). Any exception or
    malformed input yields ``""`` — never raises into the PreToolUse path."""
    try:
        if not isinstance(memories, list):
            return ""
        hit = select_pitfall(memories, min_similarity=min_similarity)
        if hit is None:
            return ""
        return surface_pitfall_directive(hit)
    except Exception:
        return ""
