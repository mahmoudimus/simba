"""PreToolUse pitfall/doctrine enforcement gate — surface a stored scar as a
STOP-and-confirm DIRECTIVE when the agent's pending move matches it.

The two-half acme "lobotomy" cure: the retrieval half (entropy-boost, shipped) makes
the right scar RANK well; this enforcement half fires it as a *directive* at the
decision point — the *"you're about to take the workaround you told me not to"* gate.
The gap it closes was measured: the agent CITED its diagnose-first doctrine yet still
proposed the revert+xfail workaround — the scar surfaced as a passive fact, not a gate.

Pure + fail-open, like ``conflict.py``/``intent.py``. The recall + TYPE filtering
(FAILURE / PREFERENCE / GOTCHA — the doctrine/scar/trap types) happens in the daemon
via the hook; this module only (1) decides whether the TOP-RANKED candidate clears a
directive FLOOR and (2) frames it. The floor is STRICTER than recall's min_similarity
because a directive interrupts the agent — it must fire only on a strong, specific
match.

Why only the top-ranked candidate: the no-false-positive guarantee is measured on the
top candidate. On the live acme store (probe, 2026-06-15) the 3 labeled moments fire
their top candidate at similarity >= 0.82 while 6 benign moves top out at <= 0.73 — a
clean gap; floor 0.78 sits in it (0/6 benign FP, 3/3 fire). A deep scan past the top
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


def surface_pitfall_directive(memory: dict) -> str:
    """Frame a matched scar/doctrine as a STOP-and-confirm directive (not passive
    context). The framing is type-aware (doctrine vs already-failed vs known-trap) so
    the model treats the memory as a gate on its current move."""
    mtype = (memory.get("type") or "").upper() if isinstance(memory, dict) else ""
    content = (
        (memory.get("content") if isinstance(memory, dict) else "") or ""
    ).strip()
    lead = _FRAMING.get(mtype, _DEFAULT_FRAMING)
    body = content or "(a prior scar matched this move)"
    return (
        "<pitfall-warning>\n"
        f"  {lead}: {body}\n"
        "  Your pending move appears to match it. Do NOT proceed on autopilot — "
        "confirm your current action honors this before continuing; if you are "
        "knowingly overriding it, say so explicitly and why.\n"
        "</pitfall-warning>"
    )


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
