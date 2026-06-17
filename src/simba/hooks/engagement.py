"""Tier-1 engagement marker (spec 27): the simba-EMITTED ``🦁☑`` ledger.

A glanceable, per-turn signal that simba actually engaged this turn and *what it
surfaced*. The distinction that makes it real (vs the probabilistic ``[✓ rules]``
self-attestation): the ledger is built deterministically by simba's hooks from
the recall/gate result, the agent merely ECHOES it, and ``Stop`` verifies the
echo against simba's own per-turn record. Presence then reflects a real simba
interaction, not an emoji the model invented.

Anchored at ``UserPromptSubmit`` (fires every turn, tools or not → even a
zero-tool turn emits a marker); ``PreToolUse`` APPENDS the gate action
(rule-warned / rewrote / blocked). Pure assembly here; the hooks wire emission +
the per-turn "did simba act" record. All behind ``hooks.engagement_marker_enabled``
(default off → byte-identical to today).
"""

from __future__ import annotations

# Single source of truth for the marker glyph — shared with the preflight ledger
# (spec 28) so both surfaces read identically.
from simba.doctrine.preflight import MARKER

__all__ = [
    "MARKER",
    "append_gate_action",
    "gate_action_label",
    "has_marker",
    "prompt_ledger",
]


def prompt_ledger(*, memory_count: int, top_similarity: float) -> str:
    """Build the ``UserPromptSubmit`` ledger line from this turn's recall result.

    ``🦁☑ recalled N (top X.XX)`` when something matched, else
    ``🦁☑ idle (nothing matched)``. Always returns a marker line so even a
    zero-recall turn is observable.
    """
    if memory_count and memory_count > 0:
        return f"{MARKER} recalled {memory_count} (top {top_similarity:.2f})"
    return f"{MARKER} idle (nothing matched)"


def append_gate_action(ledger: str, action: str) -> str:
    """Append a ``· <action>`` gate clause to an existing ledger line."""
    action = (action or "").strip()
    if not action:
        return ledger
    return f"{ledger} · {action}"


def gate_action_label(kind: str, detail: str) -> str:
    """Render a gate action label for the marker.

    ``kind`` is one of ``rewrite`` / ``block`` / ``warn`` (the PreToolUse gate
    outcomes); ``detail`` is a short command/target. Unknown kinds fall back to
    ``"<kind>: <detail>"``.
    """
    detail = (detail or "").strip()
    verbs = {"rewrite": "rewrote", "block": "blocked", "warn": "rule-warned"}
    verb = verbs.get(kind, kind)
    return f"{verb}: {detail}" if detail else verb


def has_marker(text: str) -> bool:
    """True iff ``text`` contains the engagement marker glyph (echo check)."""
    return bool(text) and MARKER in text
