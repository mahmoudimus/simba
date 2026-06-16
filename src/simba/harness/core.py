"""Harness-agnostic hook core: canonical result + dispatch.

Each lifecycle hook's logic lives in ``simba.hooks.<event>.run(payload)`` and
returns a CanonicalResult.  ``dispatch`` is the single entrypoint used by both
transports — the inline CLI and the daemon ``POST /hook/{event}`` endpoint.

All filesystem paths inside a hook's ``run`` are derived from ``payload`` (e.g.
``payload["cwd"]``), never from the process cwd, so dispatch is safe to run
inside the daemon process whose own cwd differs from the agent's.
"""

from __future__ import annotations

import dataclasses
import importlib

# canonical event name -> module exposing run(payload) -> CanonicalResult
_EVENT_MODULES = {
    "session_start": "simba.hooks.session_start",
    "prompt_submit": "simba.hooks.user_prompt_submit",
    "stop": "simba.hooks.stop",
    "pre_compact": "simba.hooks.pre_compact",
    "pre_tool": "simba.hooks.pre_tool_use",
    # v2: "post_tool"
}


@dataclasses.dataclass
class CanonicalResult:
    """Harness-agnostic hook result."""

    additional_context: str = ""
    suppress_output: bool = False
    # Count of memories recalled/injected this turn (0 when none). Surfaced to
    # callers (e.g. the pi bridge prints "[simba: N memories injected]") so recall
    # is visible, not magic.
    memory_count: int = 0
    # v2 fields (defined for forward-compat; unused in MVP):
    block_reason: str | None = None
    transform: dict | None = None
    # A directive that context-capable harnesses (Claude/Codex) inject as
    # additionalContext (already included in additional_context) but block-only
    # harnesses (pi tool_call) must enforce as a hard block. Populated for a
    # strong TOOL_RULE match. Claude/Codex render IGNORES it (byte-identical).
    escalated_block: str | None = None


def dispatch(event: str, payload: dict) -> CanonicalResult:
    """Run the canonical hook for ``event``. Raises KeyError if unknown."""
    module = importlib.import_module(_EVENT_MODULES[event])
    return module.run(payload)
