"""pi ``message_end`` hook — doctrine-verify the finalized assistant message.

pi fires ``message_end`` when a message ends and lets an extension REPLACE the
finalized message (``MessageEndEventResult.message``, keeping the role). This closes
the tools-free-output gap: an agent can state a wrong conclusion or a
doctrine-violating plan in PROSE with no tool to gate, and there is no pre-hook on
assistant text on any harness. ``message_end`` is the closest post-emission catch on
pi (``Stop``/``SubagentStop`` are the Claude/Codex equivalent).

pi-only: the bridge POSTs ``message_text`` (the finalized assistant text), and on a
violation applies the returned ``block_reason`` by annotating/replacing the message
(keeping the role) or injecting a correction so the next call re-derives. Reuses the
pitfall machinery via ``reasoning_verify`` (same as Stop/SubagentStop). Gated by
``hooks.reasoning_verify_enabled`` (default off → no block; bridge applies nothing).
Fail-open.
"""

from __future__ import annotations

from simba.harness.core import CanonicalResult


def _hooks_cfg():
    """Load the hooks config section (registers it on first access)."""
    import simba.config
    import simba.hooks.config

    _ = simba.hooks.config  # ensure the "hooks" section is registered
    return simba.config.load("hooks")


def run(hook_input: dict) -> CanonicalResult:
    """Doctrine-verify a pi ``message_end`` event's finalized message.

    Returns a ``CanonicalResult`` whose ``block_reason`` is the correction directive
    when the message violates a stored doctrine, else ``None``. Off (default) → no
    block. Fail-open."""
    cfg = _hooks_cfg()
    text = (hook_input.get("message_text") or "").strip()
    if not getattr(cfg, "reasoning_verify_enabled", False) or not text:
        return CanonicalResult()

    cwd_str = hook_input.get("cwd")
    from simba.hooks import reasoning_verify

    block = reasoning_verify.verify(text, cwd_str, cfg)
    return CanonicalResult(block_reason=block)


def main(hook_input: dict) -> str:
    """Render for the (pi-only) daemon path — JSON CanonicalResult."""
    import json

    r = run(hook_input)
    return json.dumps(
        {
            "additional_context": r.additional_context,
            "memory_count": r.memory_count,
            "block_reason": r.block_reason,
        }
    )
