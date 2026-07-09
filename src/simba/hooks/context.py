"""pi ``context`` hook — re-inject the ledger/doctrine before every LLM call.

pi fires ``context`` before EVERY LLM call with the current message list and lets an
extension return a modified list (``ContextEventResult.messages``). This closes the
mid-reasoning-drift gap that Claude/Codex cannot reach: recall fires once per prompt
there and never re-fires, so across a long reasoning chain the rules fall out of
attention. Here we re-inject a tiny ``🦁☑`` ledger (+ optional recall) so the doctrine
rides the whole chain.

pi-only: the bridge POSTs ``messages_text`` (the flattened recent conversation), and
applies the returned ``additional_context`` by appending it as a custom message. The
payload is kept MINIMAL because this fires per LLM call (latency). Gated by
``hooks.engagement_marker_enabled`` (default off → empty result, the bridge applies
nothing). Fail-open throughout.
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
    """Build the re-injection context for a pi ``context`` event.

    Returns a ``CanonicalResult`` whose ``additional_context`` is a small ``🦁☑``
    ledger (recall count + top similarity) the bridge re-injects before the LLM
    call. Off (default) or no conversation text → empty result. Fail-open."""
    cfg = _hooks_cfg()
    if not getattr(cfg, "engagement_marker_enabled", False):
        return CanonicalResult()

    text = (hook_input.get("messages_text") or "").strip()
    if not text:
        return CanonicalResult()

    cwd_str = hook_input.get("cwd")
    memories: list[dict] = []
    try:
        import simba.hooks._memory_client as memory_client

        memories = memory_client.recall_memories(
            text,
            project_path=cwd_str if cwd_str else None,
            min_similarity=cfg.prompt_min_similarity,
        )
    except Exception:
        memories = []  # fail-open: still emit the (idle) marker so it rides the chain

    try:
        import simba.hooks.engagement as eng

        top = float(memories[0].get("similarity", 0.0)) if memories else 0.0
        ledger = eng.prompt_ledger(memory_count=len(memories), top_similarity=top)
    except Exception:
        return CanonicalResult()

    return CanonicalResult(additional_context=ledger, memory_count=len(memories))


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
