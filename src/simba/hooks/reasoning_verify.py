"""Tier-2 doctrine-verify for the finish hooks (spec 27, Phase E).

The tools-free / mid-reasoning gap the ``PreToolUse`` gate cannot reach: an agent
can state a wrong conclusion or a doctrine-violating plan in PROSE with no tool to
gate. ``Stop``/``SubagentStop`` are the only Claude/Codex enforcement points after
the agent has spoken — so this promotes them from observe-only to an optional
doctrine-verify that, on a violation, returns a block reason. The adapter maps a
``Stop``/``SubagentStop`` ``block_reason`` to Claude's
``{"decision":"block","reason":…}`` (force a reconsider before finishing).

Reuses the EXACT pitfall machinery the PreToolUse gate uses (recall the
doctrine/scar TYPES, then ``pitfall.pitfall_note`` does the LLM violation check vs
the topical-match floor) — the finalized message stands in for the "pending move".
Fail-open throughout: any error / no LLM / no match returns ``None`` (never blocks
the finish spuriously). Gated by ``hooks.reasoning_verify_enabled`` (default off →
this is never called; the finish hooks stay byte-identical).
"""

from __future__ import annotations


def _pitfall_llm_client(cfg):
    """An available LLM client for violation mode, else None (→ fallback). Fail-open."""
    if getattr(cfg, "pitfall_gate_mode", "violation") != "violation":
        return None
    try:
        import simba.llm.client

        client = simba.llm.client.get_client()
        return client if client.available() else None
    except Exception:
        return None


def verify(response: str, cwd_str: str | None, cfg) -> str | None:
    """Return a block reason iff the finalized ``response`` VIOLATES a doctrine/scar.

    Recalls only the doctrine TYPES (``hooks.pitfall_gate_types``) topically close to
    the response, then defers to ``pitfall.pitfall_note`` (violation-mode LLM check,
    or the configured no-LLM fallback) — the same logic + thresholds as the PreToolUse
    pitfall gate. ``None`` when nothing fires, on no LLM in violation mode with no
    fallback, or on any error. Never raises (fail-open)."""
    if not response:
        return None
    try:
        import simba.hooks._memory_client as memory_client
        import simba.memory.pitfall

        types = [
            t.strip().upper()
            for t in getattr(
                cfg, "pitfall_gate_types", "FAILURE,PREFERENCE,GOTCHA"
            ).split(",")
            if t.strip()
        ]
        if not types:
            return None
        memories = memory_client.recall_memories(
            response,
            project_path=cwd_str if cwd_str else None,
            max_results=getattr(cfg, "pitfall_gate_max_results", 5),
            filters={"types": types},
        )
        directive = simba.memory.pitfall.pitfall_note(
            memories, response, cfg=cfg, llm_client=_pitfall_llm_client(cfg)
        )
        return directive or None
    except Exception:
        return None
