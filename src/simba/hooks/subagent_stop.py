"""SubagentStop hook — verify-before-report doctrine check (spec 27, Phase E).

``SubagentStop`` was completely unused before this — yet the ignored-doctrine
incidents this design targets happened INSIDE subagents (a subagent stating a
doctrine-violating conclusion that then reports back to the parent). So this is the
natural verify-before-report hook: when ``hooks.reasoning_verify_enabled`` is on,
doctrine-verify the subagent's finalized output and, on a violation, set a
``block_reason`` → the adapter maps a ``SubagentStop`` block to Claude's
``{"decision":"block","reason":…}`` (the subagent reconsiders before reporting).

Default-OFF → ``run`` returns an empty ``CanonicalResult`` (no block, no output) —
byte-identical to the prior state where the event was unwired. Fail-open.
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
    """Run the SubagentStop pipeline. Returns a CanonicalResult.

    Doctrine-verify (Tier 2) only — no tailor/continuous capture (the parent's
    ``Stop`` owns that). Off by default → empty result. Fail-open."""
    cfg = _hooks_cfg()
    response = hook_input.get("response", "")
    if getattr(cfg, "reasoning_verify_enabled", False) and response:
        from simba.hooks import reasoning_verify

        cwd_str = hook_input.get("cwd")
        block = reasoning_verify.verify(response, cwd_str, cfg)
        if block:
            return CanonicalResult(block_reason=block)
    return CanonicalResult()


def main(hook_input: dict) -> str:
    """Run the SubagentStop hook and render the Claude/Codex envelope."""
    import simba.harness.adapters.claude as claude

    return claude.render("SubagentStop", run(hook_input))
