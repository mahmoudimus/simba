# 25 — Guardian cost + gate-graduation (reinforcement vs enforcement)

Status: Proposal A SHIPPED (default-off lever) + Proposal B documented.

## PROBLEM

The CLAUDE.md guardian re-injects the full `SIMBA:core` block on EVERY
`UserPromptSubmit`, unconditionally (`simba.guardian.extract_core.main` called
from `src/simba/hooks/user_prompt_submit.py`). That block is ~2k tokens — pure
overhead every turn even when the model still has the rules. A `[✓ rules]`
signal checker already exists (`simba.guardian.check_signal`) but is only used
at Codex finalize, NOT to gate injection.

## FRAMING

- **guardian = *presence*** (rule in context; broad, probabilistic; fights
  context decay). guardian is *necessary*.
- **gate = *enforcement*** (redirect / TOOL_RULE / pitfall; narrow,
  deterministic). gate is *sufficient*.

A rule that has a deterministic gate no longer needs the probabilistic presence
layer — it has graduated from reinforcement to enforcement.

## Proposal A — inject conditionally, not every prompt (the token win) — SHIPPED

- Lever: `hooks.guardian_signal_gated: bool = False` (a `@configurable` field on
  `HooksConfig`). **Default False = today's behavior preserved exactly
  (byte-identical).**
- When True: inject the `SIMBA:core` block only when EITHER (a) it's the first
  prompt of the session / post-compaction (no prior-response signal recorded),
  OR (b) the model's PREVIOUS response was missing the `[✓ rules]` signal
  (= rules decayed). Otherwise SKIP the block. Always **fail-open**: any
  uncertainty or error → inject (never silently drop the rules).
- Plumbing (`src/simba/guardian/signal_flag.py`):
  - The Stop hook (`src/simba/hooks/stop.py`) sees the model's response — it
    records whether `[✓ rules]` was present for this session to a small
    per-session flag file under the temp dir
    (`<tmp>/claude-rules-signal-<session_id>.json`). Reuses
    `simba.guardian.check_signal.check_signal` for the detection logic.
  - `user_prompt_submit.run()` reads that flag to decide: present → SKIP,
    missing/absent/error → INJECT (fail-open).
  - PreCompact and SessionStart **reset** the flag (delete it) so the next
    prompt re-injects after compaction / a fresh session.

## Proposal B — gate-graduation (docs + light mechanism) — DOCUMENTED

- Workflow documented in `.claude/rules/CORE_INSTRUCTIONS.md` (mirrors the
  "SoTA levers graduate to default-ON" phrasing): a CORE rule that becomes a
  deterministic gate (redirect / TOOL_RULE / pitfall) graduates OUT of the
  `SIMBA:core` block to reclaim tokens — the gate now guarantees it.
- No automatic exclusion mechanism is shipped: marker-level "gated" annotation
  was evaluated and judged not clean enough to add silently (it would change
  `extract_core` output semantics and risk dropping a rule whose gate isn't
  actually wired). Left as documented engineering discipline — the human curates
  the block when a gate lands.

## Defer (NOT implemented)

- Proposal C (tiering / trim) — out of scope.
- Proposal D (violation tracking) — out of scope.
