# 27 — Reasoning-layer verification: the 🦁☑ engagement marker + doctrine check (harness-tiered)

**Date:** 2026-06-16
**Status:** TODO (design; not started)
**Branch:** TBD off `main`
**Builds on:** spec 25 (the `[✓ rules]` signal machinery + `guardian/signal_flag.py`),
spec 23/24 (the pi bridge + canonical core).

## Why

Two failure modes this session — the drizzle hand-edit and the PR-review-in-place —
were agents **ignoring doctrine they already had**. The gate (`PreToolUse`) catches
the wrong *command*, but two gaps remain:

1. **Tools-free wrong actions.** Bad advice, a wrong conclusion, a flawed plan
   stated in prose with **no tool** has no gate — you fundamentally cannot gate
   token emission (there is no pre-hook on assistant *text*).
2. **Mid-reasoning drift.** Recall fires once per prompt (`UserPromptSubmit`) and
   never re-fires; across a long reasoning chain the rules fall out of attention.

And separately, the user wants **observability**: a glanceable, per-turn signal
that simba actually engaged this turn and *what it surfaced* — so a human can
audit whether the agent consulted (and respected) the memory/doctrine layer.

**Key distinction (load-bearing):** a self-attested prefix (telling the agent to
print `🦁☑`) is a *claim, not proof* — the same probabilistic limit as `[✓ rules]`.
The real version is **simba-emitted**: simba's deterministic hooks emit the marker
+ a one-line ledger of what they did; the agent echoes it; `Stop` verifies the
echo. Then presence reflects a *real* simba interaction, and the ground truth is
simba's own log.

## What we discovered — harness capability matrix

Enforcement requires a hook that can *block/modify*. The granularity differs
sharply by harness:

| Capability | Claude Code | Codex | **pi** |
|---|---|---|---|
| Gate a tool (block/rewrite) | `PreToolUse` | `PreToolUse` + `PermissionRequest` | `tool_call` |
| Block / transform a prompt | `UserPromptSubmit` (block) | (subset) | `input` (`InputEventResult`) |
| Inject context at turn start | `UserPromptSubmit` | yes | `before_agent_start` (+ rewrite **system prompt**) |
| **Re-inject mid-reasoning (every LLM call)** | ❌ | ❌ | ✅ **`context`** → `{ messages? }` |
| **Replace the finalized assistant output** | ❌ | ❌ | ✅ **`message_end`** → `{ message? }` |
| Set reasoning/thinking budget | ❌ | ❌ | ✅ thinking-level handlers |
| Block the *finish* → force reconsider | `Stop` / `SubagentStop` (decision:block) | (subset) | `agent_end` / `turn_end` |

**Findings:**
- Claude Code's only non-tool enforcement points are `UserPromptSubmit` (pre) and
  `Stop`/`SubagentStop` (post). simba's `Stop` is **observe-only** today, and
  `SubagentStop` is **completely unused** — yet the incidents happened *inside
  subagents*, so that's the natural verify-before-report hook.
- **pi is in a different class.** `context` (modify the message list before
  **every** LLM call) closes the mid-reasoning-drift gap; `message_end` (replace
  the finalized message) closes the tools-free-output gap. Neither has a
  Claude/Codex equivalent. **simba's pi bridge currently wires only `tool_call`**
  → all of this is untapped headroom.

## Design — two tiers

### Tier 1 — universal observability: the 🦁☑ engagement marker (all harnesses)
- simba's hooks emit a one-line ledger **when they fire**, into
  `additional_context`. Anchored at **`UserPromptSubmit`** (every turn, tools or
  not) so even a zero-tool turn emits it; `PreToolUse` appends the gate action:
  ```
  🦁☑ recalled 3 (top 0.74) · rule-warned: Edit drizzle/meta
  🦁☑ recalled 0 · rewrote: regenerate-bdd-init-schema-docker.sh
  🦁☑ idle (nothing matched)
  ```
- Guardian/CORE instructs the agent to **echo** the marker. `Stop` verifies the
  echo (reuse `guardian.check_signal` + `guardian/signal_flag.py` from spec 25);
  a turn that *acted* but lacks a preceding `🦁☑` ledger is flagged.
- Ground truth = the daemon's recall/gate log; the marker is the glanceable
  surface. Config: `hooks.engagement_marker_enabled: bool = False`.

### Tier 2 — pi-native reasoning enforcement (pi only)
- **`context`**: re-inject doctrine/recall (+ the `🦁☑` ledger) into the message
  list before **every** LLM call → the rules ride the whole reasoning chain; no
  mid-reasoning drift. Bridge posts to a daemon endpoint, applies the returned
  `messages`.
- **`message_end`**: doctrine-verify the finalized assistant message (reuse the
  pitfall machinery); on a violation, **replace/annotate** it (keeping the role)
  or inject a correction so the next call re-derives. The tools-free output catch.
- **`before_agent_start`** (optional): rewrite the system prompt to carry the
  engagement contract.
- All pi-only; degrade cleanly to Tier 1 on Claude/Codex. The Claude/Codex
  *approximation* of `context` re-injection — making the agent provably engage
  simba before reasoning — is the **mandated-preflight** pattern in
  [spec 28](28-intent-priming-and-preflight.md).

### Stop / SubagentStop upgrade (Claude/Codex degraded enforcement)
- Promote simba's `Stop` from observe-only to an optional **doctrine-verify +
  block-to-reconsider** (`CanonicalResult.block_reason` → Claude `Stop`
  `decision: block`). Wire **`SubagentStop`** the same way — verify a subagent's
  output before it reports back to the parent.

## Phases (TDD — RED first)
- **A.** Engagement ledger emitted into `additional_context` — `UserPromptSubmit`
  always, `PreToolUse` appends the gate action; behind
  `engagement_marker_enabled` (default off). Characterization: off ⇒ byte-identical
  Claude/Codex output.
- **B.** Echo verification in `Stop` (reuse `signal_flag`/`check_signal`); flag a
  marker-missing turn that surfaced activity.
- **C.** pi `context` re-injection — `simba.ts` wires `context`, POSTs to a daemon
  endpoint, applies returned `messages` (ledger + optional recall). Golden test of
  the payload + applied result.
- **D.** pi `message_end` doctrine-verify — `simba.ts` wires `message_end`, POSTs
  the finalized message for a doctrine check (pitfall reuse), applies
  replace/annotate on violation. Golden test.
- **E.** `Stop` / `SubagentStop` block-to-reconsider on Claude/Codex —
  `CanonicalResult.block_reason` from a `Stop` run; adapter maps it to
  `decision: block`; characterization that off = today's observe-only output.
- **F.** Measurement — marker-present rate; manual spot-check that ignored-doctrine
  incidents drop; **latency** of the `context` re-injection (it fires per LLM
  call — keep the injected ledger tiny, gate by config, measure overhead).

## Config (`hooks` section)
- `engagement_marker_enabled: bool = False` — Tier 1 marker emit + echo-verify.
- `reasoning_verify_enabled: bool = False` — Tier 2 `message_end` / `Stop`
  doctrine check (costs an LLM judgment); reuse `hooks.pitfall_gate_*` for the
  check logic + `pitfall_gate_types`.

## Caveats / non-goals
- **Self-attestation alone is weak** — the marker must be *simba-emitted*, not
  agent-invented.
- **`message_end` is a replace, not a re-think** — you substitute/annotate (keeping
  the role); for a genuine re-reason, inject a correction via `context` and let the
  next call re-derive.
- **`context` fires per LLM call** → latency; keep the injected payload minimal and
  config-gated.
- **Tier 2 is pi-only.** Claude/Codex get Tier 1 + the `Stop`/`SubagentStop` block.
- You cannot gate mid-emission text on any harness — the closest is `message_end`
  (pi) / `Stop`-block (Claude/Codex), both post-emission.

## Acceptance
- Marker emitted **every turn including zero-tool turns**, reflecting *real* simba
  activity; `engagement_marker_enabled=False` ⇒ byte-identical to today.
- On pi: a doctrine-violating *finalized* message is caught + corrected/annotated
  via `message_end`; doctrine/recall rides **every** LLM call via `context`.
- `Stop` / `SubagentStop` can block-to-reconsider on Claude/Codex.
- ruff + full suite green; the off path is characterization-tested byte-identical.
