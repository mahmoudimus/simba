# 28 — Intent-primed doctrine + mandated preflight at `UserPromptSubmit`

**Date:** 2026-06-16
**Status:** TODO (design; not started)
**Branch:** TBD off `main`
**Builds on:** spec 27 (the `🦁☑` marker, `signal_flag`, Stop/SubagentStop verify),
the kira/d810 gates (the doctrine this primes), spec 26 (project-scoped recall).

## Why

`UserPromptSubmit` is the **only hook that sees the user's intent before any action
happens** — every other hook (`PreToolUse`, `Stop`) *reacts* to what the agent
already did. Today simba uses it only for generic semantic recall + CORE injection.
That's the cheapest, earliest prevention point and it's under-exploited.

Both incidents this session — the drizzle hand-edit and the PR-review-in-place —
were the agent **picking the wrong path despite having the doctrine**. Front-loading
the *right approach from stated intent* prevents the wrong action **before a gate
has to catch it**. And generic recall isn't enough: it's not intent-classified,
not tied to the *applicable gates*, and purely advisory.

## Model: prime → mandate → enforce

1. **Prime (intent-keyed doctrine).** Classify the prompt's task **cheaply** —
   embedding-match against doctrine *triggers* (the embedder is already loaded; no
   LLM call) — and inject the task-relevant doctrine + **which TOOL_RULEs/redirects
   apply** to this project. Pre-emptive guidance, not generic recall.
2. **Mandate (preflight action).** When the intent is a real task (risk/intent
   tier), inject: *"Your first action this turn is `simba preflight <task>`."* This
   converts the implicit consult into an **explicit, observable, logged tool call**
   — the `🦁☑`, grounded in a real invocation rather than a self-claimed emoji.
3. **Enforce (gate on preflight).** `PreToolUse` **blocks any mutating tool that
   runs without a preflight having fired this turn** (per-turn flag — reuse spec-25
   `guardian/signal_flag.py`). Read-only tools are unaffected. The teeth:
   "consult before acting" becomes a *precondition*, not a request.

Plus: `Stop`/`SubagentStop` verify the preflight ran (spec 27 machinery). And
(optional) a **block-to-redirect** at the human boundary — for a small curated
high-risk phrasing set, `UserPromptSubmit` *blocks* the prompt with the right
recipe instead of injecting.

**Without the gate-on-preflight half, this is just an advisory instruction — the
exact failure mode we keep hitting.** Build prime + mandate + enforce together.

## `simba preflight <task>` (new CLI + daemon endpoint)

Given task text + cwd, returns:
- (a) the **intent-relevant doctrine** memories (project-scoped recall — spec 26),
- (b) the **applicable `TOOL_RULE`s + redirect rules** for the project
  (`rules_cli` + `redirect/store` lookups you already have),
- (c) a short "here's the right approach" brief.

Side effects: sets the **per-turn preflight flag** (so the gate sees it) and emits
the `🦁☑` ledger (spec 27). It is mostly intent-keyed recall + a rules/redirects
lookup — small.

## Intent classification (cheap, hot-path-safe)

- Maintain **doctrine triggers**: each high-value doctrine/gate carries trigger
  phrases/embeddings (e.g. "PR review", "regenerate init-schema / migration /
  drizzle", "system tests"). Store alongside the doctrine memory / rule.
- On `UserPromptSubmit`, embedding-match the prompt against triggers (cheap; no LLM
  on the hot path). Above threshold → prime that doctrine; if it's a **risk-tier**
  trigger → also mandate preflight.
- Default-OFF; measure false-prime rate.

## Harness symmetry

This mandated-preflight pattern is the **Claude/Codex approximation of pi's native
`context` re-injection** (spec 27 Tier 2). pi can re-inject doctrine on every LLM
call natively, so on pi the mandate is optional — the `context` event already keeps
doctrine present throughout reasoning. Claude/Codex lack that granularity, so they
fake "the agent provably engaged simba before reasoning" by mandating a preflight
*tool* and gating on it.

## Open direction: affordance over prohibition (revisit after 27/28 land)

A redirect-deny gate enumerates the *wrong* commands and blocks them. A cleaner
model for workflows that have a known "right way" is **affordance**: package the
right workflow as a **skill / plugin command** (e.g. `simba pr-review <N>` that does
the `pr_reviews/kira-pr-<N>` worktree dance; `simba db-regen` that runs the docker
init-schema regen with `COMPOSE_PROJECT_NAME` set) and **steer the agent to it via
intent-priming** — give it the paved path instead of fencing off the cliffs.

This likely subsumes several deny-gates: the **kira PR-review** case (review-in-place
via `git show pr-N:` → use the worktree skill) and the **drizzle/init-schema** case
(hand-edit → use the regen skill). Worth exploring how simba can *register* such
skills/commands (it's already a plugin) and whether `message_end` / preflight
(specs 27/28) can verify the agent took the affordance vs. hand-rolling.

**Decision (2026-06-16):** the kira PR-review gate is **deferred** — don't ship a
redirect-deny now; revisit it through this affordance lens once the 27/28 machinery
(intent-prime → preflight → verify) exists.

## Phases (TDD — RED first)
- **A.** Doctrine-triggers store + intent matcher (embedding match), behind
  `intent_priming_enabled` (default off). Tests: a "review PR" prompt matches the
  PR-review trigger; an unrelated prompt does not.
- **B.** Intent-primed injection at `UserPromptSubmit` (matched doctrine + applicable
  gates). Characterization: off ⇒ byte-identical (today's recall + CORE only).
- **C.** `simba preflight <task>` endpoint (recall + rules/redirects lookup) + sets
  the per-turn flag + emits `🦁☑`. Tests.
- **D.** Mandate injection (risk-tier) + `PreToolUse` **gate-on-preflight**: a
  mutating tool with no preflight this turn is blocked; read-only tools allowed; a
  preflight clears the gate. Tests for all three.
- **E.** `Stop`/`SubagentStop` verify the preflight ran (reuse spec 27).
- **F.** (Optional) block-to-redirect for a curated high-risk phrasing set.
- **G.** Measurement: false-prime rate, preflight over-fire/annoyance, whether it
  reduces wrong-path incidents, intent-match latency.

## Config (`hooks` section)
- `intent_priming_enabled: bool = False` — prime matched doctrine + applicable gates.
- `preflight_mandate_enabled: bool = False` — the gate-on-preflight teeth.
- `preflight_mandate_risk_only: bool = True` — mandate only for risk-tier intents,
  not every task (over-fire guard).
- Reuse the doctrine-triggers store + `hooks.pitfall_gate_types`.

## Caveats / non-goals
- **Injection alone is advisory** — the `PreToolUse`-blocks-mutating-without-preflight
  rule is what gives it teeth. Build both halves.
- **Over-fire/annoyance** — default `risk_only`; never mandate preflight for trivial
  or read-only prompts.
- **False-primes pollute context** — threshold the intent match + measure.
- **`UserPromptSubmit` can only *prepend* context on Claude/Codex** (not rewrite the
  prompt; pi's `input` event can transform it).

## Acceptance
- A task-shaped prompt ("review PR #N", "regenerate init-schema") primes the right
  doctrine + applicable gates; an unrelated prompt does not. `intent_priming_enabled
  =False` ⇒ byte-identical to today.
- With `preflight_mandate` on: a mutating tool is blocked until `simba preflight` ran
  this turn; read-only tools unaffected; preflight clears it.
- `simba preflight` returns doctrine + applicable rules, sets the flag; `Stop`
  verifies.
- ruff + full suite green; the off path is characterization-tested byte-identical.
