# pi coding-agent harness support — design

**Date:** 2026-06-15
**Status:** approved design, pre-plan
**Author:** brainstorming session

## Problem

simba supports two coding-agent harnesses today — Claude Code and Codex — by
shipping a per-harness manifest that invokes `simba hook <Event>` as a
subprocess. Each invocation reads a JSON payload on stdin and writes a
harness-specific envelope on stdout (`hooks/_io.py`).

We want simba to support a third harness: **pi** (`@earendil-works/pi-coding-agent`,
v0.78.0), an AI coding assistant CLI the user runs daily (`~/.pi/agent/`,
deepseek-v4-flash). pi extends differently from the other two: extensions are
**in-process TypeScript modules** that subscribe to an `EventBus`, not
subprocesses driven over stdin/stdout. pi already reads `CLAUDE.md`/`AGENTS.md`
and loads skills from `~/.claude/skills`, so static rules and skills work today
with zero code. The missing capability is the **dynamic memory loop** —
recall-on-prompt and capture-on-stop — which requires an extension.

## pi's contract (verified against installed `dist/core/extensions/types.d.ts`)

An extension is `export default function (pi: ExtensionAPI) { … }` and registers
handlers via `pi.on(event, handler)`. The relevant lifecycle events map cleanly
onto simba's six canonical hooks:

| simba hook   | pi event                                   | injection / gate mechanism |
|--------------|--------------------------------------------|----------------------------|
| SessionStart | `session_start` (+ `resources_discover`)   | daemon health/start; `resources_discover` can return `skillPaths` |
| UserPromptSubmit | `before_agent_start` → `{message}` | inject `additional_context` (recall + re-injected rules) as a message — the direct analogue of Claude's `additionalContext`. Appending to `systemPrompt` is available but deferred (see increments) |
| PreToolUse   | `tool_call` → `{block, reason}`            | TOOL_RULE deny + pitfall directive expressible as `block`+`reason`. **No `updatedInput`** — redirect *rewrite* mode is not portable |
| PostToolUse  | `tool_result` → `{content, details, isError}` | transform tool results |
| Stop         | `agent_end` / `turn_end`                   | guardian signal check + tailor reflection capture |
| PreCompact   | `session_before_compact` / `session_compact` | export transcript for learning extraction |

Sessions live at `~/.pi/agent/sessions/*.jsonl` (analogous to `~/.codex/sessions`).
Extensions are registered in `~/.pi/agent/settings.json` `extensions[]` (file
paths) or `packages[]` (npm).

## Goals

- A dynamic memory loop in pi: recall on prompt, capture on stop, daemon health
  on session start, transcript export on compact (the MVP).
- Do it without duplicating hook logic in TypeScript — logic stays in Python,
  single source of truth.
- Generalize the execution model so the daemon becomes the warm execution path
  for **all** harnesses, and the CLI becomes a thin client.
- Incremental path to full parity (tool gating, post-tool transform, skills).

## Non-goals

- Porting the redirect **rewrite** mode (pi `tool_call` exposes only `block`,
  not `updatedInput`). rg-style flag fixes degrade to block-with-explanation.
- Re-implementing recall/ranking/guardian/tailor logic in TS.
- Changing Claude/Codex behavior (the refactor must be behavior-preserving for
  them).

## Architecture

### 1. One logic core, two transports

Today each hook's logic lives in `simba/hooks/<event>.py` as
`main(payload) -> json_string`, where the string is already a *Claude-shaped*
envelope. We split logic from wire-format:

- **Canonical event + payload** — normalized across harnesses:
  `session_start`, `prompt_submit`, `pre_tool`, `post_tool`, `stop`,
  `pre_compact`.
- **Canonical result** — `{ additional_context?: str, block?: {reason: str},
  transform?: {...} }`. Harness-agnostic.
- Each `hooks/<event>.py` exposes a pure `run(payload: dict) -> CanonicalResult`
  (the logic). The existing `main()` becomes a thin Claude adapter over `run()`
  so Claude/Codex behavior is preserved.

Proposed layout:

```
src/simba/harness/
  __init__.py
  core.py            # CanonicalEvent / CanonicalResult types; dispatch(event, payload) -> CanonicalResult
  adapters/
    claude.py        # native event ⇄ canonical; CanonicalResult -> Claude envelope (moves logic out of _io.py)
    codex.py         # + PermissionRequest; second-based timeouts
    pi.py            # native pi event ⇄ canonical; CanonicalResult -> pi event-result
```

`dispatch(event, payload)` normalizes input, calls `hooks.<event>.run`, and
returns the **canonical** result. Rendering to a native envelope is the
transport's job, not the core's — so the core (and the daemon endpoint) stay
harness-agnostic.

The same core is reachable two ways:

- **Inline** — `simba hook <event>` imports the core, runs in-process. Works
  with the daemon down (graceful degradation). This is today's path.
- **Daemon** — a new `POST /hook/{event}` FastAPI endpoint runs the *same*
  `dispatch()` with warm state (no Python startup, no embedder reload) and
  returns the canonical result as JSON.

The CLI becomes a thin client: try the daemon first, fall back to inline. This
generalizes the existing `_memory_client` try-daemon pattern, except the
fallback **runs the logic inline** instead of returning empty.

### 2. The `POST /hook/{event}` endpoint (keystone)

A single FastAPI route runs `dispatch(event, json_body)` and returns the
canonical result. It serves **both** the thin CLI and the pi extension, which
is what keeps the pi extension free of logic. Added to `simba/memory/server.py`
alongside the existing `/recall`, `/store`, `/health` routes.

### 3. The pi bridge extension (one TS shim, marshalling only)

A single bundled `simba.ts`:

```ts
export default function (pi: ExtensionAPI) {
  pi.on("session_start",        …)  // daemon health/start + memory count via ctx.ui
  pi.on("before_agent_start",   …)  // recall + rules -> return { message }
  pi.on("agent_end",            …)  // signal check + reflection capture
  pi.on("session_before_compact", …) // transcript export for extraction
}
```

Each handler builds the canonical payload, prefers the daemon
(`POST http://localhost:8741/hook/<event>`), falls back to spawning
`simba hook <event>` with JSON on stdin, then applies the canonical result to
pi's event-result object. No recall/ranking/guardian logic in TS — it only
marshals JSON and maps `additional_context`/`block` onto pi's shapes. This file
is the single concession to "no Node.js", framed like `.codex/hooks.json`: a
generated resource, not a home for logic.

### 4. Install / distribution

`simba pi-install [--remove]` (mirrors `codex-install`):

- Writes bundled `simba.ts` to `~/.pi/agent/extensions/simba.ts`.
- Idempotently registers it in `~/.pi/agent/settings.json` `extensions[]`
  (read-modify-write, like the Codex `[features] hooks` toggle).
- Optionally adds simba's skills dir to settings `skills[]`.
- `--remove` reverses both.

Default scope: **global** (`~/.pi/agent/`), matching how the user runs pi.
Distribution: **bundle the `.ts`** in the simba package (no separate npm
package).

### 5. Config (all via `simba config`)

A `pi` section (`@configurable`): `pi.enabled` (bool),
`pi.extension_path` (str), reusing the existing daemon URL/port config. No
hidden constants or env-only values.

## Documentation & metadata updates

Shipping a third runtime is not done until the docs say so. Each increment
updates the docs that describe runtime support:

- **`README.md`** — the runtimes badge (`Claude Code + Codex` → add pi), the
  tagline ("Claude Code _and_ Codex"), and the install section (document
  `simba pi-install` alongside `simba install` / `simba codex-install`). Add a
  pi-support section parallel to the existing `#codex-support` anchor.
- **`.claude/rules/CORE_INSTRUCTIONS.md`** — the Workflow section's hook
  inventory (lines listing "6 Claude Code hooks" and "Codex hooks") gains a pi
  line. This file is **SIMBA-marker-managed**: run `simba markers audit` after
  editing and keep any `SIMBA:core` blocks intact.
- **`CLAUDE.md`** (project) — the Repository Overview / Architecture notes that
  mention the hook system gain pi as a supported runtime.
- **`docs/plans/README.md`** — add the spec 23 index entry.
- **`src/simba/__main__.py`** — the top-of-file CLI usage docstring lists every
  `simba` subcommand; add `simba pi-install` (and `--remove`) so `simba` help is
  accurate.

These land **with the code in the same increment**, not as a follow-up — a
runtime that works but isn't documented reads as unsupported.

## Testing (TDD)

Pure-Python tests carry the weight:

- **Core** — event normalization and result mapping for each canonical event;
  Claude/Codex adapters produce byte-identical envelopes to today (regression
  guard on behavior-preservation).
- **Endpoint** — `POST /hook/{event}` via FastAPI `TestClient` returns the
  canonical result for each event.
- **CLI transport** — daemon-up routes to HTTP; daemon-down falls back to
  inline (mock the daemon).
- **pi contract** — a golden test pinning the canonical payload the extension
  sends and the pi-shaped result it applies.
- **Manual smoke** — `pi -p "…"` in a repo with the extension installed;
  confirm recall injection appears and a reflection is captured on stop.

## MVP → parity increments

- **v1 (MVP):** `session_start`, `before_agent_start` (recall + rules),
  `agent_end` (signal + capture), `session_before_compact` (export). The memory
  loop.
- **v2:** `tool_call` gate — TOOL_RULE deny + pitfall directive as
  `block`+`reason`; document that redirect rewrite mode degrades to
  block-with-explanation.
- **v3:** `tool_result` PostToolUse transform; `resources_discover` skill
  registration; optional `before_agent_start` `systemPrompt` append for guardian
  core-rules (vs the MVP's message-only injection).

## Risks / open questions

- **Hot-path latency.** `before_agent_start` fires every user turn. The daemon
  HTTP path keeps it warm; the inline fallback pays Python startup. Acceptable
  for MVP; measure before optimizing.
- **Behavior-preservation for Claude/Codex.** The refactor moves envelope
  rendering out of `_io.py`/`main()` into adapters. The byte-identical
  regression tests are the gate; this lands first, before any pi code.
- **pi version drift.** The extension contract is pinned to pi 0.78.0
  (`@earendil-works/pi-coding-agent` 0.51.3 in the verified install). Event names
  could change; the golden contract test will catch breakage at upgrade time.
