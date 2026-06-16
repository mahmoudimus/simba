# 24 — pi tool-gating (v2: canonicalize PreToolUse + wire pi `tool_call`)

**Date:** 2026-06-16
**Status:** approved scope (v2.0), pre-build
**Builds on:** spec 23 (pi harness MVP). Branch `feat/pi-tool-gating` off `main`.

## Goal

Give pi a **tool gate**: before a tool runs, simba can **block** a forbidden
command or **silently rewrite** it — the same protections Claude Code/Codex get
from the `PreToolUse` hook. This is the v2 increment on top of the shipped MVP
memory loop.

## Grounding: what `PreToolUse` actually does (src/simba/hooks/pre_tool_use.py)

`main()` produces **three** output shapes:

1. **Redirect** (Bash only, runs first, returns early):
   - rewrite mode → `_io.pretool_rewrite(command, reason)` (allow + `updatedInput.command`)
   - deny mode → `_io.pretool_deny(reason)` (permissionDecision deny + reason)
2. **Context injection** (`_io.context("PreToolUse", combined)`): context-low
   warning + **TOOL_RULE warning** + truth-constraint warning + **pitfall
   directive** + thinking-block recall. These are *warnings the model reads*,
   **not blocks**.
3. **Empty** (`_io.empty("PreToolUse")`) when nothing fired.

Key fact: **only the redirect hard-blocks.** TOOL_RULE matches and the pitfall
directive are context injections on Claude/Codex. (The only place a TOOL_RULE
*denies* today is the Codex-only `PermissionRequest` hook, at
`permission_deny_similarity` = 0.78.)

## Grounding: pi `tool_call` (verified vs `@earendil-works/pi-coding-agent` 0.78.0)

- Event: per-tool discriminated union, e.g. `BashToolCallEvent { toolName: "bash";
  input: BashToolInput }` — `input` is **typed and mutable** (`input.command` for bash).
- Result: `ToolCallEventResult { block?: boolean; reason?: string }`, and the
  doc says *"to modify arguments, mutate `event.input` in place."*

So pi `tool_call` can **block** (`{block, reason}`) **or rewrite** (mutate
`event.input`). It has **no context-injection channel** — a warning has nowhere
to go.

## v2.0 scope (this build)

pi `tool_call` enforces, all command/tool-based (no agent reasoning needed):

| PreToolUse decision | Claude/Codex (unchanged) | pi `tool_call` |
|---|---|---|
| redirect **deny** | `pretool_deny(reason)` | `{block: true, reason}` |
| redirect **rewrite** | `pretool_rewrite(cmd, reason)` | mutate `event.input.command = cmd` (silent rewrite) |
| **strong TOOL_RULE** (≥ `permission_deny_similarity`) | context warning (as today) | `{block: true, reason}` (escalated, like Codex PermissionRequest) |
| weak TOOL_RULE / context-low / recall | context injection | **dropped** (no channel) |

**v2.1 (deferred):** the **pitfall/doctrine** gate also blocking pi. It keys on
the agent's *reasoning*, which `tool_call` doesn't carry, so the bridge must pull
recent assistant text from `ctx.sessionManager` and pass it as `thinking`. Its
own slice after v2.0.

## Canonical model changes

`CanonicalResult` already has `additional_context`, `block_reason`, `transform`.
Add ONE field:

```python
# A directive that context-capable harnesses (Claude/Codex) inject as
# additionalContext (already included in additional_context) but block-only
# harnesses (pi tool_call) must enforce as a hard block. Populated for a strong
# TOOL_RULE match. Claude/Codex render IGNORES it (byte-identical).
escalated_block: str | None = None
```

- `block_reason` → hard block on **all** harnesses (redirect deny).
- `transform` → `{"command": cmd, "reason": r}` (redirect rewrite).
- `additional_context` → context injection (Claude/Codex render; pi drops).
- `escalated_block` → block only on block-only harnesses (pi); Claude/Codex ignore.

## Phases

### Phase A — canonicalize `pre_tool` (load-bearing, byte-identical)
- `pre_tool_use.py`: extract `run(payload) -> CanonicalResult`; `main()` = a
  delegate `claude.render("PreToolUse", run(payload))`. Map:
  - redirect rewrite → `CanonicalResult(transform={"command":…, "reason":…})`
  - redirect deny → `CanonicalResult(block_reason=…)`
  - context path → `CanonicalResult(additional_context=combined)`; empty → `CanonicalResult()`
  - strong TOOL_RULE → also set `escalated_block` (reuse the
    `permission_request` strong-match logic / a shared helper at
    `permission_deny_similarity`). The TOOL_RULE warning still goes into
    `additional_context` exactly as today.
- `core._EVENT_MODULES["pre_tool"] = "simba.hooks.pre_tool_use"`.
- `claude.NATIVE_TO_CANONICAL["PreToolUse"] = "pre_tool"`.
- `claude.render` PreToolUse branch: `transform` → `_io.pretool_rewrite`;
  `block_reason` → `_io.pretool_deny`; else `additional_context` →
  `_io.context`/`_io.empty`. (Add the missing `transform` handling to render.)
- Add `escalated_block` to `CanonicalResult`, the `/hook/{event}` response, and
  `hook-canonical` output.
- **Characterization tests** pinning byte-identical `main()` output for: redirect
  deny, redirect rewrite, a context case (TOOL_RULE warning present), and the
  empty case. These are the gate — Claude/Codex must not change.

### Phase B — CLI/daemon routing
- `pre_tool` already flows through `dispatch()` once it's in `_EVENT_MODULES`.
- `simba hook PreToolUse` now routes through the canonical path (it's in
  `NATIVE_TO_CANONICAL`); the characterization test guards byte-identical.
- Confirm `POST /hook/pre_tool` returns the new fields.

### Phase C — pi bridge `tool_call`
- `simba.ts`: `pi.on("tool_call", async (e, ctx) => {…})`:
  - Map pi tool name → Claude convention (`bash`→`Bash`, `edit`→`Edit`,
    `write`→`Write`, `read`→`Read`, …) and `e.input` → `tool_input`.
  - POST `/hook/pre_tool` with `{tool_name, tool_input, cwd}`.
  - Apply: `transform` → mutate `e.input.command = transform.command`, note
    `[simba: rewrote → …]`, allow; `block_reason` or `escalated_block` →
    `return {block: true, reason}`, note `[simba: blocked — …]`; else allow.
  - Extend the TS `Canonical` interface with `transform` + `escalated_block`.
- No new pi config: the extension always wires `tool_call` and the **daemon is
  authoritative** — gating reuses the existing `hooks.redirect_enabled` /
  `hooks.rule_check_enabled` / `permission_deny_similarity`. (Decided during the
  build: a separate `pi.tool_gate_enabled` would be redundant and the TS extension
  can't cheaply read daemon config per-call.)

### Phase D — docs
- No new daemon gating config (reuses `redirect_*` and
  `permission_deny_similarity`).
- README pi section + CHANGELOG (v0.9.0 — new pi capability) + this spec's index row.

## Testing
- Phase A characterization (byte-identical) — the binding gate.
- `escalated_block` populated only at ≥ `permission_deny_similarity`; unit-tested
  with a fake TOOL_RULE recall.
- `/hook/pre_tool` endpoint returns block/transform/escalated_block correctly.
- pi `tool_call` contract: a golden test of the payload the bridge sends + the
  result it applies (block vs input-mutation).
- Manual: `pi` in a repo with a redirect rule (e.g. `rg -rn`) — confirm the
  command is silently rewritten; with a TOOL_RULE — confirm the call is blocked.

## Non-goals
- Pitfall/doctrine gate on pi (v2.1).
- Changing Claude/Codex `PreToolUse` behavior in any way (byte-identical is the gate).
- Canonicalizing `PostToolUse` / `PermissionRequest` (later).
