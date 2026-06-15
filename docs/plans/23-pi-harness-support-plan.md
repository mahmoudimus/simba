# pi coding-agent harness support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the simba memory loop (recall-on-prompt, capture-on-stop, daemon health, transcript export) to the pi coding-agent harness, via a refactor that turns the daemon into a warm execution path for all harnesses and the CLI into a thin client.

**Architecture:** Each lifecycle hook's logic moves behind a `run(payload) -> CanonicalResult` function. A harness-agnostic `dispatch()` runs it; one Claude/Codex adapter renders the canonical result to today's exact stdout envelope (byte-identical). The same `dispatch()` is exposed two ways — inline via the `simba hook` / `simba hook-canonical` CLI, and over HTTP via a new daemon `POST /hook/{event}` endpoint. A thin bundled `simba.ts` pi extension subscribes to pi's EventBus, calls the daemon (falling back to the CLI), and applies the canonical result to pi's event-result shapes. No recall/ranking/guardian logic lives in TypeScript.

**Tech Stack:** Python 3.12, FastAPI (daemon), httpx, pytest (`uv run --no-sync pytest`), ruff; TypeScript (pi extension, `@mariozechner/pi-coding-agent` ≥0.51.3 / pi 0.78.0), Node ≥22 (global `fetch`).

**Spec:** `docs/plans/23-pi-harness-support.md`

---

## Scope

This plan covers the **MVP (v1)**: the four side-effect/recall events that make
the memory loop work — `session_start`, `prompt_submit` (pi
`before_agent_start`), `stop` (pi `agent_end`), `pre_compact` (pi
`session_before_compact`). Tool gating (`tool_call`), PostToolUse
(`tool_result`), and skill registration (`resources_discover`) are **v2/v3** and
get their own plan when reached (outlined at the end). The Phase 0 refactor is
deliberately scoped to only the four MVP events; the remaining hooks
(`PreToolUse`, `PostToolUse`, `PermissionRequest`) keep their current
`main()`-based path until v2.

## File Structure

**Create:**
- `src/simba/harness/__init__.py` — package marker + public exports
- `src/simba/harness/core.py` — `CanonicalResult` dataclass, `dispatch()`, canonical-event → module map
- `src/simba/harness/adapters/__init__.py`
- `src/simba/harness/adapters/claude.py` — Claude/Codex native↔canonical name map + `render(native_event, result) -> str`
- `src/simba/pi/__init__.py` — pi config section (`@configurable("pi")`)
- `src/simba/pi/extension/simba.ts` — the bundled pi bridge extension (TypeScript resource)
- `tests/harness/test_core.py`
- `tests/harness/test_claude_adapter.py`
- `tests/harness/test_hook_endpoint.py`
- `tests/harness/test_cli_transport.py`
- `tests/pi/test_pi_config.py`
- `tests/pi/test_pi_install.py`

**Modify:**
- `src/simba/hooks/session_start.py` — add `run()`, `main()` delegates
- `src/simba/hooks/user_prompt_submit.py` — add `run()`, `main()` delegates
- `src/simba/hooks/stop.py` — add `run()`, `main()` delegates
- `src/simba/hooks/pre_compact.py` — add `run()`, `main()` delegates
- `src/simba/hooks/config.py` — add `dispatch_via_daemon` flag
- `src/simba/memory/routes.py` — add `POST /hook/{event}`
- `src/simba/__main__.py` — `hook-canonical` subcommand, daemon-first routing in `_cmd_hook`, `_cmd_pi_install`, usage docstring, command dispatch
- `pyproject.toml` — package-data for `simba/pi/extension/*.ts`
- `README.md`, `.claude/rules/CORE_INSTRUCTIONS.md`, `CLAUDE.md`, `docs/plans/README.md` — docs

---

## Phase 0 — Canonical core + Claude/Codex adapter (behavior-preserving)

This phase introduces the seam and proves Claude/Codex output is unchanged. The
regression guard is a **characterization test**: it pins each hook's current
`main()` output, stays green through the refactor.

### Task 0.1: Characterization test — pin current hook output

**Files:**
- Test: `tests/harness/test_claude_adapter.py`

- [ ] **Step 1: Write the characterization test against current `main()`**

```python
"""Pins Claude/Codex hook envelopes so the canonical refactor stays byte-identical."""
from __future__ import annotations

import json

import simba.hooks.session_start
import simba.hooks.user_prompt_submit
import simba.hooks.stop
import simba.hooks.pre_compact


def test_user_prompt_submit_empty_prompt_envelope():
    out = simba.hooks.user_prompt_submit.main({"prompt": "", "cwd": "/tmp"})
    parsed = json.loads(out)
    assert parsed["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"


def test_stop_no_response_is_empty_object():
    out = simba.hooks.stop.main({"cwd": "/tmp"})
    assert json.loads(out) == {}


def test_pre_compact_missing_fields_suppresses_output():
    out = simba.hooks.pre_compact.main({})
    assert json.loads(out) == {"suppressOutput": True}


def test_session_start_returns_session_start_envelope():
    out = simba.hooks.session_start.main({"cwd": "/tmp"})
    assert json.loads(out)["hookSpecificOutput"]["hookEventName"] == "SessionStart"
```

- [ ] **Step 2: Run to verify it passes on current code**

Run: `uv run --no-sync pytest tests/harness/test_claude_adapter.py -v`
Expected: PASS (these pin existing behavior). If the daemon is down, recall
returns empty and envelopes are still well-formed.

- [ ] **Step 3: Commit**

```bash
git add tests/harness/test_claude_adapter.py
git commit -m "test(harness): characterize current hook envelopes pre-refactor"
```

### Task 0.2: `CanonicalResult` + `dispatch()`

**Files:**
- Create: `src/simba/harness/__init__.py`, `src/simba/harness/core.py`
- Test: `tests/harness/test_core.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import pytest

import simba.harness.core as core


def test_canonical_result_defaults():
    r = core.CanonicalResult()
    assert r.additional_context == ""
    assert r.suppress_output is False
    assert r.block_reason is None


def test_dispatch_unknown_event_raises():
    with pytest.raises(KeyError):
        core.dispatch("not_an_event", {})


def test_dispatch_prompt_submit_returns_canonical(monkeypatch):
    # No daemon needed: recall failure returns empty, run() still produces a result.
    r = core.dispatch("prompt_submit", {"prompt": "", "cwd": "/tmp"})
    assert isinstance(r, core.CanonicalResult)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/harness/test_core.py -v`
Expected: FAIL with `ModuleNotFoundError: simba.harness.core` (and, after the
module exists but before `run()` is added in 0.3, `AttributeError: run`).

- [ ] **Step 3: Implement `core.py`**

```python
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
    # v2: "pre_tool", "post_tool"
}


@dataclasses.dataclass
class CanonicalResult:
    """Harness-agnostic hook result."""

    additional_context: str = ""
    suppress_output: bool = False
    # v2 fields (defined for forward-compat; unused in MVP):
    block_reason: str | None = None
    transform: dict | None = None


def dispatch(event: str, payload: dict) -> CanonicalResult:
    """Run the canonical hook for ``event``. Raises KeyError if unknown."""
    module = importlib.import_module(_EVENT_MODULES[event])
    return module.run(payload)
```

```python
# src/simba/harness/__init__.py
from simba.harness.core import CanonicalResult, dispatch

__all__ = ["CanonicalResult", "dispatch"]
```

- [ ] **Step 4: Run — still fails on `run()` until 0.3**

Run: `uv run --no-sync pytest tests/harness/test_core.py::test_dispatch_unknown_event_raises -v`
Expected: PASS for the unknown-event and defaults tests; the `prompt_submit`
test fails until Task 0.3 adds `run()`.

- [ ] **Step 5: Commit**

```bash
git add src/simba/harness/__init__.py src/simba/harness/core.py tests/harness/test_core.py
git commit -m "feat(harness): canonical result + dispatch scaffold"
```

### Task 0.3: Claude/Codex adapter

**Files:**
- Create: `src/simba/harness/adapters/__init__.py`, `src/simba/harness/adapters/claude.py`
- Test: `tests/harness/test_claude_adapter.py` (extend)

- [ ] **Step 1: Add failing render tests**

```python
# append to tests/harness/test_claude_adapter.py
import simba.harness.adapters.claude as claude
from simba.harness.core import CanonicalResult


def test_render_user_prompt_submit_with_context():
    out = claude.render("UserPromptSubmit", CanonicalResult(additional_context="hi"))
    parsed = json.loads(out)
    assert parsed["hookSpecificOutput"]["additionalContext"] == "hi"


def test_render_stop_with_context_uses_stop_reason():
    out = claude.render("Stop", CanonicalResult(additional_context="WARN"))
    assert json.loads(out) == {"stopReason": "WARN"}


def test_render_stop_empty_is_empty_object():
    assert json.loads(claude.render("Stop", CanonicalResult())) == {}


def test_render_pre_compact_suppress():
    out = claude.render("PreCompact", CanonicalResult(suppress_output=True))
    assert json.loads(out) == {"suppressOutput": True}


def test_native_to_canonical_map():
    assert claude.NATIVE_TO_CANONICAL["UserPromptSubmit"] == "prompt_submit"
    assert claude.NATIVE_TO_CANONICAL["PreCompact"] == "pre_compact"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/harness/test_claude_adapter.py -v`
Expected: FAIL (`ModuleNotFoundError: simba.harness.adapters.claude`).

- [ ] **Step 3: Implement `claude.py`**

```python
"""Render a CanonicalResult to the Claude Code / Codex stdout envelope.

Claude and Codex share envelope shapes for the four MVP events, so one adapter
serves both.  Output is byte-identical to the pre-refactor hooks.<event>.main().
"""
from __future__ import annotations

import json

import simba.hooks._io
from simba.harness.core import CanonicalResult

# Claude/Codex native event name -> canonical event
NATIVE_TO_CANONICAL = {
    "SessionStart": "session_start",
    "UserPromptSubmit": "prompt_submit",
    "Stop": "stop",
    "PreCompact": "pre_compact",
    # v2: "PreToolUse": "pre_tool", "PostToolUse": "post_tool",
    #     "PermissionRequest": "permission_request",
}


def render(event: str, result: CanonicalResult) -> str:
    """Render ``result`` for Claude/Codex ``event`` as a JSON string."""
    if event in ("SessionStart", "UserPromptSubmit"):
        return simba.hooks._io.context(event, result.additional_context)
    if event == "PreCompact":
        if result.suppress_output:
            return json.dumps({"suppressOutput": True})
        return simba.hooks._io.context("PreCompact", result.additional_context)
    if event == "Stop":
        if result.additional_context:
            return json.dumps({"stopReason": result.additional_context})
        return json.dumps({})
    return simba.hooks._io.context(event, result.additional_context)
```

```python
# src/simba/harness/adapters/__init__.py  (empty marker)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --no-sync pytest tests/harness/test_claude_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simba/harness/adapters/ tests/harness/test_claude_adapter.py
git commit -m "feat(harness): Claude/Codex canonical render adapter"
```

### Task 0.4: Refactor the four hooks to `run()` + delegating `main()`

Mechanical transform per module: move the existing `main()` body into
`run(payload) -> CanonicalResult` (returning the canonical fields instead of
calling `_io`/`json.dumps`), and make `main()` a one-line delegate. Side-effects
(daemon poll, tailor capture, transcript export, stderr prints) stay inside
`run()`. **Every path must come from `payload`, never `pathlib.Path.cwd()`** —
verified by Task 1.3.

- [ ] **Step 1: `user_prompt_submit.py`** — replace the tail of `main()`:

Change the end of the current `main()` (the `combined`/print/return block) so
the function becomes:

```python
def run(hook_input: dict) -> "CanonicalResult":
    from simba.harness.core import CanonicalResult
    # ... existing body that builds `parts` (unchanged) ...
    combined = "\n\n".join(parts)
    if combined:
        tokens = len(combined) // 4
        tags = f"~{tokens} tokens"
        if core_blocks:
            tags += " | ✓ rules"
        combined += f"\n[simba: {tags}]"
        print(f"[simba: {tags}]", file=sys.stderr)
    return CanonicalResult(additional_context=combined)


def main(hook_input: dict) -> str:
    import simba.harness.adapters.claude as claude
    return claude.render("UserPromptSubmit", run(hook_input))
```

- [ ] **Step 2: `stop.py`** — `run()` returns canonical, `main()` delegates:

```python
def run(hook_input: dict) -> "CanonicalResult":
    from simba.harness.core import CanonicalResult
    cwd_str = hook_input.get("cwd")
    cwd = pathlib.Path(cwd_str) if cwd_str else None
    parts: list[str] = []
    response = hook_input.get("response", "")
    if response:
        signal_result = simba.guardian.check_signal.main(response=response, cwd=cwd)
        if signal_result:
            parts.append(signal_result)
    simba.tailor.hook.process_hook(json.dumps(hook_input))
    return CanonicalResult(additional_context="\n\n".join(parts))


def main(hook_input: dict) -> str:
    import simba.harness.adapters.claude as claude
    return claude.render("Stop", run(hook_input))
```

- [ ] **Step 3: `session_start.py`** — wrap existing body. The current `main()`
ends with `return simba.hooks._io.context("SessionStart", combined)`. Rename the
body to `run()` returning `CanonicalResult(additional_context=combined)`, and add:

```python
def main(hook_input: dict) -> str:
    import simba.harness.adapters.claude as claude
    return claude.render("SessionStart", run(hook_input))
```

- [ ] **Step 4: `pre_compact.py`** — the current `main()` returns either
`json.dumps({"suppressOutput": True})` (missing/empty cases) or
`simba.hooks._io.context(...)`. Map those to canonical:

```python
def run(hook_input: dict) -> "CanonicalResult":
    from simba.harness.core import CanonicalResult
    # ... existing guards: on each early `return json.dumps({"suppressOutput": True})`
    #     return CanonicalResult(suppress_output=True) instead ...
    # ... existing success path builds `combined` ...
    return CanonicalResult(additional_context=combined)


def main(hook_input: dict) -> str:
    import simba.harness.adapters.claude as claude
    return claude.render("PreCompact", run(hook_input))
```

- [ ] **Step 5: Run the characterization + core tests**

Run: `uv run --no-sync pytest tests/harness/ -v`
Expected: PASS — Task 0.1 envelopes unchanged, Task 0.2 `prompt_submit` dispatch
now returns a `CanonicalResult`.

- [ ] **Step 6: Run the full suite (no behavior regression anywhere)**

Run: `uv run --no-sync pytest -q && uv run --no-sync ruff check src/ tests/`
Expected: all pass, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/simba/hooks/
git commit -m "refactor(hooks): extract run() behind canonical dispatch (byte-identical main)"
```

---

## Phase 1 — Daemon `POST /hook/{event}` endpoint

### Task 1.1: Endpoint returns canonical result

**Files:**
- Modify: `src/simba/memory/routes.py`
- Test: `tests/harness/test_hook_endpoint.py`

- [ ] **Step 1: Write the failing test (FastAPI TestClient)**

```python
from __future__ import annotations

import fastapi.testclient

import simba.memory.routes


def _client() -> fastapi.testclient.TestClient:
    app = fastapi.FastAPI()
    app.include_router(simba.memory.routes.router)
    return fastapi.testclient.TestClient(app)


def test_hook_endpoint_prompt_submit_returns_canonical_fields():
    resp = _client().post("/hook/prompt_submit", json={"prompt": "", "cwd": "/tmp"})
    assert resp.status_code == 200
    body = resp.json()
    assert "additional_context" in body
    assert "suppress_output" in body


def test_hook_endpoint_unknown_event_404():
    resp = _client().post("/hook/bogus", json={})
    assert resp.status_code == 404
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/harness/test_hook_endpoint.py -v`
Expected: FAIL with 404 on `/hook/prompt_submit` (route not yet defined).

- [ ] **Step 3: Add the route to `routes.py`**

Add near the other `@router.post` definitions. It is a **sync** `def` so FastAPI
runs it in a threadpool — `dispatch()` makes a blocking httpx loopback to
`/recall`, which must not stall the event loop.

```python
import fastapi

import simba.harness.core


@router.post("/hook/{event}")
def run_hook(event: str, payload: dict) -> dict:
    """Run a canonical hook and return its CanonicalResult as JSON.

    Sync handler: dispatch() may make a blocking loopback to /recall, so FastAPI
    offloads it to a threadpool instead of blocking the event loop.
    """
    try:
        result = simba.harness.core.dispatch(event, payload)
    except KeyError:
        raise fastapi.HTTPException(status_code=404, detail=f"unknown hook event: {event}")
    return {
        "additional_context": result.additional_context,
        "suppress_output": result.suppress_output,
        "block_reason": result.block_reason,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --no-sync pytest tests/harness/test_hook_endpoint.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simba/memory/routes.py tests/harness/test_hook_endpoint.py
git commit -m "feat(daemon): POST /hook/{event} runs canonical dispatch"
```

### Task 1.2: cwd-isolation test (daemon must honor payload cwd)

**Files:**
- Test: `tests/harness/test_hook_endpoint.py` (extend)

- [ ] **Step 1: Write the test** — `stop` capture must write under the payload
cwd, not the test process cwd.

```python
import pathlib


def test_stop_capture_uses_payload_cwd(tmp_path):
    resp = _client().post(
        "/hook/stop",
        json={"cwd": str(tmp_path), "transcript_path": "", "response": "done [✓ rules]"},
    )
    assert resp.status_code == 200
    # tailor writes reflections under <cwd>/.simba/, never under the daemon's cwd
    stray = pathlib.Path.cwd() / ".simba" / "tailor"
    # If anything was written, it must be under tmp_path — assert no stray write here
    assert not (stray.exists() and any(stray.iterdir())) or (tmp_path / ".simba").exists()
```

- [ ] **Step 2: Run**

Run: `uv run --no-sync pytest tests/harness/test_hook_endpoint.py::test_stop_capture_uses_payload_cwd -v`
Expected: PASS. If it FAILS, a hook's `run()` is using `pathlib.Path.cwd()` —
fix that `run()` to derive its path from `payload["cwd"]` (Phase 0 requirement).

- [ ] **Step 3: Commit**

```bash
git add tests/harness/test_hook_endpoint.py
git commit -m "test(daemon): hook dispatch honors payload cwd, not process cwd"
```

---

## Phase 2 — CLI thin client

### Task 2.1: `simba hook-canonical <event>` (inline-or-daemon canonical output)

**Files:**
- Modify: `src/simba/__main__.py`
- Test: `tests/harness/test_cli_transport.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import json
import subprocess
import sys


def _run_cli(args: list[str], stdin: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "simba", *args],
        input=stdin, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_hook_canonical_prompt_submit_emits_canonical_json():
    out = _run_cli(["hook-canonical", "prompt_submit"], json.dumps({"prompt": "", "cwd": "/tmp"}))
    body = json.loads(out)
    assert "additional_context" in body and "suppress_output" in body
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/harness/test_cli_transport.py -v`
Expected: FAIL (`Unknown command: hook-canonical`, non-zero exit).

- [ ] **Step 3: Implement `_cmd_hook_canonical` + a shared transport helper in `__main__.py`**

```python
def _hook_via_daemon(event: str, payload: dict) -> "simba.harness.core.CanonicalResult | None":
    """Try the daemon's /hook/{event}; return None on any failure (caller runs inline)."""
    import httpx

    import simba.harness.core
    import simba.hooks._memory_client

    url = f"{simba.hooks._memory_client.daemon_url()}/hook/{event}"
    try:
        resp = httpx.post(url, json=payload, timeout=3.0)
        if resp.status_code == 200:
            b = resp.json()
            return simba.harness.core.CanonicalResult(
                additional_context=b.get("additional_context", ""),
                suppress_output=b.get("suppress_output", False),
                block_reason=b.get("block_reason"),
            )
    except (httpx.HTTPError, ValueError):
        pass
    return None


def _dispatch_canonical(event: str, payload: dict) -> "simba.harness.core.CanonicalResult":
    """Daemon-first, inline fallback. Honors hooks.dispatch_via_daemon."""
    import simba.config
    import simba.harness.core
    import simba.hooks.config

    _ = simba.hooks.config
    if simba.config.load("hooks").dispatch_via_daemon:
        result = _hook_via_daemon(event, payload)
        if result is not None:
            return result
    return simba.harness.core.dispatch(event, payload)


def _cmd_hook_canonical(args: list[str]) -> int:
    """Run a canonical hook and print its CanonicalResult as JSON."""
    if not args:
        print("Usage: simba hook-canonical <canonical_event>", file=sys.stderr)
        return 1
    event = args[0]
    payload: dict = {}
    try:
        raw = sys.stdin.read()
        if raw:
            payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        result = _dispatch_canonical(event, payload)
    except KeyError:
        print(f"Unknown canonical event: {event}", file=sys.stderr)
        return 1
    print(json.dumps({
        "additional_context": result.additional_context,
        "suppress_output": result.suppress_output,
        "block_reason": result.block_reason,
    }))
    return 0
```

Wire it into the command dispatch in `main()` (next to the existing
`elif command == "hook":`):

```python
elif command == "hook-canonical":
    return _cmd_hook_canonical(rest)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --no-sync pytest tests/harness/test_cli_transport.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simba/__main__.py tests/harness/test_cli_transport.py
git commit -m "feat(cli): hook-canonical subcommand (daemon-first, inline fallback)"
```

### Task 2.2: Route `simba hook <NativeEvent>` through canonical for MVP events

**Files:**
- Modify: `src/simba/__main__.py` (`_cmd_hook`)
- Modify: `src/simba/hooks/config.py` (add `dispatch_via_daemon`)
- Test: `tests/harness/test_cli_transport.py` (extend)

- [ ] **Step 1: Add the config flag**

In `src/simba/hooks/config.py`, add to `HooksConfig` (under "Memory client"):

```python
    # Route canonicalized hooks through the daemon when it's up (warm path);
    # fall back to running them inline. Output is byte-identical either way.
    dispatch_via_daemon: bool = True
```

- [ ] **Step 2: Write the failing test** — native `Stop` with no response still
emits the exact `{}` envelope, regardless of transport.

```python
def test_native_stop_envelope_unchanged():
    out = _run_cli(["hook", "Stop"], json.dumps({"cwd": "/tmp"}))
    assert json.loads(out) == {}
```

- [ ] **Step 3: Run — passes today (legacy path), guards the change**

Run: `uv run --no-sync pytest tests/harness/test_cli_transport.py::test_native_stop_envelope_unchanged -v`
Expected: PASS now; must stay PASS after the change.

- [ ] **Step 4: Update `_cmd_hook` to use the canonical path for MVP events**

```python
def _cmd_hook(args: list[str]) -> int:
    """Dispatch a hook event. Called by Claude Code / Codex, not users."""
    if not args:
        print("Usage: simba hook <event>", file=sys.stderr)
        print(f"Events: {', '.join(_HOOK_EVENTS)}", file=sys.stderr)
        return 1

    event = args[0]

    import simba.harness.adapters.claude as claude

    # Canonicalized (MVP) events: daemon-first, inline fallback, render to envelope.
    canonical = claude.NATIVE_TO_CANONICAL.get(event)
    if canonical is not None:
        payload: dict = {}
        try:
            raw = sys.stdin.read()
            if raw:
                payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
        result = _dispatch_canonical(canonical, payload)
        print(claude.render(event, result))
        return 0

    # Legacy path for not-yet-canonicalized events (PreToolUse/PostToolUse/PermissionRequest).
    module_name = _HOOK_EVENTS.get(event)
    if module_name is None:
        print(f"Unknown hook event: {event}", file=sys.stderr)
        return 1
    import importlib

    module = importlib.import_module(module_name)
    hook_data: dict = {}
    try:
        raw = sys.stdin.read()
        if raw:
            hook_data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    print(module.main(hook_data))
    return 0
```

- [ ] **Step 5: Run the transport + characterization + full suite**

Run: `uv run --no-sync pytest tests/harness/ -v && uv run --no-sync pytest -q`
Expected: PASS — native envelopes unchanged, both transports verified.

- [ ] **Step 6: Commit**

```bash
git add src/simba/__main__.py src/simba/hooks/config.py tests/harness/test_cli_transport.py
git commit -m "feat(cli): route MVP hooks through canonical dispatch (daemon-first)"
```

---

## Phase 3 — pi config section

### Task 3.1: `@configurable("pi")`

**Files:**
- Create: `src/simba/pi/__init__.py`
- Test: `tests/pi/test_pi_config.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import simba.config
import simba.pi  # registers the "pi" section


def test_pi_config_defaults():
    cfg = simba.config.load("pi")
    assert cfg.enabled is True
    assert cfg.extension_path.endswith("simba.ts")


def test_pi_section_is_registered():
    assert "pi" in simba.config.list_sections()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/pi/test_pi_config.py -v`
Expected: FAIL (`ModuleNotFoundError: simba.pi`).

- [ ] **Step 3: Implement `src/simba/pi/__init__.py`**

```python
"""pi coding-agent harness integration: config section."""
from __future__ import annotations

import dataclasses

import simba.config


@simba.config.configurable("pi")
@dataclasses.dataclass
class PiConfig:
    # Whether `simba install` also wires the pi extension.
    enabled: bool = True
    # Where the bundled bridge extension is written and registered.
    extension_path: str = "~/.pi/agent/extensions/simba.ts"
    # pi's settings.json (extensions[] registration target).
    settings_path: str = "~/.pi/agent/settings.json"
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --no-sync pytest tests/pi/test_pi_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simba/pi/__init__.py tests/pi/test_pi_config.py
git commit -m "feat(pi): @configurable pi section"
```

---

## Phase 4 — The pi bridge extension (TypeScript resource)

### Task 4.1: Bundle `simba.ts`

**Files:**
- Create: `src/simba/pi/extension/simba.ts`
- Modify: `pyproject.toml` (package-data)

- [ ] **Step 1: Write `simba.ts`**

```ts
/**
 * simba — memory loop bridge for the pi coding agent.
 *
 * Pure marshalling: each pi lifecycle event is forwarded to simba (daemon HTTP,
 * CLI fallback) and the canonical result is applied to pi's event result. No
 * recall/ranking/guardian logic lives here — it all runs in Python.
 */
import { spawn } from "node:child_process";
import type {
  ExtensionAPI,
  ExtensionContext,
  BeforeAgentStartEvent,
  AgentEndEvent,
} from "@mariozechner/pi-coding-agent";

const DAEMON = process.env.SIMBA_DAEMON_URL || "http://localhost:8741";

interface Canonical {
  additional_context?: string;
  suppress_output?: boolean;
  block_reason?: string | null;
}

function lastAssistantText(messages: Array<{ role?: string; content?: unknown }>): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role !== "assistant") continue;
    const c = m.content;
    if (typeof c === "string") return c;
    if (Array.isArray(c)) {
      return c
        .filter((p): p is { type: "text"; text: string } =>
          Boolean(p && typeof p === "object" && (p as { type?: string }).type === "text"))
        .map((p) => p.text)
        .join("\n");
    }
    return "";
  }
  return "";
}

function viaCli(event: string, payload: Record<string, unknown>): Promise<Canonical> {
  return new Promise((resolve) => {
    const child = spawn("simba", ["hook-canonical", event], { stdio: ["pipe", "pipe", "ignore"] });
    let out = "";
    child.stdout.on("data", (d: Buffer) => (out += d.toString()));
    child.on("close", () => {
      try {
        resolve(JSON.parse(out));
      } catch {
        resolve({});
      }
    });
    child.on("error", () => resolve({}));
    child.stdin.end(JSON.stringify(payload));
  });
}

async function callSimba(event: string, payload: Record<string, unknown>): Promise<Canonical> {
  try {
    const resp = await fetch(`${DAEMON}/hook/${event}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(3000),
    });
    if (resp.ok) return (await resp.json()) as Canonical;
  } catch {
    /* daemon down — fall back to the CLI */
  }
  return viaCli(event, payload);
}

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async (_e, ctx: ExtensionContext) => {
    const r = await callSimba("session_start", { cwd: ctx.cwd });
    if (r.additional_context && ctx.hasUI) ctx.ui.notify(r.additional_context, "info");
  });

  pi.on("before_agent_start", async (e: BeforeAgentStartEvent, ctx: ExtensionContext) => {
    const r = await callSimba("prompt_submit", { prompt: e.prompt, cwd: ctx.cwd });
    if (r.additional_context) {
      return {
        message: { customType: "simba-memory", content: r.additional_context, display: true },
      };
    }
  });

  pi.on("agent_end", async (e: AgentEndEvent, ctx: ExtensionContext) => {
    await callSimba("stop", {
      response: lastAssistantText(e.messages),
      cwd: ctx.cwd,
      transcript_path: ctx.sessionManager.getSessionFile() ?? "",
    });
  });

  pi.on("session_before_compact", async (_e, ctx: ExtensionContext) => {
    await callSimba("pre_compact", {
      cwd: ctx.cwd,
      transcript_path: ctx.sessionManager.getSessionFile() ?? "",
      session_id: ctx.sessionManager.getSessionId(),
    });
  });
}
```

- [ ] **Step 2: Add package-data to `pyproject.toml`**

Under the setuptools package-data config (alongside the existing `simba = [...]`
entry that ships GGUF/skill resources), add the `.ts` glob:

```toml
[tool.setuptools.package-data]
simba = ["pi/extension/*.ts"]
```

(Merge into the existing `simba = [ ... ]` list rather than duplicating the key.)

- [ ] **Step 3: Verify the resource is packaged**

Run: `uv run --no-sync python -c "import importlib.resources as r; print((r.files('simba')/'pi'/'extension'/'simba.ts').is_file())"`
Expected: `True`

- [ ] **Step 4: Commit**

```bash
git add src/simba/pi/extension/simba.ts pyproject.toml
git commit -m "feat(pi): bundle simba.ts bridge extension"
```

---

## Phase 5 — `simba pi-install`

### Task 5.1: Install/remove + settings.json registration

**Files:**
- Modify: `src/simba/__main__.py`
- Test: `tests/pi/test_pi_install.py`

- [ ] **Step 1: Write the failing test** (drive paths via a temp pi home)

```python
from __future__ import annotations

import json
import pathlib

import simba.__main__ as cli


def _pi_home(tmp_path, monkeypatch) -> pathlib.Path:
    home = tmp_path / ".pi" / "agent"
    monkeypatch.setattr(cli, "_pi_agent_home", lambda: home)
    return home


def test_pi_install_writes_extension_and_registers(tmp_path, monkeypatch):
    home = _pi_home(tmp_path, monkeypatch)
    rc = cli._cmd_pi_install([])
    assert rc == 0
    ext = home / "extensions" / "simba.ts"
    assert ext.is_file()
    settings = json.loads((home / "settings.json").read_text())
    assert str(ext) in settings["extensions"]


def test_pi_install_is_idempotent(tmp_path, monkeypatch):
    home = _pi_home(tmp_path, monkeypatch)
    cli._cmd_pi_install([])
    cli._cmd_pi_install([])
    settings = json.loads((home / "settings.json").read_text())
    assert settings["extensions"].count(str(home / "extensions" / "simba.ts")) == 1


def test_pi_install_remove(tmp_path, monkeypatch):
    home = _pi_home(tmp_path, monkeypatch)
    cli._cmd_pi_install([])
    cli._cmd_pi_install(["--remove"])
    settings = json.loads((home / "settings.json").read_text())
    assert str(home / "extensions" / "simba.ts") not in settings.get("extensions", [])
    assert not (home / "extensions" / "simba.ts").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/pi/test_pi_install.py -v`
Expected: FAIL (`AttributeError: _cmd_pi_install` / `_pi_agent_home`).

- [ ] **Step 3: Implement in `__main__.py`**

```python
def _pi_agent_home() -> pathlib.Path:
    """Return pi's agent home (PI_CODING_AGENT_DIR or ~/.pi/agent)."""
    env = os.environ.get("PI_CODING_AGENT_DIR")
    if env:
        return pathlib.Path(env).expanduser()
    return pathlib.Path.home() / ".pi" / "agent"


def _cmd_pi_install(args: list[str]) -> int:
    """Install or remove the bundled pi bridge extension."""
    import importlib.resources

    remove = "--remove" in args
    home = _pi_agent_home()
    ext_dir = home / "extensions"
    ext_path = ext_dir / "simba.ts"
    settings_path = home / "settings.json"

    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except (json.JSONDecodeError, OSError):
            settings = {}
    extensions = settings.setdefault("extensions", [])

    if remove:
        if ext_path.exists():
            ext_path.unlink()
        if str(ext_path) in extensions:
            extensions.remove(str(ext_path))
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        print(f"pi extension removed from {settings_path}")
        return 0

    ext_dir.mkdir(parents=True, exist_ok=True)
    src = importlib.resources.files("simba") / "pi" / "extension" / "simba.ts"
    ext_path.write_text(src.read_text())
    if str(ext_path) not in extensions:
        extensions.append(str(ext_path))
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"pi extension installed: {ext_path}")
    print(f"  registered in {settings_path}")
    print("  daemon URL: $SIMBA_DAEMON_URL or http://localhost:8741")
    return 0
```

Wire into `main()`:

```python
elif command == "pi-install":
    return _cmd_pi_install(rest)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --no-sync pytest tests/pi/test_pi_install.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simba/__main__.py tests/pi/test_pi_install.py
git commit -m "feat(cli): simba pi-install (write + register bridge extension)"
```

### Task 5.2: Manual smoke test

- [ ] **Step 1:** `uv run --no-sync simba server &` (start the daemon), then
  `uv run --no-sync simba pi-install`.
- [ ] **Step 2:** In a repo with memories, run `pi -p "what did we decide about X"`
  and confirm a `<recalled-memories>` block is injected (check pi output / the
  injected message).
- [ ] **Step 3:** Confirm `~/.pi/agent/settings.json` lists the extension and the
  daemon shows recall traffic.
- [ ] **Step 4:** `uv run --no-sync simba pi-install --remove` to clean up.

(Manual — no commit; record the result in the PR description.)

---

## Phase 6 — Documentation & metadata

### Task 6.1: Update runtime docs

**Files:** `README.md`, `.claude/rules/CORE_INSTRUCTIONS.md`, `CLAUDE.md`, `docs/plans/README.md`, `src/simba/__main__.py`

- [ ] **Step 1: `README.md`** — runtimes badge `Claude Code + Codex` → add pi;
  tagline "Claude Code _and_ Codex" → include pi; install section documents
  `simba pi-install`; add a pi-support section parallel to `#codex-support`.

- [ ] **Step 2: `.claude/rules/CORE_INSTRUCTIONS.md`** — under Workflow, add a
  line after the Codex hooks line:

```
- pi extension (`~/.pi/agent/extensions/simba.ts`): subscribes to session_start,
  before_agent_start, agent_end, session_before_compact; bridges to `simba` via
  daemon HTTP (CLI fallback). Installed by `simba pi-install`.
```

  Then run `uv run --no-sync simba markers audit` and keep any `SIMBA:core`
  blocks intact.

- [ ] **Step 3: `CLAUDE.md`** — in the Repository Overview / hooks notes, add pi
  as a third supported runtime.

- [ ] **Step 4: `docs/plans/README.md`** — add the spec 23 index entry.

- [ ] **Step 5: `src/simba/__main__.py`** — add to the top-of-file usage
  docstring:

```
    simba pi-install       Install bundled bridge extension for pi (~/.pi/agent)
    simba pi-install --remove
                           Remove the pi bridge extension
    simba hook-canonical <event>
                           Run a canonical hook, print CanonicalResult JSON
```

- [ ] **Step 6: Verify + commit**

Run: `uv run --no-sync simba markers audit && uv run --no-sync pytest -q`
Expected: markers healthy, tests pass.

```bash
git add README.md .claude/rules/CORE_INSTRUCTIONS.md CLAUDE.md docs/plans/README.md src/simba/__main__.py
git commit -m "docs: document pi runtime support + simba pi-install"
```

---

## Future increments (separate plans when reached)

- **v2 — tool gating.** Canonicalize `PreToolUse` → `pre_tool` (TOOL_RULE deny +
  pitfall directive as `block_reason`). pi `tool_call` handler returns
  `{block, reason}`. Document that redirect **rewrite** mode is not portable (pi
  `tool_call` has no `updatedInput`); it degrades to block-with-explanation.
- **v3 — post-tool + skills + system prompt.** Canonicalize `PostToolUse` →
  `post_tool` (pi `tool_result` transform); `resources_discover` returns simba's
  `skillPaths`; optional `before_agent_start` `systemPrompt` append for guardian
  core-rules.
- **pi contract golden test.** Pin the canonical payload the extension sends and
  the pi-shaped result it applies, so a pi version bump that renames an event is
  caught at upgrade time (spec risk: pinned to pi 0.78.0).

---

## Self-Review

**Spec coverage:**
- Canonical core + dual transport → Phase 0 (core, adapter, run() refactor) + Phase 1 (endpoint) + Phase 2 (CLI thin client). ✓
- `POST /hook/{event}` keystone → Task 1.1. ✓
- Thin pi bridge (marshalling only) → Phase 4 (`simba.ts`). ✓
- `simba pi-install` + settings.json registration → Phase 5. ✓
- pi config via `@configurable` → Phase 3. ✓
- Behavior-byte-identical for Claude/Codex, lands first → Phase 0 characterization test + the cwd-isolation guard (Task 1.2). ✓
- MVP scope (4 events), v2/v3 deferred → Scope section + Future increments. ✓
- Documentation updates (README badge/tagline/section, CORE_INSTRUCTIONS marker-managed, CLAUDE.md, docs/plans/README, CLI docstring) → Phase 6. ✓
- Testing strategy (core, endpoint, CLI transport, manual smoke) → Tasks 0.1–0.3, 1.1–1.2, 2.1–2.2, 5.2. ✓

**Placeholder scan:** No TBD/TODO. Every code step shows complete code; the four
`run()` refactors specify the exact transform and the canonical fields to
return. pi API accessors (`getSessionFile`, `getSessionId`, `ui.notify`,
`BeforeAgentStartEvent.prompt`) verified against the installed d.ts.

**Type consistency:** `CanonicalResult(additional_context, suppress_output,
block_reason, transform)` used identically in core, claude adapter, endpoint,
CLI helper, and tests. Canonical event names (`session_start`, `prompt_submit`,
`stop`, `pre_compact`) consistent across `_EVENT_MODULES`, `NATIVE_TO_CANONICAL`,
the endpoint path, and the extension's `callSimba` calls. The extension's
`prompt_submit`/`stop`/`pre_compact`/`session_start` strings match the canonical
names (note: the extension maps its pi event names to canonical names at the
call site — `before_agent_start` → `prompt_submit`, `agent_end` → `stop`,
`session_before_compact` → `pre_compact`).
