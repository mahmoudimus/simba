# ADR: internal daemon-API footguns (2026-07-10)

## Status

Accepted. Three rules, each tied to a live incident on 2026-07-10.

## Context

Three failure classes surfaced in production the same day: a silent `POST
/restart` that never actually restarted the daemon, a `GET /list` self-call
that materialized 45GB of vectors, and an auto-spawned daemon whose
stdout/stderr were discarded, which is why neither of the first two was
noticed until sampled live. Each is now closed by a deterministic gate
(a test, a runtime check, or both), not just a reminder.

## Rule 1 — internal bulk reads MUST project columns

**Incident:** PR #90 (2026-07-10). Internal self-HTTP `GET /list` callers
(maintenance/hygiene/decay/consolidation/reflection passes) fetched the
whole corpus with no column projection. Lance's select cost is paid
**server-side**, during the query, regardless of which fields the caller's
JSON later reads — a caller that only ever touches `total` still pays for
every column, including each row's 1024-dim `vector`, across the entire
table. Peak footprint: 45GB for a corpus whose live data was ~31MB.

**Rule:** every internal (daemon-side) call to `GET /list` must pass
`fields=` narrowed to what it actually reads. `include_vectors` is never
set unless the caller genuinely needs the embedding back.

**Enforcement:**
- Runtime gate: `/list`'s handler (`src/simba/memory/routes.py`) 400s any
  request whose `X-Simba-Client` attribution is daemon-internal (`daemon`,
  or a nested `<origin>.daemon` loopback) and carries no `fields=`.
  External/CLI/plain clients are unaffected.
- Static gate: `tests/test_internal_list_projection_lint.py` AST-scans
  `src/simba/**/*.py` for the internal calling convention and fails,
  file:line, on any call site missing `fields=`. A reviewed exception needs
  an explicit, commented allowlist entry.

## Rule 2 — post-response work MUST NOT ride response-tied BackgroundTasks

**Incident:** `POST /restart` (PR #89) returned its 202 and then never
actually exec'd — uptime kept climbing forever after. Root cause: the
drain → stop-scheduler → flush-stdio → `os.execv` sequence was scheduled via
FastAPI's `BackgroundTasks`, and the app's middleware stack includes a
`BaseHTTPMiddleware` subclass (`DiagnosticsMiddleware`). Under Starlette,
`BaseHTTPMiddleware` runs the entire downstream response — background tasks
included — as a child task of the **same** anyio task group that scopes the
request (`call_next` spawns the inner app inside its own `async with
create_task_group()`). A client disconnect right after the response is
transmitted (the normal case for a fire-and-forget restart caller) makes the
ASGI server cancel that request's task; anyio propagates the cancellation to
every child in the group, killing the sequence before `os.execv` ever ran.
ASGI-transport tests never caught it because that transport never
disconnects. Any exception in the sequence was also invisible: it ran after
the response was already sent, so nothing HTTP-visible could ever show it.

**Rule:** work that must survive past the response (and must not be
cancellable by anything tied to the request) is scheduled as a **detached**
`asyncio.create_task`, held in its own strong reference outside any
request-scoped task group and outside any registry a shutdown drain might
wait on. Any exception in it is logged at CRITICAL and surfaced through a
durable, polled channel (not the response that already left).

**Enforcement:** `simba.memory.background.schedule_restart`/`restart_task`
hold the restart sequence in a dedicated `_RESTART_TASK` slot, deliberately
**not** in the drained `background.TASKS` registry (registering it there
would make `drain()` — called from inside the sequence itself — wait on its
own caller and deadlock every restart). Failures set
`app.state.last_restart_error`, surfaced as `lastRestartError` on `GET
/health`. Covered by `tests/memory/test_restart.py`, including a test that
cancels every other task in the process and asserts the restart sequence
still completes.

## Rule 3 — spawned daemons MUST NOT discard stdio

**Incident:** the SessionStart auto-spawn (`src/simba/hooks/
session_start.py`, `_auto_start_daemon`) launched the daemon with both
stdout and stderr redirected to `DEVNULL` (confirmed live via `lsof`). This
is exactly why incidents 1 and 2 went unnoticed for as long as they did —
there was nowhere for either failure to leave a trace.

**Rule:** a spawned daemon's stdout/stderr are always captured somewhere on
disk. Simple append, no rotation — log growth is bounded by daemon
verbosity (one INFO line per request/heartbeat), not by anything unbounded
from outside.

**Enforcement:** `_auto_start_daemon` now opens `.simba/memory/daemon.log`
(append mode, resolved via the same repo-root-aware helper the sqlite
sidecar uses) and passes it as both `stdout` and `stderr` to `Popen`,
instead of `subprocess.DEVNULL`. Covered by
`tests/hooks/test_session_start.py`.
