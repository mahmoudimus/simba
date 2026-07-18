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

## Addendum (2026-07-17) — projection alone doesn't bound a scan

**Incident:** the daemon's RSS watchdog hard-tripped at ~5GB during a
PreCompact storm. Root contributor: `src/simba/episodes/consolidate.py` and
`src/simba/reflection/pass_.py` each self-called `GET /list?limit=100000`
with `fields=` narrowed to what they read (satisfying Rule 1 above) but
still requesting `content`/`context` over the **entire** ~10k-memory corpus,
just to (a) discover which sessions needed work and (b) fetch one session's
members. Rule 1's column projection stops a `vector`-shaped blowup; it does
nothing to stop a row-count-shaped one — narrowing *which columns* come back
per row doesn't bound *how many rows* carry them.

**Rule 1 addendum:** a daemon-internal `GET /list` call whose `fields=`
includes `context` (the expensive, potentially large column short of
`vector` itself) must ALSO pass a row-bounding constraint: `sessionSource=`,
`projectPath=`, `since=`, or `limit<=1000`. Un-bounded corpus-wide reads are
now split into a cheap projected **discovery** scan (`id,type,
sessionSource,projectPath,createdAt` — no content/context) used to group
memories into sessions and pre-check eligibility, followed by a **targeted**
fetch (full fields, server-side `sessionSource=` or `projectPath=` scoped)
only for the rows actually needed. `episodes.consolidate` additionally
maintains a per-project incremental-discovery watermark (`simba.episodes.
watermark`, sidecar DB) so a repeat sweep's discovery scan is bounded by
`since=<watermark>` instead of re-scanning the whole corpus every time.

**Enforcement:**
- Runtime gate: `/list`'s handler 400s a daemon-internal request whose
  `fields=` includes `context` and carries none of `sessionSource=`,
  `projectPath=`, `since=`, or `limit<=1000`. External/CLI/plain clients are
  unaffected, same as Rule 1.
- Static gate: `tests/test_internal_list_projection_lint.py` additionally
  flags any internal call site whose (statically resolvable) `fields=`
  includes `context` without a resolvable row bound
  (`CONTEXT_BOUND_ALLOWLIST` is the reviewed-exception carve-out, mirroring
  Rule 1's `ALLOWLIST`).
- New server-side `/list` filters: `sessionSource=` (exact match, same
  mechanism as `type=`/`projectPath=`) and `since=` (ISO-8601 UTC,
  `createdAt >= since`, compared as parsed datetimes — never as raw
  strings, since `createdAt` values are mixed-precision and a later
  fractional-second timestamp can sort lexicographically *before* an
  earlier whole-second one).

## Addendum (2026-07-18) — the HTTP-layer gate missed direct in-process scans

**Incident:** a native `sample` taken mid-RSS-burst under concurrent session
traffic caught the allocator inside `pyarrow _Tabular.to_pylist ->
ChunkedArray.to_pylist -> ListScalar.as_py -> arrow::MakeScalar<float> ->
operator new` — a LanceDB query materializing the 1024-dim `vector` column
into ~10 million individually heap-allocated Python/Arrow float scalars.
Rule 1 (2026-07-10) and its 2026-07-17 addendum both gate self-HTTP `GET
/list` calls — but a **direct, in-process** `table.query().to_list()` or
`table.vector_search(...).to_list()` never goes through HTTP at all, so
neither the runtime gate nor `tests/test_internal_list_projection_lint.py`
(which only AST-scans `<something>.get(url, ...)` call sites) ever saw it.
Every real call site of this shape as of 2026-07-18 fetched every column —
`vector` included — regardless of what the caller read afterward: `/stats`
(hit by every SessionStart hook, the primary burst driver), the hybrid-recall
session-expansion and anticipated-query record fetches, `vector_db.py`'s
duplicate-check/search/reembed/access-tracking helpers, the FTS-mirror boot
reconcile, `/scopes/normalize`, `/promotions/candidates`, `/reindex`,
`/reembed`, and the cross-store `reconcile.py` audit.

**Rule 1, direct-call corollary:** every in-process LanceDB `.query(`/
`.search(`-family call chain that reaches `.to_list()` must call `.select(`
somewhere in that chain, narrowed to the columns its consumers actually
read — never `vector` unless a caller genuinely needs the embedding back
(and even then, bounded, e.g. a single-row fallback fetch, never a
whole-corpus one). This is the same rule as Rule 1 above, just applied to
the call shape that isn't HTTP.

**Enforcement:**
- Static gate: `tests/test_lance_projection_lint.py` AST-scans
  `src/simba/**/*.py` for a call chain ending in `.to_list()` that
  originates from a `.query(`/`.search(`-family builder (this also matches
  `vector_search`) and fails, file:line, on any such chain missing
  `.select(`. Same same-function variable-tracking approach as the sibling
  HTTP-layer lint (including a self-referential reassignment, e.g. `query =
  query.where(...)`), and the same reviewed-exception `ALLOWLIST` carve-out
  — empty as of this pass, since every real site got a genuine projection.
- Every site fixed in the 2026-07-18 pass got a projection narrowed to its
  actual consumers (read by hand, not inferred): `/stats` selects
  `type,confidence,createdAt`; the hybrid-recall record fetches and
  `vector_db.search_memories` select the metadata/text fields the RRF/
  rerank/format pipeline reads (`id,type,content,context,confidence,
  createdAt,tags,projectPath,sessionSource`); `find_duplicates` selects
  `id,type` (LanceDB auto-includes `_distance` for a vector-search
  `.to_list()` regardless of `.select()`); the FTS-mirror-feeding sites
  (`init_fts_mirror`, `/reindex`, `/reembed`, `reconcile.py`) share
  `simba.memory.fts.REQUIRED_MEMORY_FIELDS`; `reembed_table` selects every
  column except `vector` (verified via `lancedb`'s async `table.schema()`)
  and falls back to a bounded (`.limit(1)`) single-row re-fetch of the OLD
  vector only for a row whose re-embed failed or had no content/context to
  embed.
