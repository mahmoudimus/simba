# 18 — Durable workflow engine (`simba.workflow`)

> **STATUS (2026-06-10): READY TO IMPLEMENT.** A lean, embedded, pure-Python
> durable-pipeline primitive — the substrate for the CQRS async-refresh in spec
> `17`, and a generalization of the ad-hoc `rlm_jobs` / `episode_jobs` ledgers.
> Steal the *patterns* from huey / eventsourcing / dagster (cloned in `_gitless/`);
> keep **none** of their infrastructure (no broker, no server, no daemon, no
> Postgres). Single-node, SQLite + subprocess, zero new dependencies.

## Why

simba already hand-rolls a minimal durable pipeline (hooks = event bus,
`rlm_jobs`/`episode_jobs` = idempotency ledger, detached `subprocess` = workers,
the episodes scheduler = periodic sweep). It works but is **ad-hoc and duplicated
per use-case**. This spec extracts it into one tested module so the spec-17
materialized-view refresh (and future async work) sits on a real primitive with
exactly-once enqueue, retries/backoff, stale-reclaim, resumable projections, and
asset-freshness — without adopting a heavyweight engine that violates the CORE
"no external services" constraint.

This module is **independently useful and NOT gated on spec 17's ceiling probe** —
it's foundational plumbing. It ships **default-off / unwired** (nothing in the live
hooks changes); migrating `rlm_jobs`/`episode_jobs` onto it is a **follow-up**, not
this pass.

## Constraints (from `.claude/rules/CORE_INSTRUCTIONS.md`)

- **Pure Python under `src/simba/`**, no new deps, ruff-clean (88 cols, `pathlib`,
  `TYPE_CHECKING` for annotation-only imports).
- **No external services / no broker / no server.** SQLite (via the existing
  `simba.db` peewee layer) + `subprocess`. Mirror `src/simba/rlm/jobs.py` exactly
  for the DB pattern (`simba.db.BaseModel`, `register_model`, `connect(cwd)`).
- **All config via `@configurable`** — new `workflow` section; no hidden constants.
- **Append-only friendly**: mutable status/cursor state lives in SQLite, never in
  LanceDB.
- **Determinism for tests**: every time-dependent op takes an injectable
  `now: str` (or a `clock` callable) defaulting to the real clock — so tests never
  call the wall clock. (Same reason workflow scripts ban `Date.now()`.)
- **TDD, RED first.** Every function below has a failing test written and watched
  fail before implementation.

## Reference patterns to mine (`_gitless/`, read-only — do NOT vendor)

| Repo | Read | Steal (the idea) |
|---|---|---|
| `_gitless/huey/huey/` | `api.py`, `storage.py` (`SqliteStorage`), `consumer.py` | task lifecycle; **SQLite atomic dequeue**; `retry`/`retry_delay` backoff; periodic tasks; the worker loop |
| `_gitless/eventsourcing/eventsourcing/` | `persistence.py` (notification tracking / `Tracking`), `application.py`, `system.py` (`ProcessApplication`, follow/policy) | **notification-tracking = a checkpoint cursor + exactly-once projection**; rebuild-by-replay |
| `_gitless/dagster/.../_core/definitions/` | freshness-policy / declarative-automation defs | **asset + freshness policy** ("stale when upstream changed / N events / T seconds") — the *model*, not the daemon/executor |

Mine for *design*, then write simba-native code. No copied source, no imports of these libs.

## Module layout — `src/simba/workflow/`

```
src/simba/workflow/
  __init__.py
  config.py        # @configurable("workflow")
  store.py         # peewee models (WfTask, WfCheckpoint, WfAsset) on simba.db
  queue.py         # enqueue / claim / complete / fail / reclaim_stale
  projection.py    # Projection: resumable, exactly-once, rebuildable
  asset.py         # Asset + freshness policy: is_stale / mark_materialized
  runner.py        # run_sync / dispatch_detached / fan_out / worker_loop
```

### 1. `store.py` — SQLite models (mirror `rlm/jobs.py`)

- `WfTask`: `queue:str`, `dedup_key:str|null`, `status:str` (pending|running|done|
  failed|dead), `payload:str` (JSON), `attempts:int=0`, `max_attempts:int`,
  `available_at:str`, `created_at:str`, `started_at:str|null`,
  `finished_at:str|null`, `error:str|null`, `result:str|null`. UNIQUE index on
  `(queue, dedup_key)` (partial/where dedup_key not null — emulate by storing a
  sentinel or a separate unique index; match how `rlm_jobs` does UNIQUE).
- `WfCheckpoint`: `name:str UNIQUE`, `position:int=0`, `updated_at:str`.
- `WfAsset`: `name:str UNIQUE`, `last_materialized_at:str|null`,
  `last_source_position:int=0`.
- `register_model` each, as `rlm/jobs.py` does.

### 2. `queue.py` — durable task queue

```python
def enqueue(queue, payload, *, dedup_key=None, max_attempts=None, delay_seconds=0,
            now=None, cwd=None) -> int | None
# Idempotent on (queue, dedup_key): if a non-terminal/terminal row with that key
# exists, return its id without inserting (None or existing id — pick & test one).
# Without dedup_key, always insert. available_at = now + delay_seconds.

def claim(queue, *, now=None, cwd=None) -> dict | None
# Atomically take the oldest pending task with available_at <= now: flip to
# running, set started_at, return it. Concurrent claim of the same task is
# impossible (UPDATE ... WHERE status='pending' guarded; verify exactly-once).

def complete(task_id, *, result=None, now=None, cwd=None) -> None
def fail(task_id, *, error, now=None, cwd=None) -> str
# attempts += 1; if attempts < max_attempts: status=pending,
# available_at = now + backoff(attempts) [exponential, capped by config];
# else status=dead. Return the new status.

def reclaim_stale(queue, *, stale_after_seconds, now=None, cwd=None) -> int
# running tasks with started_at older than stale_after -> pending (dead-worker
# recovery, like episodes job_timeout_hours). Return count reclaimed.
```

### 3. `projection.py` — resumable exactly-once projection (eventsourcing idea)

```python
class Projection:
    def __init__(self, name, process_fn): ...
    def run(self, events, *, cwd=None) -> int
    # events: iterable of (position:int, event) sorted ascending. Process only
    # events with position > checkpoint; call process_fn(event); advance the
    # checkpoint to the last processed position (atomically per event/batch).
    # Returns count processed. Re-running processes only new events (resume).
    def rebuild(self, all_events, *, reset_fn=None, cwd=None) -> int
    # reset checkpoint to 0, call reset_fn() (caller clears its derived table),
    # then run(all_events) — replay from zero. Rebuildability = swappability.
```
`process_fn` (caller's) does the derived-table upsert (e.g. entity-cluster). The
projection only guarantees **exactly-once advancement + resumability**.

### 4. `asset.py` — asset + freshness (dagster idea, no daemon)

```python
@dataclass
class FreshnessPolicy:
    stale_after_events: int | None = None     # N new source rows since last materialize
    stale_after_seconds: float | None = None  # wall-clock staleness

def is_stale(name, policy, *, current_source_position, now=None, cwd=None) -> bool
# True if never materialized, or (current_pos - last_source_position) >=
# stale_after_events, or (now - last_materialized_at) >= stale_after_seconds.

def mark_materialized(name, *, source_position, now=None, cwd=None) -> None
```
Drives "should I enqueue a refresh?" — called by the existing scheduler/hooks. No
new daemon.

### 5. `runner.py` — execution

```python
def run_sync(handler, task) -> ...                  # execute a claimed task in-process
def dispatch_detached(argv, *, cwd, env=None) -> None
# fire-and-forget Popen: start_new_session=True, stdin/out/err=DEVNULL — the
# rlm-engine pattern; never blocks a hook.
def fan_out(items, fn, *, max_workers=None) -> list  # concurrent.futures parallel map;
# per-item exceptions captured (return None for failures); for within-stage parallelism.
def worker_loop(queue, handler, *, max_tasks=None, now=None, cwd=None) -> int
# claim -> run_sync(handler) -> complete / fail(retry), until empty or max_tasks.
```

### 6. `config.py` — `@configurable("workflow")`

`default_max_attempts:int=3`, `retry_backoff_base_seconds:float=2.0`,
`retry_backoff_max_seconds:float=300.0`, `stale_after_seconds:int=3600`,
`fan_out_max_workers:int=8`, `worker_mode:str="detached"` (sync|detached).
`enqueue`/`fail` read defaults from here when args are None.

## TDD — tests/workflow/ (RED first, watch each fail)

- **queue**: enqueue-dedup idempotency (same key twice → one row); enqueue without
  key → new rows; claim returns task + flips running; second claim → None
  (exactly-once); claim skips a task whose `available_at` is in the future;
  complete → done; fail retries with backoff (attempts<max → pending + future
  available_at) then dead (attempts==max); reclaim_stale reclaims old running,
  leaves fresh running.
- **projection**: run advances checkpoint + processes all; re-run processes only
  events past the checkpoint (resume); event at/below checkpoint is skipped
  (exactly-once); rebuild resets + replays all (reset_fn called).
- **asset**: is_stale True when never materialized; True at event-count threshold;
  True at time threshold; False when fresh; mark_materialized resets both axes.
- **runner**: fan_out returns all results + isolates a failing item; dispatch_detached
  calls Popen with `start_new_session=True` + cwd (monkeypatch `subprocess.Popen`,
  like `tests/rlm/test_engine.py`); worker_loop drains a queue (claim→complete) and
  routes a raising handler to `fail`.
- Use `tmp_path` + the `simba.db` test pattern from `tests/episodes/test_consolidate.py`
  (`monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)`).
- Inject `now=` in every time-dependent assertion — no wall-clock calls in tests.

## Non-goals (this pass)

- Do **not** migrate `rlm_jobs` / `episode_jobs` onto this (follow-up; no behavior
  change now).
- Do **not** wire it into live hooks. Ships as a tested, unused-by-default library.
- No periodic-cron scheduler of its own (the episodes scheduler + hooks already
  trigger; `is_stale` is the decision function they call).

## Acceptance

- `uv run pytest tests/workflow` green; `uv run ruff check src/simba/workflow tests/workflow`
  clean; full `uv run pytest` still green.
- Every new symbol has a RED-first test. Time-dependent ops injectable.
- `simba config get workflow.default_max_attempts` works (section registered).
- A short module docstring in `__init__.py` states the CQRS role (durable queue +
  projection + asset-freshness; the substrate for spec 17's async view refresh).
- Commit on `feat/durable-workflow` (no Claude attribution in the message).
