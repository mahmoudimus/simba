# 19 — Collapse `rlm_jobs` / `episode_jobs` onto a `simba.workflow` lease

> **STATUS (2026-06-10): READY TO IMPLEMENT.** Builds on spec `18` (the durable
> workflow engine, already shipped on `feat/durable-workflow`). Adds a **lease**
> primitive to `simba.workflow` and retires the two ad-hoc control-plane ledgers
> (`rlm/jobs.py`, `episodes/jobs.py`) onto it. **Semantics-preserving** — the
> dedup/lock behavior is load-bearing (a regression = double LLM spend, or a
> transcript/session that never gets digested). Branch off `feat/durable-workflow`.

## Why

`rlm/jobs.py` and `episodes/jobs.py` are two near-identical ~60-line modules that
do the same thing — **lease a key** (claim a `(id, project)` lock at most once,
release it on completion; episodes adds expiry-based reclaim of dead-worker
locks). They predate `simba.workflow`. This spec extracts that shared behavior
into one tested primitive and reduces each ledger to a thin delegation, deleting
the bespoke `RlmJob` / `EpisodeJob` models. One backing table (`wf_tasks`), one
lock semantics, defined and tested once.

This is **not** a produce/consume queue use-case — it's a **lease/lock with
optional expiry**, which is why it needs its own primitive rather than reusing
`enqueue`+`claim` (those are pending→running dequeue semantics; a lease is
acquire-key-at-most-once + release-by-key).

## What exists to build on (already shipped, this branch)

- `src/simba/workflow/store.py` — `WfTask(queue, dedup_key, status, payload,
  attempts, max_attempts, available_at, created_at, started_at, finished_at,
  error, result)`, UNIQUE `(queue, dedup_key)` (SQLite NULL-distinct). **Reuse
  this table** — do not add a new one.
- `src/simba/workflow/_time.py` — `now()`, `parse()`, `add_seconds()`,
  `resolve(value|None)` (the injectable-clock seam). **Reuse for all time math.**
- `src/simba/workflow/queue.py` — `enqueue`/`claim`/`complete`/`fail`/`reclaim_stale`
  (for reference / the `simba.db.connect` + peewee patterns).
- DB pattern: `with simba.db.connect(cwd): ...`, mirroring `rlm/jobs.py`.

## Phase 0 — the lease primitive (TDD, the real new code)

`src/simba/workflow/lease.py` — a lock on `(queue, key)`, optional expiry:

```python
def acquire(queue, key, *, stale_after_seconds=None, payload=None,
            now=None, cwd=None) -> bool:
    """Try to take the (queue, key) lock. Return True iff acquired.

    Atomic, on WfTask (status used as the lock state):
      - no row for (queue,key)            -> create status='running' -> True
      - row status='done'                 -> False  (durable dedup; never reclaimed)
      - row status='running', NOT stale   -> False  (held by a live worker)
      - row status='running', stale        -> reclaim (reset started_at/attempts,
            keep status='running') -> True   [only when stale_after_seconds set and
            started_at older than now-stale_after_seconds]
    payload (JSON) is stored on create. now via _time.resolve.
    """

def release(queue, key, *, result=None, now=None, cwd=None) -> None:
    """Release the (queue,key) lock: status='done', finished_at=now,
    result=json(result). No-op if the row is absent."""
```

Implementation notes:
- The `(queue, dedup_key)` UNIQUE index already guarantees one row per key — use
  `dedup_key=key`. Handle the create-race with `try create / except IntegrityError
  -> re-read` (like `queue.enqueue`).
- "stale" = `status=='running'` and `parse(started_at) < parse(now) -
  stale_after_seconds` (use `_time`). `stale_after_seconds=None` ⇒ never reclaim
  (so a held/dead lock blocks forever — that's rlm's *current* behavior; see the
  config decision below).
- `max_attempts` is irrelevant to a lease — set a sentinel (e.g. 1) on create;
  the lease never retries. Keep the row valid for the shared table.

**TDD — `tests/workflow/test_lease.py` (RED first):**
- acquire absent → True + a `running` row exists.
- acquire twice (same key) → second False.
- acquire after release (now `done`) → False (durable dedup, even with
  `stale_after_seconds` set — done is never reclaimed).
- `stale_after_seconds=None`: a `running` lock is never reclaimed → re-acquire False.
- `stale_after_seconds=X`: a `running` lock with `started_at` older than X →
  re-acquire True (reclaimed); a fresh `running` lock → False.
- release stores `result` (read it back); release on absent key → no error.
- two queues, same key → independent locks.
- inject `now=` everywhere; no wall-clock in tests.

## Phase 1 — collapse the ledgers onto the lease

Keep the **public `claim`/`complete` signatures stable** (deliberate: lowest-risk,
and `rlm.jobs.complete` is called from `feat/rlm-personal-assistant`
(`rlm/engine.py::_complete_rlm_job`) which is *not* on this branch — stable
signatures avoid a merge break). Replace the bodies; delete the models.

```python
# src/simba/rlm/jobs.py   (DELETE the RlmJob model)
import simba.workflow.lease as lease
_QUEUE = "rlm_digest"
def _key(tid, project): return f"{tid}\x00{project}"

def claim(transcript_id, project_path, engine, *, cwd=None) -> bool:
    cfg = _rlm_cfg()  # load "rlm" section
    return lease.acquire(
        _QUEUE, _key(transcript_id, project_path),
        stale_after_seconds=(cfg.digest_stale_after_seconds or None),
        payload={"engine": engine}, cwd=cwd,
    )
def complete(transcript_id, project_path, n_stored, *, cwd=None) -> None:
    lease.release(_QUEUE, _key(transcript_id, project_path),
                  result={"n_stored": n_stored}, cwd=cwd)

# src/simba/episodes/jobs.py   (DELETE the EpisodeJob model)
_QUEUE = "episode_consolidate"
def claim(session_source, project_path, *, cwd=None, stale_after_seconds=None) -> bool:
    return lease.acquire(_QUEUE, _key(session_source, project_path),
                         stale_after_seconds=stale_after_seconds, cwd=cwd)
def complete(session_source, project_path, *, cwd=None) -> None:
    lease.release(_QUEUE, _key(session_source, project_path), cwd=cwd)
```

Call sites (verify unchanged — signatures stable): `hooks/pre_compact.py:135`,
`__main__.py:2819` (rlm claim), `__main__.py:2786` (rlm complete),
`episodes/consolidate.py:138` (episode claim), `__main__.py:2731` (episode
complete). No codemod needed at this size; if a future reshape changes
signatures, run **libcst ephemerally** (`uv run --with libcst python tools/cm.py`)
so no dep lands — not required here.

### The rlm stale-reclaim decision (resolved)

rlm currently has **no** stale-reclaim → a dead digest worker locks that
transcript from ever re-digesting (latent bug). Resolution: **add config field
`rlm.digest_stale_after_seconds: int = 0`** (0 ⇒ `None` ⇒ no reclaim ⇒ *preserve
current behavior* by default). Passed through to `lease.acquire`. This neither
smuggles a behavior change nor forecloses the fix — flip the field to enable
expiry. Episodes keeps its existing expiry (`consolidate` already passes
`stale_after_seconds = job_timeout_hours*3600`).

## Phase 2 — data: no migration

`rlm_jobs` / `episode_jobs` are ephemeral control-plane. Dropping them costs at
most one re-digest / one re-consolidate (and episodes' durable dedup is the
stored `EPISODE` itself — `consolidate` checks `_has_episode` before claiming, so
no episode dupes). **Do not migrate rows.** Leave the old tables vestigial (a
later cleanup can `DROP`); `register_model` for `RlmJob`/`EpisodeJob` is removed
when the models are deleted, so the tables simply stop being written.

## Phase 3 — tests

- **New**: `tests/workflow/test_lease.py` (Phase 0 above).
- **Rewrite** `tests/rlm/test_jobs.py` + `tests/episodes/test_jobs.py` to assert
  *public behavior* over the lease (claim True-once / False-after; complete→done;
  episodes stale-reclaim with `stale_after_seconds`; rlm no-reclaim by default,
  reclaim when `digest_stale_after_seconds` set) — not the deleted models.
- `tests/hooks/test_pre_compact.py`, `tests/episodes/test_consolidate.py`,
  `tests/test_main_cli.py` should pass **unchanged** (public API stable; they
  monkeypatch claim/complete or the db path). Fix only if any asserted on
  `RlmJob`/`EpisodeJob` directly.
- New config field test: `tests/rlm/test_config.py` → `digest_stale_after_seconds == 0`.

## Constraints (from `.claude/rules/CORE_INSTRUCTIONS.md`)

Pure Python under `src/simba/`; no new deps; SQLite via `simba.db`; all config via
`@configurable` (the new `rlm.digest_stale_after_seconds`); ruff-clean (88 cols);
TDD RED-first; inject `now=`. No external services.

## Acceptance

- `uv run pytest tests/workflow tests/rlm tests/episodes tests/hooks` green;
  full `uv run pytest` green; `uv run ruff check src/ tests/` clean.
- `RlmJob` / `EpisodeJob` models deleted; both ledgers delegate to
  `simba.workflow.lease`; public `claim`/`complete` signatures unchanged.
- `simba config get rlm.digest_stale_after_seconds` → `0`.
- Commit on `feat/jobs-on-workflow` (no Claude attribution; no push; don't touch
  `uv.lock`).
