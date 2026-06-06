# Phase 6 — Decay / Forgetting + Feedback-Aware Ranking

## Implementation Spec for simba

---

## 0. Prerequisite: Vocabulary and Invariants

**Source of truth for mutable ranking signals**: The SQLite `simba.db` table `memory_usage` is the single authority for `access_count`, `last_accessed`, `strength`, `dormant`, and `feedback_score`. LanceDB's `accessCount`/`lastAccessedAt` columns remain as they are (fire-and-forget via `update_access_tracking`) but are **never read** for ranking — the sqlite table wins. This avoids fighting LanceDB's fragment model for mutable state.

**Forgetting is reversible**: Setting `dormant = True` excludes a memory from recall. `dormant` can be cleared by the scheduler (when strength recovers, which never happens under pure decay — but feedback bumps can revive) or manually. No row is ever deleted from `memory_usage`. No memory is ever deleted from LanceDB by the decay system.

**Deterministic time contract**: Every function that computes time-sensitive values accepts `now: float` (epoch seconds) as a parameter. No function calls `time.time()` internally.

---

## 1. Usage Store

### Goal

Define a peewee model `MemoryUsage` in `simba.db` (the shared `simba.db`) keyed by `memory_id`. Provide synchronous CRUD helpers. This table is the mutable sidecar for every LanceDB memory.

### Files

- `src/simba/memory/usage.py` — model definition + CRUD helpers
- `tests/memory/test_usage.py` — test file

### Config

No new config keys in this sub-feature. The connection is established via the existing `simba.db.connect(cwd)` context manager.

### Schema / Signatures

```python
# src/simba/memory/usage.py

import simba._vendor.peewee as pw
import simba.db

class MemoryUsage(simba.db.BaseModel):
    memory_id: pw.TextField        # primary key; matches the LanceDB `id` field
    access_count: pw.IntegerField  # default=0; incremented on every recall
    last_accessed: pw.FloatField   # epoch seconds; 0.0 when never accessed
    strength: pw.FloatField        # default=1.0; recomputed by scheduler
    dormant: pw.BooleanField       # default=False; excludes from recall when True
    feedback_score: pw.FloatField  # default=0.0; range [-1.0, +1.0]
    created_at: pw.FloatField      # epoch seconds; set on first upsert

    class Meta:
        table_name = "memory_usage"
        primary_key = False  # memory_id is the PK, declared as primary_key=True below

# Actual declaration uses pw.CharField(primary_key=True) for memory_id
```

Exact field declarations:

```python
class MemoryUsage(simba.db.BaseModel):
    memory_id = pw.CharField(max_length=64, primary_key=True)
    access_count = pw.IntegerField(default=0)
    last_accessed = pw.FloatField(default=0.0)
    strength = pw.FloatField(default=1.0)
    dormant = pw.BooleanField(default=False)
    feedback_score = pw.FloatField(default=0.0)
    created_at = pw.FloatField(default=0.0)

    class Meta:
        table_name = "memory_usage"
```

CRUD helpers (all synchronous, must be called inside `simba.db.connect()` context):

```python
def get_or_create(memory_id: str, now: float) -> MemoryUsage:
    """Return existing row or INSERT a default row with created_at=now."""

def bump_access(memory_id: str, now: float) -> None:
    """Increment access_count, set last_accessed=now. Upserts if missing."""

def set_dormant(memory_id: str, *, dormant: bool) -> None:
    """Set the dormant flag. No-op if row missing (silent)."""

def apply_feedback(memory_id: str, delta: float, now: float) -> None:
    """Clamp feedback_score += delta to [-1.0, +1.0]. Upserts if missing."""

def set_strength(memory_id: str, strength: float) -> None:
    """Overwrite strength. Clamp to [0.0, 1.0]. No-op if missing."""

def get_many(memory_ids: list[str]) -> dict[str, MemoryUsage]:
    """Bulk-fetch rows keyed by memory_id. Missing ids absent from dict."""

def get_all_for_decay(
    *,
    include_dormant: bool = False,
) -> list[MemoryUsage]:
    """Return all rows, optionally including already-dormant rows."""
```

Registration: at the bottom of `usage.py`, call `simba.db.register_model(MemoryUsage)` so `connect()` creates the table automatically.

Also add a one-time migration helper called from `_init_schema_usage` (registered via `simba.db.register_schema`). This migration adds an index:

```sql
CREATE INDEX IF NOT EXISTS idx_memory_usage_dormant
    ON memory_usage(dormant, strength);
```

The schema initializer is registered via `simba.db.register_schema(_init_index)` where `_init_index` is a function that runs the `CREATE INDEX` DDL. This runs before `create_tables` in `connect()` (per the existing contract in `db.py` line 90-94).

### Implementation Steps

1. Create `src/simba/memory/usage.py`.
2. Import `simba._vendor.peewee as pw` and `simba.db`.
3. Define `MemoryUsage(simba.db.BaseModel)` with the six fields above. `memory_id` uses `pw.CharField(max_length=64, primary_key=True)`.
4. Write `_init_index(conn)` that executes `CREATE INDEX IF NOT EXISTS idx_memory_usage_dormant ON memory_usage(dormant, strength)` and call `simba.db.register_schema(_init_index)`.
5. Call `simba.db.register_model(MemoryUsage)` at module level.
6. Implement `get_or_create(memory_id, now)`: use `MemoryUsage.get_or_create(memory_id=memory_id, defaults={"created_at": now})` and return the instance.
7. Implement `bump_access(memory_id, now)`: call `get_or_create`, then `MemoryUsage.update(access_count=MemoryUsage.access_count + 1, last_accessed=now).where(MemoryUsage.memory_id == memory_id).execute()`.
8. Implement `set_dormant(memory_id, dormant)`: `MemoryUsage.update(dormant=dormant).where(MemoryUsage.memory_id == memory_id).execute()`.
9. Implement `apply_feedback(memory_id, delta, now)`: `get_or_create`, then clamp `feedback_score + delta` to `[-1.0, 1.0]`, then `MemoryUsage.update(feedback_score=clamped).where(...).execute()`.
10. Implement `set_strength(memory_id, strength)`: clamp to `[0.0, 1.0]`, then update.
11. Implement `get_many(memory_ids)`: single `MemoryUsage.select().where(MemoryUsage.memory_id.in_(memory_ids))`, return as `{row.memory_id: row}` dict.
12. Implement `get_all_for_decay(include_dormant)`: `MemoryUsage.select()` with optional `.where(MemoryUsage.dormant == False)`.

### Tests — `tests/memory/test_usage.py`

All tests use `tmp_path` (pytest fixture) as `cwd`. Each test wraps in `simba.db.connect(tmp_path)`.

```
test_get_or_create_creates_row
  - Call get_or_create("mem_abc", now=1000.0)
  - Assert row.memory_id == "mem_abc", row.access_count == 0, row.strength == 1.0
  - Assert row.dormant == False, row.feedback_score == 0.0

test_get_or_create_is_idempotent
  - Call get_or_create twice with same id and different now
  - Assert only one row in DB (count == 1)
  - Assert created_at equals the first call's now

test_bump_access_increments
  - bump_access("mem_abc", now=100.0)
  - bump_access("mem_abc", now=200.0)
  - row = get_many(["mem_abc"])["mem_abc"]
  - Assert row.access_count == 2, row.last_accessed == 200.0

test_bump_access_upserts_missing_row
  - bump_access("mem_new", now=50.0) on empty DB
  - Assert row exists with access_count == 1

test_set_dormant_true_false
  - get_or_create("mem_x", now=0.0)
  - set_dormant("mem_x", dormant=True)
  - Assert get_many(["mem_x"])["mem_x"].dormant == True
  - set_dormant("mem_x", dormant=False)
  - Assert .dormant == False

test_apply_feedback_clamps
  - get_or_create("mem_y", now=0.0)
  - apply_feedback("mem_y", 0.9, now=1.0)
  - apply_feedback("mem_y", 0.9, now=2.0)  # would exceed +1.0
  - Assert .feedback_score == 1.0

  - apply_feedback("mem_y", -3.0, now=3.0)
  - Assert .feedback_score == -1.0

test_set_strength_clamps
  - get_or_create("mem_z", now=0.0)
  - set_strength("mem_z", 1.5)  # above 1.0
  - Assert .strength == 1.0
  - set_strength("mem_z", -0.1)
  - Assert .strength == 0.0

test_get_all_for_decay_excludes_dormant_by_default
  - Insert two rows: one dormant=False, one dormant=True
  - Assert get_all_for_decay() returns length 1
  - Assert get_all_for_decay(include_dormant=True) returns length 2
```

### Acceptance

`uv run pytest tests/memory/test_usage.py -v` passes all 8 tests. `simba.db.connect(tmp_path)` creates `memory_usage` table with correct schema (verify via `PRAGMA table_info(memory_usage)`).

### Verify

```bash
uv run pytest tests/memory/test_usage.py -v
python -c "
import pathlib, simba.db, simba.memory.usage
with simba.db.connect(pathlib.Path('/tmp/test_simba')):
    simba.memory.usage.get_or_create('mem_test', 0.0)
    print('usage store ok')
"
```

### Reuse

- `src/simba/db.py:BaseModel` — base class
- `src/simba/db.py:register_model` — table registration
- `src/simba/db.py:register_schema` — schema init registration
- `src/simba/db.py:connect` — connection context manager
- `src/simba/kg/store.py:KgEdge` — peewee model reference pattern

---

## 2. Strength Model

### Goal

A pure, deterministic function `compute_strength` that maps `(age_days, access_count, feedback_score, half_life, ...)` to a `float` in `[0.0, 1.0]`. No side effects, no I/O, no time calls. This is the mathematical core that the scheduler (sub-feature 4) calls for each memory row.

### Files

- `src/simba/memory/strength.py` — pure functions only
- `tests/memory/test_strength.py` — test file

### Config

New fields on `MemoryConfig` in `src/simba/memory/config.py`:

```
memory.decay_half_life_days: float = 30.0
  # Time (days) at which an unaccessed memory's decay factor reaches 0.5.

memory.reinforcement_scale: float = 0.5
  # How much each access "lifts" strength. Logistic scale factor.
  # strength_reinforcement = 1 - exp(-access_count / reinforcement_scale)
  # With scale=0.5: 1 access → ~0.86, 2 → ~0.98, 3 → ~1.00 (saturates fast).

memory.feedback_weight: float = 0.2
  # Weight applied to feedback_score when computing final strength.
  # final_strength = decay * reinforcement * (1 + feedback_weight * feedback_score)

memory.strength_dormancy_threshold: float = 0.1
  # Memories with strength below this after a decay pass become dormant.

memory.decay_capacity_per_type: int = 0
  # Max non-dormant memories per (type, project_path). 0 = unlimited.
  # When > 0, the weakest memories beyond this cap are set dormant.
```

### Schema / Signatures

```python
# src/simba/memory/strength.py

def decay_factor(age_days: float, half_life: float) -> float:
    """Exponential decay: 0.5^(age_days / half_life).
    Returns 1.0 when half_life <= 0 (disabled). Clamps age_days to >= 0.
    """

def reinforcement_factor(access_count: int, scale: float) -> float:
    """Logistic reinforcement: 1 - exp(-access_count / scale).
    Returns 0.0 when scale <= 0 or access_count == 0.
    Returns 1.0 when scale <= 0 and access_count > 0 (treat as "always accessed").
    """

def compute_strength(
    *,
    created_at_epoch: float,
    now: float,
    access_count: int,
    feedback_score: float,
    half_life: float,
    reinforcement_scale: float,
    feedback_weight: float,
) -> float:
    """Combine decay + reinforcement + feedback into strength in [0.0, 1.0].

    Formula:
        age_days = (now - created_at_epoch) / 86400
        d = decay_factor(age_days, half_life)
        r = reinforcement_factor(access_count, reinforcement_scale)
        base = max(d, d * r)   # reinforcement lifts but never below raw decay
        feedback_term = 1.0 + feedback_weight * clamp(feedback_score, -1, 1)
        raw = base * feedback_term
        return clamp(raw, 0.0, 1.0)
    """
```

The formula above means:
- A brand-new memory (age 0, no accesses) has `d=1.0`, `r=0.0` (0 accesses), `base = max(1.0, 0.0) = 1.0`. Strength starts at 1.0.
- After half_life days with no accesses: `d=0.5`, strength = 0.5.
- Each access raises `r` rapidly; with `scale=0.5`, one access brings `r ≈ 0.86`, so `base = max(0.5, 0.5*0.86) = 0.5` — wait, this is wrong: reinforcement should *multiply* decay, not compete. The correct reading:

Revised formula (simpler, unambiguous):

```
age_days = (now - created_at_epoch) / 86400
d = 0.5 ** (age_days / half_life)             # pure time decay
r = 1.0 - exp(-access_count / scale)          # [0, 1), 0 when never accessed
base = d + (1.0 - d) * r                      # r pulls base toward 1.0 from decay floor
feedback_term = 1.0 + feedback_weight * feedback_score   # feedback_score in [-1,1]
raw = base * feedback_term
return clamp(raw, 0.0, 1.0)
```

Interpretation:
- `d + (1-d)*r`: decay provides a floor; each access raises base toward 1.0. With 0 accesses: base=d. With many accesses: base→1.0 regardless of age. This is the Generative Agents spaced-repetition intuition.
- `feedback_weight=0.2` and `feedback_score=+1.0` lifts by 20%; `score=-1.0` cuts by 20%.

The spec uses this revised formula. Update `compute_strength` docstring accordingly.

### Implementation Steps

1. Create `src/simba/memory/strength.py` with `from __future__ import annotations` and `import math`.
2. Implement `decay_factor(age_days, half_life)`: guard `half_life <= 0` returns `1.0`; `age_days = max(0.0, age_days)`; return `0.5 ** (age_days / half_life)`.
3. Implement `reinforcement_factor(access_count, scale)`: guard `access_count == 0` returns `0.0`; guard `scale <= 0` returns `1.0`; return `1.0 - math.exp(-access_count / scale)`.
4. Implement `compute_strength(...)` using the revised formula. Clamp to `[0.0, 1.0]` via `max(0.0, min(1.0, raw))`.
5. No imports from the rest of simba. This module has zero dependencies outside stdlib.

### Tests — `tests/memory/test_strength.py`

```
test_decay_factor_at_zero_age
  - decay_factor(0.0, 30.0) == 1.0

test_decay_factor_at_half_life
  - abs(decay_factor(30.0, 30.0) - 0.5) < 1e-9

test_decay_factor_disabled_when_halflife_zero
  - decay_factor(100.0, 0.0) == 1.0

test_reinforcement_zero_when_no_accesses
  - reinforcement_factor(0, 0.5) == 0.0

test_reinforcement_approaches_one
  - reinforcement_factor(10, 0.5) > 0.999  # saturates fast

test_reinforcement_scale_zero_and_accesses
  - reinforcement_factor(1, 0.0) == 1.0

test_compute_strength_brand_new
  - compute_strength(created_at_epoch=0.0, now=0.0, access_count=0,
      feedback_score=0.0, half_life=30.0, reinforcement_scale=0.5,
      feedback_weight=0.2)
  - Assert result == 1.0

test_compute_strength_after_one_half_life_no_access
  - created_at_epoch=0, now = 30 * 86400, access_count=0, feedback_score=0.0
  - half_life=30.0, reinforcement_scale=0.5, feedback_weight=0.2
  - Assert abs(result - 0.5) < 1e-6

test_compute_strength_reinforcement_lifts_above_decay
  - same as above but access_count=5
  - Assert result > 0.5  # reinforcement brings it above pure decay

test_compute_strength_positive_feedback_lifts
  - result_with = compute_strength(..., feedback_score=1.0, feedback_weight=0.2)
  - result_without = compute_strength(..., feedback_score=0.0, feedback_weight=0.2)
  - Assert result_with > result_without

test_compute_strength_negative_feedback_lowers
  - result_neg = compute_strength(..., feedback_score=-1.0, feedback_weight=0.2)
  - result_neutral = compute_strength(..., feedback_score=0.0, ...)
  - Assert result_neg < result_neutral

test_compute_strength_clamps_to_zero
  - Very old memory, no accesses, max feedback_weight=1.0, feedback_score=-1.0
  - Pass half_life=1.0, age > 300 days
  - Assert result >= 0.0  # never negative

test_compute_strength_clamps_to_one
  - Brand new, many accesses, positive feedback
  - Assert result == 1.0
```

### Acceptance

`uv run pytest tests/memory/test_strength.py -v` passes all 12 tests. Module has no imports outside `math` and `__future__`.

### Verify

```bash
uv run pytest tests/memory/test_strength.py -v
python -c "
from simba.memory.strength import compute_strength
import time
s = compute_strength(
    created_at_epoch=time.time() - 86400*30,
    now=time.time(),
    access_count=0, feedback_score=0.0,
    half_life=30.0, reinforcement_scale=0.5, feedback_weight=0.2,
)
print(f'strength after 30 days no access: {s:.3f}')  # expect ~0.5
"
```

### Reuse

- `src/simba/memory/scoring.py:_recency` — same `0.5 ** (age/half_life)` pattern; `decay_factor` is the canonical replacement

---

## 3. Recall-Time Reinforcement

### Goal

When the `/recall` daemon route returns memories, bump `access_count` and `last_accessed` in `memory_usage` (sqlite) for each returned memory id. This runs alongside the existing LanceDB fire-and-forget `update_access_tracking`. The sqlite write is the authoritative one.

### Files

- `src/simba/memory/usage.py` — `bump_access` already defined in sub-feature 1; no new functions needed
- `src/simba/memory/routes.py` — modify the fire-and-forget block at lines 353-359
- `tests/memory/test_recall_reinforcement.py` — test file

### Config

No new config keys.

### Schema / Signatures

Add a new async helper in `routes.py` (private, adjacent to the existing `_background_tasks` pattern):

```python
async def _bump_usage(memory_ids: list[str], now: float, cwd: pathlib.Path) -> None:
    """Bump access stats in memory_usage for all recalled ids. Fire-and-forget."""
    import simba.db
    import simba.memory.usage

    def _sync() -> None:
        with simba.db.connect(cwd):
            for mid in memory_ids:
                simba.memory.usage.bump_access(mid, now)

    await asyncio.to_thread(_sync)
```

Modify the fire-and-forget section in `recall_memories` (around line 353 in `routes.py`):

```python
if results:
    recalled_ids = [r["id"] for r in results]
    now_epoch = time.time()
    cwd = pathlib.Path(getattr(request.app.state, "cwd", "."))

    # Existing LanceDB access tracking (keep as-is)
    task1 = asyncio.create_task(
        simba.memory.vector_db.update_access_tracking(table, recalled_ids)
    )
    _background_tasks.add(task1)
    task1.add_done_callback(_background_tasks.discard)

    # New: sqlite usage store (authoritative for ranking signals)
    task2 = asyncio.create_task(_bump_usage(recalled_ids, now_epoch, cwd))
    _background_tasks.add(task2)
    task2.add_done_callback(_background_tasks.discard)
```

The daemon's `app.state` must expose `cwd`. In `src/simba/memory/server.py`, where `app.state` is configured, add `app.state.cwd = str(pathlib.Path.cwd())` (or the `--cwd` arg if one exists). Look for where `app.state.table` and `app.state.config` are set.

### Implementation Steps

1. Open `src/simba/memory/server.py`. Find where `app.state` fields are initialized. Add `app.state.cwd = pathlib.Path(args.cwd) if hasattr(args, "cwd") else pathlib.Path.cwd()`. (If `args` does not have `cwd`, use `pathlib.Path.cwd()`.)
2. Open `src/simba/memory/routes.py`. Add `_bump_usage` async function as specified above.
3. In `recall_memories`, replace the single fire-and-forget task block with the two-task block above.
4. Import `simba.db` and `simba.memory.usage` inside `_bump_usage` (lazy import to avoid circular).

### Tests — `tests/memory/test_recall_reinforcement.py`

Use a fake `app.state` and `httpx.AsyncClient` or direct function calls. The cleanest approach: test `_bump_usage` directly.

```
test_bump_usage_writes_to_sqlite
  - Use tmp_path as cwd; call simba.db.connect(tmp_path) to pre-init
  - await _bump_usage(["mem_a", "mem_b"], now=1000.0, cwd=tmp_path)
  - Open simba.db.connect(tmp_path)
  - rows = simba.memory.usage.get_many(["mem_a", "mem_b"])
  - Assert "mem_a" in rows and "mem_b" in rows
  - Assert rows["mem_a"].access_count == 1
  - Assert rows["mem_a"].last_accessed == 1000.0

test_bump_usage_increments_on_repeated_recall
  - await _bump_usage(["mem_x"], now=100.0, cwd=tmp_path)
  - await _bump_usage(["mem_x"], now=200.0, cwd=tmp_path)
  - Assert rows["mem_x"].access_count == 2
  - Assert rows["mem_x"].last_accessed == 200.0

test_bump_usage_handles_empty_list
  - await _bump_usage([], now=0.0, cwd=tmp_path)
  - No exception; DB still empty
```

### Acceptance

After a `POST /recall` that returns memories, `memory_usage` rows for those ids have `access_count >= 1`. LanceDB `accessCount` also increments (unchanged behavior).

### Verify

```bash
uv run pytest tests/memory/test_recall_reinforcement.py -v
# Integration: start daemon, do a recall, then inspect simba.db
python -c "
import pathlib, simba.db, simba.memory.usage
with simba.db.connect(pathlib.Path('.')):
    rows = simba.memory.usage.get_all_for_decay(include_dormant=True)
    for r in rows[:5]:
        print(r.memory_id, r.access_count, r.last_accessed)
"
```

### Reuse

- `src/simba/memory/routes.py:_background_tasks` — same pattern for fire-and-forget
- `src/simba/memory/usage.py:bump_access` — from sub-feature 1

---

## 4. Scheduler Decay Pass

### Goal

A periodic function `run_decay_pass` that reads all `memory_usage` rows, recomputes `strength` using `compute_strength`, writes it back, and sets `dormant=True` for memories below `strength_dormancy_threshold`. Additionally, if `decay_capacity_per_type > 0`, it demotes the weakest memories per `(type, project_path)` bucket beyond the cap. Integrated into `SyncScheduler.run_once` as an async step.

### Files

- `src/simba/memory/decay.py` — `run_decay_pass` + helpers
- `src/simba/sync/scheduler.py` — add decay pass to `run_once`
- `tests/memory/test_decay.py` — test file

### Config

Uses fields added in sub-feature 2:
- `memory.decay_half_life_days`
- `memory.reinforcement_scale`
- `memory.feedback_weight`
- `memory.strength_dormancy_threshold`
- `memory.decay_capacity_per_type`

Plus one new field:

```
memory.decay_enabled: bool = True
  # Master switch. Set False to disable all decay/dormancy updates.
```

### Schema / Signatures

```python
# src/simba/memory/decay.py

import dataclasses
import pathlib
import typing

@dataclasses.dataclass
class DecayResult:
    processed: int = 0         # total rows visited
    updated: int = 0           # rows where strength changed
    newly_dormant: int = 0     # rows transitioned to dormant=True this pass
    revived: int = 0           # rows transitioning from dormant=True to False (strength recovered)
    errors: int = 0


def run_decay_pass(
    *,
    now: float,                 # epoch seconds (caller provides; never time.time() inside)
    cwd: pathlib.Path,
    cfg: typing.Any,            # MemoryConfig (or any object with the decay fields)
) -> DecayResult:
    """Synchronous decay pass over all memory_usage rows.

    Steps:
    1. Load all rows via get_all_for_decay(include_dormant=True).
    2. For each row, compute new_strength = compute_strength(
           created_at_epoch=row.created_at,
           now=now,
           access_count=row.access_count,
           feedback_score=row.feedback_score,
           half_life=cfg.decay_half_life_days,
           reinforcement_scale=cfg.reinforcement_scale,
           feedback_weight=cfg.feedback_weight,
       )
    3. If new_strength != row.strength (tolerance 1e-6): set_strength, update counter.
    4. If new_strength < cfg.strength_dormancy_threshold and not row.dormant:
           set_dormant(row.memory_id, dormant=True); newly_dormant += 1.
       If new_strength >= cfg.strength_dormancy_threshold and row.dormant:
           set_dormant(row.memory_id, dormant=False); revived += 1.
    5. If cfg.decay_capacity_per_type > 0: run _apply_capacity_cap(cfg, now).
    6. Return DecayResult.
    """


def _apply_capacity_cap(cfg: typing.Any, cwd: pathlib.Path) -> tuple[int, int]:
    """Demote the weakest non-dormant memories per (type, project_path) bucket.

    Fetches all non-dormant rows. Groups by (memory_type, project_path).
    For groups larger than decay_capacity_per_type, sets the weakest
    (lowest strength) beyond the cap to dormant=True.

    Note: memory_type and project_path are NOT stored in memory_usage.
    They live in LanceDB. The capacity cap therefore requires a bulk-fetch
    of LanceDB rows to join type/project_path. This is done via the daemon
    HTTP client (GET /list) — so this function is only called from
    run_decay_pass when the daemon is running.

    If the daemon is not reachable, skip this step silently and return (0, 0).
    Returns (groups_processed, memories_demoted).
    """
```

**Daemon integration**: `run_decay_pass` is synchronous and DB-only for the strength/dormancy steps (steps 1-4). Step 5 (capacity cap) calls the daemon HTTP client to get type metadata. This is acceptable because `run_decay_pass` is called from `SyncScheduler.run_once`, which already requires the daemon to be running.

**Scheduler integration**: In `SyncScheduler.run_once`, after the existing `epi` await, add:

```python
decay_result = await loop.run_in_executor(None, self._maybe_decay)
```

And add method:

```python
def _maybe_decay(self) -> DecayResult | None:
    """Run decay pass if enabled."""
    import time
    import simba.config
    import simba.memory.config  # registers "memory" section
    import simba.memory.decay

    _ = simba.memory.config
    cfg = simba.config.load("memory")
    if not getattr(cfg, "decay_enabled", True):
        return None
    return simba.memory.decay.run_decay_pass(
        now=time.time(),
        cwd=self.cwd,
        cfg=cfg,
    )
```

Add `decay` key to the `summary` dict in `run_once`:

```python
"decay": {
    "processed": decay_result.processed if decay_result else 0,
    "updated": decay_result.updated if decay_result else 0,
    "newly_dormant": decay_result.newly_dormant if decay_result else 0,
    "revived": decay_result.revived if decay_result else 0,
} if decay_result else {"skipped": True},
```

### Implementation Steps

1. Create `src/simba/memory/decay.py`.
2. Import `pathlib`, `dataclasses`, `typing`, `logging`.
3. Import `simba.db`, `simba.memory.usage`, `simba.memory.strength` at the top (not lazy — no circularity here).
4. Define `DecayResult` dataclass.
5. Implement `run_decay_pass` as a single `with simba.db.connect(cwd):` block enclosing all DB operations. Call `compute_strength` (pure, no DB) for each row. Batch the updates: collect `(memory_id, new_strength, new_dormant)` tuples, then issue peewee updates.
6. Implement `_apply_capacity_cap`: import `httpx` lazily; call `GET /list?limit=100000`; if `httpx.HTTPError`, return `(0, 0)`. Group non-dormant rows by `(type, projectPath)`. Sort each group by strength ascending. For each row beyond the cap, call `set_dormant(id, dormant=True)` inside `simba.db.connect(cwd)`.
7. Open `src/simba/sync/scheduler.py`. Add `_maybe_decay` method. In `run_once`, add `decay_result = await loop.run_in_executor(None, self._maybe_decay)`. Update the summary dict.
8. Add the five new config fields to `MemoryConfig` in `src/simba/memory/config.py`.

### Tests — `tests/memory/test_decay.py`

```
test_decay_pass_reduces_strength_of_old_memory
  - Insert a memory_usage row: created_at = now - 60*86400 (60 days ago), access_count=0
  - cfg: half_life=30.0, reinforcement_scale=0.5, feedback_weight=0.2,
          strength_dormancy_threshold=0.1, decay_enabled=True, decay_capacity_per_type=0
  - result = run_decay_pass(now=now, cwd=tmp_path, cfg=cfg)
  - row = get_many([...])
  - Assert row.strength < 0.5  # after 2 half-lives: ~0.25
  - Assert result.processed == 1
  - Assert result.updated == 1

test_decay_pass_sets_dormant_when_below_threshold
  - Created 200 days ago, no accesses, half_life=30.0 → strength ≈ 0.5^(200/30) ≈ 0.009
  - threshold = 0.1
  - Assert result.newly_dormant == 1
  - Assert row.dormant == True

test_decay_pass_revives_when_strength_recovers
  - Pre-set dormant=True, strength=0.05
  - Manually bump access_count to 20 (simulating past accesses)
  - Re-run decay pass; new strength = compute_strength(..., access_count=20) >> threshold
  - Assert result.revived == 1
  - Assert row.dormant == False

test_decay_pass_skips_when_disabled
  - cfg.decay_enabled = False
  - result = run_decay_pass(now=now, cwd=tmp_path, cfg=cfg)
  - Assert result is None (returns early)
  -- OR the scheduler's _maybe_decay returns None

test_decay_pass_is_deterministic
  - Same inputs → same DecayResult.updated, same strength values
  - Call twice with identical now, assert row.strength identical

test_decay_pass_no_rows_is_noop
  - Empty DB → result.processed == 0, no errors
```

### Acceptance

After running a decay pass on a DB with a 90-day-old unaccessed memory (`half_life=30.0`): strength ≈ `0.5^3 = 0.125`. After 100 days: `0.5^(100/30) ≈ 0.099 < 0.1` → dormant. The scheduler summary includes a `"decay"` key.

### Verify

```bash
uv run pytest tests/memory/test_decay.py -v
# Simulate via CLI after daemon is running:
python -c "
import time, pathlib, simba.config, simba.memory.config, simba.memory.decay
_ = simba.memory.config
cfg = simba.config.load('memory')
result = simba.memory.decay.run_decay_pass(now=time.time(), cwd=pathlib.Path('.'), cfg=cfg)
print(result)
"
```

### Reuse

- `src/simba/sync/scheduler.py:SyncScheduler.run_once` — structural pattern for `_maybe_decay`
- `src/simba/memory/strength.py:compute_strength` — the computation
- `src/simba/memory/usage.py:get_all_for_decay`, `set_strength`, `set_dormant`

---

## 5. Recall Excludes Dormant + Scoring Folds Strength

### Goal

Two changes to the recall pipeline:

**5a. Filter dormant memories**: After RRF fusion and before `composite_rescore`, load `memory_usage` rows for the candidate set and remove any with `dormant=True`.

**5b. Strength term in composite_rescore**: Add a `strength` term to `composite_rescore` weighted by `score_weight_strength`.

### Files

- `src/simba/memory/hybrid.py` — add dormant filter after `rrf_fuse`
- `src/simba/memory/scoring.py` — add strength term to `composite_rescore`
- `src/simba/memory/config.py` — two new config fields
- `tests/memory/test_dormant_filter.py` — test file
- `tests/memory/test_scoring_strength.py` — test file

### Config

```
memory.score_weight_strength: float = 0.4
  # Weight of the strength term in composite_rescore.
  # strength_score = usage_row.strength if row exists else 1.0 (no penalty for missing rows)

memory.dormant_filter_enabled: bool = True
  # When True, dormant memories are excluded from recall results.
```

### Schema / Signatures

**5a: Dormant filter in `hybrid.py`**

Add a synchronous helper and call it in `hybrid_search`:

```python
def _filter_dormant(
    records: list[dict[str, typing.Any]],
    cwd: pathlib.Path,
) -> list[dict[str, typing.Any]]:
    """Remove records whose memory_usage.dormant == True.

    Missing rows (no usage record yet) are treated as non-dormant.
    Called in a thread (sync sqlite) via asyncio.to_thread.
    """
    import simba.db
    import simba.memory.usage

    ids = [r["id"] for r in records if r.get("id")]
    if not ids:
        return records
    with simba.db.connect(cwd):
        usage_map = simba.memory.usage.get_many(ids)
    return [
        r for r in records
        if not usage_map.get(r.get("id", ""), _FAKE_NON_DORMANT).dormant
    ]

class _FAKE_NON_DORMANT:
    dormant = False
```

In `hybrid_search`, after the `composite_rescore` call and before the final `[:max_results]` slice, add:

```python
if getattr(cfg, "dormant_filter_enabled", True) and cwd is not None:
    fused = await asyncio.to_thread(_filter_dormant, fused, cwd)
```

`cwd` must be threaded into `hybrid_search`. Add `cwd: pathlib.Path | None = None` as a keyword-only parameter. Update the call in `routes.py` to pass `cwd=pathlib.Path(request.app.state.cwd)`.

**5b: Strength term in `scoring.py`**

Extend `composite_rescore` signature:

```python
def composite_rescore(
    records: list[dict[str, typing.Any]],
    *,
    cfg: typing.Any,
    now: float,
    usage_map: dict[str, typing.Any] | None = None,  # NEW: keyed by memory_id
) -> list[dict[str, typing.Any]]:
```

Inside `composite_rescore`, after computing `imp`:

```python
w_str = float(getattr(cfg, "score_weight_strength", 0.0))
if w_str and usage_map is not None:
    usage = usage_map.get(rec.get("id", ""), None)
    strength_val = float(usage.strength) if usage is not None else 1.0
else:
    strength_val = 1.0
composite = w_rel * rel + w_rec * rec_score + w_imp * imp + w_str * strength_val
```

The `usage_map` is loaded in `hybrid_search` just before calling `composite_rescore`:

```python
usage_map: dict[str, typing.Any] = {}
w_str = float(getattr(cfg, "score_weight_strength", 0.0))
if w_str:
    import simba.db
    import simba.memory.usage as _usage_mod
    def _load_usage() -> dict:
        ids = [r.get("id") for r in fused if r.get("id")]
        with simba.db.connect(cwd or pathlib.Path(".")):
            return _usage_mod.get_many(ids)
    usage_map = await asyncio.to_thread(_load_usage)

fused = simba.memory.scoring.composite_rescore(
    fused, cfg=cfg, now=time.time(), usage_map=usage_map
)
```

### Implementation Steps

1. Add `score_weight_strength: float = 0.4` and `dormant_filter_enabled: bool = True` to `MemoryConfig`.
2. Modify `composite_rescore` in `scoring.py`: add `usage_map` parameter, add `w_str` branch inside the loop. The signature change is backward-compatible (defaults to `None`).
3. In `hybrid.py`:
   a. Add `cwd: pathlib.Path | None = None` to `hybrid_search` signature.
   b. Add `_filter_dormant` helper.
   c. Add `_FAKE_NON_DORMANT` sentinel.
   d. After `composite_rescore`, before `[:max_results]`, call `_filter_dormant` via `asyncio.to_thread`.
   e. Before `composite_rescore`, load `usage_map` if `score_weight_strength > 0`.
4. In `routes.py`, pass `cwd=pathlib.Path(getattr(request.app.state, "cwd", "."))` to `hybrid_search`.
5. For the non-hybrid path (`search_memories` in `routes.py`), wrap the result list with a direct call to `_filter_dormant_sync` (a synchronous version accepting the same args) in a `to_thread` call.

### Tests — `tests/memory/test_dormant_filter.py`

```
test_filter_dormant_removes_dormant_records
  - Insert usage rows: "mem_a" dormant=False, "mem_b" dormant=True
  - records = [{"id": "mem_a", ...}, {"id": "mem_b", ...}]
  - result = _filter_dormant(records, cwd=tmp_path)
  - Assert len(result) == 1
  - Assert result[0]["id"] == "mem_a"

test_filter_dormant_keeps_missing_rows
  - No usage rows in DB
  - records = [{"id": "mem_new"}]
  - result = _filter_dormant(records, cwd=tmp_path)
  - Assert len(result) == 1  # missing row = non-dormant

test_filter_dormant_handles_empty_input
  - _filter_dormant([], tmp_path) == []
```

### Tests — `tests/memory/test_scoring_strength.py`

```
test_composite_rescore_strength_term_lifts_high_strength
  - Two records: r1 with rrf_score=0.5, r2 with rrf_score=0.5 (tied)
  - usage_map: r1.strength=1.0, r2.strength=0.1
  - cfg: scoring_enabled=True, score_weight_relevance=1.0, score_weight_recency=0.0,
          score_weight_importance=0.0, score_weight_strength=1.0,
          recency_halflife_days=90.0
  - result = composite_rescore(records, cfg=cfg, now=0.0, usage_map=usage_map)
  - Assert result[0]["id"] == r1_id  # higher strength wins tie

test_composite_rescore_strength_defaults_to_1_when_no_usage_row
  - usage_map = {} (empty)
  - record with rrf_score=0.5
  - Assert composite_score is computed without error
  - Assert composite_score > 0.0

test_composite_rescore_backward_compat_no_usage_map
  - Call composite_rescore(...) without usage_map kwarg
  - No exception; order is same as before (strength=1.0 for all, no discrimination)
```

### Acceptance

`simba memory recall "test query"` never returns a dormant memory. High-strength memories win ties against low-strength ones when `score_weight_strength > 0`.

### Verify

```bash
uv run pytest tests/memory/test_dormant_filter.py tests/memory/test_scoring_strength.py -v
# Manually mark a memory dormant and confirm it disappears from recall:
python -c "
import pathlib, simba.db, simba.memory.usage
with simba.db.connect(pathlib.Path('.')):
    simba.memory.usage.set_dormant('mem_XXXX', dormant=True)
"
simba memory recall "the query that normally retrieves mem_XXXX"
# mem_XXXX should not appear
```

### Reuse

- `src/simba/memory/scoring.py:composite_rescore` — extended, not replaced
- `src/simba/memory/hybrid.py:hybrid_search` — extended with `cwd` and filter step
- `src/simba/memory/usage.py:get_many` — bulk lookup

---

## 6. Outcome Feedback

### Goal

Allow a user or automated system to mark a recalled memory as `good` or `bad`. This adjusts `feedback_score` in `memory_usage`, which then feeds into `compute_strength` on the next decay pass. Exposed as:

- CLI: `simba memory feedback <id> good|bad [--weight 0.3]`
- Daemon route: `POST /memory/<id>/feedback` with `{"signal": "good"|"bad", "weight": 0.3}`

The signal maps: `good` → `+weight`, `bad` → `-weight`. Default weight is `0.3`. `feedback_score` is clamped to `[-1.0, 1.0]` (existing `apply_feedback` contract).

### Files

- `src/simba/memory/routes.py` — new route `POST /memory/{memory_id}/feedback`
- `src/simba/__main__.py` — new `_memory_feedback` function + wire into `_cmd_memory`
- `tests/memory/test_feedback.py` — test file

### Config

```
memory.feedback_default_weight: float = 0.3
  # Default delta applied per good/bad signal. Overridable per-call.
```

### Schema / Signatures

**Daemon route** (in `routes.py`):

```python
class FeedbackRequest(pydantic.BaseModel):
    signal: str                   # "good" or "bad"
    weight: float | None = None   # override; None → cfg.feedback_default_weight

@router.post("/memory/{memory_id}/feedback")
async def memory_feedback(
    memory_id: str,
    body: FeedbackRequest,
    request: fastapi.Request,
) -> dict:
    """Adjust feedback_score for a memory. Never deletes, never affects LanceDB."""
    cfg = request.app.state.config
    cwd = pathlib.Path(getattr(request.app.state, "cwd", "."))

    if body.signal not in ("good", "bad"):
        raise fastapi.HTTPException(400, detail="signal must be 'good' or 'bad'")

    weight = body.weight if body.weight is not None else getattr(
        cfg, "feedback_default_weight", 0.3
    )
    weight = max(0.0, min(1.0, float(weight)))
    delta = weight if body.signal == "good" else -weight

    import simba.db
    import simba.memory.usage

    now = time.time()

    def _apply() -> float:
        with simba.db.connect(cwd):
            simba.memory.usage.apply_feedback(memory_id, delta, now=now)
            rows = simba.memory.usage.get_many([memory_id])
            return rows[memory_id].feedback_score if memory_id in rows else 0.0

    new_score = await asyncio.to_thread(_apply)
    return {"status": "ok", "id": memory_id, "feedback_score": new_score}
```

**CLI function** (in `__main__.py`):

```python
def _memory_feedback(args: list[str]) -> int:
    """Mark a memory as good or bad.

    Usage: simba memory feedback <id> good|bad [--weight 0.3]
    """
```

Implementation: parse `args[0]` as `memory_id`, `args[1]` as `signal` (`good`/`bad`), parse `--weight`. Call `httpx.post(f"{url}/memory/{memory_id}/feedback", json={"signal": signal, "weight": weight})`. Print `feedback_score` from response.

Wire into `_cmd_memory` dispatch table: add `elif subcmd == "feedback": return _memory_feedback(rest)`.

Update `_MEMORY_USAGE` string to include the `feedback` subcommand entry.

### Implementation Steps

1. Add `feedback_default_weight: float = 0.3` to `MemoryConfig`.
2. In `routes.py`, define `FeedbackRequest` pydantic model and `memory_feedback` route.
3. In `__main__.py`, implement `_memory_feedback(args)`.
4. Add `elif subcmd == "feedback": return _memory_feedback(rest)` in `_cmd_memory` before the final `else`.
5. Update `_MEMORY_USAGE` docstring at the top of the dispatch section.

### Tests — `tests/memory/test_feedback.py`

Test the route in isolation using a fake `app.state` (monkeypatching `request.app.state`), or use `fastapi.testclient.TestClient` with a minimal app fixture.

```
test_feedback_good_increments_score
  - Create app with state: cwd=tmp_path, config=fake_cfg (feedback_default_weight=0.3)
  - POST /memory/mem_abc/feedback {"signal": "good"}
  - Assert response status == 200
  - Assert response["feedback_score"] == 0.3

test_feedback_bad_decrements_score
  - POST /memory/mem_abc/feedback {"signal": "bad"}
  - Assert response["feedback_score"] == -0.3  # starting from 0

test_feedback_repeated_good_clamps_at_one
  - POST /memory/mem_x/feedback {"signal": "good", "weight": 0.8}
  - POST /memory/mem_x/feedback {"signal": "good", "weight": 0.8}
  - Assert response["feedback_score"] == 1.0

test_feedback_invalid_signal_returns_400
  - POST /memory/mem_abc/feedback {"signal": "neutral"}
  - Assert response.status_code == 400

test_feedback_custom_weight
  - POST /memory/mem_abc/feedback {"signal": "good", "weight": 0.1}
  - Assert response["feedback_score"] == 0.1

test_feedback_missing_usage_row_creates_it
  - Empty DB; POST /memory/mem_new/feedback {"signal": "good"}
  - Assert response["feedback_score"] == 0.3
  - Verify row exists in DB with memory_id == "mem_new"
```

### Acceptance

`simba memory feedback mem_XXXX good` increases the memory's `feedback_score` in `simba.db`. On the next scheduler decay pass, the strength of `mem_XXXX` is higher than it would be without feedback. `simba memory feedback mem_XXXX bad` decreases strength, potentially triggering dormancy faster.

### Verify

```bash
uv run pytest tests/memory/test_feedback.py -v
# With daemon running:
simba memory feedback mem_XXXX good
# Then inspect:
python -c "
import pathlib, simba.db, simba.memory.usage
with simba.db.connect(pathlib.Path('.')):
    row = simba.memory.usage.get_many(['mem_XXXX'])
    print(row['mem_XXXX'].feedback_score)
"
```

### Reuse

- `src/simba/memory/routes.py:patch_memory` — same route-per-id pattern
- `src/simba/__main__.py:_memory_delete` — same argparse pattern for id + subverb
- `src/simba/memory/usage.py:apply_feedback` — the core operation

---

## 7. Usage/Feedback Eval Fixture

### Goal

A deterministic dataset + pytest tests that prove the decay/feedback system changes recall ranking as expected. No LanceDB, no daemon, no embedding — pure sqlite + scoring layer tests.

### Files

- `tests/memory/fixtures/decay_feedback_corpus.py` — in-module fixture factory (not a JSON file — avoids file I/O in tests)
- `tests/memory/test_decay_feedback_eval.py` — evaluation tests

### Config

No new config. Tests pass config via inline dataclass instances (not `simba.config.load`).

### Schema / Signatures

**Fixture factory** (`decay_feedback_corpus.py`):

```python
# tests/memory/fixtures/decay_feedback_corpus.py

import dataclasses

@dataclasses.dataclass
class CorpusEntry:
    memory_id: str
    created_at_epoch: float       # epoch seconds
    access_count: int
    feedback_score: float
    label: str                    # human label for assertion ("strong", "weak", "dormant_candidate")

def make_corpus(now: float) -> list[CorpusEntry]:
    """Return a fixed corpus of 5 entries with known expected ranking order."""
    day = 86400.0
    return [
        CorpusEntry("mem_fresh",    now - 1*day,    0, 0.0,  "strong"),
        CorpusEntry("mem_accessed", now - 30*day,   5, 0.0,  "strong"),
        CorpusEntry("mem_loved",    now - 60*day,   1, 1.0,  "strong"),
        CorpusEntry("mem_old",      now - 90*day,   0, 0.0,  "weak"),
        CorpusEntry("mem_hated",    now - 10*day,   0, -1.0, "weak"),
    ]
```

Expected strength ordering with `half_life=30, reinforcement_scale=0.5, feedback_weight=0.2`:

- `mem_fresh`: age=1d, d≈0.977, r=0, base=0.977, s≈0.977
- `mem_accessed`: age=30d, d=0.5, r=1-exp(-5/0.5)≈1.0, base=0.5+(0.5*1.0)=1.0, s≈1.0
- `mem_loved`: age=60d, d≈0.25, r=1-exp(-1/0.5)≈0.865, base=0.25+(0.75*0.865)≈0.899, feedback=1.2, s≈1.0 (clamped)
- `mem_old`: age=90d, d≈0.125, r=0, base=0.125, s≈0.125
- `mem_hated`: age=10d, d≈0.794, r=0, base=0.794, feedback=0.8, s≈0.635

Ranking by strength (desc): `mem_accessed ≈ mem_loved > mem_fresh > mem_hated > mem_old`

### Implementation Steps

1. Create `tests/memory/fixtures/` directory with empty `__init__.py`.
2. Create `tests/memory/fixtures/decay_feedback_corpus.py` with `make_corpus`.
3. Create `tests/memory/test_decay_feedback_eval.py` with tests below.

### Tests — `tests/memory/test_decay_feedback_eval.py`

Use `tmp_path` fixture. All config is a `types.SimpleNamespace` with the decay fields.

```
test_strength_ordering_matches_expected
  - now = 1_000_000_000.0  # fixed epoch
  - corpus = make_corpus(now)
  - cfg = SimpleNamespace(decay_half_life_days=30.0, reinforcement_scale=0.5,
                          feedback_weight=0.2, strength_dormancy_threshold=0.1,
                          decay_enabled=True, decay_capacity_per_type=0)
  - For each entry, compute s = compute_strength(
        created_at_epoch=e.created_at_epoch, now=now,
        access_count=e.access_count, feedback_score=e.feedback_score,
        half_life=cfg.decay_half_life_days,
        reinforcement_scale=cfg.reinforcement_scale,
        feedback_weight=cfg.feedback_weight,
    )
  - strength_by_id = {e.memory_id: s for e, s in zip(corpus, strengths)}
  - Assert strength_by_id["mem_accessed"] > strength_by_id["mem_fresh"]
       (reinforcement beats freshness for heavily-accessed old memory)
  - Assert strength_by_id["mem_fresh"] > strength_by_id["mem_old"]
  - Assert strength_by_id["mem_hated"] < strength_by_id["mem_fresh"]
       (negative feedback penalizes)
  - Assert strength_by_id["mem_old"] < cfg.strength_dormancy_threshold * 2
       (old memory approaches dormancy)

test_decay_pass_changes_db_strength
  - Seed memory_usage table with corpus entries (insert rows via get_or_create)
  - run_decay_pass(now=now, cwd=tmp_path, cfg=cfg)
  - rows = get_many([e.memory_id for e in corpus])
  - Assert rows["mem_old"].strength < rows["mem_fresh"].strength
  - Assert rows["mem_accessed"].strength >= rows["mem_fresh"].strength

test_decay_pass_marks_mem_old_dormant_at_extreme_age
  - Override corpus: created_at = now - 200*86400 for mem_old
  - After decay pass, Assert rows["mem_old"].dormant == True

test_feedback_lifts_strength_above_unfeedback_peer
  - Two memories: same age (30d), same access_count (0)
  - peer_a: feedback_score=0.0, peer_b: feedback_score=1.0
  - s_a = compute_strength(..., feedback_score=0.0, ...)
  - s_b = compute_strength(..., feedback_score=1.0, ...)
  - Assert s_b > s_a

test_feedback_bad_can_trigger_dormancy
  - Memory: age=20d, access_count=0, feedback_score=-1.0
  - half_life=30, feedback_weight=0.5
  - s = compute_strength(...)
  - With strong negative feedback: s = 0.794 * 0.5 ≈ 0.397; above threshold=0.1
  - Adjust: half_life=5d, feedback_weight=0.9 → d=0.5^4≈0.0625, s=0.0625*0.1≈0.006 < 0.1
  - Seed DB with that row, run decay pass
  - Assert rows[memory_id].dormant == True

test_dormant_filter_excludes_from_composite_rescore_pipeline
  - Two records: r1 (dormant=False, rrf_score=0.4), r2 (dormant=True, rrf_score=0.9)
  - Seed memory_usage in tmp_path
  - _filter_dormant([r1, r2], cwd=tmp_path) → only r1 returned
  - Assert r2 not in filtered, even though it had higher rrf_score

test_strength_weight_changes_ranking
  - Two records: r1 (rrf=0.5, strength=1.0), r2 (rrf=0.6, strength=0.1)
  - usage_map = {r1_id: SimpleNamespace(strength=1.0), r2_id: SimpleNamespace(strength=0.1)}
  - cfg: scoring_enabled=True, score_weight_relevance=1.0, score_weight_recency=0.0,
          score_weight_importance=0.0, score_weight_strength=2.0, recency_halflife_days=90
  - result = composite_rescore([r1, r2], cfg=cfg, now=0.0, usage_map=usage_map)
  - Assert result[0]["id"] == r1_id
       (r1's strength bonus overcomes r2's rrf_score lead)
  - Flip score_weight_strength=0.0 → Assert result[0]["id"] == r2_id
       (without strength, pure rrf order prevails)
```

### Acceptance

All 7 tests in `test_decay_feedback_eval.py` pass. The assertions document the exact ranking semantics so a regression immediately shows which invariant broke.

### Verify

```bash
uv run pytest tests/memory/test_decay_feedback_eval.py -v --tb=short
```

---

## Cross-Cutting Implementation Notes

### Build Sequence (Phase 6 Checklist)

1. **Sub-feature 1**: `usage.py` + `tests/memory/test_usage.py`. Run tests → GREEN. Confirm `memory_usage` table created on `connect`.
2. **Sub-feature 2**: `strength.py` + `tests/memory/test_strength.py` + add 5 config fields to `MemoryConfig`. Run tests → GREEN.
3. **Sub-feature 4 (partial)**: Add 5 config fields to `MemoryConfig` (already done in step 2). Create `decay.py` without the `_apply_capacity_cap` step initially. Run `test_decay.py` → GREEN for 5 of 6 tests.
4. **Sub-feature 3**: Modify `routes.py` and `server.py` for recall-time reinforcement. Run `tests/memory/test_recall_reinforcement.py` → GREEN.
5. **Sub-feature 5**: Modify `scoring.py` (add `usage_map` param), `hybrid.py` (add `cwd`, dormant filter, usage_map load). Add 2 config fields. Run `test_dormant_filter.py` and `test_scoring_strength.py` → GREEN.
6. **Sub-feature 4 (complete)**: Wire `_maybe_decay` into `SyncScheduler.run_once`. Implement `_apply_capacity_cap` with daemon HTTP. Complete `test_decay.py` → all GREEN.
7. **Sub-feature 6**: Daemon feedback route + CLI `feedback` subcommand. Run `test_feedback.py` → GREEN.
8. **Sub-feature 7**: Eval fixture + `test_decay_feedback_eval.py`. Run → GREEN.
9. **Final**: `uv run pytest tests/memory/ -v` — all tests pass. `uv run ruff check src/simba/memory/`. `uv run ruff format src/simba/memory/`.

### Error Handling

- All `simba.db.connect` calls in background tasks are wrapped in `try/except Exception`; failures log at `DEBUG` and never propagate (fire-and-forget contract).
- `run_decay_pass` catches per-row exceptions in an inner `try/except`, increments `result.errors`, and continues.
- `_apply_capacity_cap` catches `httpx.HTTPError` and returns `(0, 0)` silently.
- `_filter_dormant` in `hybrid.py`: if `connect` fails, log `DEBUG` and return `records` unchanged (fail-open — prefer showing a dormant memory over dropping all results).
- The feedback route returns `400` for invalid `signal` values; all other errors return `500` via FastAPI's default exception handler.

### State Management

The sqlite `memory_usage` table is append-on-first-access (rows created by `get_or_create` on first recall or feedback call) and mutable thereafter. No row is ever deleted. Dormancy is toggled. `feedback_score` is adjusted, never replaced wholesale. This ensures the append-only spirit is maintained: the historical trajectory of accesses and feedback is implicitly preserved in the final clamped values (not individually audited — that is out of scope for Phase 6).

### Performance

- `get_many(ids)` uses a single `WHERE memory_id IN (...)` query, not N individual lookups.
- The decay pass is a single table scan followed by batch updates. For 10,000 memories: ~10ms on modern hardware. Run at most once per sync cycle (300s default).
- `_filter_dormant` and the `usage_map` load in `hybrid_search` are each single-query sqlite reads behind `asyncio.to_thread`. The expected latency add is <5ms per recall.
- The capacity cap step in `_apply_capacity_cap` calls `GET /list?limit=100000`. This is expensive at corpus scale. When `decay_capacity_per_type = 0` (the default), this step is skipped entirely.

### Security

- `memory_id` in the daemon route (`/memory/{memory_id}/feedback`) must be validated to exist in `memory_usage` or LanceDB before writing. The route as specified does a `get_or_create`, which means supplying an arbitrary id creates a phantom usage row with no backing LanceDB entry. To prevent phantom rows: after `apply_feedback`, check if the returned row was just created and the id does not appear in a `/list` response; if absent, call `set_dormant(id, dormant=True)` immediately. This is a secondary guard — in production, the daemon is local-only (port 8741, no auth), so the attack surface is minimal.
- `weight` is clamped to `[0.0, 1.0]` in the route before computing `delta` so an adversarial `weight=999` cannot cause an outsized jump.

### ruff / Style Rules

- All new files: `from __future__ import annotations` at top.
- `TYPE_CHECKING` guard for annotation-only imports (none expected in this workstream since all imports are runtime-needed).
- Paths via `pathlib.Path`, not `os.path`.
- Line length 88 (ruff default).
- No `time.time()` inside pure functions — `now: float` is always a parameter.
- `peewee` accessed as `simba._vendor.peewee` (never installed as a top-level dep).
