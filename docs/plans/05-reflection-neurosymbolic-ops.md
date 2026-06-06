# Simba Implementation Spec: Phase 5 / Phase 7 / Ops Hardening

---

## Group A — Phase 5: Reflection

### Task A.1 — Add `REFLECTION` to the valid memory-type registries

**Goal:** Register `REFLECTION` as a first-class memory type in every place that enforces the allowed-type list, so stores and CLI validation accept it before any reflection logic runs.

**Files:**
- `src/simba/memory/routes.py` — `VALID_TYPES` list (line 78)
- `src/simba/__main__.py` — `_VALID_MEMORY_TYPES` set (line 982)

**Config:** None. This task is pure enumeration.

**Signatures/Schema:**

```python
# routes.py — append to VALID_TYPES list
VALID_TYPES = [
    "GOTCHA", "WORKING_SOLUTION", "PATTERN", "DECISION",
    "FAILURE", "PREFERENCE", "SYSTEM", "TOOL_RULE", "EPISODE",
    "REFLECTION",   # cross-session synthesized insight (Phase 5)
]

# __main__.py — append to _VALID_MEMORY_TYPES set
_VALID_MEMORY_TYPES = {
    "WORKING_SOLUTION", "GOTCHA", "PATTERN", "DECISION",
    "FAILURE", "PREFERENCE", "EPISODE",
    "REFLECTION",   # cross-session synthesized insight (Phase 5)
}
```

**Implementation steps:**
1. In `src/simba/memory/routes.py`, append `"REFLECTION"` to the `VALID_TYPES` list after `"EPISODE"`.
2. In `src/simba/__main__.py`, add `"REFLECTION"` to `_VALID_MEMORY_TYPES`.

**Tests:** `tests/episodes/test_consolidate.py` already has `TestEpisodeType`; mirror that class.

File: `tests/reflection/test_type_registration.py`

```python
# RED — fails before step 1/2

def test_reflection_in_routes_valid_types() -> None:
    import simba.memory.routes as routes
    assert "REFLECTION" in routes.VALID_TYPES

def test_reflection_in_cli_valid_types() -> None:
    import simba.__main__ as cli
    assert "REFLECTION" in cli._VALID_MEMORY_TYPES
```

**Acceptance:** Both assertions pass; `uv run pytest tests/reflection/test_type_registration.py`.

**Verify:**
```bash
uv run pytest tests/reflection/test_type_registration.py -v
```

**Reuse:** `src/simba/memory/routes.py:VALID_TYPES`, `src/simba/__main__.py:_VALID_MEMORY_TYPES`

---

### Task A.2 — `ReflectionConfig` dataclass

**Goal:** Expose every reflection knob through `simba config` so there are no hidden constants.

**Files:**
- `src/simba/reflection/__init__.py` — empty package marker
- `src/simba/reflection/config.py` — the `@configurable` dataclass

**Config (`reflection` section):**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `enabled` | `bool` | `True` | Master on/off switch |
| `scheduler_enabled` | `bool` | `True` | Allow background scheduler to trigger reflect pass |
| `min_source_memories` | `int` | `10` | Minimum non-REFLECTION/SYSTEM memories before a pass runs |
| `max_source_memories` | `int` | `100` | Cap on memories baked into the LLM prompt |
| `importance_threshold` | `float` | `0.6` | Confidence below which a candidate reflection is discarded |
| `deduplicate_threshold` | `float` | `0.88` | Vector similarity above which a new reflection is considered duplicate of an existing one |
| `interval_cycles` | `int` | `5` | Run reflection once every N scheduler sync cycles (0 = every cycle) |
| `max_reflections_per_pass` | `int` | `3` | Hard cap on new REFLECTION memories stored per pass |
| `project_scoped` | `bool` | `True` | Scope reflection to current project_path (False = global) |

**Signatures/Schema:**

```python
# src/simba/reflection/config.py
from __future__ import annotations
import dataclasses
import simba.config

@simba.config.configurable("reflection")
@dataclasses.dataclass
class ReflectionConfig:
    enabled: bool = True
    scheduler_enabled: bool = True
    min_source_memories: int = 10
    max_source_memories: int = 100
    importance_threshold: float = 0.6
    deduplicate_threshold: float = 0.88
    interval_cycles: int = 5
    max_reflections_per_pass: int = 3
    project_scoped: bool = True
```

**Implementation steps:**
1. Create `src/simba/reflection/__init__.py` (empty).
2. Write `src/simba/reflection/config.py` as above.

**Tests:** `tests/reflection/test_config.py`

```python
# RED — fails before config.py exists

def test_reflection_config_defaults() -> None:
    import simba.reflection.config as rc
    cfg = rc.ReflectionConfig()
    assert cfg.enabled is True
    assert cfg.min_source_memories == 10
    assert cfg.importance_threshold == 0.6

def test_reflection_config_via_simba_config(monkeypatch) -> None:
    import simba.config
    import simba.reflection.config  # registers section
    cfg = simba.config.load("reflection")
    assert hasattr(cfg, "scheduler_enabled")
    assert hasattr(cfg, "deduplicate_threshold")
```

**Acceptance:** Config round-trips through `simba config get reflection.enabled`.

**Verify:**
```bash
uv run pytest tests/reflection/test_config.py -v
uv run simba config get reflection.enabled
```

**Reuse:** `src/simba/episodes/config.py:EpisodesConfig` — identical `@configurable` pattern.

---

### Task A.3 — Reflection LLM prompt builder

**Goal:** A pure function that accepts a list of memory dicts and returns the LLM prompt string for cross-session insight synthesis. Testable in isolation, no LLM dependency.

**Files:**
- `src/simba/reflection/prompt.py`

**Config:** Uses `ReflectionConfig.max_source_memories` (passed in, not imported directly).

**Signatures/Schema:**

```python
# src/simba/reflection/prompt.py
from __future__ import annotations

_REFLECTION_PROMPT = """\
You are a memory consolidator reviewing {n} memories captured across multiple \
coding sessions for project '{project}'.

Your task: identify CROSS-SESSION patterns, recurring friction points, evolving \
decisions, or durable insights that are NOT captured in any single session \
summary (EPISODE). These must be non-obvious and high-value — skip anything \
already stated verbatim in a memory.

For each insight you identify (maximum {max_reflections}), store it with:
  simba memory store --type REFLECTION \\
    --content "<≤200-char cross-session insight>" \\
    --context "<evidence: list 2-4 memory ids that support this>" \\
    --confidence <0.0-1.0> \\
    --project-path '{project}'

Rules:
- content MUST be ≤200 characters
- confidence ≥ {importance_threshold} (discard weaker candidates)
- Do NOT paraphrase a single memory — that is an EPISODE, not a REFLECTION
- Do NOT store a reflection if an existing REFLECTION memory already captures it
- Prefer actionable insights over observations

Existing REFLECTIONs (do not duplicate):
{existing_reflections}

Memories to analyse:
{memory_lines}
"""

def build_reflection_prompt(
    memories: list[dict],
    *,
    project: str,
    existing_reflections: list[dict],
    max_source_memories: int = 100,
    max_reflections: int = 3,
    importance_threshold: float = 0.6,
) -> str:
    """Return the reflection synthesis prompt for the LLM."""
    ...
```

**Implementation steps:**
1. Write `build_reflection_prompt` to slice `memories[:max_source_memories]`, format each as `"- [{type}] {content} ({context[:120]})  <{id}>"`, format `existing_reflections` similarly, and interpolate into `_REFLECTION_PROMPT`.

**Tests:** `tests/reflection/test_prompt.py`

```python
def test_prompt_includes_project_and_memories() -> None:
    from simba.reflection.prompt import build_reflection_prompt
    mems = [{"id": "m1", "type": "GOTCHA", "content": "rg has no -r", "context": ""}]
    prompt = build_reflection_prompt(mems, project="/myproj", existing_reflections=[], max_reflections=3)
    assert "/myproj" in prompt
    assert "rg has no -r" in prompt
    assert "m1" in prompt
    assert "REFLECTION" in prompt

def test_prompt_caps_source_memories() -> None:
    from simba.reflection.prompt import build_reflection_prompt
    mems = [{"id": f"m{i}", "type": "PATTERN", "content": f"c{i}", "context": ""} for i in range(200)]
    prompt = build_reflection_prompt(mems, project="/p", existing_reflections=[], max_source_memories=10)
    # Only m0..m9 should appear, m10+ must not
    assert "m9" in prompt
    assert "m10" not in prompt

def test_prompt_lists_existing_reflections() -> None:
    from simba.reflection.prompt import build_reflection_prompt
    existing = [{"id": "r1", "type": "REFLECTION", "content": "old insight", "context": ""}]
    prompt = build_reflection_prompt([], project="/p", existing_reflections=existing)
    assert "old insight" in prompt
```

**Acceptance:** All three pass; prompt never exceeds token budget for normal inputs.

**Verify:**
```bash
uv run pytest tests/reflection/test_prompt.py -v
```

**Reuse:** `src/simba/episodes/consolidate.py:_build_episode_prompt` — same list-of-dicts formatting pattern.

---

### Task A.4 — `reflect_pass` orchestrator

**Goal:** The core reflection function: fetch memories from daemon, gate on `min_source_memories`, dedup against existing reflections, dispatch the LLM via the RLM engine, fail-open.

**Files:**
- `src/simba/reflection/pass_.py`

**Config:** Accepts `ReflectionConfig` injected; loads via `simba.config.load("reflection")` when not provided.

**Signatures/Schema:**

```python
# src/simba/reflection/pass_.py
from __future__ import annotations
import dataclasses

@dataclasses.dataclass
class ReflectResult:
    status: str          # "dispatched" | "disabled" | "no_engine" | "too_few" | "skipped_interval" | "error"
    memories_considered: int = 0
    existing_reflections: int = 0
    dispatched: bool = False
    errors: int = 0


def reflect_pass(
    *,
    cwd: str,
    cycle_count: int = 0,
    rcfg: "ReflectionConfig | None" = None,
    engine: object | None = None,
    daemon_url: str | None = None,
) -> ReflectResult:
    """Run one reflection synthesis pass. Returns a ReflectResult. Never raises."""
    ...
```

**Implementation steps:**
1. Load `rcfg` from `simba.config.load("reflection")` when not given; return `ReflectResult(status="disabled")` when `not rcfg.enabled`.
2. If `rcfg.interval_cycles > 0 and cycle_count % rcfg.interval_cycles != 0`, return `ReflectResult(status="skipped_interval")`.
3. Fetch memories via `httpx.get(f"{daemon_url}/list", params={"limit": 100000})`. Filter: exclude `SYSTEM`, `REFLECTION`. Scope to `projectPath == cwd` when `rcfg.project_scoped`.
4. If `len(source_memories) < rcfg.min_source_memories`, return `ReflectResult(status="too_few", memories_considered=len(source_memories))`.
5. Fetch existing `REFLECTION` memories similarly (filter by `type == "REFLECTION"`).
6. Resolve engine via `simba.rlm.engine.get_engine(rlm_cfg)` when not injected; return `ReflectResult(status="no_engine")` if `None`.
7. Build prompt via `simba.reflection.prompt.build_reflection_prompt`.
8. Call `engine.run(prompt, cwd=cwd)`; catch all exceptions, return `ReflectResult(status="error", errors=1)` on failure.
9. Return `ReflectResult(status="dispatched", memories_considered=len(source_memories), existing_reflections=len(existing), dispatched=True)`.

**Tests:** `tests/reflection/test_pass.py`

```python
# All tests use FakeEngine + monkeypatched _list_memories, no live LLM.

class FakeEngine:
    def __init__(self): self.runs = []
    def run(self, prompt, *, cwd): self.runs.append((prompt, cwd))

def _make_mems(n, mtype="GOTCHA", project="/proj"):
    return [{"id": f"m{i}", "type": mtype, "content": f"c{i}",
             "context": "", "projectPath": project} for i in range(n)]

def test_disabled_returns_early(monkeypatch):
    from simba.reflection.pass_ import reflect_pass
    from simba.reflection.config import ReflectionConfig
    result = reflect_pass(cwd="/proj", rcfg=ReflectionConfig(enabled=False), engine=FakeEngine())
    assert result.status == "disabled"
    assert not result.dispatched

def test_interval_gate(monkeypatch):
    from simba.reflection.pass_ import reflect_pass
    from simba.reflection.config import ReflectionConfig
    import simba.reflection.pass_ as rp
    monkeypatch.setattr(rp, "_list_memories", lambda *a, **k: _make_mems(20))
    result = reflect_pass(cwd="/proj", cycle_count=1,
                          rcfg=ReflectionConfig(interval_cycles=5), engine=FakeEngine())
    assert result.status == "skipped_interval"

def test_too_few_memories(monkeypatch):
    from simba.reflection.pass_ import reflect_pass
    from simba.reflection.config import ReflectionConfig
    import simba.reflection.pass_ as rp
    monkeypatch.setattr(rp, "_list_memories", lambda *a, **k: _make_mems(3))
    result = reflect_pass(cwd="/proj", rcfg=ReflectionConfig(min_source_memories=10),
                          engine=FakeEngine())
    assert result.status == "too_few"

def test_dispatches_when_eligible(monkeypatch):
    from simba.reflection.pass_ import reflect_pass
    from simba.reflection.config import ReflectionConfig
    import simba.reflection.pass_ as rp
    monkeypatch.setattr(rp, "_list_memories", lambda *a, **k: _make_mems(15))
    engine = FakeEngine()
    result = reflect_pass(cwd="/proj", cycle_count=0,
                          rcfg=ReflectionConfig(min_source_memories=5, interval_cycles=0),
                          engine=engine)
    assert result.status == "dispatched"
    assert result.dispatched
    assert len(engine.runs) == 1
    assert "/proj" in engine.runs[0][0]

def test_no_engine_returns_no_engine(monkeypatch):
    from simba.reflection.pass_ import reflect_pass
    from simba.reflection.config import ReflectionConfig
    import simba.reflection.pass_ as rp
    import simba.rlm.engine
    monkeypatch.setattr(rp, "_list_memories", lambda *a, **k: _make_mems(20))
    monkeypatch.setattr(simba.rlm.engine, "get_engine", lambda cfg: None)
    result = reflect_pass(cwd="/proj", rcfg=ReflectionConfig(min_source_memories=5))
    assert result.status == "no_engine"

def test_engine_error_returns_error(monkeypatch):
    from simba.reflection.pass_ import reflect_pass
    from simba.reflection.config import ReflectionConfig
    import simba.reflection.pass_ as rp
    monkeypatch.setattr(rp, "_list_memories", lambda *a, **k: _make_mems(15))
    class BrokenEngine:
        def run(self, *a, **k): raise RuntimeError("boom")
    result = reflect_pass(cwd="/proj", rcfg=ReflectionConfig(min_source_memories=5,
                          interval_cycles=0), engine=BrokenEngine())
    assert result.status == "error"
    assert result.errors == 1
```

**Acceptance:** All five pass; the pass never raises under any input.

**Verify:**
```bash
uv run pytest tests/reflection/test_pass.py -v
```

**Reuse:** `src/simba/episodes/consolidate.py:consolidate_session` — same gate/dispatch pattern. `src/simba/hooks/_memory_client.py:daemon_url` for default URL.

---

### Task A.5 — Wire `reflect_pass` into `SyncScheduler.run_once`

**Goal:** Call `reflect_pass` at the end of every sync cycle, guarded by `ReflectionConfig.scheduler_enabled`. Include result in the summary dict.

**Files:**
- `src/simba/sync/scheduler.py`

**Config:** `reflection.scheduler_enabled` (read inside `_maybe_reflect`).

**Implementation steps:**
1. Add `_maybe_reflect(self) -> dict` to `SyncScheduler` following the same pattern as `_maybe_consolidate`:
   ```python
   def _maybe_reflect(self) -> dict:
       import simba.config
       import simba.reflection.config  # registers section
       import simba.reflection.pass_
       rcfg = simba.config.load("reflection")
       if not rcfg.enabled or not rcfg.scheduler_enabled:
           return {"status": "disabled"}
       result = simba.reflection.pass_.reflect_pass(
           cwd=str(self.cwd),
           cycle_count=self._cycle_count,
           rcfg=rcfg,
           daemon_url=self.daemon_url,
       )
       return {"status": result.status, "dispatched": result.dispatched}
   ```
2. In `run_once`, after the `epi` call:
   ```python
   ref = await loop.run_in_executor(None, self._maybe_reflect)
   ```
3. Add `"reflection": ref` to the returned `summary` dict.

**Tests:** `tests/sync/test_scheduler.py` — add to existing class `TestRunOnce`:

```python
@pytest.mark.asyncio
@patch(_EXT, side_effect=_ext_side_effect)
@patch(_IDX, side_effect=_idx_side_effect)
@patch("simba.sync.scheduler.SyncScheduler._maybe_reflect",
       return_value={"status": "disabled"})
@patch("simba.sync.scheduler.SyncScheduler._maybe_consolidate",
       return_value={"dispatched": []})
async def test_summary_includes_reflection(mock_cons, mock_ref, mock_index, mock_extract):
    scheduler = SyncScheduler(interval_seconds=1)
    summary = await scheduler.run_once()
    assert "reflection" in summary
    assert summary["reflection"]["status"] == "disabled"
```

**Acceptance:** `summary["reflection"]` always present with a `status` key.

**Verify:**
```bash
uv run pytest tests/sync/test_scheduler.py -v
```

**Reuse:** `src/simba/sync/scheduler.py:_maybe_consolidate` (exact structural twin).

---

## Group B — Phase 7: Neuro-Symbolic Deductive Distillation

### Caveat (bake into all sub-phases)

The solver guarantees correctness *with respect to the formalization only*. It does not guarantee that the LLM's NL→logic translation is faithful to the original natural-language facts. The LLM lives exclusively in the extraction/proposal pipeline (sub-phases DERIVE proposal and INDUCE). The Z3/Datalog engines are a consistency/closure engine, not a truth oracle. All derived/verified edges are tagged `proof="derived:<rule_id>"` to make their provenance transparent. A dormant edge (`valid_to` stamped) is never deleted — this is the append-only contract.

---

### Task B.1 — Schema extensions: `kg_derived_edges` + `kg_rules` + `dormant` flag

**Goal:** Add two new tables and one column to `kg_edges` without touching existing data. All schema changes are idempotent migrations registered via `simba.db.register_schema`.

**Files:**
- `src/simba/neuron/schema.py` — new file; owns the DDL and migration

**Config:** None — schema is not configurable.

**Signatures/Schema:**

```sql
-- kg_derived_edges: materialized edges produced by the DERIVE step
CREATE TABLE IF NOT EXISTS kg_derived_edges (
    id INTEGER PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    subject_type TEXT,
    object_type TEXT,
    proof TEXT NOT NULL,           -- "derived:<rule_id>"
    source_edge_ids TEXT NOT NULL, -- JSON array of kg_edges.id values
    rule_id INTEGER,               -- FK to kg_rules.id (nullable for ad-hoc)
    confidence REAL DEFAULT 0.8,
    valid_from TEXT NOT NULL,
    valid_to TEXT,                 -- NULL = currently valid
    occurred_at TEXT,
    project_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(subject, predicate, object, project_path, valid_from)
);

CREATE INDEX IF NOT EXISTS idx_kg_derived_project
    ON kg_derived_edges(project_path);

-- kg_rules: Horn rules that the INDUCE step promotes from recurring derivation patterns
CREATE TABLE IF NOT EXISTS kg_rules (
    id INTEGER PRIMARY KEY,
    rule_text TEXT NOT NULL,       -- Datalog/Horn clause text
    head_predicate TEXT NOT NULL,  -- the predicate this rule derives
    confidence REAL DEFAULT 0.7,
    activation_count INTEGER DEFAULT 0,
    enabled INTEGER DEFAULT 1,     -- 0 = soft-disabled (not deleted)
    created_at TEXT NOT NULL,
    last_fired_at TEXT,
    UNIQUE(rule_text)
);
```

```python
# Migration function registered with simba.db
def _migrate_dormant_flag(conn: sqlite3.Connection) -> None:
    """Add dormant column to kg_edges (marks AGM-retracted edges)."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(kg_edges)")}
    if "dormant" not in cols:
        conn.execute("ALTER TABLE kg_edges ADD COLUMN dormant INTEGER DEFAULT 0")
```

**Implementation steps:**
1. Create `src/simba/neuron/schema.py`.
2. Write `_DERIVED_SCHEMA_SQL` (the two `CREATE TABLE` + index statements above).
3. Write `_init_neuron_schema(conn)`: runs `_DERIVED_SCHEMA_SQL` then `_migrate_dormant_flag`.
4. Call `simba.db.register_schema(_init_neuron_schema)` at module level.
5. Import this module from `src/simba/neuron/__init__.py` so schema installs on first neuron import.

**Tests:** `tests/neuron/test_schema.py`

```python
import pathlib, sqlite3
import pytest
import simba.db

@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    return db_path

def test_kg_derived_edges_created(tmp_db) -> None:
    import simba.neuron.schema  # triggers registration + migration
    with simba.db.connect():
        pass
    conn = sqlite3.connect(str(tmp_db))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "kg_derived_edges" in tables
    assert "kg_rules" in tables

def test_dormant_column_added(tmp_db) -> None:
    import simba.neuron.schema
    with simba.db.connect():
        pass
    conn = sqlite3.connect(str(tmp_db))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(kg_edges)")}
    assert "dormant" in cols

def test_migrations_are_idempotent(tmp_db) -> None:
    import simba.neuron.schema
    # Run twice — should not raise
    with simba.db.connect():
        pass
    with simba.db.connect():
        pass
```

**Acceptance:** Tables exist in `.simba/simba.db` after any `simba` command that imports `simba.neuron`.

**Verify:**
```bash
uv run pytest tests/neuron/test_schema.py -v
```

**Reuse:** `src/simba/kg/store.py:_init_schema` — same registration pattern. `src/simba/kg/store.py:_migrate_occurred_at` — idempotent column-add pattern.

---

### Task B.2 — `NeuronConfig` under `simba config`

**Goal:** Move the existing `ServerConfig` dataclass (currently plain, not `@configurable`) into the config system, and add all Phase 7 knobs.

**Files:**
- `src/simba/neuron/config.py` — replace current plain dataclass

**Config (`neuron` section):**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `python_cmd` | `str` | `sys.executable` | Python interpreter for Z3 subprocess |
| `souffle_cmd` | `str` | `shutil.which("souffle") or ""` | Soufflé binary path |
| `enabled` | `bool` | `True` | Master switch for all Phase 7 operations |
| `derive_enabled` | `bool` | `True` | Run the DERIVE (Datalog materialization) step |
| `verify_enabled` | `bool` | `True` | Run the VERIFY (Z3 consistency check) step |
| `revise_enabled` | `bool` | `True` | Run the REVISE (AGM contraction) step |
| `distill_enabled` | `bool` | `True` | Write derived edges back to `kg_derived_edges` |
| `induce_enabled` | `bool` | `True` | Promote recurring patterns to `kg_rules` |
| `derive_max_edges` | `int` | `500` | Limit on edges fed to Datalog per cycle |
| `verify_timeout_seconds` | `int` | `30` | Z3 subprocess timeout |
| `induce_min_activations` | `int` | `3` | Minimum derivation repetitions before a rule is promoted |
| `induce_min_confidence` | `float` | `0.7` | Confidence gate for promoted rules |
| `contradiction_sample_size` | `int` | `200` | Max edges encoded in Z3 per verify pass |

**Signatures/Schema:**

```python
# src/simba/neuron/config.py
from __future__ import annotations
import dataclasses
import shutil
import sys
import simba.config

@simba.config.configurable("neuron")
@dataclasses.dataclass
class NeuronConfig:
    python_cmd: str = dataclasses.field(default_factory=lambda: sys.executable)
    souffle_cmd: str = dataclasses.field(default_factory=lambda: shutil.which("souffle") or "")
    enabled: bool = True
    derive_enabled: bool = True
    verify_enabled: bool = True
    revise_enabled: bool = True
    distill_enabled: bool = True
    induce_enabled: bool = True
    derive_max_edges: int = 500
    verify_timeout_seconds: int = 30
    induce_min_activations: int = 3
    induce_min_confidence: float = 0.7
    contradiction_sample_size: int = 200

# Keep backward-compat alias for existing callers of CONFIG.python_cmd / CONFIG.souffle_cmd
def _load() -> NeuronConfig:
    import simba.config as _sc
    return _sc.load("neuron")

CONFIG = _load()  # module-level for existing verify.py callers — refreshed on import
```

**Implementation steps:**
1. Replace `config.py` content with the `@configurable` dataclass above.
2. Update `src/simba/neuron/verify.py` to call `simba.config.load("neuron")` instead of referencing the module-level `CONFIG` directly (replace `simba.neuron.config.CONFIG.python_cmd` → `simba.config.load("neuron").python_cmd`).

**Tests:** `tests/neuron/test_config.py` — extend existing file:

```python
def test_neuron_config_has_phase7_fields() -> None:
    import simba.neuron.config as nc
    cfg = nc.NeuronConfig()
    assert cfg.derive_enabled is True
    assert cfg.verify_timeout_seconds == 30
    assert cfg.induce_min_activations == 3

def test_neuron_config_via_simba_config(monkeypatch) -> None:
    import simba.config
    import simba.neuron.config  # registers section
    cfg = simba.config.load("neuron")
    assert hasattr(cfg, "contradiction_sample_size")
```

**Acceptance:** `simba config get neuron.derive_enabled` returns `True`.

**Verify:**
```bash
uv run pytest tests/neuron/test_config.py -v
uv run simba config get neuron.verify_timeout_seconds
```

**Reuse:** `src/simba/episodes/config.py` — `@configurable` pattern. `src/simba/memory/config.py` — `@configurable` pattern.

---

### Task B.3 — Sub-phase DERIVE: Datalog materialization over `kg_edges`

**Goal:** Export the current `kg_edges` for a project as Soufflé `.facts` files, run seed Horn rules, collect new candidate derived edges with provenance.

**Files:**
- `src/simba/neuron/derive.py`

**Config:** `NeuronConfig.derive_enabled`, `derive_max_edges`, `souffle_cmd`.

**Signatures/Schema:**

```python
# src/simba/neuron/derive.py
from __future__ import annotations
import dataclasses

@dataclasses.dataclass
class DerivedEdge:
    subject: str
    predicate: str
    object: str
    source_edge_ids: list[int]   # kg_edges.id values that fired this rule
    rule_id: int | None           # kg_rules.id if from a stored rule, else None
    confidence: float = 0.8
    occurred_at: str | None = None

@dataclasses.dataclass
class DeriveResult:
    candidates: list[DerivedEdge]
    rules_applied: int = 0
    edges_fed: int = 0
    errors: int = 0
    souffle_output: str = ""

_SEED_RULES: str = """\
// Seed Horn rules for Phase 7 DERIVE pass
// subject(X,T1) :- edge(X,_,_,T1). etc.
// Transitivity: if A uses_tool B and B runs_on C then A runs_on C (via uses_tool)
.decl edge(sub:symbol, pred:symbol, obj:symbol, id:number)
.decl derived(sub:symbol, pred:symbol, obj:symbol, via1:number, via2:number)

derived(A, "transitively_uses", C, ID1, ID2) :-
    edge(A, "uses", B, ID1),
    edge(B, "uses", C, ID2),
    A != C.

derived(A, "co_occurs_with", B, ID1, ID2) :-
    edge(A, "causes", X, ID1),
    edge(B, "causes", X, ID2),
    A != B.

.output derived
"""

def run_derive(
    project_path: str,
    *,
    cfg: "NeuronConfig | None" = None,
    extra_rules: str = "",
) -> DeriveResult:
    """Materialize derived edges via Soufflé. Returns DeriveResult. Never raises."""
    ...
```

**Implementation steps:**
1. Load `cfg` from `simba.config.load("neuron")` when not given.
2. Return `DeriveResult(candidates=[], errors=0)` when `not cfg.derive_enabled` or `not cfg.souffle_cmd`.
3. Fetch up to `cfg.derive_max_edges` currently-valid (non-dormant) edges from `kg_edges` for `project_path`.
4. Write `edge.facts` to a `tempfile.mkdtemp()` directory: one tab-separated line per edge `"subject\tpredicate\tobject\tid"`.
5. Combine `_SEED_RULES + extra_rules` into a `.dl` temp file (using `analyze_datalog` pattern from `verify.py`).
6. Run Soufflé via `subprocess.run([cfg.souffle_cmd, "-F", facts_dir, "-D", "-", dl_file])` with timeout `60`.
7. Parse `stdout` lines of `derived.csv` format `sub,pred,obj,via1,via2` into `DerivedEdge` objects.
8. Clean up temp dirs.
9. Return `DeriveResult`.

**Tests:** `tests/neuron/test_derive.py`

```python
import pytest
from unittest.mock import patch

def test_derive_disabled_returns_empty(monkeypatch) -> None:
    from simba.neuron.derive import run_derive, DeriveResult
    from simba.neuron.config import NeuronConfig
    result = run_derive("/proj", cfg=NeuronConfig(derive_enabled=False))
    assert result.candidates == []
    assert result.errors == 0

def test_derive_no_souffle_returns_empty() -> None:
    from simba.neuron.derive import run_derive
    from simba.neuron.config import NeuronConfig
    result = run_derive("/proj", cfg=NeuronConfig(souffle_cmd=""))
    assert result.candidates == []

def test_derive_with_fake_souffle(tmp_path, monkeypatch) -> None:
    """Monkeypatch subprocess.run to return a fake Soufflé output."""
    import subprocess
    from simba.neuron.derive import run_derive
    from simba.neuron.config import NeuronConfig
    import simba.db, simba.kg.store

    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    # Insert two edges that should trigger transitivity rule
    simba.kg.store.kg_add("A", "uses", "B", "test", project_path="/proj")
    simba.kg.store.kg_add("B", "uses", "C", "test", project_path="/proj")

    fake_output = "A\ttransitively_uses\tC\t1\t2\n"
    mock_result = type("R", (), {"returncode": 0, "stdout": fake_output, "stderr": ""})()

    with patch("subprocess.run", return_value=mock_result):
        result = run_derive("/proj", cfg=NeuronConfig(souffle_cmd="souffle", derive_enabled=True))

    assert len(result.candidates) == 1
    assert result.candidates[0].subject == "A"
    assert result.candidates[0].predicate == "transitively_uses"
    assert result.candidates[0].object == "C"

def test_derive_subprocess_error_is_fail_open(monkeypatch) -> None:
    from simba.neuron.derive import run_derive
    from simba.neuron.config import NeuronConfig
    import subprocess
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: pathlib.Path("/nonexistent/simba.db"))
    with patch("subprocess.run", side_effect=FileNotFoundError("no souffle")):
        result = run_derive("/proj", cfg=NeuronConfig(souffle_cmd="souffle"))
    assert result.errors >= 1
    assert result.candidates == []
```

**Acceptance:** With Soufflé installed: `run_derive` materializes transitivity edges on a seeded graph. Without Soufflé: always fail-open.

**Verify:**
```bash
uv run pytest tests/neuron/test_derive.py -v
```

**Reuse:** `src/simba/neuron/verify.py:analyze_datalog` — temp file + subprocess pattern. `src/simba/kg/store.py:kg_query` — fetching edges.

---

### Task B.4 — Sub-phase VERIFY: Z3 constraint encoding + UNSAT core extraction

**Goal:** Encode a set of `kg_edges` + `kg_derived_edges` as Z3 Boolean constraints (one per edge assertion) under bitemporal validity, check satisfiability, and on UNSAT extract the minimal conflicting subset.

**Files:**
- `src/simba/neuron/z3_verify.py`

**Config:** `NeuronConfig.verify_enabled`, `verify_timeout_seconds`, `contradiction_sample_size`.

**Signatures/Schema:**

```python
# src/simba/neuron/z3_verify.py
from __future__ import annotations
import dataclasses

@dataclasses.dataclass
class VerifyResult:
    satisfiable: bool
    unsat_edge_ids: list[int]   # ids from kg_edges of the minimal UNSAT core
    checked_edges: int = 0
    errors: int = 0
    raw_output: str = ""

def build_z3_script(edges: list[dict]) -> str:
    """Return a self-contained Python/Z3 script string that:
    1. Declares one Bool per (subject, predicate, object) triple.
    2. Adds a constraint that if two edges share subject+object but have
       mutually-exclusive predicates (e.g., "uses" vs "does_not_use")
       they cannot both be true.
    3. Calls s.check(); on UNSAT calls s.unsat_core() and prints
       "UNSAT:<id1>,<id2>,..." to stdout; on SAT prints "SAT".
    """
    ...

def run_verify(
    project_path: str,
    *,
    cfg: "NeuronConfig | None" = None,
    extra_edges: list[dict] | None = None,
) -> VerifyResult:
    """Run Z3 consistency check. Returns VerifyResult. Never raises."""
    ...
```

**Implementation steps:**
1. Load `cfg`; return `VerifyResult(satisfiable=True, unsat_edge_ids=[])` when `not cfg.verify_enabled`.
2. Fetch `cfg.contradiction_sample_size` currently-valid edges from `kg_edges` (non-dormant). Append `extra_edges` if given.
3. Call `build_z3_script(edges)`:
   - Assign each edge an integer id and a `Bool(f"e{id}")`.
   - Mutual-exclusion constraint: for each pair sharing `(subject, object)` where predicates are antonyms (a small hardcoded set: `uses/does_not_use`, `prefers/avoids`, `fixes/breaks`), add `Not(And(e_a, e_b))`.
   - Bitemporal constraint: two edges with same `(subject, predicate, object)` and overlapping `(valid_from, valid_to)` cannot both be `True`.
   - Add `s.add(edge_bool)` for every edge (assert it holds).
   - `result = s.check()` → on `unsat`: `core = s.unsat_core()`, print `"UNSAT:" + ",".join(str(b) for b in core)`.
   - On `sat`: print `"SAT"`.
4. Call `verify_z3(script)` (the existing subprocess runner in `verify.py`).
5. Parse stdout: `"SAT"` → `satisfiable=True`; `"UNSAT:e1,e3"` → parse edge ids from bool names.
6. Return `VerifyResult`.

**Tests:** `tests/neuron/test_z3_verify.py`

```python
def test_sat_empty_graph() -> None:
    from simba.neuron.z3_verify import run_verify, VerifyResult
    from simba.neuron.config import NeuronConfig
    result = run_verify("/proj", cfg=NeuronConfig(verify_enabled=False))
    assert result.satisfiable is True
    assert result.unsat_edge_ids == []

def test_build_z3_script_sat_no_conflict() -> None:
    from simba.neuron.z3_verify import build_z3_script
    edges = [
        {"id": 1, "subject": "A", "predicate": "uses", "object": "B",
         "valid_from": "2024-01-01T00:00:00Z", "valid_to": None},
    ]
    script = build_z3_script(edges)
    assert "e1" in script
    assert "s.check()" in script
    assert "unsat_core" in script

def test_build_z3_script_detects_contradiction() -> None:
    """Two edges: A uses B and A does_not_use B — script should encode UNSAT."""
    from simba.neuron.z3_verify import build_z3_script
    edges = [
        {"id": 1, "subject": "A", "predicate": "uses", "object": "B",
         "valid_from": "2024-01-01T00:00:00Z", "valid_to": None},
        {"id": 2, "subject": "A", "predicate": "does_not_use", "object": "B",
         "valid_from": "2024-01-01T00:00:00Z", "valid_to": None},
    ]
    script = build_z3_script(edges)
    # The script must contain the Not(And(...)) mutual-exclusion constraint
    assert "Not(And(" in script or "Not(And" in script

def test_planted_contradiction_fixture(monkeypatch, tmp_path) -> None:
    """Regression fixture: verify detects a planted USES/DOES_NOT_USE pair."""
    import simba.db, simba.kg.store
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    simba.kg.store.kg_add("X", "uses", "Y", "test", project_path="/proj")
    simba.kg.store.kg_add("X", "does_not_use", "Y", "test", project_path="/proj")

    from simba.neuron.z3_verify import run_verify
    from simba.neuron.config import NeuronConfig
    # Skip if z3 not installed (CI guard)
    try:
        import z3  # noqa: F401
    except ImportError:
        pytest.skip("z3 not installed")
    result = run_verify("/proj", cfg=NeuronConfig(verify_enabled=True,
                        contradiction_sample_size=50, verify_timeout_seconds=15))
    assert result.satisfiable is False
    assert len(result.unsat_edge_ids) >= 2
```

**Acceptance:** The planted-contradiction fixture (`X uses Y` + `X does_not_use Y`) produces `satisfiable=False` and `len(unsat_edge_ids) >= 2`.

**Verify:**
```bash
uv run pytest tests/neuron/test_z3_verify.py -v
```

**Reuse:** `src/simba/neuron/verify.py:verify_z3` — subprocess runner for Z3 scripts.

---

### Task B.5 — Sub-phase REVISE: AGM contraction via entrenchment order

**Goal:** Given the UNSAT core edge ids from VERIFY, apply AGM-style contraction: drop the weaker conflicting fact by stamping it dormant (never deleted; `valid_to` stamped + `dormant=1`). Entrenchment order: higher `(occurred_at, ingestion_time=valid_from, confidence)` wins.

**Files:**
- `src/simba/neuron/revise.py`

**Config:** `NeuronConfig.revise_enabled`.

**Signatures/Schema:**

```python
# src/simba/neuron/revise.py
from __future__ import annotations
import dataclasses

@dataclasses.dataclass
class ReviseResult:
    dormant_edge_ids: list[int]   # edges stamped dormant
    retained_edge_ids: list[int]  # edges kept
    skipped: int = 0              # pairs where ordering was tied (no action)
    errors: int = 0

def entrenchment_score(edge: dict) -> tuple[str, str, float]:
    """Return (occurred_at or "", valid_from or "", confidence) for ordering.
    Higher tuple = more entrenched = kept.
    """
    ...

def revise_unsat_core(
    unsat_edge_ids: list[int],
    *,
    project_path: str,
    cfg: "NeuronConfig | None" = None,
) -> ReviseResult:
    """Mark the weaker edge in each conflicting pair as dormant. Never raises."""
    ...
```

**Implementation steps:**
1. Load `cfg`; return `ReviseResult(dormant_edge_ids=[], retained_edge_ids=[])` when `not cfg.revise_enabled`.
2. Fetch all edges with `id IN unsat_edge_ids` from `kg_edges`.
3. Group by `(subject, object)` to find conflicting pairs (same endpoints, different predicate).
4. For each pair, compute `entrenchment_score` for each side; the lower-scored edge is dormant.
5. On tie: leave both active (append to `skipped`).
6. For dormant candidates: `UPDATE kg_edges SET valid_to=now(), dormant=1 WHERE id=?` (uses `simba.db.connect()`).
7. Return `ReviseResult`.

**Tests:** `tests/neuron/test_revise.py`

```python
def test_revise_disabled_returns_empty() -> None:
    from simba.neuron.revise import revise_unsat_core
    from simba.neuron.config import NeuronConfig
    result = revise_unsat_core([1, 2], project_path="/proj",
                               cfg=NeuronConfig(revise_enabled=False))
    assert result.dormant_edge_ids == []

def test_revise_stamps_weaker_edge_dormant(tmp_path, monkeypatch) -> None:
    import simba.db, simba.kg.store
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    # Insert: older edge (lower entrenchment) conflicts with newer
    import time
    simba.kg.store.kg_add("A", "uses", "B", "test", project_path="/proj")
    time.sleep(0.01)
    simba.kg.store.kg_add("A", "does_not_use", "B", "test", project_path="/proj")

    from simba.neuron.revise import revise_unsat_core
    from simba.neuron.config import NeuronConfig
    import simba.db as db
    with db.connect():
        from simba.kg.store import KgEdge
        ids = [e.id for e in KgEdge.select().where(KgEdge.project_path == "/proj")]
    assert len(ids) == 2

    result = revise_unsat_core(ids, project_path="/proj",
                               cfg=NeuronConfig(revise_enabled=True))
    assert len(result.dormant_edge_ids) == 1
    assert len(result.retained_edge_ids) == 1

    with db.connect():
        from simba.kg.store import KgEdge
        dormant_e = KgEdge.get_by_id(result.dormant_edge_ids[0])
        assert dormant_e.dormant == 1
        assert dormant_e.valid_to is not None

def test_revise_tied_entrenchment_skips(tmp_path, monkeypatch) -> None:
    from simba.neuron.revise import entrenchment_score
    edge_a = {"id": 1, "occurred_at": "2024-01-01T00:00:00Z",
               "valid_from": "2024-01-01T00:00:00Z", "confidence": 0.8}
    edge_b = {"id": 2, "occurred_at": "2024-01-01T00:00:00Z",
               "valid_from": "2024-01-01T00:00:00Z", "confidence": 0.8}
    assert entrenchment_score(edge_a) == entrenchment_score(edge_b)
```

**Acceptance:** A planted contradiction results in exactly one dormant edge and one retained edge. The dormant edge's `valid_to` is non-null and `dormant=1`. The edge row still exists (never deleted).

**Verify:**
```bash
uv run pytest tests/neuron/test_revise.py -v
```

**Reuse:** `src/simba/kg/store.py:kg_invalidate` — pattern for stamping `valid_to`.

---

### Task B.6 — Sub-phase DISTILL: write verified derived edges to `kg_derived_edges`

**Goal:** Write `DerivedEdge` candidates (from DERIVE, surviving VERIFY+REVISE) back to `kg_derived_edges` with proof chain. Return counts of new vs duplicate.

**Files:**
- `src/simba/neuron/distill.py`

**Config:** `NeuronConfig.distill_enabled`.

**Signatures/Schema:**

```python
# src/simba/neuron/distill.py
from __future__ import annotations
import dataclasses
from simba.neuron.derive import DerivedEdge

@dataclasses.dataclass
class DistillResult:
    added: int = 0
    duplicates: int = 0
    errors: int = 0

def distill_edges(
    candidates: list[DerivedEdge],
    *,
    project_path: str,
    cfg: "NeuronConfig | None" = None,
) -> DistillResult:
    """Write verified DerivedEdge candidates to kg_derived_edges. Never raises."""
    ...
```

**Implementation steps:**
1. Load `cfg`; return `DistillResult()` when `not cfg.distill_enabled`.
2. For each candidate:
   a. Build `proof = f"derived:{candidate.rule_id or 'adhoc'}"`.
   b. Build `source_edge_ids = json.dumps(candidate.source_edge_ids)`.
   c. `INSERT OR IGNORE INTO kg_derived_edges (subject, predicate, object, proof, source_edge_ids, rule_id, confidence, valid_from, valid_to, occurred_at, project_path, created_at) VALUES (...)`.
   d. `rowcount == 0` → duplicate; else `added += 1`.
3. Return `DistillResult`.

**Tests:** `tests/neuron/test_distill.py`

```python
def test_distill_disabled_returns_empty() -> None:
    from simba.neuron.distill import distill_edges, DistillResult
    from simba.neuron.config import NeuronConfig
    from simba.neuron.derive import DerivedEdge
    result = distill_edges([DerivedEdge("A","uses","B",[1],None)],
                           project_path="/proj", cfg=NeuronConfig(distill_enabled=False))
    assert result.added == 0

def test_distill_inserts_new_edge(tmp_path, monkeypatch) -> None:
    import simba.db
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    import simba.neuron.schema  # ensure tables created
    with simba.db.connect(): pass

    from simba.neuron.distill import distill_edges
    from simba.neuron.config import NeuronConfig
    from simba.neuron.derive import DerivedEdge
    result = distill_edges([DerivedEdge("A","transitively_uses","C",[1,2],None,0.8)],
                           project_path="/proj",
                           cfg=NeuronConfig(distill_enabled=True))
    assert result.added == 1
    assert result.duplicates == 0

def test_distill_deduplicates(tmp_path, monkeypatch) -> None:
    import simba.db
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    import simba.neuron.schema
    with simba.db.connect(): pass

    from simba.neuron.distill import distill_edges
    from simba.neuron.config import NeuronConfig
    from simba.neuron.derive import DerivedEdge
    cand = DerivedEdge("A","transitively_uses","C",[1,2],None,0.8)
    distill_edges([cand], project_path="/proj", cfg=NeuronConfig(distill_enabled=True))
    result2 = distill_edges([cand], project_path="/proj", cfg=NeuronConfig(distill_enabled=True))
    assert result2.duplicates == 1
```

**Acceptance:** Unique edges are inserted; duplicate candidates are silently skipped; `kg_derived_edges` grows monotonically.

**Verify:**
```bash
uv run pytest tests/neuron/test_distill.py -v
```

**Reuse:** `src/simba/kg/store.py:kg_add` — INSERT OR IGNORE + duplicate detection pattern.

---

### Task B.7 — Sub-phase INDUCE: promote recurring derivation patterns to `kg_rules`

**Goal:** Scan `kg_derived_edges` for rules fired `>= induce_min_activations` times, and insert them into `kg_rules` when `confidence >= induce_min_confidence`.

**Files:**
- `src/simba/neuron/induce.py`

**Config:** `NeuronConfig.induce_enabled`, `induce_min_activations`, `induce_min_confidence`.

**Signatures/Schema:**

```python
# src/simba/neuron/induce.py
from __future__ import annotations
import dataclasses

@dataclasses.dataclass
class InduceResult:
    promoted: int = 0
    already_known: int = 0
    below_threshold: int = 0
    errors: int = 0

def induce_rules(
    *,
    project_path: str,
    cfg: "NeuronConfig | None" = None,
) -> InduceResult:
    """Promote recurring derivation patterns to kg_rules. Never raises."""
    ...
```

**Implementation steps:**
1. Load `cfg`; return `InduceResult()` when `not cfg.induce_enabled`.
2. Query `kg_derived_edges` grouped by `(rule_id, predicate)` counting activations. Filter: `rule_id IS NOT NULL AND activation_count >= cfg.induce_min_activations AND avg_confidence >= cfg.induce_min_confidence`.
3. For each candidate rule: `INSERT OR IGNORE INTO kg_rules (rule_text, head_predicate, confidence, activation_count, created_at)`. `rowcount == 0` → `already_known`.
4. Return `InduceResult`.

**Tests:** `tests/neuron/test_induce.py`

```python
def test_induce_disabled_returns_empty() -> None:
    from simba.neuron.induce import induce_rules
    from simba.neuron.config import NeuronConfig
    result = induce_rules(project_path="/proj", cfg=NeuronConfig(induce_enabled=False))
    assert result.promoted == 0

def test_induce_promotes_frequent_pattern(tmp_path, monkeypatch) -> None:
    import simba.db, sqlite3, json, time
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    import simba.neuron.schema
    with simba.db.connect(): pass

    # Manually seed kg_derived_edges with 4 rows for rule_id=1, head=transitively_uses
    conn = sqlite3.connect(str(db_path))
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for i in range(4):
        conn.execute(
            "INSERT OR IGNORE INTO kg_derived_edges "
            "(subject, predicate, object, proof, source_edge_ids, rule_id, "
            "confidence, valid_from, project_path, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f"A{i}", "transitively_uses", f"C{i}",
             "derived:1", json.dumps([i, i+1]), 1, 0.8, now, "/proj", now)
        )
    conn.commit()
    conn.close()

    from simba.neuron.induce import induce_rules
    from simba.neuron.config import NeuronConfig
    result = induce_rules(project_path="/proj",
                         cfg=NeuronConfig(induce_enabled=True,
                                          induce_min_activations=3,
                                          induce_min_confidence=0.7))
    assert result.promoted >= 1

def test_induce_skips_below_threshold(tmp_path, monkeypatch) -> None:
    import simba.db
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    import simba.neuron.schema
    with simba.db.connect(): pass

    from simba.neuron.induce import induce_rules
    from simba.neuron.config import NeuronConfig
    # No rows in kg_derived_edges → nothing to promote
    result = induce_rules(project_path="/proj",
                         cfg=NeuronConfig(induce_enabled=True, induce_min_activations=3))
    assert result.promoted == 0
    assert result.below_threshold == 0
```

**Acceptance:** A rule fired 4 times at 0.8 confidence (above both thresholds) gets promoted to `kg_rules`; a rule fired only twice does not.

**Verify:**
```bash
uv run pytest tests/neuron/test_induce.py -v
```

**Reuse:** `src/simba/neuron/distill.py:distill_edges` — INSERT OR IGNORE + rowcount pattern.

---

### Task B.8 — `distillation_pass` orchestrator + scheduler wiring

**Goal:** Wire DERIVE → VERIFY → REVISE → DISTILL → INDUCE into a single `distillation_pass` function, expose a result dataclass, and hook it into `SyncScheduler.run_once`.

**Files:**
- `src/simba/neuron/pipeline.py`
- `src/simba/sync/scheduler.py` (modify)

**Config:** `NeuronConfig.enabled` (master gate).

**Signatures/Schema:**

```python
# src/simba/neuron/pipeline.py
from __future__ import annotations
import dataclasses

@dataclasses.dataclass
class DistillationResult:
    status: str            # "ok" | "disabled" | "error"
    candidates: int = 0    # from DERIVE
    satisfiable: bool = True
    unsat_core_size: int = 0
    dormant: int = 0       # from REVISE
    distilled: int = 0     # from DISTILL
    promoted_rules: int = 0
    errors: int = 0

def distillation_pass(
    *,
    project_path: str,
    cfg: "NeuronConfig | None" = None,
) -> DistillationResult:
    """Run the full 5-step deductive distillation pipeline. Never raises."""
    ...
```

**Implementation steps:**
1. Load `cfg`; return `DistillationResult(status="disabled")` when `not cfg.enabled`.
2. Call `run_derive(project_path, cfg=cfg)` → `derive_res`.
3. Call `run_verify(project_path, cfg=cfg, extra_edges=[...])` → `verify_res`.
4. If `not verify_res.satisfiable`: call `revise_unsat_core(verify_res.unsat_edge_ids, project_path=project_path, cfg=cfg)` → `revise_res`.
5. Call `distill_edges(derive_res.candidates, project_path=project_path, cfg=cfg)` → `distill_res`.
6. Call `induce_rules(project_path=project_path, cfg=cfg)` → `induce_res`.
7. Return `DistillationResult(status="ok", candidates=len(derive_res.candidates), ...)`.

Wire into `SyncScheduler._maybe_distill(self) -> dict` following `_maybe_consolidate` / `_maybe_reflect` pattern, add to `run_once` summary as `"distillation"`.

**Tests:** `tests/neuron/test_pipeline.py`

```python
from unittest.mock import patch, MagicMock

def test_disabled_returns_disabled() -> None:
    from simba.neuron.pipeline import distillation_pass
    from simba.neuron.config import NeuronConfig
    result = distillation_pass(project_path="/proj", cfg=NeuronConfig(enabled=False))
    assert result.status == "disabled"

def test_full_pipeline_sat_path(monkeypatch) -> None:
    """All sub-phases called; SAT path — no revise."""
    from simba.neuron import pipeline as p
    from simba.neuron.derive import DeriveResult, DerivedEdge
    from simba.neuron.z3_verify import VerifyResult
    from simba.neuron.revise import ReviseResult
    from simba.neuron.distill import DistillResult
    from simba.neuron.induce import InduceResult
    from simba.neuron.config import NeuronConfig

    monkeypatch.setattr(p, "run_derive",
        lambda *a, **k: DeriveResult(candidates=[DerivedEdge("A","r","B",[1],None)], edges_fed=1))
    monkeypatch.setattr(p, "run_verify",
        lambda *a, **k: VerifyResult(satisfiable=True, unsat_edge_ids=[], checked_edges=1))
    monkeypatch.setattr(p, "distill_edges",
        lambda *a, **k: DistillResult(added=1))
    monkeypatch.setattr(p, "induce_rules",
        lambda *a, **k: InduceResult(promoted=0))

    result = p.distillation_pass(project_path="/proj",
                                 cfg=NeuronConfig(enabled=True))
    assert result.status == "ok"
    assert result.candidates == 1
    assert result.distilled == 1
    assert result.dormant == 0

def test_full_pipeline_unsat_path(monkeypatch) -> None:
    """UNSAT path triggers revise."""
    from simba.neuron import pipeline as p
    from simba.neuron.derive import DeriveResult, DerivedEdge
    from simba.neuron.z3_verify import VerifyResult
    from simba.neuron.revise import ReviseResult
    from simba.neuron.distill import DistillResult
    from simba.neuron.induce import InduceResult
    from simba.neuron.config import NeuronConfig

    monkeypatch.setattr(p, "run_derive",
        lambda *a, **k: DeriveResult(candidates=[], edges_fed=5))
    monkeypatch.setattr(p, "run_verify",
        lambda *a, **k: VerifyResult(satisfiable=False, unsat_edge_ids=[1,2], checked_edges=5))
    revise_called = []
    monkeypatch.setattr(p, "revise_unsat_core",
        lambda ids, **k: (revise_called.append(ids), ReviseResult(dormant_edge_ids=[1], retained_edge_ids=[2]))[1])
    monkeypatch.setattr(p, "distill_edges", lambda *a, **k: DistillResult())
    monkeypatch.setattr(p, "induce_rules", lambda *a, **k: InduceResult())

    result = p.distillation_pass(project_path="/proj", cfg=NeuronConfig(enabled=True))
    assert result.satisfiable is False
    assert result.unsat_core_size == 2
    assert result.dormant == 1
    assert revise_called == [[1, 2]]
```

**Acceptance:** SAT path skips REVISE; UNSAT path calls REVISE. Summary dict from `SyncScheduler.run_once` includes `"distillation"` key.

**Verify:**
```bash
uv run pytest tests/neuron/test_pipeline.py -v
uv run pytest tests/sync/test_scheduler.py -v
```

**Reuse:** `src/simba/sync/scheduler.py:_maybe_consolidate` — structural twin for `_maybe_distill`.

---

### Task B.9 — KG density metric + contradiction-injection test fixture

**Goal:** (1) Add a `kg_density()` function to `simba.kg.store` as the primary phase-7 progress metric. (2) Provide a reusable pytest fixture that plants a contradiction and asserts the VERIFY step finds it.

**Files:**
- `src/simba/kg/store.py` (add `kg_density`)
- `tests/neuron/conftest.py` (fixture)
- `tests/neuron/test_contradiction_detection.py`

**Signatures/Schema:**

```python
# src/simba/kg/store.py — append
def kg_density(project_path: str | None = None) -> dict[str, float]:
    """Return graph density metrics for a project.

    Returns:
        {
          "edge_count": int,
          "derived_edge_count": int,
          "node_count": int,   # distinct subjects + objects
          "density": float,    # edges / (nodes*(nodes-1)) or 0 for tiny graphs
          "derived_ratio": float,  # derived_edge_count / (edge_count + derived_edge_count)
        }
    """
    ...
```

```python
# tests/neuron/conftest.py
import pytest

@pytest.fixture()
def planted_contradiction(tmp_path, monkeypatch):
    """Seed the KG with a USES/DOES_NOT_USE pair and return their edge ids."""
    import simba.db, simba.kg.store
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    simba.kg.store.kg_add("ToolA", "uses", "LibB", "test", project_path="/proj")
    simba.kg.store.kg_add("ToolA", "does_not_use", "LibB", "test", project_path="/proj")
    with simba.db.connect():
        ids = [e.id for e in simba.kg.store.KgEdge.select()
               .where(simba.kg.store.KgEdge.project_path == "/proj")]
    return ids, "/proj"
```

**Tests:** `tests/neuron/test_contradiction_detection.py`

```python
import pytest

def test_verify_finds_planted_contradiction(planted_contradiction) -> None:
    try:
        import z3  # noqa: F401
    except ImportError:
        pytest.skip("z3 not installed")
    edge_ids, project_path = planted_contradiction
    from simba.neuron.z3_verify import run_verify
    from simba.neuron.config import NeuronConfig
    result = run_verify(project_path, cfg=NeuronConfig(verify_enabled=True,
                        contradiction_sample_size=10, verify_timeout_seconds=15))
    assert result.satisfiable is False
    assert len(result.unsat_edge_ids) >= 2

def test_kg_density_baseline(planted_contradiction) -> None:
    _, project_path = planted_contradiction
    from simba.kg.store import kg_density
    metrics = kg_density(project_path)
    assert metrics["edge_count"] == 2
    assert metrics["node_count"] >= 2   # ToolA, LibB
    assert 0.0 <= metrics["density"] <= 1.0
    assert metrics["derived_ratio"] == 0.0   # no derived edges yet
```

**Acceptance:** `test_verify_finds_planted_contradiction` passes (with z3 installed); `test_kg_density_baseline` always passes.

**Verify:**
```bash
uv run pytest tests/neuron/test_contradiction_detection.py -v
```

**Reuse:** `src/simba/kg/store.py:kg_query` — same `simba.db.connect()` context.

---

## Group C — Ops Hardening

### Task C.1 — Latency p50/p95 metrics in `DiagnosticsTracker` + `/metrics` endpoint

**Goal:** Track per-endpoint latency (p50 + p95) in `DiagnosticsTracker` using an in-memory reservoir. Expose aggregated stats as a `GET /metrics` route (JSON, no auth). No external dependency — use a fixed-size reservoir with a `statistics.quantiles` call at report time.

**Files:**
- `src/simba/memory/diagnostics.py` (extend)
- `src/simba/memory/routes.py` (add route)

**Config:** `memory.diagnostics_reservoir_size: int = 1000` (max latency samples kept per endpoint).

`src/simba/memory/config.py` — add one field to `MemoryConfig`:

```python
diagnostics_reservoir_size: int = 1000
```

**Signatures/Schema:**

```python
# diagnostics.py additions

import statistics
import collections

class DiagnosticsTracker:
    # Existing fields unchanged.

    def __init__(self, report_interval: int = 50, reservoir_size: int = 1000) -> None:
        ...
        self._reservoir_size = reservoir_size
        self._latency_samples: dict[str, list[float]] = collections.defaultdict(list)

    def record_latency(self, endpoint: str, latency_ms: float) -> None:
        """Record a latency sample, evicting oldest when at capacity."""
        buf = self._latency_samples[endpoint]
        if len(buf) >= self._reservoir_size:
            buf.pop(0)
        buf.append(latency_ms)

    def latency_percentiles(self, endpoint: str) -> dict[str, float]:
        """Return {"p50": float, "p95": float, "n": int} for one endpoint."""
        buf = self._latency_samples.get(endpoint, [])
        if len(buf) < 2:
            return {"p50": 0.0, "p95": 0.0, "n": len(buf)}
        qs = statistics.quantiles(buf, n=20)  # 5% increments
        return {"p50": qs[9], "p95": qs[18], "n": len(buf)}

    def all_latency_stats(self) -> dict[str, dict[str, float]]:
        """Return percentiles for all endpoints that have samples."""
        return {ep: self.latency_percentiles(ep) for ep in self._latency_samples}
```

```python
# routes.py addition
@router.get("/metrics")
async def metrics(request: fastapi.Request) -> dict:
    diag = getattr(request.app.state, "diagnostics", None)
    uptime = int(time.time() - request.app.state.start_time)
    latency = diag.all_latency_stats() if diag else {}
    return {
        "uptime_seconds": uptime,
        "latency": latency,
        "total_requests": diag._total_requests if diag else 0,
    }
```

Update `DiagnosticsMiddleware.dispatch` to call `diag.record_latency(request.url.path, elapsed_ms)` where `elapsed_ms = (time.monotonic() - t0) * 1000`.

**Implementation steps:**
1. Add `diagnostics_reservoir_size` field to `MemoryConfig`.
2. Extend `DiagnosticsTracker.__init__` to accept `reservoir_size` parameter.
3. Add `record_latency`, `latency_percentiles`, `all_latency_stats` methods.
4. In `DiagnosticsMiddleware.dispatch`: capture `t0 = time.monotonic()` before `await call_next(request)`; after response, call `diag.record_latency(request.url.path, (time.monotonic()-t0)*1000)`.
5. Add `GET /metrics` route to `routes.py`.
6. Update `create_app` to pass `reservoir_size=config.diagnostics_reservoir_size` to `DiagnosticsTracker`.

**Tests:** `tests/memory/test_diagnostics.py` — add to existing class:

```python
def test_record_latency_and_percentiles() -> None:
    tracker = DiagnosticsTracker(report_interval=50, reservoir_size=100)
    for i in range(20):
        tracker.record_latency("POST /recall", float(i))
    stats = tracker.latency_percentiles("POST /recall")
    assert stats["n"] == 20
    assert stats["p50"] > 0.0
    assert stats["p95"] >= stats["p50"]

def test_reservoir_evicts_oldest() -> None:
    tracker = DiagnosticsTracker(reservoir_size=5)
    for i in range(10):
        tracker.record_latency("/store", float(i))
    assert len(tracker._latency_samples["/store"]) == 5

def test_all_latency_stats_returns_all_endpoints() -> None:
    tracker = DiagnosticsTracker()
    tracker.record_latency("/recall", 12.0)
    tracker.record_latency("/recall", 15.0)
    tracker.record_latency("/store", 8.0)
    tracker.record_latency("/store", 9.0)
    stats = tracker.all_latency_stats()
    assert set(stats.keys()) == {"/recall", "/store"}
```

**Acceptance:** `GET /metrics` returns JSON with `"latency"` dict containing p50/p95 per endpoint.

**Verify:**
```bash
uv run pytest tests/memory/test_diagnostics.py -v
# Integration smoke (daemon must be running):
# curl -s http://localhost:8741/metrics | python -m json.tool
```

**Reuse:** `src/simba/memory/diagnostics.py:DiagnosticsTracker` — existing class extended in-place.

---

### Task C.2 — Memory-hygiene scheduler pass: TTL aging for stale `TOOL_RULE` memories

**Goal:** A scheduled hygiene pass that marks `TOOL_RULE` memories older than `memory.tool_rule_max_age_days` as dormant (via `DELETE` on LanceDB since vector DB is the source of truth for memories, matching the existing delete route). Solves the known false-warning injection from stale `TOOL_RULE` entries.

**Files:**
- `src/simba/memory/hygiene.py` — new module
- `src/simba/sync/scheduler.py` (add `_maybe_hygiene`)

**Config (`memory` section) — add to `MemoryConfig`:**

```python
tool_rule_max_age_days: int = 30       # 0 = disabled
hygiene_scheduler_enabled: bool = True
```

**Signatures/Schema:**

```python
# src/simba/memory/hygiene.py
from __future__ import annotations
import dataclasses

@dataclasses.dataclass
class HygieneResult:
    expired_count: int = 0
    checked_count: int = 0
    errors: int = 0

def run_hygiene_pass(
    *,
    daemon_url: str,
    cfg: "MemoryConfig | None" = None,
) -> HygieneResult:
    """Expire stale TOOL_RULE memories via daemon DELETE. Never raises."""
    ...
```

**Implementation steps:**
1. Add `tool_rule_max_age_days` and `hygiene_scheduler_enabled` to `MemoryConfig`.
2. Create `src/simba/memory/hygiene.py`:
   a. Load `cfg`; return `HygieneResult()` when `cfg.tool_rule_max_age_days == 0`.
   b. `GET {daemon_url}/list?type=TOOL_RULE&limit=10000` to collect all TOOL_RULE memories.
   c. Compute cutoff: `now - timedelta(days=cfg.tool_rule_max_age_days)`.
   d. For each memory where `createdAt < cutoff`: `DELETE {daemon_url}/memory/{id}`.
   e. Count deletions; return `HygieneResult`.
3. Add `_maybe_hygiene` to `SyncScheduler`:
   ```python
   def _maybe_hygiene(self) -> dict:
       import simba.config, simba.memory.config, simba.memory.hygiene
       cfg = simba.config.load("memory")
       if not cfg.hygiene_scheduler_enabled or cfg.tool_rule_max_age_days == 0:
           return {"status": "disabled"}
       result = simba.memory.hygiene.run_hygiene_pass(daemon_url=self.daemon_url, cfg=cfg)
       return {"status": "ok", "expired": result.expired_count}
   ```
4. Add `hyg = await loop.run_in_executor(None, self._maybe_hygiene)` in `run_once`; include `"hygiene": hyg` in summary.

**Tests:** `tests/memory/test_hygiene.py`

```python
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone

def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def _now() -> datetime:
    return datetime.now(tz=timezone.utc)

def test_hygiene_disabled_when_zero_age() -> None:
    from simba.memory.hygiene import run_hygiene_pass
    from simba.memory.config import MemoryConfig
    result = run_hygiene_pass(daemon_url="http://x", cfg=MemoryConfig(tool_rule_max_age_days=0))
    assert result.expired_count == 0
    assert result.errors == 0

def test_hygiene_deletes_stale_tool_rules(monkeypatch) -> None:
    from simba.memory.hygiene import run_hygiene_pass
    from simba.memory.config import MemoryConfig
    import httpx

    old = _iso(_now() - timedelta(days=40))
    fresh = _iso(_now() - timedelta(days=5))
    memories = [
        {"id": "m1", "type": "TOOL_RULE", "createdAt": old},
        {"id": "m2", "type": "TOOL_RULE", "createdAt": fresh},
    ]

    deleted: list[str] = []

    def fake_get(url, **kw):
        r = MagicMock()
        r.raise_for_status = lambda: None
        r.json.return_value = {"memories": memories, "total": 2}
        return r

    def fake_delete(url, **kw):
        mid = url.split("/")[-1]
        deleted.append(mid)
        r = MagicMock(); r.raise_for_status = lambda: None; return r

    with patch("httpx.get", side_effect=fake_get):
        with patch("httpx.delete", side_effect=fake_delete):
            result = run_hygiene_pass(
                daemon_url="http://localhost:8741",
                cfg=MemoryConfig(tool_rule_max_age_days=30)
            )
    assert result.expired_count == 1
    assert deleted == ["m1"]

def test_hygiene_http_error_is_fail_open(monkeypatch) -> None:
    from simba.memory.hygiene import run_hygiene_pass
    from simba.memory.config import MemoryConfig
    import httpx
    with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
        result = run_hygiene_pass(
            daemon_url="http://localhost:8741",
            cfg=MemoryConfig(tool_rule_max_age_days=30)
        )
    assert result.errors >= 1
    assert result.expired_count == 0
```

**Acceptance:** Memories with `createdAt` older than `tool_rule_max_age_days` are deleted; fresh ones survive; all errors are fail-open.

**Verify:**
```bash
uv run pytest tests/memory/test_hygiene.py -v
```

**Reuse:** `src/simba/episodes/consolidate.py:_list_memories` — same `httpx.get /list` pagination pattern.

---

### Task C.3 — Lighter install: `[project.optional-dependencies]` split

**Goal:** Allow `pip install simba-ai` (or `uv pip install simba-ai`) to succeed without `llama-cpp-python` and `lancedb` by splitting heavy ML dependencies into an optional extra. A "core" install supports the KG, sync, config CLI, and hooks but not the in-process embedding server.

**Files:**
- `pyproject.toml`

**Config:** N/A — this is packaging metadata only.

**Implementation steps:**

1. In `pyproject.toml`, move `lancedb>=0.4`, `llama-cpp-python>=0.3`, `huggingface-hub>=0.26` from `[project.dependencies]` to a new `[project.optional-dependencies]` table under the key `embed`:

```toml
[project]
dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.34",
    "httpx>=0.28",
    "mcp[cli]",
    "z3-solver",
    "tomli-w>=1.0",
]

[project.optional-dependencies]
embed = [
    "lancedb>=0.4",
    "llama-cpp-python>=0.3",
    "huggingface-hub>=0.26",
]
full = ["simba-ai[embed]"]
```

2. All imports of `lancedb`, `llama_cpp`, `huggingface_hub` in `src/simba/memory/` are already guarded inside `async def init_database` / `EmbeddingService.start` (lazy imports). Verify no top-level `import lancedb` exists at module init time. If any exist, move them behind `TYPE_CHECKING` guards or into the function body.

3. Update `README.md` install section to document `pip install simba-ai[embed]` for the full daemon.

**Tests:** `tests/test_core_import.py` — new file

```python
"""Smoke test: simba core modules import without lancedb/llama_cpp."""

def test_config_importable_without_embed(monkeypatch) -> None:
    """simba.config must import with no ML deps."""
    import sys
    # Block lancedb and llama_cpp to simulate core-only install
    for name in list(sys.modules):
        if "lancedb" in name or "llama_cpp" in name:
            del sys.modules[name]
    # Should not raise
    import simba.config  # noqa: F401
    import simba.kg.store  # noqa: F401
    import simba.sync.scheduler  # noqa: F401

def test_memory_server_lazy_imports() -> None:
    """simba.memory.server module-level import must not trigger lancedb."""
    import importlib, sys
    # Remove lancedb from sys.modules to detect eager import
    lancedb_backup = sys.modules.pop("lancedb", None)
    try:
        if "simba.memory.server" in sys.modules:
            del sys.modules["simba.memory.server"]
        import simba.memory.server  # noqa: F401
        assert "lancedb" not in sys.modules, "lancedb was imported at module level"
    finally:
        if lancedb_backup is not None:
            sys.modules["lancedb"] = lancedb_backup
```

**Acceptance:** `pip install simba-ai` (no extras) runs; `simba config list` works; `simba memory start` fails with a clear `ImportError: install simba-ai[embed]` message.

**Verify:**
```bash
uv run pytest tests/test_core_import.py -v
# In a fresh venv:
# pip install -e . && simba config get memory.port   # must succeed
# simba memory start   # must fail with clear ImportError, not a cryptic one
```

**Reuse:** `src/simba/memory/embeddings.py:EmbeddingService.start` — already uses lazy import pattern.

---

### Task C.4 — Fix `release.yml` artifact glob

**Goal:** Replace `files: dist/*` with explicit globs to exclude `.gitignore` or other stray files that `uv build` may place in `dist/`.

**Files:**
- `.github/workflows/release.yml`

**Config:** N/A.

**Implementation steps:**

1. In `.github/workflows/release.yml`, in the `Create GitHub Release` step, change:

```yaml
# Before (line 87)
files: dist/*

# After
files: |
  dist/*.whl
  dist/*.tar.gz
```

This is the complete change. No Python code is modified.

**Tests:** No pytest test needed. Manual verification only.

**Acceptance:** A test release build (`uv build && ls dist/`) shows only `*.whl` + `*.tar.gz`; no other files. The workflow's `files:` pattern matches exactly those two.

**Verify:**
```bash
uv build
ls dist/   # Confirm only .whl + .tar.gz present
# Read the workflow YAML and confirm the glob is specific
grep "files:" .github/workflows/release.yml
```

**Reuse:** N/A.

---

### Task C.5 — Move hidden constant `prompt_min_similarity` into `HooksConfig`

**Goal:** The value `0.45` used as `prompt_min_similarity` in `src/simba/hooks/config.py` is already a `@configurable` field (`HooksConfig.prompt_min_similarity`), so this constant is technically exposed. However, `user_prompt_submit.py` accesses it via `cfg.prompt_min_similarity` (already correct). The actual hidden constant to fix is the hardcoded `_MIN_SIMILARITY = 0.45` comment in `hooks/config.py` (line 26) and any remaining hardcoded literals in hooks.

Audit and fix all remaining hardcoded numeric thresholds in hook files that should be `@configurable` fields.

**Files to audit:**
- `src/simba/hooks/user_prompt_submit.py` — already reads `cfg.prompt_min_similarity` ✓
- `src/simba/hooks/config.py` — `prompt_min_similarity: float = 0.45` is already a field ✓
- `src/simba/hooks/pre_tool_use.py` — audit for hardcoded floats
- `src/simba/hooks/pre_compact.py` — audit for hardcoded floats

**Implementation steps:**
1. Run `rg -n '[0-9]\.[0-9][0-9]' src/simba/hooks/` to find any remaining literal floats not routed through `HooksConfig`.
2. For each found: add a named field to `HooksConfig` with the same default, and replace the literal with `cfg.<field_name>`.
3. Specific known case: if any hook uses `min_similarity=0.45` as a literal keyword argument rather than loading from config, replace it.

**Tests:** `tests/hooks/test_user_prompt_submit.py` — add:

```python
def test_prompt_min_similarity_reads_from_config(monkeypatch) -> None:
    """Verify the hook uses cfg.prompt_min_similarity, not a literal."""
    import simba.hooks.user_prompt_submit as ups
    import simba.hooks._memory_client as mc
    import simba.hooks.config as hcfg
    import simba.config

    captured = {}

    def fake_recall(query, *, project_path=None, min_similarity=None):
        captured["min_similarity"] = min_similarity
        return []

    monkeypatch.setattr(mc, "recall_memories", fake_recall)
    monkeypatch.setattr(simba.config, "load",
        lambda section: hcfg.HooksConfig(prompt_min_similarity=0.77)
        if section == "hooks" else simba.config.load.__wrapped__(section))

    ups.main({"prompt": "x" * 20, "cwd": "/proj"})
    assert captured.get("min_similarity") == 0.77

def test_no_hardcoded_similarity_in_hooks_source() -> None:
    """Regression: no magic 0.45 literal outside of HooksConfig default."""
    import pathlib
    src = pathlib.Path("src/simba/hooks")
    for py in src.glob("*.py"):
        if py.name == "config.py":
            continue  # defaults are allowed here
        text = py.read_text()
        # 0.45 appearing as a standalone float literal (not in a comment/string) is a flag
        import re
        matches = re.findall(r'(?<!["\'\w])0\.45(?!["\'\w])', text)
        assert matches == [], f"Hardcoded 0.45 found in {py}: {matches}"
```

**Acceptance:** `simba config set hooks.prompt_min_similarity 0.5` takes effect on the next hook invocation without code changes. No magic floats remain in hook implementation files.

**Verify:**
```bash
uv run pytest tests/hooks/test_user_prompt_submit.py -v
rg -n 'min_similarity.*0\.[0-9]' src/simba/hooks/ --type py
uv run simba config get hooks.prompt_min_similarity
```

**Reuse:** `src/simba/hooks/config.py:HooksConfig` — existing `@configurable` class. `src/simba/hooks/user_prompt_submit.py:_cfg()` — config accessor pattern.

---

## Implementation Build Sequence

### Phase A — Reflection (prerequisite for nothing; ship independently)

- [ ] A.1 — Add `REFLECTION` to type registries (`routes.py`, `__main__.py`)
- [ ] A.2 — `ReflectionConfig` dataclass (`src/simba/reflection/config.py`)
- [ ] A.3 — `build_reflection_prompt` (`src/simba/reflection/prompt.py`)
- [ ] A.4 — `reflect_pass` orchestrator (`src/simba/reflection/pass_.py`)
- [ ] A.5 — Wire into `SyncScheduler.run_once` + summary key

### Phase B — Neuro-Symbolic Distillation (each sub-phase shippable independently)

- [ ] B.1 — Schema: `kg_derived_edges` + `kg_rules` + `dormant` column
- [ ] B.2 — `NeuronConfig` as `@configurable`
- [ ] B.3 — `run_derive` (DERIVE step)
- [ ] B.4 — `run_verify` + `build_z3_script` (VERIFY step)
- [ ] B.5 — `revise_unsat_core` (REVISE step)
- [ ] B.6 — `distill_edges` (DISTILL step)
- [ ] B.7 — `induce_rules` (INDUCE step)
- [ ] B.8 — `distillation_pass` pipeline + scheduler wiring
- [ ] B.9 — `kg_density` metric + `planted_contradiction` fixture

### Phase C — Ops Hardening (each task independent)

- [ ] C.1 — Latency p50/p95 in `DiagnosticsTracker` + `GET /metrics`
- [ ] C.2 — `run_hygiene_pass` + scheduler wiring + `tool_rule_max_age_days` config
- [ ] C.3 — `pyproject.toml` optional-dependencies split
- [ ] C.4 — `release.yml` glob fix
- [ ] C.5 — Audit + fix remaining hardcoded thresholds in hooks
