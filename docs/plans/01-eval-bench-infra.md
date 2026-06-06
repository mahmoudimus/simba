# Eval-Program Infrastructure Spec: A4 · A5 · A6

## Preamble: Patterns & Conventions Found

**Config registration** (`src/simba/eval/config.py:12`): `@simba.config.configurable("section")` stacked above `@dataclasses.dataclass`. Load via `simba.config.load("section")`.

**CLI dispatch** (`src/simba/__main__.py:2076`): `_cmd_eval(args)` does `if args and args[0] == "build": return _eval_build(args[1:])`. The same pattern will host `bench` and `leaderboard` as branches inside `_cmd_eval`. Manual `while i < len(args)` arg parsing; no argparse.

**run_recall signature** (`src/simba/eval/benchmarks/run.py:19`): returns `dict[str, Any]` with keys `n_conversations`, `n_cases`, `overall`, `by_category`.

**run_qa signature** (`src/simba/eval/benchmarks/judge.py:147`): returns `dict[str, Any]` with keys `n_graded`, `n_skipped`, `overall`, `by_category`.

**EvalReport.to_dict()** (`src/simba/eval/runner.py:49`): returns `dataset_name`, `n_cases`, `ks`, `aggregate`, `per_case`.

**sync_embedders cache parameter** (`src/simba/eval/run.py:52`): `sync_embedders(cfg, *, cache=None)` — pass `EmbeddingCache` instance.

**Dataset loaders**: `locomo.load_locomo(path)` and `longmemeval.load_longmemeval(path)` return `list[Dataset]`; both accept slicing with `[:n]`.

**Append-only JSONL**: existing pattern in `src/simba/memory/` — open in `"a"` mode, one JSON object per line.

**Test fixtures**: `conftest.py` provides `tmp_path`, `simba_db`, `monkeypatch`; tests monkeypatch module-level functions (e.g., `monkeypatch.setattr(cli, "_install_codex_skills", fake)`). No live network or LLM in tests.

**Pathlib only**: `TID251` bans `os.path`; use `pathlib.Path` throughout.

**TYPE_CHECKING guard**: annotation-only imports placed under `if typing.TYPE_CHECKING:`.

---

## Task A4 — `simba eval bench` CLI

### Goal

Replace the three `scripts/run_*.py` ad-hoc scripts with a single reproducible command: `simba eval bench locomo|longmemeval [--qa] [--n N | --per N | all] [--k K] [--split dev|test] [--json]`. It loads the named benchmark dataset from a config-registered path, drives `run_recall` (always) and optionally `run_qa`, wires both caches, prints results, and returns exit code 0 on success.

### Files

| Action | Path |
|---|---|
| Create | `/Users/mahmoud/src/ai/simba/src/simba/eval/bench_config.py` |
| Modify | `/Users/mahmoud/src/ai/simba/src/simba/__main__.py` |
| Delete (recommended) | `/Users/mahmoud/src/ai/simba/scripts/run_locomo.py` |
| Delete (recommended) | `/Users/mahmoud/src/ai/simba/scripts/run_longmemeval.py` |
| Delete (recommended) | `/Users/mahmoud/src/ai/simba/scripts/run_qa.py` |

### Config

New `@configurable("bench")` section in `src/simba/eval/bench_config.py`:

```
bench.locomo_path: str = ""
    Filesystem path to locomo10.json. Empty = must be supplied via CLI or error.

bench.longmemeval_path: str = ""
    Filesystem path to longmemeval_*.json. Empty = must be supplied.

bench.embedding_cache_path: str = ".simba/eval/embedding_cache.db"
    SQLite path for EmbeddingCache. Relative paths resolved from cwd.

bench.judge_cache_path: str = ".simba/eval/judge_cache.db"
    SQLite path for JudgeCache.

bench.default_k: int = 10
    Default top-k for QA retrieval context window.

bench.default_n: int = 50
    Default number of QA cases when --n / --per / all not given.

bench.results_path: str = ".simba/eval/results.jsonl"
    Append-only results log (A5 writes here too).

bench.max_results_eval: int = 20
    MemoryConfig.max_results override used during bench runs.

bench.max_results_broad_eval: int = 20
    MemoryConfig.max_results_broad override used during bench runs.

bench.fts_candidate_pool_eval: int = 40
    MemoryConfig.fts_candidate_pool override used during bench runs.

bench.fts_candidate_pool_broad_eval: int = 60
    MemoryConfig.fts_candidate_pool_broad override used during bench runs.

bench.llm_rerank_enabled_eval: bool = False
    Disable LLM reranker during bench to isolate raw retrieval.

bench.scoring_enabled_eval: bool = False
    Disable scoring during bench.

bench.expansion_enabled_eval: bool = False
    Disable expansion during bench.
```

### Signatures

**`src/simba/eval/bench_config.py`** — new file:

```python
from __future__ import annotations
import dataclasses
import simba.config

@simba.config.configurable("bench")
@dataclasses.dataclass
class BenchConfig:
    locomo_path: str = ""
    longmemeval_path: str = ""
    embedding_cache_path: str = ".simba/eval/embedding_cache.db"
    judge_cache_path: str = ".simba/eval/judge_cache.db"
    default_k: int = 10
    default_n: int = 50
    results_path: str = ".simba/eval/results.jsonl"
    max_results_eval: int = 20
    max_results_broad_eval: int = 20
    fts_candidate_pool_eval: int = 40
    fts_candidate_pool_broad_eval: int = 60
    llm_rerank_enabled_eval: bool = False
    scoring_enabled_eval: bool = False
    expansion_enabled_eval: bool = False

    def eval_memory_config_overrides(self) -> dict[str, object]:
        """Return the MemoryConfig field overrides for bench runs."""
        return {
            "max_results": self.max_results_eval,
            "max_results_broad": self.max_results_broad_eval,
            "fts_candidate_pool": self.fts_candidate_pool_eval,
            "fts_candidate_pool_broad": self.fts_candidate_pool_broad_eval,
            "llm_rerank_enabled": self.llm_rerank_enabled_eval,
            "scoring_enabled": self.scoring_enabled_eval,
            "expansion_enabled": self.expansion_enabled_eval,
        }
```

**`src/simba/__main__.py`** — new private helper `_eval_bench(args)` inserted alongside `_eval_build`; `_cmd_eval` gains one new branch:

```python
def _eval_bench(args: list[str]) -> int:
    """simba eval bench locomo|longmemeval [--qa] [--n N|--per N|all]
       [--k K] [--split dev|test] [--path PATH] [--json]"""
    ...

# Inside _cmd_eval, after the existing "build" check:
if args and args[0] == "bench":
    return _eval_bench(args[1:])
```

Complete `_eval_bench` body:

```python
def _eval_bench(args: list[str]) -> int:
    import dataclasses
    import json as _json
    import pathlib

    import simba.config
    import simba.eval.bench_config  # registers "bench" section
    import simba.eval.benchmarks.locomo as locomo
    import simba.eval.benchmarks.longmemeval as lme
    import simba.eval.benchmarks.run as bench_run
    import simba.eval.run as run
    import simba.memory.config
    import simba.memory.embedding_cache as ec
    import simba.eval.benchmarks.judge_cache as jc
    from simba.eval.bench_results import append_result, current_git_sha

    # --- arg parsing ---
    if not args or args[0].startswith("--"):
        print(
            "Usage: simba eval bench locomo|longmemeval [--qa] "
            "[--n N | --per N | all] [--k K] [--split dev|test] "
            "[--path PATH] [--json]",
            file=sys.stderr,
        )
        return 1

    dataset_name = args[0]
    if dataset_name not in ("locomo", "longmemeval"):
        print(
            f"eval bench: unknown dataset {dataset_name!r}; "
            "choose locomo or longmemeval",
            file=sys.stderr,
        )
        return 1

    run_qa_flag = False
    n_arg: str = ""
    k: int = 0
    split_arg: str = ""
    path_arg: str = ""
    as_json = False

    i = 1
    while i < len(args):
        if args[i] == "--qa":
            run_qa_flag = True
            i += 1
        elif args[i] in ("--n", "--per") and i + 1 < len(args):
            n_arg = args[i] + args[i + 1]   # e.g. "--n50" or "--per10"
            # Encode flag+value together for later parsing.
            # Store as "--n 50" or "--per 10" convention.
            n_arg = f"{args[i][2:]}{args[i + 1]}"  # "n50" or "per10"
            i += 2
        elif args[i] == "all":
            n_arg = "all"
            i += 1
        elif args[i] == "--k" and i + 1 < len(args):
            k = int(args[i + 1])
            i += 2
        elif args[i] == "--split" and i + 1 < len(args):
            split_arg = args[i + 1]
            i += 2
        elif args[i] == "--path" and i + 1 < len(args):
            path_arg = args[i + 1]
            i += 2
        elif args[i] == "--json":
            as_json = True
            i += 1
        else:
            print(f"eval bench: unknown option {args[i]!r}", file=sys.stderr)
            return 1
    ...
```

The signature section above is intentionally partial — the complete body is spelled out in the Implementation Steps below.

### Implementation Steps

1. **Create `src/simba/eval/bench_config.py`** with `BenchConfig` as shown in Signatures. No imports besides `dataclasses` and `simba.config`.

2. **Create `src/simba/eval/bench_results.py`** (shared between A4 and A5). This module owns two helpers needed by `_eval_bench`:
   - `current_git_sha() -> str`: runs `subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)`, returns stdout stripped or `"unknown"` on any error.
   - `append_result(path: pathlib.Path, record: dict[str, object]) -> None`: `path.parent.mkdir(parents=True, exist_ok=True)`, opens in `"a"` mode, writes `json.dumps(record) + "\n"`. Append-only, never truncates.
   - `config_snapshot(mcfg: object, bcfg: object) -> dict[str, object]`: returns `{"memory": dataclasses.asdict(mcfg), "bench": dataclasses.asdict(bcfg)}`.

3. **Write `_eval_bench(args)` in `src/simba/__main__.py`**, placed immediately after `_eval_build`. Full logic:

   a. Parse args as described in Signatures section. Ambiguous `--n` / `--per` encoding: store a `("n", int)` or `("per", int)` or `("all", None)` tuple after parsing; `n_arg` in the mapping above is simplified — use a local `n_mode: str = "n"` and `n_val: int = 0` pair instead, set `n_mode = "per"` for `--per`, `n_mode = "all"` for bare `all`.

   b. Load configs: `bcfg = simba.config.load("bench")`, `mcfg = simba.config.load("memory")`. Apply `BenchConfig.eval_memory_config_overrides()` to a copy: `mcfg = dataclasses.replace(mcfg, **bcfg.eval_memory_config_overrides())`.

   c. Resolve dataset path: prefer `--path` arg, else `bcfg.locomo_path` / `bcfg.longmemeval_path`. If empty, print error and return 1.

   d. Resolve cache paths relative to `pathlib.Path.cwd()` when not absolute: `_resolve_path(bcfg.embedding_cache_path)` — a one-liner lambda.

   e. Load embedding model: `embed_doc, embed_query = run.sync_embedders(mcfg, cache=ec.EmbeddingCache(_resolve_path(bcfg.embedding_cache_path)))`. Wrap in try/except, return 1 on failure with a message.

   f. Load datasets: call `locomo.load_locomo(path)` or `lme.load_longmemeval(path)`. Apply `[:n_val]` slice when `n_mode == "n"` and `n_val > 0`.

   g. Run recall: `recall_report = bench_run.run_recall(datasets, embed_doc=embed_doc, embed_query=embed_query, cfg=mcfg)`.

   h. If `--qa`: import `simba.eval.benchmarks.judge as judge`, `simba.llm.client as llm_client`. Call `judge.sample_cases` with the appropriate mode (`per_category=n_val` for `n_mode=="per"`, `n=n_val` for `n_mode=="n"`, no sampling for `"all"`). `k_val = k or bcfg.default_k`. `llm = llm_client.get_client()`. Build `JudgeCache`. Call `judge.run_qa(qa_datasets, embed_doc=embed_doc, embed_query=embed_query, cfg=mcfg, llm=llm, k=k_val, cache=jc.JudgeCache(_resolve_path(bcfg.judge_cache_path)))`.

   i. Assemble result record: `{"timestamp": __import__("time").time(), "git_sha": current_git_sha(), "dataset": dataset_name, "split": split_arg or None, "config": config_snapshot(mcfg, bcfg), "recall": recall_report, "qa": qa_report or None}`.

   j. Append to results: `append_result(_resolve_path(bcfg.results_path), record)`.

   k. Print: if `--json`, `print(_json.dumps({"recall": recall_report, "qa": qa_report}, indent=2))`; else print the human-readable block modelled on `run_locomo.py`'s print loop.

   l. Return 0.

4. **Wire into `_cmd_eval`**: add `if args and args[0] == "bench": return _eval_bench(args[1:])` as the first branch, before the existing `"build"` check.

5. **Update `__doc__`** at module top: add `simba eval bench DATASET [opts]   Run recall@k (+ QA) on locomo/longmemeval benchmarks` to the usage block.

6. **Delete** `scripts/run_locomo.py`, `scripts/run_longmemeval.py`, `scripts/run_qa.py` (git-rm). The CLI is the replacement.

### Tests

**File**: `/Users/mahmoud/src/ai/simba/tests/test_eval_bench.py`

All tests use `monkeypatch` + fakes — no GGUF load, no live datasets.

**Fakes needed** (define at top of test file):

```python
def _fake_sync_embedders(cfg, *, cache=None):
    return (lambda t: [0.1] * 4), (lambda t: [0.2] * 4)

def _fake_run_recall(datasets, *, embed_doc, embed_query, cfg):
    return {
        "n_conversations": len(datasets),
        "n_cases": 2,
        "overall": {"recall@1": 0.5, "recall@3": 0.6, "recall@5": 0.7,
                    "recall@10": 0.8, "mrr": 0.55, "ndcg@1": 0.5,
                    "ndcg@3": 0.6, "ndcg@5": 0.7, "ndcg@10": 0.8},
        "by_category": {},
    }

# Minimal Dataset/Memory/EvalCase fixtures for load fakes
```

**Test list**:

- `test_bench_missing_dataset_name_returns_1(monkeypatch)`: call `_eval_bench([])`, assert return value is 1. RED: function doesn't exist. GREEN: arg check fires.

- `test_bench_unknown_dataset_returns_1(monkeypatch)`: call `_eval_bench(["badname"])`, assert return value is 1.

- `test_bench_locomo_missing_path_returns_1(monkeypatch, tmp_path)`: monkeypatch `simba.config.load` so `bench.locomo_path = ""`, call `_eval_bench(["locomo"])`, assert 1.

- `test_bench_locomo_recall_runs_and_appends_result(monkeypatch, tmp_path)`: monkeypatch `simba.config.load` returning a `BenchConfig` with `locomo_path="/fake/locomo.json"` and `results_path=str(tmp_path/".simba/eval/results.jsonl")`; monkeypatch `locomo.load_locomo` to return two fake `Dataset` objects; monkeypatch `run.sync_embedders` with `_fake_sync_embedders`; monkeypatch `bench_run.run_recall` with `_fake_run_recall`; monkeypatch `bench_results.current_git_sha` to return `"abc1234"`. Call `_eval_bench(["locomo"])`. Assert return 0. Assert results JSONL exists and has exactly one line. Assert `json.loads(line)["git_sha"] == "abc1234"`. Assert `json.loads(line)["dataset"] == "locomo"`.

- `test_bench_n_flag_slices_datasets(monkeypatch, tmp_path)`: same setup but `_eval_bench(["locomo", "--n", "1"])`. Assert the fake `load_locomo` was called and `run_recall` received exactly 1 dataset (capture via a spy closure).

- `test_bench_json_flag_prints_json(monkeypatch, tmp_path, capsys)`: same setup, `_eval_bench(["locomo", "--json"])`. Assert `capsys.readouterr().out` is valid JSON with key `"recall"`.

- `test_bench_unknown_flag_returns_1(monkeypatch)`: `_eval_bench(["locomo", "--notaflags"])`, assert 1.

- `test_bench_config_memory_overrides_applied(monkeypatch, tmp_path)`: capture the `cfg` arg passed to `bench_run.run_recall` inside the spy. Assert `cfg.llm_rerank_enabled == False` and `cfg.max_results == 20` (the eval override values from BenchConfig defaults).

- `test_bench_embedding_cache_passed_to_sync_embedders(monkeypatch, tmp_path)`: spy on `run.sync_embedders`; assert it was called with a `cache=` kwarg that is an `EmbeddingCache` instance.

- `test_current_git_sha_returns_string()`: import `simba.eval.bench_results`; assert `isinstance(current_git_sha(), str)`. (This test always passes even in non-git envs because failure returns `"unknown"`.)

- `test_append_result_creates_file_and_appends(tmp_path)`: call `append_result` twice with different dicts; assert file has two lines; assert both parse as JSON.

### Acceptance

`simba eval bench locomo --path /tmp/locomo10.json --json` exits 0 and prints valid JSON containing `recall.overall.recall@5`. The file `.simba/eval/results.jsonl` gains exactly one new line per invocation.

### Verify

```bash
uv run simba eval bench locomo --path /tmp/locomo10.json --json
wc -l .simba/eval/results.jsonl
uv run simba config get bench.locomo_path
uv run simba config set bench.locomo_path /tmp/locomo10.json
uv run simba eval bench locomo
```

### Reuse

- `src/simba/__main__.py:_eval_build` — arg-parse loop pattern.
- `src/simba/__main__.py:_cmd_eval:2136` — `sync_embedders` error handling pattern.
- `scripts/run_locomo.py:main` — MemoryConfig override pattern for eval runs.
- `scripts/run_qa.py:main` — `sample_cases` + `run_qa` wiring.
- `src/simba/eval/run.py:sync_embedders` — cache= parameter.

---

## Task A5 — Results Store + Leaderboard

### Goal

`append_result` (already defined in A4's `bench_results.py`) is the write side. Add `simba eval leaderboard` which reads `.simba/eval/results.jsonl`, groups runs by `(dataset, split or "full")`, finds the latest run per group plus the previous run, diffs the key metrics, and renders `BENCHMARKS.md` in the repo root (committed). Also prints a tabular summary to stdout.

### Files

| Action | Path |
|---|---|
| Create (shared with A4) | `/Users/mahmoud/src/ai/simba/src/simba/eval/bench_results.py` |
| Create | `/Users/mahmoud/src/ai/simba/src/simba/eval/leaderboard.py` |
| Modify | `/Users/mahmoud/src/ai/simba/src/simba/__main__.py` |

### Config

No new config fields beyond what A4 already defines (`bench.results_path`). The leaderboard output path is derived: `BENCHMARKS.md` at the repo root (resolved the same way `simba.db.find_repo_root` works — already imported in `simba.config`).

Add one field to `BenchConfig`:

```
bench.leaderboard_path: str = "BENCHMARKS.md"
    Path to the rendered leaderboard file (relative to repo root or absolute).
```

### Signatures

**`src/simba/eval/bench_results.py`** — complete module (A4 + A5 share this):

```python
from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
import typing


def current_git_sha() -> str: ...

def append_result(path: pathlib.Path, record: dict[str, object]) -> None: ...

def config_snapshot(mcfg: object, bcfg: object) -> dict[str, object]: ...

def load_results(path: pathlib.Path) -> list[dict[str, object]]:
    """Read all JSONL records from results_path; skip malformed lines."""
    ...

def latest_two_by_group(
    records: list[dict[str, object]],
) -> dict[str, tuple[dict[str, object], dict[str, object] | None]]:
    """Group by (dataset, split), sort by timestamp, return (latest, prev|None)."""
    ...
```

**`src/simba/eval/leaderboard.py`** — new module:

```python
from __future__ import annotations

import pathlib
import typing

if typing.TYPE_CHECKING:
    pass


_RECALL_METRICS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr")
_QA_METRICS = ("accuracy",)


def _delta_str(current: float, prev: float | None) -> str:
    """Format '+0.012' / '-0.005' / '' for the delta column."""
    ...


def render_markdown(
    groups: dict[str, tuple[dict[str, object], dict[str, object] | None]],
) -> str:
    """Render a full BENCHMARKS.md string from latest_two_by_group output."""
    ...


def render_stdout(
    groups: dict[str, tuple[dict[str, object], dict[str, object] | None]],
) -> str:
    """Render the compact terminal table."""
    ...


def write_leaderboard(
    results_path: pathlib.Path,
    output_path: pathlib.Path,
) -> str:
    """Load results, compute groups, write output_path, return rendered markdown."""
    ...
```

**`src/simba/__main__.py`** — new private helper:

```python
def _eval_leaderboard(args: list[str]) -> int:
    """simba eval leaderboard [--json] [--no-write]"""
    ...
```

Wired in `_cmd_eval`: `if args and args[0] == "leaderboard": return _eval_leaderboard(args[1:])`.

### Implementation Steps

1. **Complete `src/simba/eval/bench_results.py`**:

   - `load_results(path)`: `if not path.exists(): return []`; iterate lines with `for line in path.read_text().splitlines()`; `json.loads(line.strip())` each; skip blank lines and lines that raise `json.JSONDecodeError`; return list.

   - `latest_two_by_group(records)`: group key = `(r["dataset"], r.get("split") or "full")`; sort each group by `r["timestamp"]` descending; take first two. Return `{key: (records[0], records[1] if len >= 2 else None)}`.

2. **Implement `src/simba/eval/leaderboard.py`**:

   - `_delta_str(current, prev)`: if `prev is None` return `""`. `d = current - prev`. Return `f"+{d:.3f}"` if `d > 0` else `f"{d:.3f}"`.

   - `render_markdown(groups)`: produce a markdown document. Header: `# Benchmark Results`. For each group key in sorted order, write `## {dataset} ({split})`. Write a table:
     ```
     | Metric | Latest ({sha[:7]}) | vs Previous | Previous ({sha[:7]}) |
     |---|---|---|---|
     | recall@5 | 0.570 | +0.012 | 0.558 |
     ```
     For recall metrics pull from `latest["recall"]["overall"]`. For QA if `latest.get("qa")` is not None, add a second sub-table for QA metrics from `latest["qa"]["overall"]`. Append timestamp line: `_Run at {iso_timestamp}._`.

   - `render_stdout(groups)`: simpler table with fixed-width columns, no markdown, suitable for terminal.

   - `write_leaderboard(results_path, output_path)`: `records = load_results(results_path)`, `groups = latest_two_by_group(records)`, `md = render_markdown(groups)`, `output_path.write_text(md)`, return `md`.

3. **Implement `_eval_leaderboard(args)` in `src/simba/__main__.py`**:

   ```python
   def _eval_leaderboard(args: list[str]) -> int:
       import pathlib
       import simba.config
       import simba.eval.bench_config  # registers "bench"
       import simba.eval.leaderboard as lb
       import simba.db

       as_json = "--json" in args
       no_write = "--no-write" in args

       bcfg = simba.config.load("bench")
       root = simba.db.find_repo_root(pathlib.Path.cwd()) or pathlib.Path.cwd()
       results_path = _resolve_path_from(bcfg.results_path, root)
       output_path = (
           root / bcfg.leaderboard_path
           if not pathlib.Path(bcfg.leaderboard_path).is_absolute()
           else pathlib.Path(bcfg.leaderboard_path)
       )

       if not results_path.exists():
           print("leaderboard: no results found (run simba eval bench first)",
                 file=sys.stderr)
           return 1

       if no_write:
           import simba.eval.bench_results as br
           records = br.load_results(results_path)
           groups = br.latest_two_by_group(records)
           print(lb.render_stdout(groups))
           return 0

       md = lb.write_leaderboard(results_path, output_path)
       import simba.eval.bench_results as br
       groups = br.latest_two_by_group(br.load_results(results_path))
       print(lb.render_stdout(groups))
       print(f"\nWrote {output_path}")
       return 0
   ```

4. **Wire `leaderboard` branch** in `_cmd_eval` immediately after the `bench` branch.

5. **Update module `__doc__`**: add `simba eval leaderboard [--no-write]  Render BENCHMARKS.md from results log`.

6. **Add `leaderboard_path` field** to `BenchConfig` in `bench_config.py`.

### BENCHMARKS.md format (exact shape)

```markdown
# Benchmark Results

<!-- Generated by `simba eval leaderboard`. Do not edit by hand. -->

## locomo (full)

| Metric | Latest (abc1234, 2026-06-06) | Delta | Previous (def5678, 2026-06-05) |
|---|---|---|---|
| recall@1 | 0.450 | +0.020 | 0.430 |
| recall@3 | 0.560 | +0.010 | 0.550 |
| recall@5 | 0.570 | +0.012 | 0.558 |
| recall@10 | 0.600 | +0.005 | 0.595 |
| mrr | 0.490 | -0.003 | 0.493 |

_Run at 2026-06-06T14:30:00Z._
```

### Tests

**File**: `/Users/mahmoud/src/ai/simba/tests/test_eval_leaderboard.py`

- `test_load_results_empty_when_file_missing(tmp_path)`: call `load_results(tmp_path / "nonexistent.jsonl")`; assert result `== []`.

- `test_load_results_skips_malformed_lines(tmp_path)`: write a JSONL with one valid and one invalid line; assert `load_results` returns exactly 1 record.

- `test_append_result_is_loadable_back(tmp_path)`: `append_result(p, {"dataset": "locomo", "timestamp": 1.0, ...})`, `load_results(p)` returns that dict.

- `test_latest_two_by_group_returns_correct_order(tmp_path)`: create 3 records for same group with timestamps 1.0, 2.0, 3.0. Assert `latest_two_by_group` returns group with `(record_t3, record_t2)`.

- `test_latest_two_by_group_two_datasets_separate_groups()`: records for `locomo` and `longmemeval`; assert two group keys in output.

- `test_delta_str_positive()`: `_delta_str(0.570, 0.558) == "+0.012"`.

- `test_delta_str_negative()`: `_delta_str(0.490, 0.493) == "-0.003"`.

- `test_delta_str_no_previous()`: `_delta_str(0.5, None) == ""`.

- `test_render_markdown_contains_recall5(tmp_path)`: build a minimal `groups` dict with one entry; call `render_markdown(groups)`; assert `"recall@5"` in output and `"## locomo"` in output.

- `test_render_markdown_no_qa_section_when_qa_is_none()`: groups entry has `qa=None`; assert `"accuracy"` not in rendered markdown.

- `test_write_leaderboard_creates_file(tmp_path)`: write two results records to JSONL; call `write_leaderboard(results_path, tmp_path/"BENCHMARKS.md")`; assert file exists and contains `"# Benchmark Results"`.

- `test_leaderboard_cmd_no_write_prints_table(monkeypatch, tmp_path, capsys)`: monkeypatch `simba.config.load` for `"bench"`, monkeypatch `bench_results.load_results` returning one record; call `_eval_leaderboard(["--no-write"])`; assert capsys stdout non-empty, return 0.

- `test_leaderboard_cmd_returns_1_when_no_results(monkeypatch, tmp_path)`: results file does not exist; assert return 1.

### Acceptance

After two `simba eval bench` runs, `simba eval leaderboard` exits 0, prints a table with delta column, and `BENCHMARKS.md` is created or overwritten with the current vs previous metrics.

### Verify

```bash
# after two bench runs
uv run simba eval leaderboard
cat BENCHMARKS.md
# check git status shows BENCHMARKS.md modified
git diff --stat BENCHMARKS.md
uv run simba eval leaderboard --no-write   # prints but skips file write
```

### Reuse

- `src/simba/eval/bench_results.py:append_result` (A4) — same JSONL append pattern as `simba.memory.fts` keyword mirror.
- `src/simba/__main__.py:_eval_build:2159` — the `_eval_*` helper convention.
- `simba.db.find_repo_root` — already used throughout `__main__.py` to find `.simba/` root.

---

## Task A6 — CI Smoke Test

### Goal

A synthetic 2-document / 2-case dataset exercised in the existing pytest suite so the `simba eval bench` code path is integration-tested in CI without requiring real datasets, a GGUF model download, or live LLM. The synthetic dataset is a small JSON fixture checked into the test tree. The test monkeypatches the embedding functions with a deterministic fake that makes recall scores predictable (gold document is always ranked first).

### Files

| Action | Path |
|---|---|
| Create | `/Users/mahmoud/src/ai/simba/tests/fixtures/smoke_bench.json` |
| Create | `/Users/mahmoud/src/ai/simba/tests/test_eval_bench_smoke.py` |
| Modify | `/Users/mahmoud/src/ai/simba/.github/workflows/ci.yml` (no-op — existing `uv run pytest -q` already picks up new tests) |

### Fixture: `tests/fixtures/smoke_bench.json`

```json
{
  "name": "smoke",
  "corpus": [
    {
      "id": "m1",
      "content": "Alice loves hiking on weekends",
      "type": "PATTERN",
      "context": "",
      "project_path": "",
      "session_source": "",
      "created_at": "",
      "confidence": 0.9
    },
    {
      "id": "m2",
      "content": "Bob prefers indoor climbing",
      "type": "PATTERN",
      "context": "",
      "project_path": "",
      "session_source": "",
      "created_at": "",
      "confidence": 0.9
    }
  ],
  "cases": [
    {
      "id": "q1",
      "query": "What does Alice like to do?",
      "relevant_ids": ["m1"],
      "intent": "single-hop",
      "note": "",
      "split": "",
      "answer": "hiking"
    },
    {
      "id": "q2",
      "query": "What sport does Bob prefer?",
      "relevant_ids": ["m2"],
      "intent": "single-hop",
      "note": "",
      "split": "",
      "answer": "indoor climbing"
    }
  ]
}
```

### Signatures

**`tests/test_eval_bench_smoke.py`** — key functions:

```python
import pathlib
import pytest

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "smoke_bench.json"

def _perfect_embed(text: str) -> list[float]:
    """Returns a unique deterministic vector keyed on content hash so
    the gold document always has the highest cosine similarity to its
    own query."""
    ...

def _make_fake_embed_doc(corpus_map: dict[str, str]):
    """Return an embed_doc callable: content -> its identity vector."""
    ...

def _make_fake_embed_query(corpus_map: dict[str, str], case_map: dict[str, str]):
    """Return an embed_query callable: query -> vector of its gold doc."""
    ...
```

### Implementation Steps

1. **Create `tests/fixtures/smoke_bench.json`** with the JSON shown above.

2. **Create `tests/test_eval_bench_smoke.py`**. The key insight for fake embeddings: assign each corpus document a unique unit vector (a one-hot vector in a 4-dim space is fine for 2 documents). Map query to the same vector as its gold document so cosine similarity = 1.0 and the gold document is always rank 1.

   Concrete fake:
   ```python
   _DOC_VECTORS = {"m1": [1.0, 0.0, 0.0, 0.0], "m2": [0.0, 1.0, 0.0, 0.0]}
   _QUERY_GOLD = {"q1": "m1", "q2": "m2"}

   def _embed_doc(text: str) -> list[float]:
       for mid, vec in _DOC_VECTORS.items():
           # match by substring presence in the corpus content
           if mid == "m1" and "Alice" in text:
               return vec
           if mid == "m2" and "Bob" in text:
               return vec
       return [0.25, 0.25, 0.25, 0.25]

   def _embed_query(text: str) -> list[float]:
       if "Alice" in text:
           return _DOC_VECTORS["m1"]
       if "Bob" in text:
           return _DOC_VECTORS["m2"]
       return [0.25, 0.25, 0.25, 0.25]
   ```

3. **Write tests**. Because `build_retriever` uses LanceDB internally, the test must monkeypatch at the `run_recall` level (which is already tested separately) or at the `recall_adapter.build_retriever` level. The cleanest approach: monkeypatch `simba.eval.recall_adapter.build_retriever` to return a fake retriever closure keyed on the known gold mapping, then call `bench_run.run_recall` directly. This exercises `run_recall`'s aggregation logic without touching LanceDB.

   Alternatively (simpler and still proves the code path): call `bench_run.run_recall` with a monkeypatched `build_retriever` that returns `lambda query: ["m1"] if "Alice" in query else ["m2"]`. The test then asserts `recall@1 == 1.0`.

4. **No changes to `ci.yml`** needed: the existing `uv run pytest -q` step already discovers all `tests/test_*.py` files. The `UV_EXTRA_INDEX_URL` for llama-cpp-python is already set, but the smoke test never imports `EmbeddingService` so the model is never loaded — CI passes without the GGUF download.

### Tests

**File**: `/Users/mahmoud/src/ai/simba/tests/test_eval_bench_smoke.py`

- `test_smoke_bench_fixture_is_valid_json()`: `json.loads(FIXTURE.read_text())` — assert no exception; assert `len(data["corpus"]) == 2`; assert `len(data["cases"]) == 2`. RED: fixture file missing. GREEN: fixture created.

- `test_smoke_bench_fixture_loads_as_dataset()`: `simba.eval.dataset.load_dataset(FIXTURE)` — assert `dataset.name == "smoke"`; assert `len(dataset.corpus) == 2`. RED: fixture invalid. GREEN: passes after step 1.

- `test_smoke_run_recall_perfect_retriever(monkeypatch)`: monkeypatch `simba.eval.recall_adapter.build_retriever` to return the perfect fake retriever. Call `bench_run.run_recall([dataset], embed_doc=_embed_doc, embed_query=_embed_query, cfg=MemoryConfig())`. Assert `report["overall"]["recall@1"] == 1.0` and `report["n_cases"] == 2`. RED: `run_recall` import error or aggregation broken. GREEN: core path works.

- `test_smoke_run_recall_zero_retriever(monkeypatch)`: fake retriever always returns `[]`. Assert `report["overall"]["recall@1"] == 0.0`. Verifies the lower bound.

- `test_smoke_append_and_load_roundtrip(tmp_path)`: write two records via `append_result`, `load_results`; assert 2 records returned and both have correct `dataset` key. Verifies A5's store is exercised end-to-end.

- `test_smoke_bench_cmd_end_to_end(monkeypatch, tmp_path)`: the highest-value test — calls `_eval_bench(["locomo", "--path", str(FIXTURE), "--json"])` with all internals monkeypatched (load_locomo replaced by `load_dataset(FIXTURE)` wrapped as a list, sync_embedders faked, build_retriever faked). Assert return 0, results file created, JSON output parseable. This proves the full CLI dispatch path down to file append without any external dep.

### Acceptance

`uv run pytest tests/test_eval_bench_smoke.py -v` passes in under 5 seconds with zero network calls. The existing CI job `uv run pytest -q` passes without modification.

### Verify

```bash
uv run pytest tests/test_eval_bench_smoke.py -v
uv run pytest -q                          # full suite still green
```

### Reuse

- `src/simba/eval/benchmarks/run.py:run_recall` — called directly in smoke tests.
- `src/simba/eval/dataset.py:load_dataset` — used to load the fixture.
- `tests/conftest.py:simba_db` — pattern for monkeypatching module attributes.

---

## Complete File List

**Create**:
- `/Users/mahmoud/src/ai/simba/src/simba/eval/bench_config.py`
- `/Users/mahmoud/src/ai/simba/src/simba/eval/bench_results.py`
- `/Users/mahmoud/src/ai/simba/src/simba/eval/leaderboard.py`
- `/Users/mahmoud/src/ai/simba/tests/fixtures/smoke_bench.json`
- `/Users/mahmoud/src/ai/simba/tests/test_eval_bench.py`
- `/Users/mahmoud/src/ai/simba/tests/test_eval_leaderboard.py`
- `/Users/mahmoud/src/ai/simba/tests/test_eval_bench_smoke.py`

**Modify**:
- `/Users/mahmoud/src/ai/simba/src/simba/__main__.py` — add `_eval_bench`, `_eval_leaderboard` helpers; add two branches in `_cmd_eval`; extend module docstring.
- `/Users/mahmoud/src/ai/simba/src/simba/eval/bench_config.py` — add `leaderboard_path` field (A5 addition on top of A4 initial creation).

**Delete**:
- `/Users/mahmoud/src/ai/simba/scripts/run_locomo.py`
- `/Users/mahmoud/src/ai/simba/scripts/run_longmemeval.py`
- `/Users/mahmoud/src/ai/simba/scripts/run_qa.py`

**No changes needed**:
- `/Users/mahmoud/src/ai/simba/.github/workflows/ci.yml` — existing `uv run pytest -q` already discovers new tests; CI job matrix unchanged.

---

## Data Flow

```
simba eval bench locomo --path /tmp/locomo10.json --json
  │
  ├─ _cmd_eval(["bench", "locomo", ...])
  │     └─ _eval_bench(["locomo", "--path", ...])
  │           │
  │           ├─ simba.config.load("bench")  → BenchConfig
  │           ├─ simba.config.load("memory") → MemoryConfig
  │           │     └─ dataclasses.replace(mcfg, **bcfg.eval_memory_config_overrides())
  │           │
  │           ├─ locomo.load_locomo(path)    → list[Dataset]
  │           │
  │           ├─ run.sync_embedders(mcfg, cache=EmbeddingCache(...))
  │           │     └─ loads GGUF once; wraps with cached_embedder
  │           │
  │           ├─ bench_run.run_recall(datasets, ...)
  │           │     └─ per Dataset: build_retriever → run_eval → CaseResult[]
  │           │     → dict {overall, by_category, n_cases, n_conversations}
  │           │
  │           ├─ [--qa] judge.run_qa(qa_datasets, ..., cache=JudgeCache(...))
  │           │     → dict {n_graded, n_skipped, overall, by_category}
  │           │
  │           ├─ bench_results.append_result(.simba/eval/results.jsonl, record)
  │           │     record = {timestamp, git_sha, dataset, split, config, recall, qa}
  │           │
  │           └─ print JSON / human table → stdout

simba eval leaderboard
  │
  ├─ _cmd_eval(["leaderboard"])
  │     └─ _eval_leaderboard([])
  │           │
  │           ├─ bench_results.load_results(.simba/eval/results.jsonl)
  │           │     → list[dict]  (skips malformed lines)
  │           │
  │           ├─ bench_results.latest_two_by_group(records)
  │           │     → {(dataset, split): (latest_record, prev_record|None)}
  │           │
  │           ├─ leaderboard.write_leaderboard(results_path, BENCHMARKS.md)
  │           │     └─ render_markdown(groups) → str
  │           │     └─ BENCHMARKS.md.write_text(md)
  │           │
  │           └─ leaderboard.render_stdout(groups) → print to terminal
```

---

## Build Sequence (Checklist)

**Phase 1 — Foundation (A4 data layer)**
- [ ] Create `src/simba/eval/bench_config.py` with `BenchConfig` (all fields including `leaderboard_path`).
- [ ] Create `src/simba/eval/bench_results.py` with `current_git_sha`, `append_result`, `config_snapshot`, `load_results`, `latest_two_by_group`.
- [ ] Write RED tests: `tests/test_eval_bench.py::test_append_result_*`, `test_current_git_sha_*`, `tests/test_eval_leaderboard.py::test_load_results_*`, `test_latest_two_by_group_*`.
- [ ] Make tests GREEN (implement the functions).

**Phase 2 — CLI A4**
- [ ] Write RED tests: `tests/test_eval_bench.py::test_bench_missing_*`, `test_bench_locomo_recall_runs_*`, `test_bench_n_flag_*`, `test_bench_json_flag_*`.
- [ ] Implement `_eval_bench` in `__main__.py`.
- [ ] Wire `if args[0] == "bench"` branch in `_cmd_eval`.
- [ ] Make tests GREEN.
- [ ] Git-rm the three scripts.

**Phase 3 — Leaderboard A5**
- [ ] Write RED tests: `tests/test_eval_leaderboard.py::test_delta_str_*`, `test_render_markdown_*`, `test_write_leaderboard_*`, `test_leaderboard_cmd_*`.
- [ ] Implement `src/simba/eval/leaderboard.py` (`_delta_str`, `render_markdown`, `render_stdout`, `write_leaderboard`).
- [ ] Implement `_eval_leaderboard` in `__main__.py`.
- [ ] Wire `if args[0] == "leaderboard"` branch.
- [ ] Make tests GREEN.

**Phase 4 — CI Smoke A6**
- [ ] Create `tests/fixtures/smoke_bench.json`.
- [ ] Write RED tests: `tests/test_eval_bench_smoke.py` (all 6 tests).
- [ ] Implement fake embedders in test file.
- [ ] Make tests GREEN (they should go green once A4 implementation is in place, since they monkey-patch LanceDB away).
- [ ] Verify `uv run pytest -q` passes in full.

**Phase 5 — Validation**
- [ ] `uv run ruff check src/ tests/` — zero violations.
- [ ] `uv run ruff format src/ tests/` — no changes.
- [ ] `uv run pytest -q` — all tests green.
- [ ] Manual: `uv run simba config get bench.locomo_path` returns `""`.
- [ ] Manual: `uv run simba config set bench.locomo_path /tmp/test.json`, then `get` confirms value.
- [ ] Manual: `uv run simba eval bench --help` prints usage (the error-path usage string).

---

## Critical Details

**No argparse**: follow the `while i < len(args)` pattern from `_cmd_eval`/`_eval_build`. Do not introduce `argparse` — it would be inconsistent with every other command in `__main__.py`.

**`dataclasses.replace` for mcfg override**: never mutate the loaded `MemoryConfig`. `dataclasses.replace(mcfg, **bcfg.eval_memory_config_overrides())` creates a clean copy for the bench run while the original stays in the registry.

**Relative path resolution**: `bench.results_path` and `bench.embedding_cache_path` are relative strings. Resolve them with `pathlib.Path.cwd() / path` when not absolute. Use a local helper `_resolve_bench_path(s: str) -> pathlib.Path` inside `_eval_bench` — one line, not a module-level function, to avoid polluting `__main__` namespace.

**Append-only guarantee**: `append_result` must open in `"a"` mode. Never call `.write_text()` on the results JSONL. `BENCHMARKS.md` is the one file that _is_ overwritten — it is derived state (computed from append-only JSONL), not primary storage.

**`BENCHMARKS.md` is generated, not primary**: include a `<!-- Generated by simba eval leaderboard -->` comment so humans know not to hand-edit it. The JSONL is the source of truth.

**Timestamp provenance**: `time.time()` called inside `_eval_bench` at the moment of the run. No time injection needed for the CLI — only tests that assert on the timestamp value need to monkeypatch `time.time`.

**`config_snapshot` must use `dataclasses.asdict`**: both `MemoryConfig` and `BenchConfig` are dataclasses, so `dataclasses.asdict` works. Include both in the record so a future diff can pinpoint which config field changed between runs.

**`run_qa` embedding cache reuse**: the same `EmbeddingCache` instance opened for recall is passed to `sync_embedders`. Documents already embedded for the recall phase are already cached — the QA phase hits cache for all of them and only pays for new query texts.

**CI safety**: the smoke test must never call `EmbeddingService._load_model`. The monkeypatch target is `simba.eval.recall_adapter.build_retriever` — patch it before `run_recall` is called. Since `run_recall` calls `build_retriever` internally, patching at that level prevents any LanceDB or GGUF import from executing within the test.

**`ruff TID251`**: every file uses `pathlib.Path`, never `os.path`. The `bench_results.py` module uses `pathlib.Path.read_text().splitlines()` not `open(...).readlines()`.
