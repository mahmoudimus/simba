# 26 — Hierarchical (ancestor-prefix) project memory recall

**Date:** 2026-06-16
**Status:** TODO (design; not started)
**Branch:** TBD off `main`

## Why

Today's project scoping is **flat and inconsistent**:

- **Plain memories** are scoped by the **literal** `--project-path` string (or cwd),
  and recall does a **strict exact match** — `src/simba/memory/vector_db.py`
  `search_memories` skips a row when `r["projectPath"] != filter_project`. So
  `/repo/api` memories never recall from `/repo` root, and vice-versa.
- **TOOL_RULEs** are canonicalized to the **git-root `project_id`** (via
  `simba.db.resolve_project_id`), so they're already shared across `/repo` and
  `/repo/api`. → memories and rules use **two different scoping models**.
- When a project filter is set, **global memories** (empty path) are excluded too
  (a global mem's `""` ≠ the filter), so a project recall is an island.

**Monorepo pain (proj):** 36 facts seeded under `/proj/api` don't recall from the
`/proj` root. Content-dedup (0.92) blocks dual-homing the same fact under two
paths — so there's no clean way to make a fact visible from both today.

## Model: ancestor-prefix recall

Recall at cwd `C` returns memories scoped to `C` **∪ every ancestor of `C` up to
the git root ∪ global**:

```
recall(/proj/api) → {/proj/api} ∪ {/proj} ∪ {global}
recall(/proj)     →               {/proj} ∪ {global}
```

Root facts inherit **down** to every package; package-specific facts stay put and
**don't leak** to siblings or root. Mirrors how CLAUDE.md cascades up the tree.

**This dissolves the dedup pain:** place a shared fact at `/proj` **once**; `api`
inherits it. You never dual-home, so content-dedup stays global and correct (each
fact lives at exactly one node).

## Design

The **client** computes the scope chain (it knows its cwd + git root); the
**daemon stays filesystem-agnostic** (string-membership match only — the client's
paths may not even exist on the daemon's host).

### Storage — normalize
- On store, normalize `projectPath` to an absolute, symlink-resolved path:
  `str(pathlib.Path(p).resolve())` (no trailing slash). Apply in
  `routes.store_memory` and the CLI store path (`__main__`). Pre-existing literal
  paths remain matchable (membership still works); a one-off
  `simba memory migrate-paths` can normalize old rows (optional).

### Recall — widen the filter
- Recall request gains optional `project_scopes: list[str]` (alias
  `projectScopes`) — the client-computed chain `[cwd-resolved, …ancestors…,
  git-root-resolved]`.
- `search_memories`: when `project_scopes` is present **and**
  `memory.hierarchical_recall` is on → keep a row if `m.projectPath ∈
  project_scopes`, OR (`hierarchical_recall_include_global` and `m.projectPath`
  is empty). Otherwise (legacy) → exact `== project_path` (byte-identical).
- **Both retrieval arms must honor the same scope set**: the BM25/FTS path
  (`src/simba/memory/fts.py`) and the hybrid fusion, not just the vector arm —
  else hierarchical recall is inconsistent across retrievers.
- Client (`src/simba/hooks/_memory_client.py` recall + CLI `memory recall`): when
  `hierarchical_recall` is on, compute the chain via `simba.db.find_repo_root(cwd)`
  + walking parents from the resolved cwd up to the root; send `project_scopes`.
  When off, send the single `project_path` (legacy).

### Config (`memory` section, `src/simba/memory/config.py`)
- `hierarchical_recall: bool = False` — default **OFF** (unmeasured; widening the
  candidate pool can dilute precision). Graduate to ON only after a measured
  no-regression on recall@k (per the SoTA-lever rule).
- `hierarchical_recall_include_global: bool = True` — global memories are the root
  of the tree (fixes the "global excluded under a project filter" quirk). Separate
  lever so it can be measured independently.
- Bound = the git root by default (the chain stops at `find_repo_root`); no extra
  config initially. (Future `hierarchical_recall_max_depth` only if a cross-repo
  climb is ever wanted.)

## Phases (TDD — RED first)
- **A.** Normalize `projectPath` on store (`routes` + CLI). Test: a stored memory's
  `projectPath` is absolute/resolved.
- **B.** `search_memories` ancestor-membership filter behind `hierarchical_recall`
  (default off). Tests: `recall(/repo/api)` includes `/repo`-scoped + global,
  **excludes** `/repo/web`-scoped; `recall(/repo)` excludes `/repo/api`-scoped;
  default-off path is exact-match legacy (characterization).
- **C.** FTS/BM25 arm + hybrid fusion honor the same scope set. Test: hybrid recall
  returns the inherited memory.
- **D.** Client chain computation (hook + CLI) via `find_repo_root`; default-off
  sends a single path. Tests for both modes.
- **E.** Config default-assertions (`hierarchical_recall is False`,
  `include_global is True`); `simba config get/set` round-trip.
- **F.** Measurement: eval recall@k on the held-out split, hierarchical ON vs OFF
  (LoCoMo / `longmemeval_s`), ablation + latency p50/p95. Graduate to default-ON
  only if no regression.

## Caveats
- **Precision dilution.** A bigger candidate pool may surface less-relevant
  ancestor/global memories; the reranker + similarity floor + RRF should rank them
  out — but **measure** (Phase F) before default-on.
- **Path normalization / symlinks.** proj has symlinks (symlinked MEMORY.md);
  `.resolve()` both the stored paths and the chain, or ancestry breaks.
- **Bound at the git root** by default (don't leak `/path/to` across
  repos).
- `include_global` flips today's behavior (project recall starts including
  globals) — measure separately.

## Alternatives considered
- **Collapse-to-git-root-id** (canonicalize memory store/recall like `rule add`):
  a tiny diff that instantly shares `/repo ≡ /repo/api` and fixes the
  memory-vs-rule inconsistency — **but flattens the repo** (an `api` fact would
  also surface in `web/`). It's the degenerate tree (store only at the root node);
  the full tree **subsumes** it. Ship the tree. (Collapse is a reasonable
  same-day stopgap to make the proj facts recall from root before the tree lands.)

## Non-goals
- Dual-homing (obviated by the tree).
- Cross-repo ancestor climb by default.
- Auto-migrating existing memories (a separate `simba memory update
  --project-path` pass if desired).

## Acceptance
- With `hierarchical_recall` on: `recall(/repo/api)` surfaces `/repo`-scoped +
  global facts; `recall(/repo)` does **not** surface `/repo/api`-scoped facts;
  the default-off path is byte-identical to today.
- **No recall@k regression** on the held-out split (else keep default-off);
  `ruff check` + full suite green.
- (Stretch) memories and TOOL_RULEs share one scoping model.
