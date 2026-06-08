# Simba

> *"Remember who you are."* — Mufasa

[![CI](https://github.com/mahmoudimus/simba/actions/workflows/ci.yml/badge.svg)](https://github.com/mahmoudimus/simba/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Runtimes: Claude Code + Codex](https://img.shields.io/badge/runtimes-Claude%20Code%20%2B%20Codex-8a2be2.svg)](#codex-support)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A unified memory + reasoning plugin for **Claude Code _and_ Codex**. It combines semantic memory, CLAUDE.md rule enforcement, neuro-symbolic logic (Z3 + Datalog), project-aware search, and **RLM lossless transcript recall** into a single pure-Python package — with full hook integration and native skills for **both** runtimes.

## Quick Start

```bash
# Install (editable — picks up source changes immediately)
uv tool install -e /path/to/simba

# Register hooks in Claude Code
simba install

# Done. Start a Claude Code session and everything auto-starts.
```

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Install from source (editable)

Recommended during development. Source changes take effect immediately.

```bash
git clone git@github.com:mahmoudimus/simba.git
cd simba
uv tool install -e .
```

The `simba` binary is installed to `~/.local/bin/simba` (ensure this is on your PATH).

### Install from git

```bash
# latest main
uv tool install git+https://github.com/mahmoudimus/simba.git
# a specific release
uv tool install git+https://github.com/mahmoudimus/simba.git@v0.2.0
```

### Install from PyPI

Published as **`simba-ai`** (the name `simba` was taken); the import package and
the CLI are still `simba`:

```bash
uv tool install simba-ai          # or: pip install simba-ai
simba install --global
```

The **core** install (KG, sync, config CLI, hooks, neuro-symbolic logic) has no
heavy ML dependencies. The in-process semantic-memory daemon needs the optional
`embed` extra (LanceDB + llama-cpp-python + huggingface-hub):

```bash
uv tool install 'simba-ai[embed]'   # or: pip install 'simba-ai[embed]'
```

Without `[embed]`, `simba memory start` exits with a clear ImportError telling
you to install the extra; everything else works.

### Upgrading

An **editable** install (`uv tool install -e .`) picks up source changes whenever
its checkout is updated — no reinstall. A **non-editable** install is a frozen
copy: upgrade it explicitly, and re-run `simba install` so the new hooks + skills
are written into `~/.claude/` (the installer now *updates* existing skills):

```bash
uv tool upgrade simba          # or: uv tool install --reinstall git+...@v0.2.0
simba install --global         # refresh hooks + bundled skills
```

See [CHANGELOG.md](CHANGELOG.md) for what's in each release. Releases are cut from
`v*` tags (`.github/workflows/release.yml`): the tag builds the wheel + sdist and
publishes a GitHub Release with the matching CHANGELOG section.

### Register hooks

```bash
simba install             # Register hooks + skills in current project
simba install --global    # Register hooks + skills globally (~/.claude/)
simba install --remove    # Remove hooks and skills
```

## Codex Support

Simba ships both Codex-native skills and a full Codex hook integration.

### Hooks

`.codex/hooks.json` registers the same handlers Claude Code uses, so memory
recall, rule reinforcement, and TOOL_RULE checks all fire under Codex too.
Codex's event set differs slightly: there is no `PreCompact`, but there is a
`PermissionRequest` event that Claude Code doesn't expose.

| Event | Claude Code | Codex |
|-------|-------------|-------|
| SessionStart | ✓ | ✓ |
| UserPromptSubmit | ✓ | ✓ |
| PreToolUse | ✓ | ✓ |
| PostToolUse | ✓ | ✓ |
| Stop | ✓ | ✓ |
| PreCompact | ✓ | — |
| PermissionRequest | — | ✓ |

Run `simba codex-install` to set up everything Codex needs:

```bash
simba codex-install           # Installs Codex skills + sets [features] hooks = true
                              # (also migrates the deprecated codex_hooks key)
simba codex-install --remove  # Removes both skills and the feature flag
```

The first time you open a Codex session in a project that ships `.codex/hooks.json`,
Codex will prompt you to trust it — accept so hooks load. The hooks invoke
the same `simba hook <Event>` dispatcher used by Claude Code; no extra step
is needed beyond having `simba` on `PATH`.

### Skills + CLI

```bash
simba codex-install           # Install bundled Codex skills to $CODEX_HOME/skills
simba codex-install --remove  # Remove bundled Codex skills
simba codex-status            # Check daemon health + pending extraction
simba codex-extract           # Print extraction prompt for latest transcript
simba codex-recall "<query>"  # Query semantic memory via /recall
simba codex-finalize          # Run end-of-task signal/error checks
simba codex-automation        # Print a suggested Codex automation directive
```

Default install path:

- `${CODEX_HOME:-$HOME/.codex}/skills/simba-onboard/`
- `${CODEX_HOME:-$HOME/.codex}/skills/simba-codex-lifecycle/`

Ask Codex to use:
- `simba-onboard` for project instruction onboarding
- `simba-codex-lifecycle` to enforce `codex-status` / `codex-extract` / `codex-finalize`

## Workflow

### Setting Up a New Project

```bash
cd ~/src/my-project

# 1. Install simba hooks and skills into this project
simba install

# 2. In Claude Code, run the onboarding skill
/simba-onboard
```

The `/simba-onboard` skill walks you through:
- Reading all your project's markdown files (CLAUDE.md, AGENTS.md, `.claude/**/*.md`)
- Extracting key instructions into categories (constraints, build commands, environment, etc.)
- Generating `.claude/rules/CORE_INSTRUCTIONS.md` with SIMBA marker sections
- Wiring references into CLAUDE.md and AGENTS.md

### Day-to-Day Usage

Once installed, simba works automatically via hooks. No manual steps needed during normal Claude Code sessions.

**What happens on each session:**

1. **SessionStart** — memory daemon auto-starts, tailor context injected, project stats shown
2. **Every prompt** — relevant memories recalled, core rules reinforced, search context injected
3. **Every tool call** — vector DB queried based on Claude's thinking; context-low warning when approaching compaction
4. **Session end** — transcript exported for learning extraction; error patterns captured

### Managing Core Instructions

SIMBA markers (`<!-- BEGIN SIMBA:name -->`) define sections in markdown files that simba can discover, audit, and update.

```bash
# See what markers exist in your project
simba markers list

# Check marker health (stale content, user-defined markers)
simba markers audit

# Update managed sections with latest templates
simba markers update

# Convert old markers (NEURON:*, CORE, etc.) to SIMBA format
simba markers migrate
```

Edit `.claude/rules/CORE_INSTRUCTIONS.md` directly to refine rules. Marker sections are preserved — simba only touches content between its own `<!-- BEGIN SIMBA:* -->` tags.

### Configuring Simba

```bash
# See all configurable settings
simba config list

# Adjust memory recall sensitivity
simba config set memory.min_similarity 0.40

# Use a custom port for the daemon
simba config set memory.port 9000

# Set a global default (applies to all projects)
simba config set --global memory.max_results 5

# View effective config (local > global > defaults)
simba config show
```

### Global vs Per-Project Install

| Scope | Command | Settings File | Skills Dir |
|-------|---------|---------------|------------|
| Project | `simba install` | `.claude/settings.local.json` | `.claude/skills/` |
| Global | `simba install --global` | `~/.claude/settings.json` | `~/.claude/skills/` |

Per-project is the default. Use global when you want simba active in every Claude Code session regardless of project.

## What It Does

Simba hooks into the Claude Code **and** Codex lifecycle events to provide persistent context across sessions (see [Codex Support](#codex-support) for the Codex event mapping):

| Hook | Purpose |
|------|---------|
| **SessionStart** | Start memory daemon, inject tailor context, show project memory stats |
| **UserPromptSubmit** | Recall semantic memories, reinforce CLAUDE.md core rules, inject search context |
| **PreToolUse** | Query semantic memory based on Claude's thinking; warn when context is low |
| **PostToolUse** | Track file reads, edits, searches, and commands in an activity log |
| **PreCompact** | Export transcript to disk before context compaction |
| **Stop** | Check for rule compliance signal, capture errors from final transcript |
| **PermissionRequest** *(Codex only)* | Deny Bash/`apply_patch`/MCP calls that match a high-confidence TOOL_RULE memory |

## Memory Daemon

FastAPI server backed by LanceDB for vector storage. Supports two embedding backends.

### In-process mode (default)

Loads a GGUF model via llama-cpp-python. No external services needed. The model (~81 MB) auto-downloads from Hugging Face on first startup.

```bash
simba server                                    # Default — auto-downloads nomic-embed-text
simba server --model-path /path/to/model.gguf   # Use a local GGUF file
simba server --n-gpu-layers 0                   # CPU-only mode
simba server --port 9000 --db-path /path/to/db  # Custom port and database
```

### External server mode

Delegates embedding to an external OpenAI-compatible server.

```bash
simba server --embed-url http://localhost:8080
```

Works with llama-cpp-python server, vLLM, text-embeddings-inference, or any OpenAI-compatible endpoint.

### Swapping the embedder

The embedder is fully `simba config`-driven (`embed_provider`, `model_repo`/`model_file`/`embedding_dims`, and the per-model task prefixes). Backends: `gguf` (in-process, default), `http` (`embed_url`), or `llm-cli` (`llm embed`; note: only as local as the chosen `llm` model — cloud models cross the no-external-service line). Example — switch to **Qwen3-Embedding-0.6B**:

```bash
simba config set memory.model_repo Qwen/Qwen3-Embedding-0.6B-GGUF
simba config set memory.model_file Qwen3-Embedding-0.6B-Q8_0.gguf
simba config set memory.embedding_dims 1024
simba config set memory.embed_doc_prefix ""
simba config set memory.embed_query_prefix "Instruct: Given a query, retrieve relevant memories
Query: "
# restart the daemon (loads the new model), then rebuild the corpus at the new dim:
simba memory reembed
```

A dimension change requires re-embedding the whole corpus — that's `simba memory reembed` (explicit, never automatic). **Measure before you switch:** `scripts/embedder_bakeoff.py` scores candidates on the eval datasets. On the current (small, saturating) datasets nomic-Q8 and Qwen3-0.6B are **a wash vs nomic-Q4** (Q8 even regressed the temporal set), so the default stays nomic-Q4 — a real embedder decision needs the hardened/real-corpus eval.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/store` | Store a typed memory with embedding |
| POST | `/recall` | Hybrid search over memories (vector + BM25, RRF-fused) |
| POST | `/sync` | Trigger a one-off sync cycle (index + extract) |
| POST | `/reindex` | Rebuild the BM25 keyword mirror from LanceDB |
| GET | `/health` | Health check with model info |
| GET | `/stats` | Memory count and database stats |
| GET | `/list` | List all memories |
| PATCH | `/memory/:id` | Update a memory's `projectPath` / `sessionSource` |
| DELETE | `/memory/:id` | Delete a specific memory |

### Hybrid recall (BM25 + vector)

`/recall` fuses two arms with [Reciprocal Rank Fusion](https://en.wikipedia.org/wiki/Learning_to_rank) (RRF):

- a **vector arm** — LanceDB cosine similarity (gated by `memory.min_similarity`), and
- a **keyword arm** — a SQLite FTS5 `bm25`/`trigram` index that catches exact
  identifiers and literal strings the embedding under-ranks. It is *not*
  cosine-gated, so it also widens coverage.

The keyword index is a **derived mirror** at `<db-path>/memory_fts.db`, kept in
sync by the daemon on store/delete/patch and rebuilt from LanceDB on startup (or
on demand via `simba memory reindex`). It is fully fail-safe: any keyword-arm
error degrades recall to vector-only, and `memory.hybrid_enabled false` forces
the pure-vector path. Scoping is strict — the keyword arm never surfaces another
project's memories.

```bash
simba config set memory.hybrid_enabled true   # on by default
simba config set memory.rrf_k 60              # RRF rank constant
simba config set memory.fts_candidate_pool 20 # candidates pulled per arm
simba config set memory.fts_tokenize trigram  # trigram | porter | unicode61
simba config set memory.vector_weight 1.0     # RRF weight, vector arm
simba config set memory.keyword_weight 1.0    # RRF weight, keyword arm
simba memory reindex                          # rebuild the keyword mirror
```

### Query intelligence

Before fusing the arms, `/recall` adapts to the query — all LLM-free, so it runs
inside hooks with no extra model call:

- **Intent-aware breadth** — a lightweight, marker-based classifier tags each
  query `broad` (aggregation / history / exploration) or `precise` (point fact).
  Broad queries lower the cosine floor (`min_similarity_broad`) **and** widen the
  result count and candidate pool (`max_results_broad` / `fts_candidate_pool_broad`);
  precise queries keep the strict defaults. An explicit client `minSimilarity` /
  `maxResults` always wins.
- **Entity-biased keyword arm** — the keyword arm is fed a small set of
  high-signal terms (identifiers, paths, proper nouns), capped at `fts_max_terms`,
  instead of the whole query. This stops a long thinking block from OR-ing
  hundreds of tokens into bm25 and matching almost everything.
- **Recency-labeled injection** — recalled memories are annotated with their
  `created` date and the most-recently-created one is flagged `recency="newest"`,
  so the model can prefer fresher facts when two memories conflict (the relevance
  order itself is untouched).
- **Multi-arm HyDE** — with `memory.expansion_enabled` (on by default), a **second
  vector arm** embeds the focused-term string as its own query and is fused into
  RRF alongside the full-query vector arm and the keyword arm. It often nails
  identifiers/entities the full-query embedding blurs, at the cost of one extra
  embed per recall.

```bash
simba config set memory.intent_aware true           # adapt breadth to query intent
simba config set memory.min_similarity_broad 0.28   # cosine floor for broad queries
simba config set memory.max_results_broad 8         # results returned for broad queries
simba config set memory.fts_candidate_pool_broad 40 # RRF candidate pool for broad queries
simba config set memory.fts_max_terms 12            # cap on high-signal keyword-arm terms
simba config set memory.expansion_enabled true      # 2nd HyDE vector arm (opt-in)
```

### Supersession on store

By default `/store` rejects an exact duplicate (cosine ≥ `duplicate_threshold`,
0.92). With **`memory.supersede_enabled`** on, a *near*-duplicate of the **same
type and project** — similarity in `[supersede_threshold, duplicate_threshold)` —
**replaces** the older memory instead of appending another near-copy: the old row
is deleted from both LanceDB and the keyword mirror and `/store` returns
`{"status": "superseded", "supersededId": ...}`. This keeps the freshest version
of an evolving note. On by default (experimental).

```bash
simba config set memory.supersede_enabled false   # keep every near-dupe instead
simba config set memory.supersede_threshold 0.85  # band floor (below duplicate_threshold)
```

### Composite scoring (recency + importance)

RRF orders purely by relevance. With **`memory.scoring_enabled`** on, a post-fusion step blends that relevance with **recency** (exponential decay on `createdAt`) and **importance** (the stored `confidence`) so the freshest / most-trusted memory wins a near-tie — the [Generative Agents](https://arxiv.org/abs/2304.03442) retrieval idea. Relevance stays **dominant**: recency/importance are tie-breakers, never the sole signal (a recency-only ranker just returns the newest memory regardless of the query).

```bash
simba config set memory.scoring_enabled true        # opt in to the measured blend
simba config set memory.score_weight_recency 0.5    # tie-breaker weights (defaults shown)
simba config set memory.score_weight_importance 0.3
simba config set memory.recency_halflife_days 90
```

Measured with the eval harness (live embedder): on the time-sensitive `simba-temporal` set — each query's answer is the *fresh* version of a near-duplicate fact — scoring lifts **recall@1 from 0.63 → 1.00** (MRR 0.81 → 1.00); on the general `simba-seed` set (uniform dates) it is an exact **no-op**. On by default (experimental).

### LLM reranker, extraction & providers

Simba can call an LLM for two memory tasks via a small **CLI-backed client** (`simba.llm`) — no SDK dependency, fully `simba config`-driven, and **fail-open** (any error degrades to the non-LLM path):

- **Reranker** (`memory.llm_rerank_enabled`) — the cross-encoder's role: after RRF + composite scoring, the LLM re-orders the candidate pool by relevance before truncation. Measured on `simba-seed` with `claude`/haiku it lifts **recall@1 0.71 → 0.90** and **MRR 0.90 → 1.00**, fixing exactly the confusable cases dense recall missed. **Non-blocking in the daemon:** reranking can't be precomputed like sync (it's query-specific), so recall serves the fast RRF+composite order *immediately* and reranks **off the hot path**, caching the result by (query, candidate-set) — a recurring query/candidate-set is then served the reranked order with no LLM call. So novel queries pay no latency (and get the fast order); recurring ones get the rerank for free. Set **`memory.llm_rerank_mode=sync`** to instead block on the rerank every recall (useful for testing/measuring a model live — at the cost of latency on every hook). The eval harness / explicit CLI recall always use the **synchronous** path (no cache) so they measure the rerank ceiling.
- **Extraction** (`sync.extract_strategy`, default `llm+regex`) — the **primary KG feed**: for every new memory the LLM extracts typed `(subject, predicate, object)` triples (reusing the project's existing entity vocabulary), unioned with the regex heuristics and canonicalized via entity resolution on write. So the bitemporal / entity-res / multi-hop KG is richly fed instead of "schema ahead of data." Runs in the background sync pipeline (not the hot path), bounded by `sync.llm_extract_max_per_cycle` so the first backlog sweep stays cheap; degrades to regex-only when no provider. Set `extract_strategy` to `regex` or `llm` to change the mix.

Providers (`llm.provider`): `claude-cli`, `llm-cli` (cloud); **100% local** `llama-cli` (llama.cpp) / `mlx-lm` (Apple MLX), set `llm.model_path`. Those CLIs **reload the model every call**, so for repeated calls (the reranker, LLM-judged eval) prefer a **persistent OpenAI-compatible server** that loads once: `mlx-server` (Apple Silicon, auto-spawned) / `llama-server` (llama.cpp, cross-platform/CUDA, auto-spawned) / `openai-http` (any running Ollama / llama.cpp / vLLM — local or a **remote GPU box**). All three set `llm.base_url`; drive any other server via `llm.serve_cmd`. See [`docs/eval-remote-gpu.md`](docs/eval-remote-gpu.md). A DeepSeek-style backend works via `llm.base_url` (claude-cli) or the `llm` CLI.

```bash
# claude-cli | llm-cli | llama-cli | mlx-lm | mlx-server | llama-server | openai-http | none
simba config set llm.provider llm-cli            # pick a backend
simba config set llm.model deepseek-chat         # model name the chosen CLI/server expects
simba config set llm.thinking xhigh              # reasoning-effort hint (best-effort)
simba config set llm.model_path ~/models/q.gguf  # for llama-cli / mlx-lm (100% local)
simba config set llm.provider openai-http        # persistent server: also set llm.base_url (see docs/eval-remote-gpu.md)
simba config set llm.provider none               # disable all LLM features
```

> **Experimental defaults:** simba now ships with its experimental features **on by default** — composite scoring, multi-arm HyDE, supersession, entity resolution, RLM pointer injection + autonomous engine, LLM extraction, and the LLM reranker. The LLM features need a working `llm.provider`/`rlm.engine` (default `claude-cli`); point them at a fast/cheap or local model, or set the relevant flag (or `llm.provider none`) to dial back cost/latency.

## Eval — Recall Benchmark

Recall quality used to be unmeasured — every ranking change was a guess. The eval harness fixes that: it scores the **real** recall stack (`plan_recall` → `hybrid_search` → RRF, the exact path `/recall` uses) against a curated dataset and reports standard IR metrics.

```bash
simba eval run                          # score the bundled simba-seed dataset
simba eval run --dataset temporal       # a bundled dataset by name
simba eval run --dataset path/to.json   # a custom dataset by path
simba eval run --split test             # score only the held-out test split
simba eval run --ks 1,3,5 --json        # custom cutoffs / machine-readable

# Build a benchmark from your REAL corpus (LLM-generated queries, real distractors):
simba eval build --n 50 --out real.json   # samples memories, makes a question per one
simba eval run --dataset real.json --split test
```

- **Metrics**: `recall@k`, `precision@k`, `hit@k`, `ndcg@k`, and `mrr` (means over cases); the text report also lists the worst cases by MRR so failures are visible, not averaged away.
- **In-process & non-invasive**: the harness builds a throwaway LanceDB table + FTS mirror from the dataset corpus — it never touches your real memory store — and embeds with the same local GGUF model as production.
- **Dataset format** — one JSON file: `{"corpus": [{id, content, type}...], "cases": [{id, query, relevant_ids}...]}`. Loading validates that every `relevant_id` resolves and corpus ids are unique, so a typo fails loudly instead of silently scoring zero.
- **Bundled datasets**: `simba-seed` is deliberately hard (tight clusters that share an entity but differ in the fact, adversarial keyword-overlap distractors, broad multi-relevant aggregation cases) — a perfect score would mean the dataset is too easy, not that recall is solved. `simba-temporal` pairs each fact with the stale version it superseded, to measure recency/importance scoring.
- **Real-corpus builder** (`simba eval build`): samples your actual memories and asks the LLM to write a natural question each one answers — the source memory is the gold, the rest of the sample are real distractors. This measures recall on real content with non-author-biased queries, instead of small synthetic sets that saturate.
- **Held-out split**: every case is deterministically assigned to `dev` or `test` (stable hash of its id, unless it pins `"split"`). Tune on `--split dev`, report on `--split test`, so tuning can't silently overfit the number you quote.

Config: `eval.ks` (default `1,3,5,10`) and `eval.dataset` (empty ⇒ bundled seed), both via `simba config`.

### External benchmarks (LoCoMo · LongMemEval · HotpotQA · HaluMem)

Beyond the bundled sets, simba scores the same recall stack against published datasets — plus an LLM-judged QA layer and a hallucination eval:

```bash
simba eval bench locomo --qa --per 30          # recall@k + LLM-judged QA on LoCoMo
simba eval bench longmemeval --qa --abstention # + abstention accuracy
simba eval bench hotpotqa                       # fullwiki multi-hop recall
simba eval halumem --user-num 5                 # HaluMem: operation-level memory-hallucination
simba eval leaderboard                          # render BENCHMARKS.md from results.jsonl
```

- **HaluMem** measures *not surfacing wrong/stale memories* (correct / hallucination / omission + boundary abstention) — the inverse of recall@k, the eval where forgetting / supersession can finally show value. It feeds a recency-annotated context (mirroring what the daemon injects); the decisive lever for the temporal categories is recency-aware retrieval, not forgetting.
- **Eval LLM serving is config-driven** and can run on a remote GPU box — `mlx-server` / `llama-server` / `openai-http`; see [`docs/eval-remote-gpu.md`](docs/eval-remote-gpu.md). The answerer and judge are separate models (no self-grading) and recorded in each result.
- **Multi-hop instruments (default-OFF, measured)**: entity-bridge (`memory.entity_bridge_enabled`, the one mechanism with a positive external signal) and Track B retrieval-time GraphRAG (`memory.kg_ppr_enabled`, a measured negative kept as an instrument). Both fold a third graph arm into recall before rescore; off until a proven in-repo delta.
- Every run appends to `.simba/eval/results.jsonl` (git SHA + config snapshot, incl. answerer/judge model) and feeds `simba eval leaderboard` → the committed `BENCHMARKS.md`.

## Neuron — Neuro-Symbolic Logic Server

Neuron is an MCP (Model Context Protocol) server that gives Claude Code access to formal verification tools (Z3 theorem prover, Souffle Datalog) and a truth database.

### Setup

```bash
# Register the MCP server with Claude Code
simba neuron install
```

### MCP Tools

Neuron exposes verification + knowledge-graph tools via the Model Context Protocol (plus 6 RLM recall tools — see [RLM](#rlm--lossless-transcript-recall) below):

| Tool | Purpose |
|------|---------|
| `verify_z3` | Execute a Z3 proof script in an isolated process |
| `analyze_datalog` | Run a Souffle Datalog analysis program |
| `truth_add` | Record a proven fact into the Truth DB (SQLite) |
| `truth_query` | Query the Truth DB for existing proven facts |
| `kg_add` | Insert an open temporal edge (subject/predicate/object + optional `occurred_at`) |
| `kg_query` | FTS/bm25 + bitemporal query (`as_of`, `occurred_after`/`occurred_before`, `expand_hops`) |
| `kg_neighbors` | Multi-hop BFS traversal from an entity (`depth`, `direction`) |
| `kg_invalidate` | Close matching open edges (stamp `valid_to`) |

### Temporal knowledge graph (bitemporal)

The KG (`.simba/simba.db`, `kg_edges` + an FTS5/bm25 mirror) stores facts as
subject–predicate–object triples on **two independent time axes**:

- **Belief time** — `valid_from` / `valid_to`: when the fact was on record.
  `kg_query(as_of=…)` snapshots it; an edge with `valid_to = NULL` is currently
  valid, and `kg_invalidate` closes it.
- **Event time** — `occurred_at`: when the fact was true in the world. Populated
  from narrative dates during fact extraction (e.g. "shipped March 5, 2024" or
  "yesterday" resolved against the memory's timestamp) and bounded with
  `kg_query(occurred_after=…, occurred_before=…)`.

`kg_edges` supersedes the legacy `proven_facts` table (migrated automatically on
first connect). `simba db facts` lists currently-valid edges, printing the
`occurred:` event date when known.

#### Entity resolution

Without resolution the graph fragments: `GITHUB_TOKEN`, `github_token`, and `the GITHUB_TOKEN` become three separate nodes. With **`kg.entity_resolution_enabled`** on, `kg_add` canonicalizes each subject/object against the entities **already in the same project** (case, articles, quotes, possessives, and trailing punctuation are normalized — code identifiers keep their underscores, so `github_token` and `github token` stay distinct). Resolution is **project-scoped**, so nodes never merge across repos. A surface-form variant collapses to the canonical name first seen.

```bash
simba config set kg.entity_resolution_enabled true   # collapse surface variants (opt-in)
```

The normalization layer (`kg/entities.py`) also exposes `resolve(name, existing, embed=…)` for **synonym** merges that don't share a normalized key (`Bob` ⇄ `Robert`) via embedding similarity — used by callers that have an embedder. LLM-based fact extraction (the opt-in `researcher` agent path) is fed the project's existing entity vocabulary so it **reuses** canonical names at the source, complementing the on-write merge.

#### Multi-hop traversal

`kg_query` matches edges directly; `kg_neighbors(entity, depth, direction)` walks the graph **outward** from an entity by breadth-first traversal, returning every edge reachable within `depth` hops (each tagged with its `hop` distance). `direction` is `out` (subject→object), `in`, or `both`. Bitemporal filters apply at **every** hop — a retracted edge cuts off the paths beyond it — and the crawl is project-scoped and bounded by `kg.max_neighbor_edges`. `kg_query(..., expand_hops=N)` composes the two: it returns the directly-matched edges **plus** the connected subgraph within `N` hops, turning a point lookup into multi-hop recall.

```bash
# Run the MCP server directly
simba neuron run --root-dir .
```

## RLM — Lossless Transcript Recall

Simba implements the **Recursive Language Model (RLM)** paradigm ([Zhang et al., arXiv:2512.24601](https://arxiv.org/abs/2512.24601)): rather than cramming a long context into the model, the full text is treated as *external data the agent navigates with code* — `grep`/`peek`/`window` over it, recursing into only the relevant slices.

Simba already exports every session's **full transcript** on `PreCompact` (to `~/.claude/transcripts/{id}/`). Normally those are mined once for ≤200-char memories and then sit idle. RLM turns them into a **queryable, lossless store**: a normal vector recall becomes a *pointer* into the full transcript (every memory carries its `sessionSource`), and the agent navigates the original text to reconstruct exactly what happened — no information loss from summarization.

**How it works (Claude drives the recursion):**

1. `rlm_recall("what did we decide about X")` → project-scoped vector recall returns **pointers** `{snippet, transcript_id, similarity, available}`.
2. `rlm_grep(transcript_id, "X")` → matches with line numbers + char offsets.
3. `rlm_peek` / `rlm_window` → read the exact region losslessly; repeat across regions/transcripts as needed.

By default Simba runs **no LLM** — it exposes navigation primitives as MCP tools on the [Neuron](#neuron--neuro-symbolic-logic-server) server and the agent already in the loop performs the recursion. An **opt-in autonomous engine** (`rlm.engine`) can also run it without any agent present — see [below](#autonomous-engine-opt-in).

### RLM MCP Tools

| Tool | Purpose |
|------|---------|
| `rlm_recall` | Find transcripts relevant to a query (project-scoped); returns navigable pointers |
| `rlm_grep` | Regex-search a transcript; returns matches with line numbers + char offsets |
| `rlm_peek` | Return an exact character range of a transcript |
| `rlm_window` | Return transcript text within ±radius chars of an offset (expand a grep hit) |
| `rlm_head` / `rlm_tail` | Return the first / last N lines of a transcript |

Recall is **project-scoped** (it reuses the leak-free LanceDB recall), so a transcript from another repo never surfaces. Configure via `simba config` (`rlm` section): `max_search_matches`, `regex_timeout_seconds`, `lru_documents`, `transcript_source`, `default_max_pointers`.

### Autonomous engine (opt-in)

By default RLM is **agent-driven** and passive injection is off. Two opt-in knobs:

- **Passive pointers** — `simba config set rlm.inject_pointers true` makes `UserPromptSubmit` surface navigable transcripts (an `<rlm-pointers>` block) every turn, so the agent knows it can go lossless.
- **Autonomous engine** — `simba config set rlm.engine claude-cli` lets Simba run the recursion itself in **agentless** contexts. After a session compacts (`PreCompact`), it spawns a **detached, cheap** `claude -p` (default `--model haiku`, never Opus) that navigates the transcript via the `rlm_*` tools and stores memories. Default `rlm.engine=claude` ⇒ off (zero extra cost).

```bash
simba config set rlm.engine claude-cli   # opt in (default model: haiku)
simba rlm digest --latest                # manual one-shot on the newest transcript
simba rlm complete <id> --stored N       # the spawned agent calls this to close its job
```

Guardrails: opt-in, cheap-by-default, **detached** (never blocks a hook), rate-limited (`engine_min_new_exchanges`), and deduped via the `rlm_jobs` table. Point it at a cheaper backend (DeepSeek/OpenRouter/local) with `engine_base_url` + `engine_api_key_env`.

> **Roadmap:** RLM navigation + lossless recall, the autonomous engine **Phase 1 (`claude-cli`)**, **episodic consolidation (L2)**, **hybrid BM25+vector recall (L3)**, and the **temporal entity-relationship knowledge graph (L4)** are implemented. Planned: engine phases **`api`** (OpenAI-compatible / DeepSeek) and **`local-gguf`** (offline).

### Episodic consolidation (L2)

Raw memories are per-fact (≤200 chars). **Episodic consolidation** rolls a whole session's memories into one coarser `EPISODE` memory — a tier between per-fact memories and the L4 knowledge graph. It reuses the **same RLM engine** (so it is opt-in and agentless): the configured engine spawns a detached agent that reads the session's memories, synthesizes one episode, and stores it with `--type EPISODE`. Because an episode is a normal memory, it is embedded, project-scoped, and surfaced by ordinary (and hybrid) recall.

```bash
simba config set rlm.engine claude-cli              # episodes reuse the RLM engine (opt-in)
simba memory consolidate                            # this project's eligible sessions
simba memory consolidate --session <id>             # one session
simba memory consolidate --all                      # every project's eligible sessions
simba config set episodes.min_memories 5            # min memories before a session is worth an episode
simba config set episodes.auto_on_precompact true   # auto at session end (default)
simba config set episodes.scheduler_enabled true    # also via the sync scheduler (default)
```

A session is *eligible* once it has ≥ `episodes.min_memories` memories and no episode yet — that rule naturally defers a just-ended session until its memories land, so consolidation never races the digest. Dispatch is engine-gated (no engine ⇒ no-op), **detached** (never blocks a hook), and deduped via the `episode_jobs` table; the agent calls `simba episodes complete <id>` to close its job.

## Orchestration — Agent Dispatch Server

Orchestration is an MCP server for async agent dispatch, status tracking, and process management.

### Setup

```bash
# Register the MCP server with Claude Code and bootstrap agent definitions
simba orchestration install

# With hot-reload proxy (useful during development)
simba orchestration install --proxy
```

This registers Orchestration as an MCP server in `.claude/settings.json` and creates agent definition files in `.claude/agents/`.

### MCP Tools

Orchestration exposes 4 tools via the Model Context Protocol:

| Tool | Purpose |
|------|---------|
| `dispatch_agent` | Launch an async subagent (non-blocking) |
| `agent_status_update` | Report subagent progress |
| `agent_status_check` | Check subagent completion status |
| `reload_server` | Hot-reload the MCP backend (proxy mode) |

### Agent Management

```bash
# Update agent status (called by subagents, not users)
simba orchestration status <ticket_id> completed
simba orchestration status <ticket_id> failed -m "error message"
```

### Managed Sections

Orchestration manages auto-updated sections in CLAUDE.md and agent definition files using `<!-- BEGIN SIMBA:name -->` markers. These sections contain tool instructions, workflows, and agent-specific guidance.

```bash
# Update managed sections in CLAUDE.md and agent files
simba orchestration sync

# Inject markers into agent definition files
simba orchestration agents --inject

# Update content in existing markers
simba orchestration agents --update
```

## Context-Low Early Warning

The PreToolUse hook monitors the session transcript file size. When it exceeds a configurable threshold (`hooks.context_low_bytes`, default 20 MB to suit 1M-context sessions), it injects a one-time `<context-low-warning>` into Claude's context recommending that it summarize its current work state before auto-compact triggers.

This gives Claude time to document progress, pending tasks, and branch context so the pre-compact transcript export captures a clean snapshot for learning extraction.

## Sync Pipeline

Automatic indexing and learning extraction from project files and session transcripts.

```bash
simba sync run                              # Full cycle: index + extract
simba sync schedule --interval 300          # Run every 5 minutes
simba server --sync-interval 300            # Start daemon with periodic sync
```

The sync pipeline:
1. **Index** — scans project files and builds the search database
2. **Extract** — processes exported transcripts and stores learnings in semantic memory

Sync can also be triggered via `POST /sync` on the daemon, which the SessionStart hook does automatically.

## Tool-call redirect

Steer bare commands to better tooling — e.g. `cargo …` → `soldr cargo …`, `python …` → `uv run python …`. A PreToolUse check parses each Bash command (handling `env VAR=…` prefixes, `&&`/`;`/`|` segments, nested `bash -c "…"`, and `uv run <tool>`) and matches the invoked program against your redirect rules.

Two modes (`hooks.redirect_mode`):
- **`deny`** (default) — blocks the call with a `permissionDecision: deny` whose reason names the corrected command; the model re-issues it. Works in every case.
- **`rewrite`** — silently substitutes the command via PreToolUse `updatedInput` (no model retry) for simple leading-program commands (`cargo build` → `soldr cargo build`); anything fancier (env-prefixed, multi-segment, `uv run`, nested shell) safely falls back to `deny` so a broken command is never synthesized. **Opt-in and not yet verified live — sanity-check that silent rewrites actually take effect in your setup (esp. under `--dangerously-skip-permissions`) before relying on it; `deny` is the reliable default.**

Rules come from **both** a version-controlled `.simba/redirects.toml` and a CLI-managed store, merged and project-scoped:

```bash
simba rule redirect add cargo "soldr cargo" --reason "use the pinned toolchain"
simba rule redirect add python "uv run python"
simba rule redirect list
simba rule redirect rm cargo
simba config set hooks.redirect_mode rewrite   # opt into silent rewrite
```

```toml
# .simba/redirects.toml
[[redirect]]
program = "cargo"
replacement = "soldr cargo"
reason = "use the pinned rustup toolchain"
```

> **Activate it:** redirect rules are **empty by default**, so the feature is a no-op until you add rules per-repo (the CLI or `.simba/redirects.toml` above). It's the deterministic sibling of the semantic [tool-rule deny](#guardian--claudemd-rule-enforcement) (which blocks by similarity to learned errors).

## Guardian — CLAUDE.md Rule Enforcement

Extracts content between `<!-- BEGIN SIMBA:core -->` tags from CLAUDE.md and injects it as context on every prompt. On session stop, checks whether Claude's response contains the `[✓ rules]` compliance signal.

Mark critical rules in your CLAUDE.md:

```markdown
## Critical Constraints
<!-- BEGIN SIMBA:core -->
- Never delete files without confirmation
- Always run tests before committing
<!-- END SIMBA:core -->
```

These rules are reinforced on every prompt regardless of context window state.

## Markers CLI

Discover, audit, update, and migrate SIMBA markers across `.md` files.

```bash
simba markers list                    # Scan for all markers, show file/section/line
simba markers audit                   # Find unused, orphaned, or stale sections
simba markers update                  # Bulk-update all markers with current templates
simba markers show completion_protocol # Print a section's template content
simba markers migrate --path /project # Convert NEURON:*, CORE, etc. to SIMBA format
simba markers migrate --dry-run       # Preview without modifying files
```

The `migrate` command converts non-SIMBA markers (`<!-- BEGIN NEURON:name -->`, `<!-- CORE -->`, bare `<!-- BEGIN X -->`) to proper `<!-- BEGIN SIMBA:name -->` format, preserving body content.

## Skills

Skills are slash-command invocable capabilities defined in `skills/`. They run as forked agents with restricted tool access.

### `/simba-onboard` — Project Onboarding

Interactive skill that analyzes your project's markdown files and generates consolidated core instructions with SIMBA markers:

1. Reads CLAUDE.md, AGENTS.md, and all `.claude/**/*.md` files
2. Extracts key instructions into categories (constraints, build commands, environment, code style, workflow, agent rules)
3. Generates `.claude/rules/CORE_INSTRUCTIONS.md` with SIMBA marker sections
4. Presents content for user verification before writing
5. Wires references into CLAUDE.md and AGENTS.md

```bash
# In Claude Code, after simba install:
/simba-onboard
```

The skill is automatically installed by `simba install`. It creates a structure similar to:

```markdown
# .claude/rules/CORE_INSTRUCTIONS.md
<!-- BEGIN SIMBA:constraints -->
## Critical Constraints
- Never delete files without confirmation
<!-- END SIMBA:constraints -->

<!-- BEGIN SIMBA:build_commands -->
## Build & Test
make && make test
<!-- END SIMBA:build_commands -->
```

### `/memories-learn` — Extract Learnings from Transcripts

Automatically triggered after context compaction:
1. **PreCompact hook** exports the session transcript to `~/.claude/transcripts/{sessionId}/` with `status: "pending_extraction"`
2. **SessionStart hook** on the next session detects the pending export and injects extraction instructions
3. A **memory-extractor agent** reads the transcript, extracts 5-15 learnings, and POSTs each to the memory daemon (`/store`)

Memory types: `GOTCHA`, `WORKING_SOLUTION`, `PATTERN`, `DECISION`, `FAILURE`, `PREFERENCE`

Extraction follows quality rules baked into the prompt (shared by the hook and
the skill): keep content under 200 chars, preserve proper nouns / file paths /
identifiers verbatim, preserve numeric precision (never weaken an exact value),
and resolve relative dates to absolute ones.

Can also be invoked manually with `/memories-learn`.

### `/memories-sanitize` — Review and Clean Up Memories

Manual skill for auditing the memory database:
1. Lists all memories via `GET /list`
2. Identifies invalid, outdated, superseded, or misleading entries
3. Deletes bad memories via `DELETE /memory/{id}`
4. Optionally stores corrected replacements

Use when memory quality degrades or after bulk extraction sessions.

### `/memories-recall-verify` — Self-Correcting Recall

Use before answering a memory-dependent question when the recalled memories look
**ambiguous, conflicting, scope-mismatched, or insufficient**. Instead of
answering from the first plausible hit, the skill drives a correction loop:

1. Recall via `simba memory recall "<question>"`.
2. Detect the failure mode (multiple instances of a generic referent, conflicting
   values, wrong-entity match, or nothing relevant).
3. **Re-query** with a narrower entity/attribute; broaden only if empty.
4. Resolve conflicts by recency (`recency="newest"` / KG `valid_to`).
5. Answer with a clear winner, **ask** to disambiguate when still unclear, or say
   "not in memory" — never fabricate.

### Memory Pipeline Flow

```
Session ends → PreCompact exports transcript
    → Next session: SessionStart detects pending extraction
    → memory-extractor agent reads transcript.md
    → POSTs learnings to /store (semantic memories in LanceDB)
    → simba sync extract converts memories to proven facts (SQLite)
    → UserPromptSubmit + PreToolUse hooks recall memories on future prompts
```

## Project Search

Per-project session memory and semantic search. Combines SQLite FTS5 with QMD semantic search.

```bash
simba search init                    # Initialize project memory
simba search add-session "summary" '["files"]' '["tools"]' "tags"
simba search add-knowledge "area" "description" "details"
simba search add-fact "fact text" "category"
simba search search "query"          # Search project memory
simba search context "query"         # Get combined RAG context
simba search recent 5                # View recent sessions
simba search stats                   # Show statistics
```

Optional external tools for enhanced search:
- [ripgrep](https://github.com/BurntSushi/ripgrep) — fast file discovery
- [fzf](https://github.com/junegunn/fzf) — fuzzy filtering for file suggestions
- [qmd](https://github.com/tobi/qmd) — semantic markdown search

## CLI Reference

```
simba install                 Register hooks + skills (project-local)
simba install --global        Register hooks + skills (~/.claude/)
simba install --remove        Remove hooks and skills
simba codex-install           Install bundled Codex skills (~/.codex/skills)
simba codex-install --remove  Remove bundled Codex skills
simba codex-status            Check daemon health + pending extraction
simba codex-extract           Print extraction prompt for latest transcript
simba codex-recall <query>    Query semantic memory via /recall
simba codex-finalize          Run end-of-task signal/error checks
simba codex-automation        Print a suggested Codex automation directive
simba server [opts]           Start memory daemon
simba neuron run              Run neuron MCP server (truth/verify)
simba orchestration install   Register orchestration MCP server
simba orchestration run       Run orchestration MCP server
simba orchestration proxy     Run via hot-reload proxy
simba orchestration sync      Update managed sections
simba orchestration agents    Manage agent definition files
simba orchestration status    Update agent task status
simba config list              List all configurable sections
simba config get <key>         Print effective value (e.g. memory.port)
simba config set <key> <val>   Set a config value (local)
simba config set --global ...  Set a config value (global)
simba config reset <key>       Remove a local override
simba config show              Dump full effective config
simba config edit [--global]   Open config.toml in $EDITOR
simba markers list             List all SIMBA markers in .md files
simba markers audit            Compare markers vs MANAGED_SECTIONS
simba markers update           Update markers with current templates
simba markers show <section>   Print raw template for a section
simba markers migrate          Convert non-SIMBA markers to SIMBA format
simba search <cmd>            Project memory operations
simba sync run           Run a full sync cycle (index + extract)
simba sync index         Index project files only
simba sync extract       Extract learnings from transcripts only
simba sync status        Show sync pipeline status
simba sync schedule      Run sync on a periodic interval
simba stats              Token economics and project statistics
simba hook <event>       Run a hook (called by Claude Code, not users)
```

## Development

```bash
git clone git@github.com:mahmoudimus/simba.git
cd simba
uv sync

# Run tests
uv run pytest                     # all tests (~625)
uv run pytest -x                  # stop on first failure
uv run pytest -k "test_name"      # specific test

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run ruff check --fix src/

# Type check
uv run pyrefly check
```

### Code Style

- Module-level imports only: `import X` then `X.Y`, not `from X import Y`
- Exemptions: `from __future__ import annotations`, `from typing import TYPE_CHECKING`
- `pathlib.Path` over `os.path`

### Pre-commit Hooks

```bash
uv run pre-commit install
```

## Data Storage

All project-level data lives under a single `.simba/` directory in the project root:

```
.simba/
  simba.db             SQLite — knowledge graph (kg_edges + FTS5), activities, reflections, agent runs, sessions/knowledge/facts, sync watermarks
  memory/              LanceDB vector database (semantic memories) + memory_fts.db keyword mirror
  search/              Search activity log
  tailor/              Error reflection journal (JSONL)
  config.toml          Local configuration overrides (optional)
```

This directory is gitignored. The embedding model cache lives in `~/.cache/huggingface/hub/`.

All SQLite access goes through a **vendored [peewee](https://github.com/coleifer/peewee) ORM** (`src/simba/_vendor/`, kept in-tree to preserve the zero-dependency install). Tables are peewee models; the FTS5 virtual tables + sync triggers and a few introspection/migration paths stay as raw SQL by necessity. LanceDB remains the source of truth for vectors; the `memory_fts.db` keyword mirror is rebuildable.

## Unified Configuration

All simba settings are configurable via TOML files with git-style scoping:

```
~/.config/simba/config.toml     # --global (user-wide defaults)
.simba/config.toml              # local (project-specific overrides)
```

Precedence: **local > global > code defaults**. CLI arguments still override config file values.

```bash
simba config list                              # Show all sections + fields
simba config get memory.port                   # Print effective value
simba config set memory.min_similarity 0.40    # Write to local config
simba config set --global memory.port 9000     # Write to global config
simba config reset memory.min_similarity       # Revert to default
simba config show                              # Dump full effective config
```

### Configurable Sections

| Section | Config Class | Key Fields |
|---------|-------------|------------|
| `memory` | `MemoryConfig` | port, min_similarity, max_results, sync_interval, ... |
| `hooks` | `HooksConfig` | health_timeout, daemon_port, context_low_bytes, permission_deny_similarity, ... |
| `sync` | `SyncConfig` | daemon_url, batch_limit, retry_count, default_interval, ... |
| `search` | `SearchConfig` | max_context_tokens, min_query_length, memory_token_budget, ... |

New config sections can be added by decorating a dataclass with `@simba.config.configurable("section_name")`.

## Configuration Reference

### Memory Daemon

| Field | Default | Description |
|-------|---------|-------------|
| `port` | 8741 | Daemon listen port |
| `db_path` | `""` (cwd/.simba/memory) | Database directory |
| `embedding_dims` | 768 | Embedding vector dimensions |
| `model_repo` | nomic-ai/nomic-embed-text-v1.5-GGUF | HuggingFace repo |
| `model_file` | nomic-embed-text-v1.5.Q4_K_M.gguf | GGUF file name |
| `model_path` | `""` (auto-download) | Local path to GGUF file |
| `n_gpu_layers` | -1 | GPU layers (-1=all, 0=CPU only) |
| `embed_url` | `""` (in-process) | External embedding server URL |
| `min_similarity` | 0.35 | Minimum cosine similarity for recall (precise queries) |
| `max_results` | 3 | Maximum memories returned per query (precise queries) |
| `duplicate_threshold` | 0.92 | Similarity threshold for dedup |
| `supersede_enabled` | false | Replace a near-duplicate same-type memory on store |
| `supersede_threshold` | 0.85 | Supersede band floor (below `duplicate_threshold`) |
| `max_content_length` | 200 | Maximum memory content length (chars) |
| `sync_interval` | 0 | Sync interval in seconds (0=disabled) |
| `diagnostics_after` | 50 | Emit diagnostics report every N requests |
| `shutdown_timeout` | 10 | Graceful shutdown timeout in seconds |
| `hybrid_enabled` | true | Fuse BM25 keyword arm with the vector arm (RRF) |
| `rrf_k` | 60 | Reciprocal Rank Fusion rank constant |
| `fts_candidate_pool` | 20 | Candidates pulled per arm before fusion |
| `fts_tokenize` | trigram | FTS5 tokenizer (trigram \| porter \| unicode61) |
| `vector_weight` | 1.0 | RRF weight for the vector arm |
| `keyword_weight` | 1.0 | RRF weight for the keyword arm |
| `intent_aware` | true | Adapt recall breadth to query intent (broad vs precise) |
| `min_similarity_broad` | 0.28 | Cosine floor for broad/aggregation queries |
| `max_results_broad` | 8 | Maximum memories returned for broad queries |
| `fts_candidate_pool_broad` | 40 | Candidate pool for broad queries |
| `fts_max_terms` | 12 | Cap on high-signal terms fed to the keyword arm |
| `expansion_enabled` | false | 2nd HyDE vector arm over the focused-term string |

### Neuron

| Field | Default | Description |
|-------|---------|-------------|
| `db_path` | `.simba/simba.db` | Truth DB + temporal knowledge graph (`kg_edges`) |

### Orchestration

| Field | Default | Description |
|-------|---------|-------------|
| `agents_dir` | `.claude/agents` | Agent definition files (Claude Code convention) |
| `db_path` | `.simba/simba.db` | Agent runs and logs (agent_runs, agent_logs tables) |

## License

[MIT](LICENSE)
