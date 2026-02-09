# Simba

> *"Remember who you are."* — Mufasa

A unified Claude Code plugin that combines semantic memory, CLAUDE.md rule enforcement, neuro-symbolic logic (Z3 + Datalog), and project-aware search into a single Python package.

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
uv tool install git+https://github.com/mahmoudimus/simba.git
```

### Register hooks

```bash
simba install           # Register all 6 hooks in ~/.claude/settings.json
simba install --remove  # Remove simba hooks from settings
```

## What It Does

Simba hooks into six Claude Code lifecycle events to provide persistent context across sessions:

| Hook | Purpose |
|------|---------|
| **SessionStart** | Start memory daemon, inject tailor context, show project memory stats |
| **UserPromptSubmit** | Recall semantic memories, reinforce CLAUDE.md core rules, inject search context |
| **PreToolUse** | Query semantic memory based on Claude's thinking; warn when context is low |
| **PostToolUse** | Track file reads, edits, searches, and commands in an activity log |
| **PreCompact** | Export transcript to disk before context compaction |
| **Stop** | Check for rule compliance signal, capture errors from final transcript |

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

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/store` | Store a typed memory with embedding |
| POST | `/recall` | Semantic search over memories |
| POST | `/sync` | Trigger a one-off sync cycle (index + extract) |
| GET | `/health` | Health check with model info |
| GET | `/stats` | Memory count and database stats |
| GET | `/list` | List all memories |
| DELETE | `/memory/:id` | Delete a specific memory |

## Neuron — Neuro-Symbolic Logic Server

Neuron is an MCP (Model Context Protocol) server that gives Claude Code access to formal verification tools (Z3 theorem prover, Souffle Datalog) and a truth database.

### Setup

```bash
# Register the MCP server with Claude Code
simba neuron install
```

### MCP Tools

Neuron exposes 4 tools via the Model Context Protocol:

| Tool | Purpose |
|------|---------|
| `truth_add` | Record a proven fact into the Truth DB (SQLite) |
| `truth_query` | Query the Truth DB for existing proven facts |
| `verify_z3` | Execute a Z3 proof script in an isolated process |
| `analyze_datalog` | Run a Souffle Datalog analysis program |

### Truth Database

A local SQLite database (`.simba/simba.db`, `proven_facts` table) that stores proven facts as subject-predicate-object triples with their proof text. Claude queries this before making assumptions about the codebase.

```bash
# Run the MCP server directly
simba neuron run --root-dir .
```

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

The PreToolUse hook monitors the session transcript file size. When it exceeds a configurable threshold (default 4 MB), it injects a one-time `<context-low-warning>` into Claude's context recommending that it summarize its current work state before auto-compact triggers.

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

### `/memories-learn` — Extract Learnings from Transcripts

Automatically triggered after context compaction:
1. **PreCompact hook** exports the session transcript to `~/.claude/transcripts/{sessionId}/` with `status: "pending_extraction"`
2. **SessionStart hook** on the next session detects the pending export and injects extraction instructions
3. A **memory-extractor agent** reads the transcript, extracts 5-15 learnings, and POSTs each to the memory daemon (`/store`)

Memory types: `GOTCHA`, `WORKING_SOLUTION`, `PATTERN`, `DECISION`, `FAILURE`, `PREFERENCE`

Can also be invoked manually with `/memories-learn`.

### `/memories-sanitize` — Review and Clean Up Memories

Manual skill for auditing the memory database:
1. Lists all memories via `GET /list`
2. Identifies invalid, outdated, superseded, or misleading entries
3. Deletes bad memories via `DELETE /memory/{id}`
4. Optionally stores corrected replacements

Use when memory quality degrades or after bulk extraction sessions.

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
simba install                 Register hooks in ~/.claude/settings.json
simba install --remove        Remove hooks
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
uv run pytest                     # all tests (~516)
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
  simba.db             SQLite — proven facts, activities, reflections, agent runs, sync watermarks
  memory/              LanceDB vector database (semantic memories)
  search/              Search activity log
  tailor/              Error reflection journal (JSONL)
  config.toml          Local configuration overrides (optional)
```

This directory is gitignored. The embedding model cache lives in `~/.cache/huggingface/hub/`.

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
| `hooks` | `HooksConfig` | health_timeout, daemon_port, context_low_bytes, ... |
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
| `min_similarity` | 0.35 | Minimum cosine similarity for recall |
| `max_results` | 3 | Maximum memories returned per query |
| `duplicate_threshold` | 0.92 | Similarity threshold for dedup |
| `max_content_length` | 1000 | Maximum memory content length (chars) |
| `sync_interval` | 0 | Sync interval in seconds (0=disabled) |
| `diagnostics_after` | 50 | Emit diagnostics report every N requests |

### Neuron

| Field | Default | Description |
|-------|---------|-------------|
| `db_path` | `.simba/simba.db` | Truth database (proven_facts table) |

### Orchestration

| Field | Default | Description |
|-------|---------|-------------|
| `agents_dir` | `.claude/agents` | Agent definition files (Claude Code convention) |
| `db_path` | `.simba/simba.db` | Agent runs and logs (agent_runs, agent_logs tables) |

## License

[MIT](LICENSE)
