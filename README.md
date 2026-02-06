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
| **PreToolUse** | Query semantic memory based on Claude's thinking before tool execution |
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
| GET | `/health` | Health check with model info |
| GET | `/stats` | Memory count and database stats |
| GET | `/list` | List all memories |
| DELETE | `/memory/:id` | Delete a specific memory |

## Neuron — Neuro-Symbolic Logic Server

Neuron is an MCP (Model Context Protocol) server that gives Claude Code access to formal verification tools and async agent orchestration.

### Setup

```bash
# Register the MCP server with Claude Code and bootstrap agent definitions
simba neuron install

# With hot-reload proxy (useful during development)
simba neuron install --proxy
```

This registers Neuron as an MCP server in `.claude/settings.json` and creates agent definition files in `.claude/agents/`.

### MCP Tools

Neuron exposes 7 tools via the Model Context Protocol:

| Tool | Purpose |
|------|---------|
| `truth_add` | Record a proven fact into the Truth DB (SQLite) |
| `truth_query` | Query the Truth DB for existing proven facts |
| `verify_z3` | Execute a Z3 proof script in an isolated process |
| `analyze_datalog` | Run a Souffle Datalog analysis program |
| `dispatch_agent` | Launch an async subagent (non-blocking) |
| `agent_status_update` | Report subagent progress |
| `agent_status_check` | Check subagent completion status |

### Truth Database

A local SQLite database (`.claude/truth.db`) that stores proven facts as subject-predicate-object triples with their proof text. Claude queries this before making assumptions about the codebase.

```bash
# Run the MCP server directly
simba neuron run --root-dir .

# Or via hot-reload proxy (restarts backend on code changes)
simba neuron proxy --root-dir .
```

### Agent Orchestration

Neuron can dispatch Claude Code subagents as background processes:

```bash
# Update agent status (called by subagents, not users)
simba neuron status <ticket_id> completed
simba neuron status <ticket_id> failed -m "error message"
```

### Managed Sections

Neuron manages auto-updated sections in CLAUDE.md and agent definition files using `<!-- BEGIN SIMBA:name -->` markers. These sections contain tool instructions, workflows, and agent-specific guidance.

```bash
# Update managed sections in CLAUDE.md and agent files
simba neuron sync

# Inject markers into agent definition files
simba neuron agents --inject

# Update content in existing markers
simba neuron agents --update
```

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
simba install            Register hooks in ~/.claude/settings.json
simba install --remove   Remove hooks
simba server [opts]      Start memory daemon
simba neuron install     Register MCP server and bootstrap agents
simba neuron run         Run MCP server directly
simba neuron proxy       Run MCP server via hot-reload proxy
simba neuron sync        Update managed sections in CLAUDE.md and agents
simba neuron agents      Manage agent definition files
simba neuron status      Update agent task status
simba search <cmd>       Project memory operations
simba stats              Token economics and project statistics
simba hook <event>       Run a hook (called by Claude Code, not users)
```

## Development

```bash
git clone git@github.com:mahmoudimus/simba.git
cd simba
uv sync

# Run tests
uv run pytest                     # all tests (~386)
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

## Configuration

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

### Neuron

| Field | Default | Description |
|-------|---------|-------------|
| `db_path` | `.claude/truth.db` | Truth database location |
| `agents_dir` | `.claude/agents` | Agent definition files |

## License

[MIT](LICENSE)
