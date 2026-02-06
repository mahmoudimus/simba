# Simba

> *"Remember who you are."* — Mufasa

A unified Claude Code plugin that combines error memory, semantic recall, CLAUDE.md rule enforcement, and project-aware search into a single Python package. Replaces four separate JS/Bash plugins (claude-tailor, claude-memory, claude-md-memory-guardian, claude-turbo-search) with one coherent system.

## Quick Start

```bash
# Install globally (editable — picks up source changes immediately)
uv tool install -e /path/to/simba

# Register hooks in Claude Code
simba install

# Done. Start a Claude Code session and the memory daemon auto-starts.
```

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Install from source (editable)

This is the recommended approach during development. Changes to Python source files take effect immediately — no reinstall needed.

```bash
git clone git@github.com:mahmoudimus/simba.git
cd simba
uv tool install -e .
```

The `simba` binary is installed to `~/.local/bin/simba` (ensure this is on your PATH).

### Install from git (non-editable)

For a stable install that doesn't change when you edit files:

```bash
uv tool install git+https://github.com/mahmoudimus/simba.git
```

### Register hooks

After installing the `simba` binary:

```bash
simba install           # Register all 6 hooks in ~/.claude/settings.json
simba install --remove  # Remove simba hooks from settings
```

This writes hook entries to `~/.claude/settings.json` so Claude Code calls `simba hook <Event>` at each lifecycle point.

## What It Does

Simba hooks into six Claude Code lifecycle events to provide persistent context across sessions:

| Hook | Purpose |
|------|---------|
| **SessionStart** | Start memory daemon, inject tailor context, show project memory stats, detect pending extractions |
| **UserPromptSubmit** | Recall semantic memories, reinforce CLAUDE.md core rules, inject project search context |
| **PreToolUse** | Query semantic memory based on Claude's thinking before tool execution |
| **PostToolUse** | Track file reads, edits, searches, and commands in an activity log |
| **PreCompact** | Export transcript to disk before context compaction for later learning extraction |
| **Stop** | Check for rule compliance signal, capture errors from final transcript |

## Memory Daemon

The memory daemon is a FastAPI server backed by LanceDB for vector storage. It supports two embedding backends:

### In-process mode (default)

Loads a GGUF model directly via llama-cpp-python. No external services needed. The model (~81 MB) auto-downloads from Hugging Face on first startup.

```bash
# Default — auto-downloads nomic-embed-text and loads in-process
simba server

# Use a specific local GGUF model file
simba server --model-path /path/to/nomic-embed-text-v1.5.Q4_K_M.gguf

# CPU-only mode (no GPU offloading)
simba server --n-gpu-layers 0

# Custom port and database path
simba server --port 9000 --db-path /path/to/db
```

### External server mode (--embed-url)

Delegates embedding to an external OpenAI-compatible server. Useful when you're already running llama-cpp-server, vLLM, or any other compatible endpoint.

```bash
# Start llama-cpp-server separately (example)
python -m llama_cpp.server \
    --model /path/to/nomic-embed-text-v1.5.Q4_K_M.gguf \
    --host 0.0.0.0 --port 8080

# Point simba at it
simba server --embed-url http://localhost:8080
```

When `--embed-url` is set, simba sends POST requests to `{embed_url}/v1/embeddings` using the OpenAI embeddings API format. The model is never loaded in-process, so simba uses minimal memory.

This works with any server that implements `POST /v1/embeddings`:
- `llama-cpp-python` server (`python -m llama_cpp.server`)
- vLLM
- text-embeddings-inference
- Any OpenAI-compatible proxy

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/store` | Store a typed memory with embedding |
| POST | `/recall` | Semantic search over memories |
| GET | `/health` | Health check with model info |
| GET | `/stats` | Memory count and database stats |
| GET | `/list` | List all memories |
| DELETE | `/memory/:id` | Delete a specific memory |

The SessionStart hook auto-starts the daemon if it is not already running.

## CLI Reference

```bash
simba install            # Register hooks in ~/.claude/settings.json
simba install --remove   # Remove hooks
simba server             # Start memory daemon (in-process embeddings)
simba server --embed-url URL   # Start with external embedding server
simba server --model-path PATH # Use a local GGUF file
simba server --n-gpu-layers N  # GPU layers (-1=all, 0=CPU)
simba server --port PORT       # Custom port (default: 8741)
simba server --db-path PATH    # Custom database directory
simba search <subcommand>      # Project memory operations
simba hook <Event>             # Run a hook (called by Claude Code, not users)
```

## Project Search

Per-project session memory and semantic search integration. Combines a local SQLite database (sessions, knowledge areas, facts) with QMD semantic search to automatically inject relevant context before each prompt.

```bash
# Initialize project memory (creates .claude-memory/memory.db)
simba search init

# Add a session summary
simba search add-session "Refactored auth module" '["src/auth.py"]' '["Read","Edit"]' "auth,refactor"

# Add codebase knowledge
simba search add-knowledge "src/auth" "JWT-based auth with refresh tokens" "15min access, 7d refresh"

# Add a project fact
simba search add-fact "Uses PostgreSQL with SQLAlchemy" "architecture"

# Search project memory
simba search search "authentication"

# Get combined context for a query
simba search context "how does auth work"

# View recent sessions
simba search recent 5

# Show statistics
simba search stats
```

Optional external tools for enhanced search:
- [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) -- fast file discovery
- [fzf](https://github.com/junegunn/fzf) -- fuzzy filtering for file suggestions
- [qmd](https://github.com/tobi/qmd) -- semantic markdown search

## Subsystems

### Tailor (`simba.tailor`)

Reads Claude Code transcript JSONL after each session, detects errors via regex patterns, normalizes them for clustering (line numbers, hex addresses, paths replaced with tokens), and appends structured reflection entries to `.claude-tailor/memory/reflections.jsonl`.

### Guardian (`simba.guardian`)

Extracts content between `<!-- CORE -->` tags from CLAUDE.md and injects it as context on every prompt. On session stop, checks whether Claude's response contains the `[✓ rules]` compliance signal. If missing, re-injects the full CLAUDE.md with a re-audit instruction.

### Skills

Seven custom slash commands:

| Skill | Purpose |
|-------|---------|
| `/memories-learn` | Extract learnings from a session transcript and store them |
| `/memories-sanitize` | Review stored memories and remove invalid entries |
| `/turbo-index` | Index the project for optimized search |
| `/qmd` | Search indexed documentation with QMD |
| `/remember` | Save the current work session to project memory |
| `/memory-stats` | Show project memory database statistics |
| `/token-stats` | Token savings dashboard |

## Development

```bash
# Set up development environment
git clone git@github.com:mahmoudimus/simba.git
cd simba
uv sync

# Run tests
uv run pytest                              # all tests
uv run pytest -x                           # stop on first failure
uv run pytest -k "test_name"               # specific test
uv run pytest tests/memory/               # specific directory

# Lint and format
uv run ruff check src/ tests/              # lint
uv run ruff format src/ tests/             # format
uv run ruff check --fix src/               # auto-fix lint issues

# Type check
uv run pyrefly check

# Import style enforcement
uv run python tools/enforce_module_imports.py --check src/ tests/
```

### Code Style

- Module-level imports only: `import X` then `X.Y`, not `from X import Y`. Enforced by `tools/enforce_module_imports.py` in pre-commit.
- Exemptions: `from __future__ import annotations`, `from typing import TYPE_CHECKING`, and imports inside `if TYPE_CHECKING:` blocks.
- `pathlib.Path` over `os.path` (enforced via ruff TID ban).

### Pre-commit Hooks

```bash
uv run pre-commit install
```

Runs on every commit: ruff lint (with auto-fix), ruff format, pyrefly type check, and module import enforcement.

## Configuration

### Memory Daemon

`MemoryConfig` fields with defaults:

| Field | Default | Description |
|-------|---------|-------------|
| `port` | 8741 | Daemon listen port |
| `db_path` | `""` (cwd/.simba/memory) | Database directory |
| `embedding_model` | nomic-embed-text | Display name for embeddings |
| `embedding_dims` | 768 | Embedding vector dimensions |
| `model_repo` | nomic-ai/nomic-embed-text-v1.5-GGUF | HuggingFace repo for auto-download |
| `model_file` | nomic-embed-text-v1.5.Q4_K_M.gguf | GGUF file name in repo |
| `model_path` | `""` (auto-download) | Local path to GGUF file |
| `n_gpu_layers` | -1 | GPU layers (-1=all, 0=CPU only) |
| `embed_url` | `""` (in-process) | URL of external OpenAI-compatible embedding server |
| `min_similarity` | 0.35 | Minimum cosine similarity for recall |
| `max_results` | 3 | Maximum memories returned per query |
| `duplicate_threshold` | 0.92 | Similarity threshold for dedup |
| `max_content_length` | 200 | Maximum memory content length |

### Project Search

Project memory is stored at `{repo_root}/.claude-memory/memory.db` and uses SQLite with FTS5 for full-text search. No configuration needed beyond initializing the database.

## Project Structure

```
src/simba/
  __main__.py          CLI entry point (install, hook, server, search)
  memory/
    server.py          FastAPI daemon with LanceDB
    routes.py          6 REST endpoints with recall logging
    vector_db.py       Cosine similarity, search, dedup
    embeddings.py      Dual-backend embedding (in-process GGUF or HTTP)
    config.py          Configuration dataclass
  search/
    project_memory.py  SQLite FTS5 for sessions, knowledge, facts
    rag_context.py     RAG orchestrator combining SQLite + QMD
    activity_tracker.py  Pipe-separated activity log with rotation
    qmd.py             QMD CLI wrapper with stop-word extraction
    deps.py            External tool availability checker
    __main__.py        CLI (simba search)
    file_suggestion.sh Shell script for fileSuggestion.command
  tailor/
    hook.py            Error detection, normalization, reflection entries
    install.py         Project installer (dirs, hooks, CLAUDE.md)
    session_start.py   Git status, checkpoints, time context
    status.py          Error counting, co-occurrence analysis
  guardian/
    extract_core.py    CLAUDE.md CORE tag extraction
    check_signal.py    Compliance signal check
  hooks/
    session_start.py   Daemon + tailor + project memory + extraction check
    user_prompt_submit.py  Memory recall + CORE blocks + search context
    pre_tool_use.py    Thinking-based recall with dedup cache
    post_tool_use.py   Activity tracking for session memory
    pre_compact.py     Transcript export to ~/.claude/transcripts/
    stop.py            Guardian signal check + tailor error capture

tests/                 ~290 tests across 20+ files
tools/
  enforce_module_imports.py  libcst-based import style checker/fixer
skills/                7 slash command skills
.claude-plugin/
  plugin.json          Plugin manifest
  hooks.json           Hook event to command mapping
```

## License

Private.
