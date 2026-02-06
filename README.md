# Simba

A unified Claude Code plugin that combines error memory, semantic recall, CLAUDE.md rule enforcement, and project-aware search into a single Python package. Replaces four separate JS/Bash plugins (claude-tailor, claude-memory, claude-md-memory-guardian, claude-turbo-search) with one coherent system.

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

## Subsystems

### Memory Daemon (`simba.memory`)

A FastAPI server backed by LanceDB for vector storage and Ollama for embeddings. Stores typed memories (GOTCHA, WORKING_SOLUTION, PATTERN, DECISION, FAILURE, PREFERENCE) with cosine similarity search, duplicate detection, and project-scoped filtering.

**Endpoints:** POST `/store`, POST `/recall`, GET `/health`, GET `/stats`, GET `/list`, DELETE `/memory/:id`

### Project Search (`simba.search`)

Per-project session memory and semantic search integration. Combines a local SQLite database (sessions, knowledge areas, facts) with QMD semantic search to automatically inject relevant context before each prompt.

Three layers of context retrieval:
- **SQLite project memory** -- FTS5 keyword search over accumulated session history, codebase knowledge, and project facts
- **QMD semantic search** -- BM25 and vector search over indexed markdown documentation
- **Activity tracking** -- logs tool usage (files read, edited, searched) for session summaries

The SQLite database lives at `{repo_root}/.claude-memory/memory.db` and requires no daemon or external services. QMD integration is optional and activates when the `qmd` CLI tool is installed.

### Tailor (`simba.tailor`)

Reads Claude Code transcript JSONL after each session, detects errors via regex patterns, normalizes them for clustering (line numbers, hex addresses, paths replaced with tokens), and appends structured reflection entries to `.claude-tailor/memory/reflections.jsonl`. Provides error frequency analysis, co-occurrence detection, and signature-based grouping.

### Guardian (`simba.guardian`)

Extracts content between `<!-- CORE -->` tags from CLAUDE.md and injects it as context on every prompt. On session stop, checks whether Claude's response contains the `[✓ rules]` compliance signal. If missing, re-injects the full CLAUDE.md with a re-audit instruction.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.ai/) running with the `nomic-embed-text` model

```
ollama pull nomic-embed-text
```

Optional external tools for enhanced search:
- [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) -- fast file discovery
- [fzf](https://github.com/junegunn/fzf) -- fuzzy filtering for file suggestions
- [qmd](https://github.com/tobi/qmd) -- semantic markdown search

Check tool availability with:
```bash
uv run python -c "import simba.search.deps; print(simba.search.deps.check_all())"
```

## Setup

```bash
git clone <repo-url> && cd simba
uv sync
```

## Usage

### Start the Memory Daemon

```bash
# Run from any directory; database defaults to .simba/memory/ in cwd
uv run python -m simba.memory.server

# Specify a custom database path
uv run python -m simba.memory.server --db-path /path/to/db

# Custom port
uv run python -m simba.memory.server --port 9000
```

The SessionStart hook auto-starts the daemon if it is not already running.

### Project Memory CLI

```bash
# Initialize project memory (creates .claude-memory/memory.db)
uv run python -m simba.search init

# Add a session summary
uv run python -m simba.search add-session "Refactored auth module" '["src/auth.py"]' '["Read","Edit"]' "auth,refactor"

# Add codebase knowledge
uv run python -m simba.search add-knowledge "src/auth" "JWT-based auth with refresh tokens" "15min access, 7d refresh"

# Add a project fact
uv run python -m simba.search add-fact "Uses PostgreSQL with SQLAlchemy" "architecture"

# Search project memory
uv run python -m simba.search search "authentication"

# Get combined context for a query
uv run python -m simba.search context "how does auth work"

# View recent sessions
uv run python -m simba.search recent 5

# Show statistics
uv run python -m simba.search stats
```

### Install Into a Project

```bash
uv run python -m simba.tailor.install
uv run python -m simba.tailor.install --dry-run    # preview only
uv run python -m simba.tailor.install --force       # overwrite existing files
```

This creates `.claude-tailor/memory/`, `.claude/commands/`, registers hooks in `.claude/settings.local.json`, and optionally manages CLAUDE.md (`--claude-md=skip|overwrite|merge`).

### Plugin Registration

The `.claude-plugin/` directory contains the plugin manifest and hook definitions. Each hook maps to a `uv run python -m simba.hooks.<module>` command with appropriate timeouts.

### Skills

Seven custom slash commands are provided as skills:

| Skill | Purpose |
|-------|---------|
| `/memories-learn` | Extract learnings from a session transcript and store them in the memory daemon |
| `/memories-sanitize` | Review stored memories and remove invalid, outdated, or superseded entries |
| `/turbo-index` | Index the project for optimized search (dependency check, QMD indexing, memory init) |
| `/qmd` | Search indexed documentation with QMD (BM25 keyword, semantic, or hybrid) |
| `/remember` | Save the current work session to project memory |
| `/memory-stats` | Show project memory database statistics and recent entries |
| `/token-stats` | Token savings dashboard comparing search-first vs blind exploration |

## Project Structure

```
src/simba/
  memory/
    server.py          FastAPI daemon with LanceDB + Ollama
    routes.py          6 REST endpoints with recall logging
    vector_db.py       Cosine similarity, search, dedup
    embeddings.py      Ollama embedding service with async queue
    config.py          Configuration dataclass
  search/
    project_memory.py  SQLite FTS5 for sessions, knowledge, facts
    rag_context.py     RAG orchestrator combining SQLite + QMD
    activity_tracker.py  Pipe-separated activity log with rotation
    qmd.py             QMD CLI wrapper with stop-word extraction
    deps.py            External tool availability checker
    __main__.py        CLI (python -m simba.search)
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

tests/                 285 tests across 20 files
tools/
  enforce_module_imports.py  libcst-based import style checker/fixer
skills/
  memories-learn/      Extract learnings from transcripts
  memories-sanitize/   Review and clean stored memories
  turbo-index/         Index project for optimized search
  qmd/                 Semantic search guide
  remember/            Save session to project memory
  memory-stats/        Project memory statistics
  token-stats/         Token savings dashboard
.claude-plugin/
  plugin.json          Plugin manifest
  hooks.json           Hook event to command mapping
```

## Development

```bash
uv run pytest                              # run all 285 tests
uv run pytest -x                           # stop on first failure
uv run pytest -k "test_name"               # run specific test
uv run ruff check src/ tests/              # lint
uv run ruff format src/ tests/             # format
uv run pyrefly check                       # type check
uv run python tools/enforce_module_imports.py --check src/ tests/  # import style
```

### Pre-commit Hooks

```bash
uv run pre-commit install
```

Runs on every commit: ruff lint (with auto-fix), ruff format, pyrefly type check, and module import enforcement.

### Code Style

- Module-level imports only: `import X` then `X.Y`, not `from X import Y`. Enforced by `tools/enforce_module_imports.py` in pre-commit.
- Exemptions: `from __future__ import annotations`, `from typing import TYPE_CHECKING`, and imports inside `if TYPE_CHECKING:` blocks.
- `pathlib.Path` over `os.path` (enforced via ruff TID ban).

## Configuration

### Memory Daemon

`MemoryConfig` fields with defaults:

| Field | Default | Description |
|-------|---------|-------------|
| `port` | 8741 | Daemon listen port |
| `db_path` | `""` (cwd/.simba/memory) | Database directory |
| `embedding_model` | nomic-embed-text | Ollama model for embeddings |
| `embedding_dims` | 768 | Embedding vector dimensions |
| `ollama_url` | http://localhost:11434 | Ollama API endpoint |
| `min_similarity` | 0.35 | Minimum cosine similarity for recall |
| `max_results` | 3 | Maximum memories returned per query |
| `duplicate_threshold` | 0.92 | Similarity threshold for dedup |
| `timeout_ms` | 10000 | Embedding request timeout |
| `max_content_length` | 200 | Maximum memory content length |

### Project Search

Project memory is stored at `{repo_root}/.claude-memory/memory.db` and uses SQLite with FTS5 for full-text search. No configuration needed beyond initializing the database.

RAG context injection is controlled by constants in `simba.search.rag_context`:

| Constant | Value | Description |
|----------|-------|-------------|
| `_MAX_CONTEXT_TOKENS` | 1500 | Total token budget for injected context |
| `_MEMORY_TOKEN_BUDGET` | 500 | Token budget for SQLite memory portion |
| `_MAX_CODE_RESULTS` | 3 | Maximum QMD search results |
| `_MIN_QUERY_LENGTH` | 15 | Minimum prompt length to trigger search |

## License

Private.
