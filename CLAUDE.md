# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a monorepo ("simba") containing 4 independent Claude Code plugin/tooling projects. Each project has its own git history, package.json, and distribution channel. They all extend Claude Code's capabilities through the hooks system.

| Project | Purpose | Language | Deps |
|---------|---------|----------|------|
| `claude-tailor` | Error memory + persistent patterns via JSONL reflections | JS (ESM) | Zero |
| `claude-memory` | Semantic memory daemon with vector DB (LanceDB + llama-cpp-python) | JS (CJS) | express, @lancedb/lancedb, uuid |
| `claude-turbo-search` | Fast file search + semantic indexing via ripgrep/fzf/QMD | Bash + JS | System tools (ripgrep, fzf, jq, bun, qmd) |
| `claude-md-memory-guardian` | CLAUDE.md rule reinforcement via `<!-- CORE -->` tag extraction | Bash | None |

## Architecture

All projects use **Claude Code hooks** to inject behavior at specific lifecycle points:

```
SessionStart       → Auto-start daemons, inject patterns/rules, show memory count
UserPromptSubmit   → Recall memories, suggest files, reinforce core rules
PreToolUse         → Inject context from vector DB based on Claude's thinking
PreCompact         → Export transcripts for learning extraction
StopHook           → Monitor for [✓ rules] signal (memory-guardian)
```

Hook output follows the Claude Code hook protocol: read JSON from stdin, write `hookSpecificOutput` JSON to stdout with `additionalContext` for context injection.

### claude-tailor (tailor-nano)

- `src/hook.js` — Post-conversation hook that reads transcript JSONL, detects errors via regex patterns, normalizes them for clustering (line numbers → `:LINE:COL`, paths → `/PATH/`), and appends reflection entries to `.claude-tailor/memory/reflections.jsonl`
- `src/install.js` — Installer that creates dirs, copies hooks, registers in `.claude/settings.local.json`, handles CLAUDE.md merge/overwrite/skip
- `src/session-start.sh` — Reinjects patterns at session start
- `commands/` — Slash commands: `/mark` (checkpoints), `/recall` (search history), `/test` (run tests), `/reflect` (analyze patterns)
- Budget constraint: <300 LOC total, zero npm dependencies, Node.js built-ins only

### claude-memory

- `server.js` — Express daemon (port 8741) with LanceDB vector database
- `services/embeddings.js` — In-process GGUF embedding via llama-cpp-python with async queue
- `services/vector-db.js` — LanceDB operations (store, search, duplicate detection at 0.92 similarity threshold)
- `routes/` — REST API: POST `/store`, POST `/recall`, GET `/health`, GET `/stats`, GET `/list`, DELETE `/memory/:id`
- `hooks/` — 4 hooks: session-start (daemon auto-start), user-prompt-submit (prompt recall), pre-tool-use (thinking-based recall), pre-compact (transcript export)
- `skills/` — `/memories-learn` (extract learnings from transcripts), `/memories-sanitize` (review/remove invalid memories)
- Memory types: GOTCHA, WORKING_SOLUTION, PATTERN, DECISION, FAILURE, PREFERENCE

### claude-turbo-search

- `scripts/` — Setup scripts for dependencies, file suggestions, MCP server, memory DB, vector search, hooks
- `hooks/` — Pre-prompt search (simple file suggestions) and RAG context injection
- `skills/` — `/turbo-index` (5-phase setup), `/qmd` (semantic search), `/remember`, `/memory-stats`, `/token-stats`
- Plugin distribution via `.claude-plugin/plugin.json` and `marketplace.json`

### claude-md-memory-guardian

- `hooks/extract-core.sh` — Extracts content between `<!-- CORE -->` tags from CLAUDE.md
- `hooks/check-signal.sh` — Checks responses for `[✓ rules]` signal; missing → re-inject full CLAUDE.md

## Python Project (src/simba)

Uses uv for dependency management, hatchling for builds, ruff for linting/formatting.

```bash
uv sync                           # Create venv and install deps
uv run pytest                     # Run tests
uv run pytest -x                  # Stop on first failure
uv run pytest -k "test_name"      # Run specific test
uv run ruff check src/ tests/     # Lint
uv run ruff format src/ tests/    # Format (black-compatible)
uv run ruff check --fix src/      # Auto-fix lint issues
```

Source layout: `src/simba/` (PEP 621 src layout with hatchling).

## Build & Test Commands

### claude-tailor (only project with tests)

```bash
cd claude-tailor
npm test                          # Run all 48 tests (node:test runner)
npm run test:watch                # Watch mode
node src/install.js               # Install into current project
node src/install.js --dry-run     # Simulate installation
node src/install.js --force       # Overwrite existing files
```

Test framework: Node.js native `node:test` + `node:assert`. Tests use temporary directories, subprocess spawning, and Arrange-Act-Assert pattern.

### claude-memory

```bash
cd claude-memory
npm install                       # Install dependencies
npm start                         # Start daemon on port 8741
npm run dev                       # Start with --watch for development
./install.sh                      # Install to system location (~/.local/share/claude-memory or ~/Library/Application Support/claude-memory)
./uninstall.sh                    # Remove installation
```

The embedding model (nomic-embed-text GGUF) is auto-downloaded from Hugging Face on first daemon startup.

### claude-turbo-search

```bash
# Install as Claude Code plugin
claude plugin marketplace add iagocavalcante/claude-turbo-search
claude plugin install claude-turbo-search@claude-turbo-search-dev

# Or run setup scripts directly
cd claude-turbo-search
./scripts/install-deps.sh         # Install ripgrep, fzf, jq, bun, qmd
./scripts/setup-hooks.sh          # Simple mode
./scripts/setup-hooks.sh --rag    # RAG mode (recommended)
./scripts/setup-hooks.sh --remove # Remove hooks
```

### claude-md-memory-guardian

```bash
# Install as Claude Code plugin
claude plugin marketplace add brannon-bowden/claude-md-memory-guardian
claude plugin install memory-guardian@brannon-bowden
```

## Key Patterns

**Module systems differ**: `claude-tailor` uses ESM (`"type": "module"`), `claude-memory` uses CommonJS (`"type": "commonjs"`). Respect this when modifying imports.

**Hook I/O protocol**: Hooks receive JSON on stdin with fields like `transcript_path`, `cwd`, `source`. They output JSON to stdout with structure `{ hookSpecificOutput: { hookEventName, additionalContext } }`.

**Error normalization** (claude-tailor): Errors are normalized before clustering — line numbers, hex addresses, large numbers, and paths are replaced with tokens to group similar errors together.

**Append-only storage**: Both `claude-tailor` (reflections.jsonl) and `claude-memory` (LanceDB) use append-only patterns. Never overwrite memory files.

**Daemon lifecycle** (claude-memory): The SessionStart hook checks daemon health, auto-starts if down, and polls up to 15 times at 300ms intervals before giving up.

## Node Version

All projects require Node.js >= 18.0.0.
