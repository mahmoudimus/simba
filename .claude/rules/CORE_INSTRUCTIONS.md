# Core Instructions

These instructions apply to ALL contexts (main session + subagents).
Managed by SIMBA markers — run `simba markers audit` to check health.

This file lives in `.claude/rules/` so Claude Code auto-loads it every session.
Critical rules are wrapped in `SIMBA:core` markers so the guardian hook
re-injects them after context compaction.

---

<!-- BEGIN SIMBA:core -->
## Critical Constraints

- **Pure Python**: All code lives under `src/simba/`. No Node.js, no external services.
- **Append-only storage**: Never overwrite memory files — LanceDB for vectors, JSONL for transcripts/reflections
- **Hook I/O protocol**: Read JSON from stdin, write `{ hookSpecificOutput: { hookEventName, additionalContext } }` to stdout
- **Hooks are plugin-level**: Registered via `.claude-plugin/hooks.json`, invoked as `uv run python -m simba.hooks.<name>`
- **No external embedding services**: llama-cpp-python loads nomic-embed-text GGUF in-process. Task prefixes: "search_document" (store), "search_query" (recall)
- **Memory content max 200 chars**: Enforced on storage. Use context field for details.
- **All config via `simba config`**: Every configurable value must be a field on a `@configurable` dataclass, gettable/settable via `simba config get/set <section>.<key>`. No hidden constants or env-var-only config.
<!-- END SIMBA:core -->

---

<!-- BEGIN SIMBA:build_commands -->
## Build & Test Commands

```bash
uv sync                           # Install deps
uv run pytest                     # Run all tests
uv run pytest -x                  # Stop on first failure
uv run pytest -k "test_name"      # Run specific test
uv run ruff check src/ tests/     # Lint
uv run ruff format src/ tests/    # Format
uv run ruff check --fix src/      # Auto-fix lint
```
<!-- END SIMBA:build_commands -->

---

<!-- BEGIN SIMBA:environment -->
## Environment

- Source layout: `src/simba/` (PEP 621, setuptools build backend)
- Python >= 3.12, uv for dependency management, ruff for lint/format
- Memory daemon port: 8741 (configurable via `simba config set memory.port`)
- Embedding model: nomic-embed-text-v1.5 Q4_K_M (~81MB GGUF, auto-downloaded)
- DB path: `.simba/memory/memories.lance` (LanceDB)
- Similarity: 0.35 min for recall, 0.92 for duplicate detection
<!-- END SIMBA:environment -->

---

<!-- BEGIN SIMBA:workflow -->
## Workflow

- 6 active hooks: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, PreCompact, Stop
- Daemon auto-starts on SessionStart (polls 15x at 300ms intervals)
- PreCompact exports transcripts to ~/.claude/transcripts/{sessionId}/
- Guardian extracts SIMBA:core blocks and re-injects on every prompt
- Silent failure: hooks exit 0 gracefully when conditions aren't met
<!-- END SIMBA:workflow -->
