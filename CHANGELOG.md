# Changelog

## [Unreleased] — 2025-02-06

### Added

- **Context-low early warning**: PreToolUse hook monitors transcript file size
  and injects a one-time `<context-low-warning>` when context approaches the
  auto-compact threshold (default 4 MB). Gives Claude time to summarize work
  state before compaction.

- **Sync pipeline**: New `simba sync` CLI with subcommands (`run`, `index`,
  `extract`, `status`, `schedule`) and `SyncScheduler` for periodic background
  sync. `POST /sync` endpoint on the daemon triggers one-off cycles.
  SessionStart hook fires a sync on every new session.

- **Diagnostics tracker**: Periodic reporting of endpoint hits, recall/store
  stats, and automatic LanceDB compaction after every N requests.

- **Access tracking**: Recalled memories now have `lastAccessedAt` and
  `accessCount` updated via fire-and-forget background tasks.

- **`simba server --sync-interval`**: Start the daemon with automatic periodic
  sync.

### Fixed

- **Vector search returning 0 results**: Added `table.checkout_latest()` before
  every vector search to refresh stale LanceDB table handles. Fragment buildup
  (one per `table.add()`) caused searches to silently return empty results.

- **Silent error swallowing in vector_db.py**: Replaced bare
  `except Exception: return []` with `logger.warning(..., exc_info=True)` so
  search failures are visible in daemon logs.

- **`table.update()` parameter name**: Fixed `values=` (wrong) to `updates=`
  (correct LanceDB API) in access tracking.

- **PreCompact TypeError on `/compact`**: Fixed crash when transcript entries
  contain nested `tool_result` content (list instead of string). Added
  `isinstance(val, str)` guard.

- **Content length limit**: Bumped `max_content_length` from 200 to 1000 to
  prevent 400 errors during learning extraction.

- **LanceDB table compaction**: Added `compact_table()` function and periodic
  compaction during diagnostics to prevent fragment buildup.

### Changed

- **Test infrastructure overhaul**: Removed all dangerous mock classes
  (`MockTable`, `MockVectorSearch`, `MockQuery`, `TrackingMockTable`) that
  masked production bugs. Tests now use real LanceDB with `tmp_path` fixtures
  and `httpx.AsyncClient` for async route testing.

- **PreToolUse refactored**: `main()` now collects context parts into a list
  (context-low warning, memory recall, truth DB) and joins them, replacing the
  earlier early-return pattern.

### Test Coverage

- 516 tests, all passing
- Lint clean (`ruff check` + `ruff format`)
