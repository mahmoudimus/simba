"""Shared test fixtures for simba tests."""

from __future__ import annotations

import json
import pathlib
import time

import pytest

import simba.db


@pytest.fixture(autouse=True)
def _reset_db_globals():
    """Reset module-global peewee state between tests.

    The shared ``simba.db.database`` proxy and the per-path schema caches are
    process-global. Resetting them after each test prevents a DB binding or open
    connection from one test leaking into the next (defensive isolation for the
    ORM's global state).
    """
    yield
    try:
        simba.db._schema_ready.clear()
        if simba.db.database.database is not None and not simba.db.database.is_closed():
            simba.db.database.close()
    except Exception:
        pass
    try:
        import simba.memory.fts as _fts

        _fts._initialized.clear()
        if _fts._db.database is not None and not _fts._db.is_closed():
            _fts._db.close()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _reset_background_globals():
    """Reset the shutdown-aware background-task registry between tests.

    ``simba.memory.background`` (handoff item 10) tracks fire-and-forget
    daemon tasks and a process-level shutdown flag in module globals ---
    mirroring the existing ``_reset_db_globals`` pattern above, since a flag
    left set (or a stale task reference) by one test must never leak into
    the next.
    """
    yield
    import simba.memory.background as _background

    _background.reset_for_tests()


@pytest.fixture(autouse=True)
def _block_real_model_loads(request, monkeypatch):
    """Globally forbid real GGUF reranker / local-LLM loads in the unit suite so no
    test reaches Hugging Face. The default ``reranker_mode="cross-encoder"`` would
    otherwise fetch + load a real GGUF on the rerank hot path — an HF 429 there hung
    a CI run. The accessors raise → the reranker fail-opens (candidates unchanged);
    tests that exercise reorder logic inject a fake scorer (their monkeypatch runs
    after this autouse fixture and overrides it). Exempt tests marked ``gguf`` (the
    opt-in real-model integration tests, run via ``-m gguf``). Promoted here from
    tests/memory/conftest.py so it covers EVERY directory (the flaky fetch came from
    a test outside tests/memory/)."""
    if request.node.get_closest_marker("gguf"):
        return
    import simba.memory.reranker

    def _forbidden(cfg):
        raise RuntimeError(
            "real GGUF model load blocked in tests (mark `gguf` to allow)"
        )

    monkeypatch.setattr(simba.memory.reranker, "_get_cross_encoder", _forbidden)
    monkeypatch.setattr(simba.memory.reranker, "_get_local_llm", _forbidden)


@pytest.fixture
def tmp_project(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a temporary project directory structure."""
    return tmp_path


@pytest.fixture
def claude_md_with_core(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a CLAUDE.md with SIMBA:core-tagged sections."""
    content = """\
# Project Rules

## Critical Constraints
<!-- BEGIN SIMBA:core -->
- Never delete files without confirmation
- Always run tests before committing
<!-- END SIMBA:core -->

Extended explanation of constraints...

## Code Style
<!-- BEGIN SIMBA:core -->
- Use descriptive variable names
- Keep functions short
<!-- END SIMBA:core -->

Detailed style guidelines...

## Memory Signal
<!-- BEGIN SIMBA:core -->
End every response with: [✓ rules]
<!-- END SIMBA:core -->
"""
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(content)
    return claude_md


@pytest.fixture
def claude_md_no_core(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a CLAUDE.md without any CORE tags."""
    content = "# Project Rules\n\nSome rules here.\n"
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(content)
    return claude_md


@pytest.fixture
def mock_reflection():
    """Factory for creating mock reflection entries."""

    def _create(overrides: dict | None = None) -> dict:
        base = {
            "id": f"nano-{int(time.time() * 1000)}-test",
            "ts": "2024-01-01T00:00:00Z",
            "error_type": "error",
            "snippet": "Error: test error",
            "context": {
                "file": "test.js",
                "operation": "testFunc",
                "module": "test-module",
            },
            "signature": "error-test",
        }
        if overrides:
            base.update(overrides)
        return base

    return _create


@pytest.fixture
def simba_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Provide a temporary simba.db with all schemas initialized."""
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    with simba.db.get_db(tmp_path) as conn:
        yield conn


@pytest.fixture
def settings_file(tmp_path: pathlib.Path):
    """Factory for creating a settings.local.json file."""

    def _create(settings: dict | None = None) -> pathlib.Path:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        path = claude_dir / "settings.local.json"
        if settings is None:
            settings = {"hooks": {}}
        path.write_text(json.dumps(settings, indent=2))
        return path

    return _create
