"""Shared test fixtures for simba tests."""

from __future__ import annotations

import json
import pathlib
import time

import pytest

import simba.db


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
