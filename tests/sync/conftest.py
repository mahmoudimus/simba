"""Shared fixtures for sync subsystem tests."""

from __future__ import annotations

import pytest

import simba.db


@pytest.fixture(autouse=True)
def _use_tmp_db(tmp_path, monkeypatch):
    """Redirect simba.db to a temp directory so tests never touch real data."""
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
