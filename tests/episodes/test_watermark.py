"""Tests for the episodic-discovery watermark (episodes/watermark.py).

Mutable sweep state (NOT config) -- the per-project high-water mark of the
max `createdAt` seen by the last COMPLETED discovery sweep
(`simba.episodes.consolidate.consolidate_eligible`, 2026-07-17 RSS-storm
fix). Lives in the peewee sidecar DB next to the `episode_jobs` lease table.
"""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.episodes.watermark as watermark


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def test_get_returns_none_when_unset() -> None:
    assert watermark.get("/proj") is None


def test_advance_then_get_roundtrip() -> None:
    watermark.advance("/proj", "2026-07-17T00:00:00Z")
    assert watermark.get("/proj") == "2026-07-17T00:00:00Z"


def test_advance_overwrites_prior_value() -> None:
    watermark.advance("/proj", "2026-07-17T00:00:00Z")
    watermark.advance("/proj", "2026-07-18T00:00:00Z")
    assert watermark.get("/proj") == "2026-07-18T00:00:00Z"


def test_watermark_is_per_project() -> None:
    watermark.advance("/proj-a", "2026-07-17T00:00:00Z")
    watermark.advance("/proj-b", "2026-01-01T00:00:00Z")
    assert watermark.get("/proj-a") == "2026-07-17T00:00:00Z"
    assert watermark.get("/proj-b") == "2026-01-01T00:00:00Z"
    assert watermark.get("/proj-c") is None


def test_all_projects_sentinel_is_distinct_from_any_project_path() -> None:
    """`all_projects=True` sweeps use a dedicated sentinel key, distinct
    from any real (even empty-string) projectPath value, so the two kinds
    of sweep never share -- or clobber -- state."""
    watermark.advance("", "2026-01-01T00:00:00Z")  # untagged-project sweep
    watermark.advance("/proj", "2026-02-01T00:00:00Z", all_projects=True)

    assert watermark.get("") == "2026-01-01T00:00:00Z"
    assert watermark.get("/proj", all_projects=True) == "2026-02-01T00:00:00Z"
    # The all_projects sweep never touches the "" (untagged) project's row.
    assert watermark.get("") == "2026-01-01T00:00:00Z"
    # And a normal per-project get for "/proj" never sees the all_projects row.
    assert watermark.get("/proj") is None
