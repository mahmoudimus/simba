"""Tests for the workflow SQLite models (store.py)."""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.workflow.store as store


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def test_tables_created_on_connect():
    with simba.db.connect():
        assert store.WfTask.table_exists()
        assert store.WfCheckpoint.table_exists()
        assert store.WfAsset.table_exists()


def test_models_registered():
    for model in (store.WfTask, store.WfCheckpoint, store.WfAsset):
        assert model in simba.db._MODELS


def test_wftask_defaults():
    with simba.db.connect():
        task = store.WfTask.create(
            queue="q",
            payload="{}",
            status="pending",
            max_attempts=3,
            available_at="2026-01-01T00:00:00Z",
            created_at="2026-01-01T00:00:00Z",
        )
    assert task.attempts == 0


def test_dedup_unique_index():
    import simba._vendor.peewee as pw

    with simba.db.connect():
        store.WfTask.create(
            queue="q",
            dedup_key="k1",
            payload="{}",
            status="pending",
            max_attempts=3,
            available_at="2026-01-01T00:00:00Z",
            created_at="2026-01-01T00:00:00Z",
        )
        with pytest.raises(pw.IntegrityError):
            store.WfTask.create(
                queue="q",
                dedup_key="k1",
                payload="{}",
                status="pending",
                max_attempts=3,
                available_at="2026-01-01T00:00:00Z",
                created_at="2026-01-01T00:00:00Z",
            )
