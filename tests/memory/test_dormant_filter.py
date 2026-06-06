"""Tests for the dormant-memory recall filter (src/simba/memory/hybrid.py)."""

from __future__ import annotations

import pathlib

import simba.db
import simba.memory.usage as usage
from simba.memory.hybrid import _filter_dormant


def test_filter_dormant_removes_dormant_records(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_a", now=0.0)
        usage.get_or_create("mem_b", now=0.0)
        usage.set_dormant("mem_b", dormant=True)

    records = [{"id": "mem_a", "content": "a"}, {"id": "mem_b", "content": "b"}]
    result = _filter_dormant(records, cwd=tmp_path)
    assert len(result) == 1
    assert result[0]["id"] == "mem_a"


def test_filter_dormant_keeps_missing_rows(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        pass  # empty DB / schema only

    records = [{"id": "mem_new"}]
    result = _filter_dormant(records, cwd=tmp_path)
    assert len(result) == 1


def test_filter_dormant_handles_empty_input(tmp_path: pathlib.Path) -> None:
    assert _filter_dormant([], cwd=tmp_path) == []
