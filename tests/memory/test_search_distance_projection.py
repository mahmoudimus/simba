"""Vector-search projections must request ``_distance`` EXPLICITLY.

Lance deprecation (observed live 2026-07-20): a search that passes output
columns without ``_distance`` currently gets it auto-included with a WARN
("Call disable_scoring_autoprojection to adopt the future behavior") and a
future lancedb release will STOP including it — silently breaking duplicate
detection (0.92 threshold) and recall similarity, both computed from
``result["_distance"]``.
"""

from __future__ import annotations

import typing

import pytest

import simba.memory.vector_db as vdb


class _RecordingSearch:
    def __init__(self, owner: _FakeTable) -> None:
        self._owner = owner

    def select(self, columns: list[str]) -> _RecordingSearch:
        self._owner.selected_columns = list(columns)
        return self

    def column(self, *_a: typing.Any) -> _RecordingSearch:
        return self

    def distance_type(self, *_a: typing.Any) -> _RecordingSearch:
        return self

    def nprobes(self, *_a: typing.Any) -> _RecordingSearch:
        return self

    def limit(self, *_a: typing.Any) -> _RecordingSearch:
        return self

    def where(self, *_a: typing.Any, **_k: typing.Any) -> _RecordingSearch:
        return self

    async def to_list(self) -> list[dict]:
        return []


class _FakeTable:
    def __init__(self) -> None:
        self.selected_columns: list[str] | None = None

    def search(self, *_a: typing.Any, **_k: typing.Any) -> _RecordingSearch:
        return _RecordingSearch(self)

    def vector_search(self, *_a: typing.Any, **_k: typing.Any) -> _RecordingSearch:
        return _RecordingSearch(self)


def test_search_result_fields_include_distance() -> None:
    assert "_distance" in vdb._SEARCH_RESULT_FIELDS


@pytest.mark.asyncio
async def test_find_duplicates_selects_distance() -> None:
    table = _FakeTable()
    await vdb.find_duplicates(table, [0.0] * 4, 0.92)
    assert table.selected_columns is not None
    assert "_distance" in table.selected_columns
