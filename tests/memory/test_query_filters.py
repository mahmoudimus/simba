"""Tests for structured recall query filters."""

from __future__ import annotations

import simba.memory.query_filters as qf


def test_parse_type_and_clean_query() -> None:
    parsed = qf.parse("type:gotcha ruff linter")
    assert parsed.query == "ruff linter"
    assert parsed.route_filters["types"] == ["GOTCHA"]


def test_parse_post_filters() -> None:
    query = 'tag:python path:"src/simba" symbol:MemoryUsage after:2026-01-01'
    parsed = qf.parse(query)
    assert parsed.query == query
    assert parsed.post_filters["tag"] == "python"
    assert parsed.post_filters["path"] == "src/simba"
    assert parsed.post_filters["symbol"] == "MemoryUsage"
    assert parsed.post_filters["after"] == "2026-01-01"


def test_apply_tag_and_symbol_filters() -> None:
    records = [
        {
            "id": "a",
            "content": "uses MemoryUsage",
            "context": "",
            "tags": '["python"]',
            "createdAt": "2026-06-01T00:00:00Z",
        },
        {
            "id": "b",
            "content": "other",
            "context": "",
            "tags": '["ruby"]',
            "createdAt": "2026-06-01T00:00:00Z",
        },
    ]
    out = qf.apply(records, {"tag": "python", "symbol": "MemoryUsage"})
    assert [r["id"] for r in out] == ["a"]
