"""Tests for append-only anticipated-query memory metadata."""

from __future__ import annotations

import simba.db
import simba.memory.anticipated as anticipated


def test_append_queries_dedupes_and_limits(tmp_path) -> None:
    with simba.db.connect(tmp_path):
        rows = anticipated.append_queries(
            memory_id="mem_a",
            queries=[
                "How do I restart Simba?",
                "how do i restart simba?",
                "  status  ",
            ],
            source="cli",
            now=1000.0,
            limit=2,
        )

        assert [row.query for row in rows] == ["How do I restart Simba?", "status"]
        stored = anticipated.list_for("mem_a")
        assert [row.query for row in stored] == ["How do I restart Simba?", "status"]
        assert stored[0].source == "cli"


def test_normalize_queries_drops_empty(tmp_path) -> None:
    assert anticipated.normalize_queries(["", "  ", "alpha"], limit=5) == ["alpha"]


def test_search_matches_anticipated_query_sidecar(tmp_path) -> None:
    with simba.db.connect(tmp_path):
        anticipated.append_queries(
            memory_id="mem_a",
            queries=["How do I fix opaque bearer auth failures?"],
            source="test",
            now=1000.0,
            limit=5,
        )
        anticipated.append_queries(
            memory_id="mem_b",
            queries=["How do I tune hybrid recall?"],
            source="test",
            now=1001.0,
            limit=5,
        )

        hits = anticipated.search("opaque bearer failure", limit=5)

    assert [hit["memory_id"] for hit in hits] == ["mem_a"]
