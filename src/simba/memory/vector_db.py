"""LanceDB operations — cosine similarity, search, deduplication, CRUD.

Ported from claude-memory/services/vector-db.js.
"""

from __future__ import annotations

import math
import typing


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for i in range(len(a)):
        dot_product += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]

    denominator = math.sqrt(norm_a) * math.sqrt(norm_b)
    return 0.0 if denominator == 0 else dot_product / denominator


async def find_duplicates(
    table: typing.Any, embedding: list[float], threshold: float
) -> dict[str, typing.Any]:
    """Check for duplicate memories based on cosine similarity."""
    try:
        results = (
            await table.vector_search(embedding)
            .column("vector")
            .distance_type("cosine")
            .limit(5)
            .to_list()
        )
        for result in results:
            if result.get("type") == "SYSTEM":
                continue
            similarity = 1 - (result.get("_distance", 0))
            if similarity >= threshold:
                return {
                    "is_duplicate": True,
                    "existing_id": result["id"],
                    "similarity": similarity,
                }
    except Exception:
        pass

    return {"is_duplicate": False}


async def search_memories(
    table: typing.Any,
    embedding: list[float],
    min_similarity: float,
    max_results: int,
    filters: dict[str, typing.Any] | None = None,
) -> list[dict[str, typing.Any]]:
    """Search memories by vector similarity."""
    if filters is None:
        filters = {}

    try:
        results = (
            await table.vector_search(embedding)
            .column("vector")
            .distance_type("cosine")
            .limit(max_results * 3)
            .to_list()
        )

        memories = []
        for r in results:
            similarity = 1 - (r.get("_distance", 0))
            if r.get("type") == "SYSTEM":
                continue
            if similarity < min_similarity:
                continue

            filter_types = filters.get("types", [])
            if filter_types and r.get("type") not in filter_types:
                continue

            filter_project = filters.get("projectPath")
            if (
                filter_project
                and r.get("projectPath")
                and r["projectPath"] != filter_project
            ):
                continue

            memories.append({**r, "similarity": similarity})

        memories.sort(key=lambda m: m["similarity"], reverse=True)
        return memories[:max_results]
    except Exception:
        return []


async def count_rows(table: typing.Any) -> int:
    """Count total rows in a table."""
    try:
        return await table.count_rows()
    except Exception:
        return 0
