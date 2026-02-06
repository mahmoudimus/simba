"""LanceDB operations — search, deduplication, CRUD.

Ported from claude-memory/services/vector-db.js.
"""

from __future__ import annotations

import logging
import time
import typing

logger = logging.getLogger("simba.memory")


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


async def update_access_tracking(table: typing.Any, memory_ids: list[str]) -> None:
    """Update lastAccessedAt and increment accessCount for recalled memories.

    Fire-and-forget: exceptions are logged but never propagated.
    """
    try:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for mid in memory_ids:
            # Read current accessCount so we can increment it.
            search = await table.search()
            rows = await search.where(f"id = '{mid}'").limit(1).to_list()
            current_count = rows[0].get("accessCount", 0) if rows else 0
            await table.update(
                where=f"id = '{mid}'",
                values={
                    "lastAccessedAt": now,
                    "accessCount": current_count + 1,
                },
            )
    except Exception:
        logger.debug(
            "access-tracking update failed for ids=%s",
            memory_ids,
            exc_info=True,
        )
