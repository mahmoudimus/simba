"""LanceDB operations — search, deduplication, CRUD.

Ported from claude-memory/services/vector-db.js.
"""

from __future__ import annotations

import logging
import time
import typing

logger = logging.getLogger("simba.memory")


class EmbeddingDimMismatchError(RuntimeError):
    """Raised when the query embedding dim doesn't match the stored vectors.

    Happens after the configured embedder changes (e.g. the bge-large default is
    1024-d; an older store is 768-d). The fix is a one-time migration.
    """


def check_embedding_dim(query_dim: int, table_dim: int | None) -> None:
    """Raise an actionable error if the query/store embedding dims disagree.

    ``table_dim`` of ``None`` (couldn't be determined) is a no-op so recall is
    never blocked by an inability to introspect the store.
    """
    if table_dim is not None and query_dim != table_dim:
        raise EmbeddingDimMismatchError(
            f"Embedding dimension mismatch: the store has {table_dim}-d vectors "
            f"but the configured embedder produces {query_dim}-d. The embedding "
            f"model changed — run `simba memory reembed` to migrate the store."
        )


def _table_vector_dim(table: typing.Any) -> int | None:
    """Best-effort: read the stored ``vector`` dim from the table schema.

    Returns ``None`` if it can't be determined (older handle, no schema, etc.) —
    callers treat that as "skip the guard", never as an error.
    """
    try:
        schema = getattr(table, "schema", None)
        if schema is None:
            return None
        field = schema.field("vector")
        list_size = getattr(field.type, "list_size", None)
        return int(list_size) if list_size and list_size > 0 else None
    except Exception:
        return None


async def find_duplicates(
    table: typing.Any, embedding: list[float], threshold: float
) -> dict[str, typing.Any]:
    """Check for duplicate memories based on cosine similarity."""
    try:
        if hasattr(table, "checkout_latest"):
            await table.checkout_latest()

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
        logger.warning("find_duplicates failed", exc_info=True)

    return {"is_duplicate": False}


async def search_memories(
    table: typing.Any,
    embedding: list[float],
    min_similarity: float,
    max_results: int,
    filters: dict[str, typing.Any] | None = None,
) -> list[dict[str, typing.Any]]:
    """Search memories by vector similarity.

    When ``filters['projectPath']`` is set, results are scoped strictly to
    that project: only memories tagged with exactly that project are kept, so
    neither other projects' nor untagged memories leak into recall.
    """
    if filters is None:
        filters = {}

    try:
        # Refresh the table handle to see newly-added fragments.
        if hasattr(table, "checkout_latest"):
            await table.checkout_latest()

        # Guard: a changed embedder (e.g. nomic-768 -> bge-1024) makes the query
        # vector incompatible with the stored vectors. Surface a clear migration
        # message instead of a silent empty recall.
        check_embedding_dim(len(embedding), _table_vector_dim(table))

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
            # Strict scope: keep only exact-project matches (drops both
            # other-project and untagged/global memories).
            if filter_project and r.get("projectPath") != filter_project:
                continue

            memories.append({**r, "similarity": similarity})

        memories.sort(key=lambda m: m["similarity"], reverse=True)
        return memories[:max_results]
    except EmbeddingDimMismatchError as exc:
        # Loud + actionable: the store needs migration, not a silent empty recall.
        logger.error("recall disabled — %s", exc)
        return []
    except Exception:
        logger.warning("search_memories failed", exc_info=True)
        return []


async def reembed_table(
    db_path: typing.Any,
    embed_fn: typing.Callable[[str], typing.Awaitable[list[float]]],
) -> tuple[typing.Any, int]:
    """Re-embed every memory's content and rewrite the table at the new dim.

    Reads all rows from the ``memories`` table, re-embeds ``content`` (+ context
    when present) with ``embed_fn`` (the daemon's doc embedder — async), drops and
    recreates the table so a changed embedding dimension takes effect, and returns
    ``(new_table, count)``. The caller is responsible for rebuilding the FTS
    mirror and swapping the app's table handle. A per-row embed failure keeps the
    old vector so one bad row can't abort the rebuild.
    """
    import lancedb

    db = await lancedb.connect_async(str(db_path))
    table = await db.open_table("memories")
    rows = await table.query().to_list()

    new_rows = []
    for raw in rows:
        row = dict(raw)
        row.pop("_distance", None)
        text = (row.get("content") or "").strip()
        ctx = (row.get("context") or "").strip()
        if ctx:
            text = f"{text} {ctx}".strip()
        if text:
            try:
                row["vector"] = await embed_fn(text)
            except Exception:
                logger.warning("reembed failed for %s; kept old vector", row.get("id"))
        new_rows.append(row)

    await db.drop_table("memories")
    new_table = await db.create_table("memories", new_rows)
    return new_table, len(new_rows)


async def compact_table(table: typing.Any) -> bool:
    """Compact table fragments to improve search performance.

    LanceDB creates one fragment per ``add()`` call.  Periodic compaction
    merges them into fewer, larger files.  Returns True on success.
    """
    try:
        stats = await table.optimize()
        logger.info("[compact] optimized: %s", stats)
        return True
    except Exception:
        logger.debug("compact_table failed", exc_info=True)
        return False


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
            rows = await table.query().where(f"id = '{mid}'").limit(1).to_list()
            current_count = rows[0].get("accessCount", 0) if rows else 0
            await table.update(
                updates={
                    "lastAccessedAt": now,
                    "accessCount": current_count + 1,
                },
                where=f"id = '{mid}'",
            )
    except Exception:
        logger.debug(
            "access-tracking update failed for ids=%s",
            memory_ids,
            exc_info=True,
        )
