"""LanceDB operations — search, deduplication, CRUD.

Ported from claude-memory/services/vector-db.js.
"""

from __future__ import annotations

import inspect
import logging
import time
import typing

logger = logging.getLogger("simba.memory")

# Columns a vector-search caller actually reads off each result row -- NEVER
# `vector` itself. Mirrors `simba.memory.hybrid`'s `_session_record`/
# `_from_vector` field set (duplicated rather than imported: `hybrid` imports
# THIS module, so the reverse import would cycle). `_distance` IS listed
# explicitly: Lance auto-includes it today but WARNs it as deprecated
# ("disable_scoring_autoprojection"), and a future release will stop --
# which would silently break similarity scoring (observed live 2026-07-20).
# (`.select()` still excludes `vector` even on a vector-search query --
# verified against lancedb 0.27.)
_SEARCH_RESULT_FIELDS: tuple[str, ...] = (
    "_distance",
    "id",
    "type",
    "content",
    "context",
    "confidence",
    "createdAt",
    "tags",
    "projectPath",
    "sessionSource",
)


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


def _vector_dim_from_schema(schema: typing.Any) -> int | None:
    """Pull the fixed-size ``vector`` dimension out of a resolved pyarrow schema."""
    try:
        field = schema.field("vector")
        list_size = getattr(field.type, "list_size", None)
        return int(list_size) if list_size and list_size > 0 else None
    except Exception:
        return None


async def _resolve_table_dim(table: typing.Any) -> int | None:
    """Best-effort: read the stored ``vector`` dim from the table schema.

    Handles both LanceDB table flavours: the sync ``Table`` exposes ``schema`` as
    a plain property, while the daemon's ``AsyncTable`` exposes it as a coroutine
    method (``await table.schema()``). Reading it synchronously (the old bug) got
    the bound coroutine on the live path, so the dim was never determined and the
    migration guard silently no-opped. Returns ``None`` when it can't be
    determined — callers treat that as "skip the guard", never as an error.
    """
    try:
        schema = getattr(table, "schema", None)
        if schema is None:
            return None
        if callable(schema):  # AsyncTable.schema is an async method
            schema = schema()
        if inspect.isawaitable(schema):
            schema = await schema
        return _vector_dim_from_schema(schema)
    except Exception:
        return None


def resolve_worktree_root(path: typing.Any) -> typing.Any:
    """MAIN repository root when ``path`` is inside a LINKED git worktree.

    A linked worktree's ``.git`` is a FILE — ``gitdir: <main>/.git/worktrees/
    <name>`` — so the fold needs no subprocess (spec 33 Phase 3). Returns
    ``None`` for primary checkouts, non-repos, and anything unparseable
    (fail-soft: the caller keeps the plainly-resolved path).
    """
    import pathlib

    current = path if isinstance(path, pathlib.Path) else pathlib.Path(str(path))
    try:
        current = current.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    while True:
        gitpath = current / ".git"
        try:
            if gitpath.is_dir():
                return None  # a primary checkout owns this tree
            if gitpath.is_file():
                content = gitpath.read_text(errors="ignore").strip()
                if not content.startswith("gitdir:"):
                    return None
                gitdir = pathlib.Path(content[len("gitdir:") :].strip())
                if not gitdir.is_absolute():
                    gitdir = (current / gitdir).resolve()
                if (
                    gitdir.parent.name == "worktrees"
                    and gitdir.parent.parent.name == ".git"
                ):
                    return gitdir.parent.parent.parent.resolve()
                return None
        except OSError:
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent


def normalize_project_path(
    project_path: str | None, *, resolve_worktrees: bool = False
) -> str:
    """Normalize a project path for hierarchical scoping (spec 26).

    Resolves to an absolute, symlink-resolved path with no trailing slash
    (``str(pathlib.Path(p).resolve())``) so the client's ancestor chain — also
    resolved — matches it by string membership. An empty path stays empty (a
    *global* memory, the root of the tree). Fail-soft: any resolution error
    returns the original string (membership still works on literal paths).

    ``resolve_worktrees`` (spec 33 Phase 3) additionally folds a linked git
    worktree onto its MAIN repository root, so one repo's memories share one
    scope across all its worktrees. Gated by
    ``memory.scope_normalize_worktrees`` at the call sites (default off — a
    recall-behavior change).
    """
    p = (project_path or "").strip()
    if not p:
        return ""
    import pathlib

    try:
        resolved = pathlib.Path(p).resolve()
        if resolve_worktrees:
            root = resolve_worktree_root(resolved)
            if root is not None:
                return str(root)
        return str(resolved)
    except (OSError, ValueError, RuntimeError):
        return p


def _scope_match(
    project_path: str | None,
    project_scopes: typing.Iterable[str],
    *,
    include_global: bool,
) -> bool:
    """Hierarchical (ancestor-prefix) scope membership for one memory (spec 26).

    The client computes ``project_scopes`` = ``[cwd-resolved, …ancestors…,
    git-root-resolved]`` (the path-aware work lives client-side; the daemon does
    pure string membership). A memory is in scope when its ``project_path`` is one
    of those scopes — so a ``/repo`` (root) fact inherits *down* to a ``/repo/api``
    cwd (because ``/repo`` is in the chain) while a ``/repo/api`` fact does NOT
    leak *up* to a ``/repo`` recall (``/repo/api`` is not in that chain). Global
    (empty-path) memories are the root of the tree and match when ``include_global``
    is set.
    """
    pp = project_path or ""
    if not pp:
        return include_global
    return pp in set(project_scopes)


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
            # Only `id`/`type`/`_distance` are read below (2026-07-18).
            # `_distance` is requested EXPLICITLY: Lance's auto-inclusion is
            # deprecated (WARN observed live 2026-07-20) and a future release
            # will drop it, silently breaking the similarity threshold.
            .select(["_distance", "id", "type"])
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

    Project scoping has two modes:

    * **Strict (legacy, default).** When ``filters['projectPath']`` is set,
      results are scoped strictly to that project: only memories tagged with
      exactly that project are kept, so neither other projects' nor untagged
      memories leak into recall.
    * **Hierarchical (spec 26).** When ``filters['hierarchical_recall']`` is set
      and ``filters['project_scopes']`` (the client-computed cwd→git-root chain)
      is present, a memory is kept when its ``projectPath`` is one of those scopes
      — so ancestor (root) facts inherit down — plus global memories when
      ``filters['hierarchical_recall_include_global']`` is set.
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
        check_embedding_dim(len(embedding), await _resolve_table_dim(table))

        results = (
            await table.vector_search(embedding)
            .column("vector")
            .distance_type("cosine")
            # Metadata/text fields the RRF/rerank/format pipeline reads
            # (see `_SEARCH_RESULT_FIELDS`) -- never the stored `vector`
            # itself (2026-07-18).
            .select(list(_SEARCH_RESULT_FIELDS))
            .limit(max_results * 3)
            .to_list()
        )

        # Hierarchical scoping is active only when the lever is on AND the client
        # supplied a scope chain; otherwise fall through to the strict legacy path
        # (byte-identical behavior when off).
        hierarchical = bool(filters.get("hierarchical_recall")) and bool(
            filters.get("project_scopes")
        )
        project_scopes = filters.get("project_scopes") or []
        include_global = bool(filters.get("hierarchical_recall_include_global", True))

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

            if hierarchical:
                # Ancestor-membership scope (spec 26): keep exact-or-ancestor
                # matches and (optionally) global memories.
                if not _scope_match(
                    r.get("projectPath"),
                    project_scopes,
                    include_global=include_global,
                ):
                    continue
            else:
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


class ReembedError(RuntimeError):
    """Re-embedding aborted before any write; the live table is untouched."""

    def __init__(self, message: str, failed_ids: list[str]) -> None:
        super().__init__(message)
        self.failed_ids = failed_ids


class ReembedResult(typing.NamedTuple):
    table: typing.Any
    count: int
    backup: str | None  # name of the pre-reembed copy, or None if not kept


async def reembed_table(
    db_path: typing.Any,
    embed_fn: typing.Callable[[str], typing.Awaitable[list[float]]],
    *,
    backup: bool = True,
) -> ReembedResult:
    """Re-embed every memory's content and rewrite the table at the new dim.

    All-or-nothing: every row is embedded BEFORE anything is written, and any
    embed failure (or a width disagreement) raises :class:`ReembedError` with
    the live table untouched. Keeping a row's old vector instead would mix two
    models' vector spaces in one table -- silently wrong at the same width, and
    a mixed-width table at a different one. Rows with no content and no context
    (the SYSTEM init row) get a zero vector of the new width.

    With ``backup`` the current rows (old vectors included) are first copied to
    a ``memories_backup_<UTC timestamp>`` table in the same database, so the
    migration can be undone. The copy is Arrow-native (latest version only, no
    LanceDB history), and the new vectors are packed into one float32 buffer,
    so no step materializes the corpus as Python floats (2026-07-18 incident).

    Returns ``(new_table, count, backup_name)``. The caller rebuilds the FTS
    mirror and swaps the app's table handle.
    """
    import datetime

    import lancedb
    import numpy as np
    import pyarrow as pa

    db = await lancedb.connect_async(str(db_path))
    table = await db.open_table("memories")
    schema = await table.schema()
    columns = [f.name for f in schema]
    old = await table.query().select(columns).to_arrow()

    ids = [str(x) for x in old.column("id").to_pylist()]
    contents = old.column("content").to_pylist() if "content" in columns else []
    contexts = old.column("context").to_pylist() if "context" in columns else []

    vectors: np.ndarray | None = None
    empty_rows: list[int] = []
    failed: list[str] = []
    for i, memory_id in enumerate(ids):
        text = (contents[i] if contents else "") or ""
        ctx = (contexts[i] if contexts else "") or ""
        text = f"{text.strip()} {ctx.strip()}".strip()
        if not text:
            empty_rows.append(i)
            continue
        try:
            vec = await embed_fn(text)
        except Exception as exc:
            logger.warning("reembed failed for %s: %s", memory_id, exc)
            failed.append(memory_id)
            continue
        if vectors is None:
            vectors = np.zeros((len(ids), len(vec)), dtype=np.float32)
        if len(vec) != vectors.shape[1]:
            failed.append(memory_id)
            continue
        vectors[i] = vec

    if failed:
        raise ReembedError(
            f"re-embed aborted: {len(failed)} of {len(ids)} rows failed; "
            "the table was not modified",
            failed,
        )
    if vectors is None:  # nothing embeddable (only empty rows): nothing to do
        return ReembedResult(table, len(ids), None)

    dim = int(vectors.shape[1])
    vec_type = pa.list_(pa.float32(), dim)
    new_vectors = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()), dim
    )
    pos = columns.index("vector")
    new = old.set_column(pos, pa.field("vector", vec_type), new_vectors)

    backup_name: str | None = None
    if backup:
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        backup_name = f"memories_backup_{stamp}"
        await db.create_table(backup_name, old)
    new_table = await db.create_table("memories", new, mode="overwrite")
    logger.info(
        "[reembed] %d rows -> %d-d (%d empty); backup=%s",
        len(ids),
        dim,
        len(empty_rows),
        backup_name,
    )
    return ReembedResult(new_table, len(ids), backup_name)


async def compact_table(
    table: typing.Any,
    *,
    cleanup_older_than: typing.Any | None = None,
    delete_unverified: bool = False,
) -> typing.Any | None:
    """Compact table fragments and prune old LanceDB versions.

    LanceDB creates one fragment per ``add()`` call.  Periodic compaction
    merges them into fewer, larger files and removes old version history once it
    is past the requested retention window.  Returns LanceDB's optimize stats on
    success, or None if compaction failed.
    """
    try:
        kwargs: dict[str, typing.Any] = {}
        if cleanup_older_than is not None:
            kwargs["cleanup_older_than"] = cleanup_older_than
        if delete_unverified:
            kwargs["delete_unverified"] = True
        stats = await table.optimize(**kwargs)
        logger.info("[compact] optimized: %s", stats)
        return stats
    except Exception:
        logger.debug("compact_table failed", exc_info=True)
        return None


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
            # Read current accessCount so we can increment it (only column
            # read below -- never `vector`, 2026-07-18).
            rows = (
                await table.query()
                .where(f"id = '{mid}'")
                .select(["accessCount"])
                .limit(1)
                .to_list()
            )
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
