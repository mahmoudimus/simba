"""SQLite FTS5 keyword mirror for hybrid memory recall (L3).

A *derived* ``bm25`` index over memory ``content``/``context``, co-located with
the LanceDB at ``<data_dir>/memory_fts.db``.  The daemon owns the LanceDB table,
so it owns this mirror: it is written on ``/store``, ``/delete``, ``/patch`` and
reconciled against LanceDB on startup.  ``SYSTEM`` memories are never indexed
(recall excludes them).

Append-only applies to the LanceDB *source* of truth; this index is rebuildable
and may be deleted from (same as the KG's external-content FTS).

Backed by a vendored peewee ``FTS5Model``.  The virtual-table DDL (which carries
the configurable tokenizer name — not bindable) is still emitted as a small,
allowlist-guarded ``CREATE`` via the bound connection.
"""

from __future__ import annotations

import contextlib
import re
import typing

import simba._vendor.peewee as pw
from simba._vendor.playhouse.sqlite_ext import FTS5Model, RowIDField, SearchField

FTS_FILENAME = "memory_fts.db"

_DEFAULT_TOKENIZE = "trigram"
# Tokenizer name is interpolated into DDL (can't be a bound parameter), so it is
# constrained to a known-safe allowlist.
_ALLOWED_TOKENIZE = frozenset({"trigram", "porter", "unicode61", "ascii"})
# Trigram MATCH needs >= 3-char terms; shorter terms are dropped (the vector arm
# covers those).
_MIN_TOKEN_LEN = 3
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+")

# Dedicated database for the keyword mirror (a separate file from simba.db),
# bound to a concrete path by ``connect()``.
_db = pw.SqliteDatabase(None)
_initialized: set[str] = set()


class MemoryFTS(FTS5Model):
    rowid = RowIDField()
    memory_id = SearchField(unindexed=True)
    project_path = SearchField(unindexed=True)
    type = SearchField(unindexed=True)
    confidence = SearchField(unindexed=True)
    created_at = SearchField(unindexed=True)
    content = SearchField()
    context = SearchField()

    class Meta:
        database = _db
        table_name = "memory_fts"


def schema_sql(tokenize: str = _DEFAULT_TOKENIZE) -> str:
    """Return the ``CREATE VIRTUAL TABLE`` DDL for the mirror."""
    tok = tokenize if tokenize in _ALLOWED_TOKENIZE else _DEFAULT_TOKENIZE
    return (
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5("
        "memory_id UNINDEXED, project_path UNINDEXED, type UNINDEXED, "
        "confidence UNINDEXED, created_at UNINDEXED, content, context, "
        f"tokenize='{tok}')"
    )


@contextlib.contextmanager
def connect(
    path: typing.Any, tokenize: str = _DEFAULT_TOKENIZE
) -> typing.Iterator[pw.SqliteDatabase]:
    """Bind the mirror DB to ``path`` and yield it (re-entrant, table ensured)."""
    p = str(path)
    if _db.database != p:
        if not _db.is_closed():
            _db.close()
        _db.init(p)
    with _db.connection_context():
        if p not in _initialized:
            _db.execute_sql(schema_sql(tokenize))
            _initialized.add(p)
        yield _db


def init(path: typing.Any, tokenize: str = _DEFAULT_TOKENIZE) -> None:
    """Create the mirror table if it does not exist."""
    with connect(path, tokenize):
        pass


def _insert(memory: dict[str, typing.Any]) -> bool:
    """Insert one memory row. Returns False if skipped (SYSTEM or no id)."""
    if memory.get("type") == "SYSTEM":
        return False
    mid = memory.get("id") or memory.get("memory_id")
    if not mid:
        return False
    MemoryFTS.insert(
        memory_id=mid,
        project_path=memory.get("projectPath", "") or "",
        type=memory.get("type", "") or "",
        confidence=float(memory.get("confidence", 0.0) or 0.0),
        created_at=memory.get("createdAt", "") or "",
        content=memory.get("content", "") or "",
        context=memory.get("context", "") or "",
    ).execute()
    return True


def upsert(memory: dict[str, typing.Any]) -> None:
    """Idempotently index a memory (delete-then-insert by ``memory_id``).

    ``SYSTEM`` memories and rows without an id are skipped.
    """
    mid = memory.get("id") or memory.get("memory_id")
    if not mid:
        return
    # DELETE-then-insert: also purges the row if the memory became SYSTEM
    # (then _insert is a no-op), keeping the mirror free of SYSTEM rows.
    MemoryFTS.delete().where(MemoryFTS.memory_id == mid).execute()
    _insert(memory)


def delete(memory_id: str) -> None:
    """Remove a memory from the mirror."""
    MemoryFTS.delete().where(MemoryFTS.memory_id == memory_id).execute()


def set_project(memory_id: str, project_path: str) -> None:
    """Update the scoping ``project_path`` for a memory (keeps /patch in sync)."""
    MemoryFTS.update(project_path=project_path or "").where(
        MemoryFTS.memory_id == memory_id
    ).execute()


def count() -> int:
    """Return the number of indexed rows."""
    return MemoryFTS.select().count()


def rebuild(memories: typing.Iterable[dict[str, typing.Any]]) -> int:
    """Replace the whole mirror from ``memories`` (skips SYSTEM). Returns count."""
    MemoryFTS.delete().execute()
    n = 0
    for m in memories:
        if _insert(m):
            n += 1
    return n


def _build_match(query: str, min_token_len: int = _MIN_TOKEN_LEN) -> str:
    """Build a safe FTS5 ``MATCH`` expression: OR of quoted literal terms.

    Quoting each term neutralizes FTS5 operators, so arbitrary user text can't
    produce a malformed expression from term content.  Returns ``""`` when no
    usable terms remain (caller treats that as "no keyword hits").
    """
    seen: set[str] = set()
    terms: list[str] = []
    for m in _TOKEN_RE.finditer(query or ""):
        tok = m.group()
        if len(tok) < min_token_len:
            continue
        low = tok.lower()
        if low in seen:
            continue
        seen.add(low)
        terms.append('"' + tok.replace('"', '""') + '"')
    return " OR ".join(terms)


def search(
    query: str,
    *,
    project_path: str | None = None,
    project_scopes: list[str] | None = None,
    include_global: bool = True,
    types: list[str] | None = None,
    limit: int = 20,
) -> list[dict[str, typing.Any]]:
    """Return up to ``limit`` bm25-ranked memories matching ``query``.

    Project scoping mirrors the vector arm (spec 26): when ``project_scopes`` (the
    client-computed cwd→git-root chain) is given, the keyword arm keeps memories
    whose ``project_path`` is one of those scopes — ancestor (root) facts inherit
    down — plus global (empty-path) memories when ``include_global``. Otherwise it
    falls back to the strict exact ``project_path`` match. ``types`` filters by
    memory type. Never raises: a malformed ``MATCH`` or missing table yields ``[]``.
    """
    match = _build_match(query)
    if not match:
        return []

    q = MemoryFTS.select().where(MemoryFTS.match(match))
    if project_scopes:
        # Hierarchical scope: project_path ∈ scopes, plus "" globals when included.
        scope_clause = MemoryFTS.project_path.in_(list(project_scopes))
        if include_global:
            scope_clause = scope_clause | (MemoryFTS.project_path == "")
        q = q.where(scope_clause)
    elif project_path:
        q = q.where(MemoryFTS.project_path == project_path)
    if types:
        q = q.where(MemoryFTS.type.in_(types))
    q = q.order_by(MemoryFTS.bm25()).limit(limit)

    try:
        rows = list(q)
    except Exception:
        return []

    results = []
    for r in rows:
        try:
            conf = float(r.confidence) if r.confidence not in (None, "") else 0.0
        except (TypeError, ValueError):
            conf = 0.0
        results.append(
            {
                "memory_id": r.memory_id,
                "type": r.type,
                "content": r.content,
                "context": r.context,
                "confidence": conf,
                "createdAt": r.created_at,
                "projectPath": r.project_path,
            }
        )
    return results
