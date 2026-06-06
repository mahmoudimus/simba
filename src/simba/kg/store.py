"""Temporal knowledge-graph store backed by ``kg_edges`` + FTS5/bm25.

Edges are bitemporal-lite: each row carries ``valid_from`` (set on insert) and
``valid_to`` (``NULL`` while the fact holds, stamped when invalidated).  This
lets ``kg_query`` return the *currently* valid graph by default, or the graph
"as of" an arbitrary timestamp.  An external-content FTS5 table mirrors the
``subject``/``predicate``/``object`` columns (trigram tokenizer) so substring
recall and bm25 ranking come for free.  Edges are scoped per-project via
``project_path`` to prevent cross-project leakage.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
import typing

import simba._vendor.peewee as pw
import simba.db
from simba._vendor.playhouse.sqlite_ext import FTS5Model, RowIDField, SearchField

_SCHEMA_BASE_SQL = """\
CREATE TABLE IF NOT EXISTS kg_edges (
    id INTEGER PRIMARY KEY,
    subject TEXT, predicate TEXT, object TEXT,
    subject_type TEXT, object_type TEXT,
    proof TEXT, transcript_id TEXT, char_start INTEGER,
    valid_from TEXT, valid_to TEXT, occurred_at TEXT,
    project_path TEXT NOT NULL, created_at TEXT,
    dormant INTEGER DEFAULT 0,
    UNIQUE(subject, predicate, object, project_path, valid_from)
);

CREATE INDEX IF NOT EXISTS idx_kg_edges_subject
    ON kg_edges(subject, project_path);
CREATE INDEX IF NOT EXISTS idx_kg_edges_project
    ON kg_edges(project_path);
"""

_SCHEMA_FTS_SQL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS kg_edges_fts USING fts5(
    subject, predicate, object,
    content='kg_edges', content_rowid='id', tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS kg_edges_ai AFTER INSERT ON kg_edges BEGIN
    INSERT INTO kg_edges_fts(rowid, subject, predicate, object)
    VALUES (NEW.id, NEW.subject, NEW.predicate, NEW.object);
END;

CREATE TRIGGER IF NOT EXISTS kg_edges_ad AFTER DELETE ON kg_edges BEGIN
    INSERT INTO kg_edges_fts(kg_edges_fts, rowid, subject, predicate, object)
    VALUES ('delete', OLD.id, OLD.subject, OLD.predicate, OLD.object);
END;

CREATE TRIGGER IF NOT EXISTS kg_edges_au AFTER UPDATE ON kg_edges BEGIN
    INSERT INTO kg_edges_fts(kg_edges_fts, rowid, subject, predicate, object)
    VALUES ('delete', OLD.id, OLD.subject, OLD.predicate, OLD.object);
    INSERT INTO kg_edges_fts(rowid, subject, predicate, object)
    VALUES (NEW.id, NEW.subject, NEW.predicate, NEW.object);
END;
"""


def backup_and_drop_proven_facts(conn: sqlite3.Connection) -> None:
    """Retire the legacy ``proven_facts`` table, preserving its rows once.

    The temporal ``kg_edges`` store supersedes ``proven_facts``.  On first
    connect we rename any surviving ``proven_facts`` to ``proven_facts_bak``
    (dropping any prior ``proven_facts_bak`` first) so its rows stay available
    for later re-analysis, and drop the even-older ``proven_facts_legacy`` if
    present.  Idempotent: a no-op once ``proven_facts`` is gone.
    """
    has_proven = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='proven_facts'"
    ).fetchone()
    if has_proven:
        conn.execute("DROP TABLE IF EXISTS proven_facts_bak")
        conn.execute("ALTER TABLE proven_facts RENAME TO proven_facts_bak")
    conn.execute("DROP TABLE IF EXISTS proven_facts_legacy")
    conn.commit()


def _migrate_occurred_at(conn: sqlite3.Connection) -> None:
    """Add the bitemporal ``occurred_at`` (event time) column to a legacy table.

    Idempotent: a no-op once the column exists.  New databases get the column
    from ``_SCHEMA_BASE_SQL`` directly; this backfills pre-existing ones.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(kg_edges)")}
    if "occurred_at" not in cols:
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("ALTER TABLE kg_edges ADD COLUMN occurred_at TEXT")


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create ``kg_edges`` plus its FTS5 mirror and sync triggers.

    Core table/indexes are always created; the FTS5 virtual table and its
    triggers are installed only when the SQLite build supports FTS5 (otherwise
    ``kg_query`` falls back to non-FTS filtering).  The legacy ``proven_facts``
    table is retired here so the migration runs once on first connect.
    """
    conn.executescript(_SCHEMA_BASE_SQL)
    _migrate_occurred_at(conn)
    with contextlib.suppress(sqlite3.OperationalError):
        conn.executescript(_SCHEMA_FTS_SQL)
    backup_and_drop_proven_facts(conn)


simba.db.register_schema(_init_schema)


class KgEdge(simba.db.BaseModel):
    # Maps the kg_edges table (created by the raw _init_schema above, which also
    # owns the external-content FTS5 mirror + sync triggers peewee can't model).
    subject = pw.TextField(null=True)
    predicate = pw.TextField(null=True)
    object = pw.TextField(null=True)
    subject_type = pw.TextField(null=True)
    object_type = pw.TextField(null=True)
    proof = pw.TextField(null=True)
    transcript_id = pw.TextField(null=True)
    char_start = pw.IntegerField(null=True)
    valid_from = pw.TextField(null=True)  # belief time: when recorded
    valid_to = pw.TextField(null=True)  # belief time: when retracted (NULL=open)
    occurred_at = pw.TextField(null=True)  # event time: when it was true in world
    project_path = pw.TextField()
    created_at = pw.TextField(null=True)
    # AGM-retraction marker (Phase 7). Column added by simba.neuron.schema's
    # idempotent migration; defaults to 0 (active). Declared here so peewee can
    # read/filter it. Nullable so legacy rows (pre-migration) load cleanly.
    dormant = pw.IntegerField(null=True, default=0)

    class Meta:
        table_name = "kg_edges"


class KgEdgeFTS(FTS5Model):
    # Query-only view of the external-content FTS mirror (creation + sync stay
    # in the raw DDL/triggers).  Used for MATCH + bm25 ranking in kg_query.
    rowid = RowIDField()
    subject = SearchField()
    predicate = SearchField()
    object = SearchField()

    class Meta:
        database = simba.db.database
        table_name = "kg_edges_fts"


def _now() -> str:
    """Return the current UTC time as an ISO-8601 ``Z`` timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _project_entities(project_path: str) -> set[str]:
    """Return the distinct subject/object surface forms already in the project."""
    names: set[str] = set()
    rows = KgEdge.select(KgEdge.subject, KgEdge.object).where(
        KgEdge.project_path == project_path
    )
    for r in rows:
        if r.subject:
            names.add(r.subject)
        if r.object:
            names.add(r.object)
    return names


def _canonicalize(subject: str, object: str, project_path: str) -> tuple[str, str]:
    """Resolve subject/object to canonical entities when enabled (else passthrough).

    Project-scoped: a variant only collapses to a canonical form already present
    in the *same* project, so cross-project nodes never merge. Normalization-only
    here (no embedder dependency in the store); synonym merging via embeddings is
    available through ``entities.resolve(embed=...)`` for callers that have one.
    """
    import simba.config
    import simba.kg.config  # registers the "kg" section
    import simba.kg.entities

    _ = simba.kg.config
    cfg = simba.config.load("kg")
    if not getattr(cfg, "entity_resolution_enabled", False):
        return subject, object

    existing = _project_entities(project_path)
    canon_subject = simba.kg.entities.resolve(subject, existing)
    # Let the object also resolve against the just-canonicalized subject.
    canon_object = simba.kg.entities.resolve(object, existing | {canon_subject})
    return canon_subject, canon_object


def _apply_temporal(
    q: typing.Any,
    *,
    as_of: str | None,
    include_expired: bool,
    occurred_after: str | None,
    occurred_before: str | None,
) -> typing.Any:
    """Apply the belief-time + event-time filters shared by query/traversal."""
    if as_of is not None:
        q = q.where(KgEdge.valid_from <= as_of).where(
            KgEdge.valid_to.is_null() | (as_of < KgEdge.valid_to)
        )
    elif not include_expired:
        q = q.where(KgEdge.valid_to.is_null())
    if occurred_after is not None:
        q = q.where(KgEdge.occurred_at >= occurred_after)
    if occurred_before is not None:
        q = q.where(KgEdge.occurred_at <= occurred_before)
    return q


def _row_to_dict(edge: KgEdge) -> dict[str, object]:
    """Project a ``KgEdge`` row into a stable public dict."""
    return {
        "id": edge.id,
        "subject": edge.subject,
        "predicate": edge.predicate,
        "object": edge.object,
        "subject_type": edge.subject_type,
        "object_type": edge.object_type,
        "proof": edge.proof,
        "transcript_id": edge.transcript_id,
        "char_start": edge.char_start,
        "valid_from": edge.valid_from,
        "valid_to": edge.valid_to,
        "occurred_at": edge.occurred_at,
    }


def kg_add(
    subject: str,
    predicate: str,
    object: str,
    proof: str,
    *,
    subject_type: str = "concept",
    object_type: str = "concept",
    transcript_id: str | None = None,
    char_start: int | None = None,
    project_path: str | None = None,
    occurred_at: str | None = None,
) -> str:
    """Insert an *open* edge (``valid_from`` = now, ``valid_to`` = NULL).

    ``valid_from`` is *belief* time (when we recorded the fact); ``occurred_at``
    is *event* time (when the fact was true in the world) — pass it when the
    narrative date is known, else leave ``None`` (unspecified).

    Returns ``"added"`` on success, or ``"exists"`` when an edge with the same
    ``(subject, predicate, object, project_path, valid_from)`` already exists
    (UNIQUE collision).  ``project_path`` defaults to the current repo's
    stable project id.
    """
    if project_path is None:
        project_path = simba.db.resolve_project_id()
    now = _now()
    with simba.db.connect():
        subject, object = _canonicalize(subject, object, project_path)
        try:
            KgEdge.create(
                subject=subject,
                predicate=predicate,
                object=object,
                subject_type=subject_type,
                object_type=object_type,
                proof=proof,
                transcript_id=transcript_id,
                char_start=char_start,
                valid_from=now,
                valid_to=None,
                occurred_at=occurred_at,
                project_path=project_path,
                created_at=now,
            )
            return "added"
        except pw.IntegrityError:
            return "exists"


def kg_invalidate(
    subject: str,
    predicate: str,
    object: str,
    *,
    project_path: str | None = None,
) -> int:
    """Close every matching *open* edge by stamping ``valid_to`` = now.

    Returns the number of edges closed.  ``project_path`` defaults to the
    current repo's stable project id.
    """
    if project_path is None:
        project_path = simba.db.resolve_project_id()
    with simba.db.connect():
        return (
            KgEdge.update(valid_to=_now())
            .where(
                (KgEdge.subject == subject)
                & (KgEdge.predicate == predicate)
                & (KgEdge.object == object)
                & (KgEdge.project_path == project_path)
                & (KgEdge.valid_to.is_null())
            )
            .execute()
        )


def kg_neighbors(
    entity: str,
    *,
    project_path: str | None = None,
    depth: int = 1,
    direction: str = "both",
    as_of: str | None = None,
    include_expired: bool = False,
    occurred_after: str | None = None,
    occurred_before: str | None = None,
    max_edges: int | None = None,
) -> list[dict[str, object]]:
    """Breadth-first traversal of the graph outward from ``entity``.

    Returns the edges reachable within ``depth`` hops, each annotated with its
    1-based ``hop`` distance. ``direction`` follows ``out`` (subject→object),
    ``in`` (object→subject), or ``both``. Bitemporal/event-time filters apply at
    every hop (so a retracted edge cuts off the paths beyond it), results are
    scoped to ``project_path`` when given (``None`` = all projects), and the
    crawl is bounded by ``max_edges`` (defaults to ``kg.max_neighbor_edges``).
    """
    if max_edges is None:
        import simba.kg.config as _kgcfg  # registers "kg"; aliased to not shadow simba
        from simba.config import load as _load_section

        _ = _kgcfg
        max_edges = _load_section("kg").max_neighbor_edges

    collected: dict[int, dict[str, object]] = {}
    visited: set[str] = {entity}
    frontier: set[str] = {entity}

    with simba.db.connect():
        for hop in range(1, depth + 1):
            if not frontier or len(collected) >= max_edges:
                break
            names = list(frontier)
            if direction == "out":
                cond = KgEdge.subject.in_(names)
            elif direction == "in":
                cond = KgEdge.object.in_(names)
            else:
                cond = KgEdge.subject.in_(names) | KgEdge.object.in_(names)

            q = KgEdge.select().where(cond)
            if project_path:
                q = q.where(KgEdge.project_path == project_path)
            q = _apply_temporal(
                q,
                as_of=as_of,
                include_expired=include_expired,
                occurred_after=occurred_after,
                occurred_before=occurred_before,
            )

            new_frontier: set[str] = set()
            for edge in q:
                if edge.id not in collected:
                    row = _row_to_dict(edge)
                    row["hop"] = hop
                    collected[edge.id] = row
                    if len(collected) >= max_edges:
                        break
                for end in (edge.subject, edge.object):
                    if end and end not in visited:
                        new_frontier.add(end)
            visited |= new_frontier
            frontier = new_frontier

    return list(collected.values())


def kg_density(project_path: str | None = None) -> dict[str, float]:
    """Return graph density metrics for a project (Phase 7 progress metric).

    ``density`` is the directed-graph edge density ``edges / (n*(n-1))`` over the
    distinct subject/object node set, clamped to ``[0, 1]`` and ``0`` for graphs
    with fewer than two nodes. ``derived_ratio`` is the share of derived edges
    among all edges. Counts only currently-valid, non-dormant base edges.
    """
    edge_count = 0
    derived_edge_count = 0
    nodes: set[str] = set()
    with simba.db.connect() as db:
        q = KgEdge.select().where(KgEdge.valid_to.is_null())
        if project_path:
            q = q.where(KgEdge.project_path == project_path)
        for edge in q:
            if getattr(edge, "dormant", 0):
                continue
            edge_count += 1
            if edge.subject:
                nodes.add(edge.subject)
            if edge.object:
                nodes.add(edge.object)

        try:
            if project_path:
                row = db.execute_sql(
                    "SELECT COUNT(*) FROM kg_derived_edges "
                    "WHERE project_path=? AND valid_to IS NULL",
                    (project_path,),
                ).fetchone()
            else:
                row = db.execute_sql(
                    "SELECT COUNT(*) FROM kg_derived_edges WHERE valid_to IS NULL"
                ).fetchone()
            derived_edge_count = int(row[0]) if row else 0
        except Exception:
            derived_edge_count = 0

    node_count = len(nodes)
    density = edge_count / (node_count * (node_count - 1)) if node_count >= 2 else 0.0
    density = max(0.0, min(1.0, density))

    total = edge_count + derived_edge_count
    derived_ratio = derived_edge_count / total if total > 0 else 0.0

    return {
        "edge_count": edge_count,
        "derived_edge_count": derived_edge_count,
        "node_count": node_count,
        "density": density,
        "derived_ratio": derived_ratio,
    }


def kg_query(
    query: str | None = None,
    subject: str | None = None,
    predicate: str | None = None,
    *,
    project_path: str | None = None,
    as_of: str | None = None,
    include_expired: bool = False,
    occurred_after: str | None = None,
    occurred_before: str | None = None,
    limit: int = 10,
    expand_hops: int = 0,
) -> list[dict[str, object]]:
    """Query the knowledge graph, with FTS/bm25 ranking and bitemporal filters.

    When *query* is set, results are matched against the trigram FTS index and
    ordered by ``bm25``; otherwise rows are filtered by *subject*/*predicate*.
    *project_path* scopes results when given.

    Two independent time axes:

    - **Belief time** (``valid_from``/``valid_to``) — when the fact was on
      record.  Unless *include_expired*, only currently-valid edges
      (``valid_to IS NULL``) are returned; *as_of* snapshots it
      (``valid_from <= as_of AND (valid_to IS NULL OR as_of < valid_to)``).
    - **Event time** (``occurred_at``) — when the fact was true in the world.
      *occurred_after*/*occurred_before* bound it (inclusive); edges with an
      unknown (``NULL``) ``occurred_at`` are excluded once either bound is set.

    Returns a list of row dicts.  A malformed FTS ``MATCH`` expression is
    swallowed and yields ``[]``.
    """
    with simba.db.connect():
        if query:
            q = (
                KgEdge.select()
                .join(KgEdgeFTS, on=(KgEdgeFTS.rowid == KgEdge.id))
                .where(KgEdgeFTS.match(query))
            )
        else:
            q = KgEdge.select()
            if subject:
                q = q.where(KgEdge.subject == subject)
            if predicate:
                q = q.where(KgEdge.predicate == predicate)

        if project_path:
            q = q.where(KgEdge.project_path == project_path)

        q = _apply_temporal(
            q,
            as_of=as_of,
            include_expired=include_expired,
            occurred_after=occurred_after,
            occurred_before=occurred_before,
        )

        if query:
            q = q.order_by(KgEdgeFTS.bm25())
        q = q.limit(limit)

        try:
            rows = list(q)
        except Exception:
            return []

        result = [_row_to_dict(row) for row in rows]

        # Multi-hop expansion: walk out from the seed edges' entities so a query
        # returns the connected subgraph, not just the directly-matched edges.
        if expand_hops > 0:
            seen = {r["id"] for r in result}
            seeds: set[str] = set()
            for r in result:
                for end in (r["subject"], r["object"]):
                    if end:
                        seeds.add(str(end))
            for ent in seeds:
                for nb in kg_neighbors(
                    ent,
                    project_path=project_path,
                    depth=expand_hops,
                    as_of=as_of,
                    include_expired=include_expired,
                    occurred_after=occurred_after,
                    occurred_before=occurred_before,
                ):
                    if nb["id"] not in seen:
                        seen.add(nb["id"])
                        result.append(nb)
    return result
