"""Shared SQLite database for all simba subsystems.

Single database at .simba/simba.db. Each subsystem owns its tables
but shares the connection. Schema is initialized lazily on first connect.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
import sqlite3
import uuid
from typing import TYPE_CHECKING

import simba.config

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

_SCHEMA_INITIALIZERS: list[Callable[[sqlite3.Connection], None]] = []


@simba.config.configurable("project")
@dataclasses.dataclass
class ProjectConfig:
    """Project identity used to scope facts/memories to one repo.

    ``project_id`` is a stable, opaque token written into the *local*
    ``.simba/config.toml``.  Because that file lives inside the repo, it
    travels with a plain ``mv`` of the folder, so the id survives a move.
    Override it with ``simba config set project.project_id <id>`` (e.g. to
    share an identity across working copies, or after wiping ``.simba/``).
    """

    project_id: str = ""


def register_schema(init_fn: Callable[[sqlite3.Connection], None]) -> None:
    """Register a schema initializer.

    Each subsystem calls this at module level to register its
    ``CREATE TABLE IF NOT EXISTS`` statements. All registered
    initializers run on the first ``get_db()`` call.
    """
    _SCHEMA_INITIALIZERS.append(init_fn)


def find_repo_root(cwd: pathlib.Path) -> pathlib.Path | None:
    """Walk up from *cwd* looking for a ``.git`` directory.

    Returns the repo root path, or ``None`` if not found.
    """
    current = cwd.resolve()
    while True:
        if (current / ".git").is_dir():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def get_db_path(cwd: pathlib.Path | None = None) -> pathlib.Path:
    """Return the path to ``.simba/simba.db``.

    Uses the repository root if one is found, otherwise falls back
    to *cwd* (or ``Path.cwd()``).
    """
    if cwd is None:
        cwd = pathlib.Path.cwd()
    root = find_repo_root(cwd)
    base = root if root is not None else cwd
    return base / ".simba" / "simba.db"


def _init_schemas(conn: sqlite3.Connection) -> None:
    """Run all registered schema initializers."""
    for init_fn in _SCHEMA_INITIALIZERS:
        init_fn(conn)


@contextlib.contextmanager
def get_db(cwd: pathlib.Path | None = None) -> Generator[sqlite3.Connection]:
    """Yield a connection to ``simba.db``, creating schema if needed.

    The connection is closed when the context manager exits.
    """
    db_path = get_db_path(cwd)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        _init_schemas(conn)
        yield conn
    finally:
        conn.close()


def resolve_project_id(cwd: pathlib.Path | None = None) -> str:
    """Return the stable project id for the repo containing *cwd*.

    Reads ``project.project_id`` from config (scoped to the repo root that
    owns the ``.simba/`` DB).  If unset, generates a uuid and persists it to
    the *local* config so it stays stable across folder moves.  Persisting is
    best-effort: a read-only filesystem never breaks fact lookup.
    """
    root = get_db_path(cwd).parent.parent
    cfg = simba.config.load("project", root=root)
    if cfg.project_id:
        return cfg.project_id

    new_id = uuid.uuid4().hex
    with contextlib.suppress(Exception):
        simba.config.set_value(
            "project", "project_id", new_id, scope="local", root=root
        )
    return new_id


def get_connection(cwd: pathlib.Path | None = None) -> sqlite3.Connection | None:
    """Open a connection to ``simba.db`` if the file exists.

    Returns ``None`` when the database file has not been created yet.
    The caller is responsible for closing the connection.
    """
    db_path = get_db_path(cwd)
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _init_schemas(conn)
    return conn
