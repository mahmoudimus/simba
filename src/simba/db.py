"""Shared SQLite database for all simba subsystems.

Single database at .simba/simba.db. Each subsystem owns its tables
but shares the connection. Schema is initialized lazily on first connect.
"""

from __future__ import annotations

import contextlib
import pathlib
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

_SCHEMA_INITIALIZERS: list[Callable[[sqlite3.Connection], None]] = []


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
