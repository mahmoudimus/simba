"""Static gate: every direct in-process LanceDB bulk read MUST project columns.

Live incident (2026-07-18): a native ``sample`` taken mid-RSS-burst caught the
allocator inside ``pyarrow _Tabular.to_pylist -> ChunkedArray.to_pylist ->
ListScalar.as_py`` -- a LanceDB query materializing the 1024-dim ``vector``
column into ~10 million individually heap-allocated Python/Arrow float
scalars. The 2026-07-10 incident (docs/adr/2026-07-10-internal-api-footguns.md)
already closed this shape for self-HTTP ``/list`` calls (runtime gate +
``tests/test_internal_list_projection_lint.py``'s AST scan of HTTP call
sites) -- but a direct, in-process ``table.query().to_list()`` or
``table.vector_search(...).to_list()`` never goes through HTTP at all, so
that gate never saw it. Every real call site of this shape as of 2026-07-18
(``routes.py``'s ``/stats``, the hybrid-recall session-expansion/anticipated-
query record fetches, ``vector_db.py``'s duplicate-check/search/reembed/
access-tracking helpers, the FTS-mirror boot reconcile, ``/list``'s
neighboring ``/scopes/normalize``, ``/promotions/candidates``, ``/reindex``,
``/reembed``, and the cross-store reconcile audit) fetched EVERY column --
including the 1024-dim ``vector`` -- regardless of what the caller actually
read off each row, because Lance's select cost is paid server-side during the
query itself (see ``routes.py``'s ``_LIST_DEFAULT_FIELDS`` comment).

This test is the backstop for that class of bug, going forward: it AST-scans
every ``.py`` file under ``src/simba`` for a call chain that ends in
``.to_list()`` and originates (directly, or via a same-function variable
assignment -- including a chain rebuilt through reassignment, e.g. ``query =
query.where(...)``) from a ``.query(`` or ``.search(``-family builder (this
also matches ``vector_search``, LanceDB's ANN entry point) and fails,
file:line, on any such chain that never calls ``.select(``. A call chain with
no ``.query(``/``.search(``-family origin at all is out of scope (fail-open,
same philosophy as the sibling HTTP-layer lint) -- this scan only ever needs
to reason about the LanceDB builder shape actually used in this codebase.

This scan does NOT judge which columns end up inside ``.select(...)`` (that
requires reading each site's consumers by hand, done once for every real
2026-07-18 site -- see the per-site comments at each fixed call site); it
only enforces that SOME projection call is present in the chain, matching
this repo's one real remaining risk: a caller that forgets `.select(` entirely
and pays for every column, `vector` included, server-side.
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "simba"

# (path relative to the repo root, the `.to_list()` call's line number) for a
# reviewed call site that is genuinely exempt from the projection rule below.
# Add an entry here ONLY alongside a one-line justifying comment -- as of the
# 2026-07-18 pass every real call site in this codebase got a real, narrowed
# `.select(...)`, so this starts empty.
ALLOWLIST: set[tuple[str, int]] = set()


def _is_query_or_search_name(name: str) -> bool:
    """Method names that mark a call chain as originating from a LanceDB
    query/search builder. Exact ``"query"``, or anything containing
    ``"search"`` -- covers ``vector_search`` (LanceDB's ANN entry point)
    alongside a plain ``query()``."""
    return name == "query" or "search" in name.lower()


def _unwrap_await(node: ast.expr) -> ast.expr:
    """Strip a leading ``Await`` -- Python's grammar applies ``await`` to the
    WHOLE trailing call chain (``await x.y().z()`` parses as ``Await(Call(...
    Call(Name('x'))))``), never to an intermediate link, so one unwrap at the
    top is enough."""
    return node.value if isinstance(node, ast.Await) else node


def _chain_flags(
    expr: ast.expr, chains: dict[str, tuple[bool, bool]]
) -> tuple[bool, bool]:
    """Walk ``expr``'s method-call chain back to its base, returning
    ``(has_query_or_search, has_select)``.

    Each ``Call(func=Attribute(value=<inner>, attr=<name>))`` link
    contributes its own ``name`` before descending into ``<inner>``, so this
    sees every ``.foo(...)`` in the chain regardless of how deep
    ``.to_list()`` sits. If the chain bottoms out at a bare ``Name`` that
    ``chains`` already has flags for (a same-function preceding assignment,
    e.g. ``query = table.query()...`` read later as ``query.to_list()``, or
    a self-referential reassignment like ``query = query.where(...)``), that
    variable's flags are folded in too -- same-function only, no
    interprocedural resolution (matching the sibling HTTP-layer lint's
    stated scope).
    """
    has_query_or_search = False
    has_select = False
    cur = _unwrap_await(expr)
    while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
        name = cur.func.attr
        if _is_query_or_search_name(name):
            has_query_or_search = True
        if name == "select":
            has_select = True
        cur = cur.func.value
    if isinstance(cur, ast.Name) and cur.id in chains:
        prev_query_or_search, prev_select = chains[cur.id]
        has_query_or_search = has_query_or_search or prev_query_or_search
        has_select = has_select or prev_select
    return has_query_or_search, has_select


class _ToListScanner(ast.NodeVisitor):
    """Walks a module in source order, tracking same-function ``name =
    <call chain>`` assignments (fresh OR self-referential reassignment) so a
    query builder split across statements -- the exact shape ``/list``'s own
    handler and the hybrid-recall session-expansion helper both use -- still
    resolves at the eventual ``.to_list()`` call. Scoped per function (reset
    on every def) since the same local name (``query``, ``rows``, ...) is
    reused across unrelated helpers.
    """

    def __init__(self) -> None:
        self.violations: list[int] = []
        self._chains: dict[str, tuple[bool, bool]] = {}

    def _visit_scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        saved = self._chains
        self._chains = {}
        for stmt in node.body:
            self.visit(stmt)
        self._chains = saved

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped(node)

    def _track(self, target: ast.expr, value: ast.expr) -> None:
        if not isinstance(target, ast.Name):
            return
        self._chains[target.id] = _chain_flags(value, self._chains)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._track(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._track(node.target, node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "to_list"):
            return
        has_query_or_search, has_select = _chain_flags(node, self._chains)
        if has_query_or_search and not has_select:
            self.violations.append(node.lineno)


def _scan(path: pathlib.Path) -> _ToListScanner | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return None
    scanner = _ToListScanner()
    scanner.visit(tree)
    return scanner


def _find_violations() -> list[tuple[str, int]]:
    violations: list[tuple[str, int]] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        scanner = _scan(path)
        if scanner is None or not scanner.violations:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno in scanner.violations:
            if (rel, lineno) in ALLOWLIST:
                continue
            violations.append((rel, lineno))
    return violations


def test_lance_to_list_calls_are_projected() -> None:
    violations = _find_violations()
    assert not violations, (
        "a LanceDB `.query(`/`.search(`-family call chain reached `.to_list()` "
        "without ever calling `.select(` -- Lance's select cost is paid "
        "server-side during the query, so an unprojected read materializes "
        "every column (including the 1024-dim `vector`) for the WHOLE result "
        "set regardless of what the caller reads afterward (see the "
        "2026-07-18 RSS-burst incident, docs/adr/2026-07-10-internal-api-"
        "footguns.md). Offending call sites (file:line) -- add `.select([...])` "
        "narrowed to what the row's consumers actually read, or an explicit, "
        "commented ALLOWLIST entry in this test if the full row is genuinely "
        "required:\n"
        + "\n".join(f"  {file}:{line}" for file, line in sorted(violations))
    )


def _all_to_list_call_lines(path: pathlib.Path) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Attribute) and node.func.attr == "to_list"
        ):
            lines.add(node.lineno)
    return lines


def test_allowlist_entries_are_still_to_list_calls() -> None:
    """Guards the allowlist itself: an entry whose call site moved, was
    fixed to call `.select(`, or was deleted outright must be pruned here,
    not silently ignored (a stale entry would quietly narrow the scan)."""
    all_lines: dict[str, set[int]] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        try:
            rel = path.relative_to(REPO_ROOT).as_posix()
            all_lines[rel] = _all_to_list_call_lines(path)
        except (SyntaxError, UnicodeDecodeError):
            continue

    stale = sorted(
        key for key in ALLOWLIST if key[1] not in all_lines.get(key[0], set())
    )
    assert not stale, f"stale ALLOWLIST entries (call site moved/removed): {stale}"


def _scan_source(tmp_path: pathlib.Path, source: str) -> _ToListScanner:
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    scanner = _scan(path)
    assert scanner is not None
    return scanner


class TestScannerUnit:
    """Unit coverage for the scanner itself, over synthetic source -- proves
    each supported shape (inline chain, split-variable, reassignment-merged
    chain, non-Lance `.to_list()`) before trusting it over the real tree."""

    def test_inline_query_chain_without_select_is_flagged(
        self, tmp_path: pathlib.Path
    ) -> None:
        src = (
            "async def f(table):\n    return await table.query().where('x').to_list()\n"
        )
        scanner = _scan_source(tmp_path, src)
        assert scanner.violations == [2]

    def test_inline_query_chain_with_select_is_allowed(
        self, tmp_path: pathlib.Path
    ) -> None:
        src = (
            "async def f(table):\n"
            "    return await table.query().select(['id']).to_list()\n"
        )
        scanner = _scan_source(tmp_path, src)
        assert scanner.violations == []

    def test_vector_search_chain_without_select_is_flagged(
        self, tmp_path: pathlib.Path
    ) -> None:
        src = (
            "async def f(table, emb):\n"
            "    return await table.vector_search(emb).limit(5).to_list()\n"
        )
        scanner = _scan_source(tmp_path, src)
        assert scanner.violations == [2]

    def test_vector_search_chain_with_select_is_allowed(
        self, tmp_path: pathlib.Path
    ) -> None:
        src = (
            "async def f(table, emb):\n"
            "    return (\n"
            "        await table.vector_search(emb)\n"
            "        .column('vector')\n"
            "        .select(['id', 'type'])\n"
            "        .limit(5)\n"
            "        .to_list()\n"
            "    )\n"
        )
        scanner = _scan_source(tmp_path, src)
        assert scanner.violations == []

    def test_split_variable_without_select_is_flagged(
        self, tmp_path: pathlib.Path
    ) -> None:
        src = (
            "async def f(table, sid):\n"
            "    query = table.query().where(sid).limit(10)\n"
            "    rows = await query.to_list()\n"
            "    return rows\n"
        )
        scanner = _scan_source(tmp_path, src)
        assert scanner.violations == [3]

    def test_split_variable_with_select_is_allowed(
        self, tmp_path: pathlib.Path
    ) -> None:
        src = (
            "async def f(table, sid):\n"
            "    query = table.query().select(['id']).where(sid).limit(10)\n"
            "    rows = await query.to_list()\n"
            "    return rows\n"
        )
        scanner = _scan_source(tmp_path, src)
        assert scanner.violations == []

    def test_self_referential_reassignment_carries_select_forward(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Mirrors `/list`'s real shape: `query = table.query().select(...)`,
        then conditionally `query = query.where(...)` before `.to_list()`."""
        src = (
            "async def f(table, where_clause):\n"
            "    query = table.query().select(['id'])\n"
            "    if where_clause:\n"
            "        query = query.where(where_clause)\n"
            "    return await query.to_list()\n"
        )
        scanner = _scan_source(tmp_path, src)
        assert scanner.violations == []

    def test_self_referential_reassignment_without_select_is_flagged(
        self, tmp_path: pathlib.Path
    ) -> None:
        src = (
            "async def f(table, where_clause):\n"
            "    query = table.query()\n"
            "    if where_clause:\n"
            "        query = query.where(where_clause)\n"
            "    return await query.to_list()\n"
        )
        scanner = _scan_source(tmp_path, src)
        assert scanner.violations == [5]

    def test_non_lance_to_list_is_out_of_scope(self, tmp_path: pathlib.Path) -> None:
        """A `.to_list()` with no `.query(`/`.search(`-family origin at all
        (e.g. some unrelated object) is never this rule's business -- fail
        open, matching the sibling HTTP-layer lint's philosophy."""
        src = "def f(thing):\n    return thing.to_list()\n"
        scanner = _scan_source(tmp_path, src)
        assert scanner.violations == []

    def test_unrelated_variable_is_not_confused_with_tracked_chain(
        self, tmp_path: pathlib.Path
    ) -> None:
        src = (
            "async def f(table):\n"
            "    where_clause = table.query().select(['id'])\n"
            "    other = 1\n"
            "    return await other.to_list()\n"
        )
        scanner = _scan_source(tmp_path, src)
        assert scanner.violations == []

    def test_scope_is_reset_per_function(self, tmp_path: pathlib.Path) -> None:
        """A `query` tracked with `.select(` in one function must not leak
        into an unrelated function reusing the same local name without it."""
        src = (
            "async def good(table):\n"
            "    query = table.query().select(['id'])\n"
            "    return await query.to_list()\n"
            "\n"
            "async def bad(table):\n"
            "    query = table.query()\n"
            "    return await query.to_list()\n"
        )
        scanner = _scan_source(tmp_path, src)
        assert scanner.violations == [7]

    def test_allowlist_mechanism_filters_a_reviewed_site(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Exercises the ALLOWLIST filtering mechanism (the real constant
        stays empty -- see its module-level comment) against a fabricated
        violation, without depending on any real exception ever existing."""
        raw_violation = (str(tmp_path / "mod.py"), 2)
        src = "async def f(table):\n    return await table.query().to_list()\n"
        scanner = _scan_source(tmp_path, src)
        assert scanner.violations == [2]

        fake_allowlist = {raw_violation}
        filtered = [
            lineno
            for lineno in scanner.violations
            if (str(tmp_path / "mod.py"), lineno) not in fake_allowlist
        ]
        assert filtered == []
