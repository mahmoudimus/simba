"""Static gate: every internal self-HTTP ``/list`` call must project fields=.

Live incident (2026-07-10): daemon-internal callers (maintenance/hygiene/
decay/consolidation/reflection passes) fetched ``GET /list`` with no column
projection, materializing the whole corpus server-side --- including every
row's 1024-dim ``vector`` --- for a 45GB peak footprint (Lance's select cost
is paid server-side regardless of what the caller reads from the JSON
afterward; see routes.py's ``_LIST_DEFAULT_FIELDS`` comment). PR #90 fixed
every internal call site that existed then (maintenance.py, hygiene.py,
decay.py, sync/extractor.py, episodes/consolidate.py, reflection/pass_.py,
hooks/_memory_client.py) by adding an explicit ``fields=`` projection --- but
nothing stopped the NEXT contributor from adding an unprojected one. This
test is that backstop: it scans every ``.py`` file under ``src/simba`` for
the calling convention those fixes share and fails, file:line, on any call
that doesn't also narrow via ``fields=``.

Scope: an httpx-style ``<something>.get(url, ...)`` call whose URL is (or
resolves, via a simple same-function preceding assignment, to) a string or
f-string containing ``/list``. This mirrors the shape of every real
internal call site (see the modules listed above) without requiring a full
type-aware analysis. Human-facing CLI commands (``simba memory list``,
``simba memory prune``, ``simba eval build``, ``simba rules list``, ``simba
rules prune``) hit the same endpoint but are explicitly out of this rule's
scope --- see the runtime gate's "External/CLI/plain clients unaffected"
carve-out in routes.py's ``/list`` handler docstring --- and are allowlisted
below with a comment rather than silently excluded by path, so the scan
still covers every file in the tree.
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "simba"

# (path relative to the repo root, the `.get(...)` call's line number) for
# every call site reviewed and confirmed to be a human-facing CLI command,
# never an automated daemon-internal pass. Add an entry here ONLY alongside a
# comment justifying it --- every other call site MUST pass fields=.
ALLOWLIST: set[tuple[str, int]] = {
    # `simba memory list`: interactive display, default limit=20 (not a
    # corpus-wide scan).
    ("src/simba/__main__.py", 2134),
    # `simba memory prune`: human-invoked CLI, not an automated daemon pass.
    ("src/simba/__main__.py", 2884),
    # `simba eval build`: human-invoked CLI; needs real content/context to
    # build eval cases.
    ("src/simba/__main__.py", 3879),
    # `simba rules list`: human-facing CLI, type-filtered + limit=50.
    ("src/simba/rules_cli.py", 104),
    # `simba rules prune`: human-facing CLI, type-filtered + limit=1000.
    ("src/simba/rules_cli.py", 168),
}


def _literal_text(node: ast.expr) -> str | None:
    """Literal text of a string constant or f-string (``JoinedStr``) --- for
    an f-string, only the LITERAL (non-interpolated) pieces are concatenated,
    e.g. ``f"{daemon_url}/list"`` -> ``"/list"``. ``None`` for anything else.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            piece.value
            for piece in node.values
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str)
        )
    return None


def _dict_keys(node: ast.expr) -> set[str] | None:
    """String keys of a dict literal, or ``None`` if not a dict literal."""
    if not isinstance(node, ast.Dict):
        return None
    keys: set[str] = set()
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
    return keys


def _is_http_get_call(call: ast.Call) -> bool:
    """``<anything>.get(...)`` --- the calling convention every internal
    ``/list`` helper uses (``httpx.get(...)``, or a pre-configured ``httpx.
    Client``'s ``client.get(...)``). Matched on the attribute name alone
    (not import-resolved) --- see the module docstring for why that's
    enough in practice; ``visit_FunctionDef``/``visit_AsyncFunctionDef``
    below skip ``@router.get(...)`` route-registration decorators, the one
    real false-positive shape in this codebase.
    """
    return isinstance(call.func, ast.Attribute) and call.func.attr == "get"


class _ListCallScanner(ast.NodeVisitor):
    """Walks a module in source order, tracking simple same-function ``name
    = <literal>`` assignments (string/f-string AND dict-literal alike) so a
    URL or a ``params`` dict built one statement above the call still
    resolves --- not just the fully-inline ``httpx.get(f"...",
    params={...})`` shape. Scoped per function (reset on every def, and
    decorators are skipped) since the same local name (``url``, ``params``,
    ...) is reused across unrelated helpers, and a route decorator isn't a
    call site at all.
    """

    def __init__(self) -> None:
        self.violations: list[int] = []
        self._strings: dict[str, str] = {}
        self._dicts: dict[str, set[str]] = {}

    def _visit_scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        saved = (self._strings, self._dicts)
        self._strings, self._dicts = {}, {}
        # Body + parameter defaults only --- NOT decorator_list: a route
        # decorator like `@router.get("/list")` matches `.get(...)` too, but
        # it registers the endpoint, it never calls it.
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        for stmt in node.body:
            self.visit(stmt)
        self._strings, self._dicts = saved

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped(node)

    def _track(self, target: ast.expr, value: ast.expr) -> None:
        if not isinstance(target, ast.Name):
            return
        text = _literal_text(value)
        keys = _dict_keys(value)
        if text is not None:
            self._strings[target.id] = text
        if keys is not None:
            self._dicts[target.id] = keys

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._track(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._track(node.target, node.value)
        self.generic_visit(node)

    def _resolved_text(self, node: ast.expr) -> str | None:
        text = _literal_text(node)
        if text is not None:
            return text
        if isinstance(node, ast.Name):
            return self._strings.get(node.id)
        return None

    def _has_fields(self, call: ast.Call) -> bool:
        for kw in call.keywords:
            if kw.arg == "fields":
                return True
            if kw.arg == "params":
                keys = _dict_keys(kw.value)
                if keys is None and isinstance(kw.value, ast.Name):
                    keys = self._dicts.get(kw.value.id)
                if keys and "fields" in keys:
                    return True
        return False

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        if not (_is_http_get_call(node) and node.args):
            return
        text = self._resolved_text(node.args[0])
        if text is not None and "/list" in text and not self._has_fields(node):
            self.violations.append(node.lineno)


def _scan(path: pathlib.Path) -> _ListCallScanner | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return None
    scanner = _ListCallScanner()
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


def test_internal_list_calls_project_fields() -> None:
    violations = _find_violations()
    assert not violations, (
        "internal /list callers must pass fields= projection -- an "
        "unprojected GET /list materializes the whole corpus server-side, "
        "including the 1024-dim `vector` column if `include_vectors` is "
        "ever set (see routes.py's `_LIST_DEFAULT_FIELDS` comment and the "
        "2026-07-10 incident). Offending call sites (file:line) -- add "
        "fields=, or an explicit, commented ALLOWLIST entry in this test "
        "if the full row is genuinely required:\n"
        + "\n".join(f"  {file}:{line}" for file, line in sorted(violations))
    )


def test_allowlist_entries_are_still_get_calls() -> None:
    """Guards the allowlist itself: an entry whose call site moved, was
    fixed to pass fields=, or was deleted outright must be pruned here, not
    silently ignored (a stale entry would quietly narrow the scan)."""
    all_get_call_lines: dict[str, set[int]] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        scanner = _scan(path)
        if scanner is None:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        lines: set[int] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and _is_http_get_call(node) and node.args:
                lines.add(node.lineno)
        all_get_call_lines[rel] = lines

    stale = sorted(
        key for key in ALLOWLIST if key[1] not in all_get_call_lines.get(key[0], set())
    )
    assert not stale, f"stale ALLOWLIST entries (call site moved/removed): {stale}"
