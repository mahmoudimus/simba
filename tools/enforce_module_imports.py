#!/usr/bin/env python3
"""Enforce module-level imports: `import X` over `from X import Y`.

Usage:
    python tools/enforce_module_imports.py --check src/ tests/
    python tools/enforce_module_imports.py --check file.py
    python tools/enforce_module_imports.py --fix file.py

Exemptions:
    - `from __future__ import ...`
    - `from typing import TYPE_CHECKING`
    - Any import inside `if TYPE_CHECKING:` blocks
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import libcst


class _InTypeCheckingBlock(libcst.CSTVisitor):
    """Track whether we're inside an `if TYPE_CHECKING:` block."""

    def __init__(self) -> None:
        self.in_type_checking = False
        self._stack: list[bool] = []

    def visit_If(self, node: libcst.If) -> bool:
        is_tc = False
        if isinstance(node.test, libcst.Name) and node.test.value == "TYPE_CHECKING":
            is_tc = True
        self._stack.append(is_tc)
        if is_tc:
            self.in_type_checking = True
        return True

    def leave_If(self, original_node: libcst.If) -> None:
        was_tc = self._stack.pop()
        if was_tc:
            self.in_type_checking = False


class ImportViolationChecker(libcst.CSTVisitor):
    """Find all `from X import Y` violations."""

    METADATA_DEPENDENCIES = (libcst.metadata.PositionProvider,)

    def __init__(self) -> None:
        super().__init__()
        self.violations: list[tuple[int, str, str]] = []
        self._tc_tracker = _InTypeCheckingBlock()
        self._tc_stack: list[bool] = []

    def visit_If(self, node: libcst.If) -> bool:
        self._tc_tracker.visit_If(node)
        return True

    def leave_If(self, original_node: libcst.If) -> None:
        self._tc_tracker.leave_If(original_node)

    def visit_ImportFrom(self, node: libcst.ImportFrom) -> None:
        if self._tc_tracker.in_type_checking:
            return

        if isinstance(node.module, libcst.Attribute | libcst.Name):
            module_name = _get_module_name(node.module)
        else:
            return

        if module_name == "__future__":
            return

        if module_name == "typing" and isinstance(node.names, (list, tuple)):
            names = [
                n.name.value
                for n in node.names
                if isinstance(n, libcst.ImportAlias) and isinstance(n.name, libcst.Name)
            ]
            if names == ["TYPE_CHECKING"]:
                return

        pos = self.metadata[libcst.metadata.PositionProvider][node]  # pyright: ignore[reportUnknownMemberType]
        line: int = pos.start.line  # type: ignore[union-attr]

        if isinstance(node.names, (list, tuple)):
            imported = ", ".join(
                n.name.value
                for n in node.names
                if isinstance(n, libcst.ImportAlias) and isinstance(n.name, libcst.Name)
            )
        else:
            imported = "*"

        self.violations.append((line, module_name, imported))


class ImportFixer(libcst.CSTTransformer):
    """Transform `from X import Y` to `import X` and rewrite references."""

    METADATA_DEPENDENCIES = (libcst.metadata.PositionProvider,)

    def __init__(self) -> None:
        super().__init__()
        self._renames: dict[str, str] = {}
        self._new_imports: set[str] = set()
        self._existing_imports: set[str] = set()
        self._tc_tracker = _InTypeCheckingBlock()

    def visit_If(self, node: libcst.If) -> bool:
        self._tc_tracker.visit_If(node)
        return True

    def leave_If(self, original_node: libcst.If, updated_node: libcst.If) -> libcst.If:
        self._tc_tracker.leave_If(original_node)
        return updated_node

    def visit_Import(self, node: libcst.Import) -> None:
        if isinstance(node.names, (list, tuple)):
            for alias in node.names:
                if isinstance(alias, libcst.ImportAlias) and isinstance(
                    alias.name, libcst.Name | libcst.Attribute
                ):
                    self._existing_imports.add(_get_module_name(alias.name))

    def leave_ImportFrom(
        self, original_node: libcst.ImportFrom, updated_node: libcst.ImportFrom
    ) -> libcst.ImportFrom | libcst.RemovalSentinel:
        if self._tc_tracker.in_type_checking:
            return updated_node

        if not isinstance(updated_node.module, libcst.Attribute | libcst.Name):
            return updated_node

        module_name = _get_module_name(updated_node.module)

        if module_name == "__future__":
            return updated_node

        if module_name == "typing" and isinstance(updated_node.names, (list, tuple)):
            names = [
                n.name.value
                for n in updated_node.names
                if isinstance(n, libcst.ImportAlias) and isinstance(n.name, libcst.Name)
            ]
            if names == ["TYPE_CHECKING"]:
                return updated_node

        if not isinstance(updated_node.names, (list, tuple)):
            return updated_node

        for alias in updated_node.names:
            if not isinstance(alias, libcst.ImportAlias):
                continue
            if not isinstance(alias.name, libcst.Name):
                continue

            orig_name = alias.name.value
            if alias.asname and isinstance(alias.asname, libcst.AsName):
                if isinstance(alias.asname.name, libcst.Name):
                    used_name = alias.asname.name.value
                else:
                    used_name = orig_name
            else:
                used_name = orig_name

            qualified = f"{module_name}.{orig_name}"
            self._renames[used_name] = qualified

        self._new_imports.add(module_name)
        return libcst.RemovalSentinel.REMOVE

    def leave_Name(
        self, original_node: libcst.Name, updated_node: libcst.Name
    ) -> libcst.Name | libcst.Attribute:
        if updated_node.value in self._renames:
            result = _build_attribute(self._renames[updated_node.value])
            if isinstance(result, libcst.Name | libcst.Attribute):
                return result
        return updated_node

    def leave_Module(
        self, original_node: libcst.Module, updated_node: libcst.Module
    ) -> libcst.Module:
        new_stmts = []
        for mod_name in sorted(self._new_imports):
            if mod_name in self._existing_imports:
                continue
            import_node = libcst.parse_statement(f"import {mod_name}\n")
            new_stmts.append(import_node)

        if not new_stmts:
            return updated_node

        body = list(updated_node.body)
        insert_idx = 0
        for i, stmt in enumerate(body):
            if isinstance(stmt, libcst.SimpleStatementLine):
                for item in stmt.body:
                    if isinstance(item, libcst.ImportFrom | libcst.Import):
                        insert_idx = i + 1
        for i, new_stmt in enumerate(new_stmts):
            body.insert(insert_idx + i, new_stmt)

        return updated_node.with_changes(body=body)


def _get_module_name(node: libcst.Name | libcst.Attribute) -> str:
    if isinstance(node, libcst.Name):
        return node.value
    parts: list[str] = []
    current: libcst.BaseExpression = node
    while isinstance(current, libcst.Attribute):
        parts.append(current.attr.value)
        current = current.value
    if isinstance(current, libcst.Name):
        parts.append(current.value)
    return ".".join(reversed(parts))


def _build_attribute(dotted: str) -> libcst.BaseExpression:
    parts = dotted.split(".")
    node: libcst.BaseExpression = libcst.Name(parts[0])
    for part in parts[1:]:
        node = libcst.Attribute(value=node, attr=libcst.Name(part))
    return node


def check_file(filepath: pathlib.Path) -> list[tuple[int, str, str]]:
    """Check a file for import violations. Returns list of (line, module, names)."""
    source = filepath.read_text()
    try:
        tree = libcst.parse_module(source)
    except libcst.ParserSyntaxError:
        return []

    wrapper = libcst.metadata.MetadataWrapper(tree)
    checker = ImportViolationChecker()
    wrapper.visit(checker)
    return checker.violations


def fix_file(filepath: pathlib.Path) -> bool:
    """Fix a file's imports in place. Returns True if changes were made."""
    source = filepath.read_text()
    try:
        tree = libcst.parse_module(source)
    except libcst.ParserSyntaxError:
        return False

    wrapper = libcst.metadata.MetadataWrapper(tree)
    fixer = ImportFixer()
    new_tree = wrapper.visit(fixer)
    new_source = new_tree.code

    if new_source == source:
        return False

    filepath.write_text(new_source)
    return True


def collect_files(paths: list[str]) -> list[pathlib.Path]:
    """Collect .py files from paths (files or directories)."""
    files = []
    for p in paths:
        path = pathlib.Path(p)
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Check for violations")
    group.add_argument("--fix", action="store_true", help="Fix violations in place")
    parser.add_argument("paths", nargs="+", help="Files or directories to process")
    args = parser.parse_args()

    files = collect_files(args.paths)
    if not files:
        return 0

    if args.check:
        total = 0
        for f in files:
            violations = check_file(f)
            for line, module, names in violations:
                print(f"{f}:{line}: from {module} import {names}")
                total += 1
        if total:
            print(f"\n{total} violation(s) found. Run with --fix to auto-fix.")
            return 1
        return 0

    if args.fix:
        fixed = 0
        for f in files:
            if fix_file(f):
                print(f"Fixed: {f}")
                fixed += 1
        print(f"\n{fixed} file(s) fixed.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
