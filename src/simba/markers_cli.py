"""CLI for discovering, auditing, and updating SIMBA markers in .md files.

Usage:
    simba markers list [--path DIR]     List all markers found in .md files
    simba markers audit [--path DIR]    Compare found markers vs MANAGED_SECTIONS
    simba markers update [--path DIR]   Update all markers with current template content
    simba markers show <section>        Print a MANAGED_SECTIONS template by name
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

import simba.markers
import simba.orchestration.templates

_EXCLUDE_DIRS = {"_gitless", "node_modules", ".git"}

_MARKER_RE = re.compile(r"<!--\s*BEGIN\s+SIMBA:(\w+)\s*-->")


class MarkerHit(NamedTuple):
    """A single marker occurrence found during scanning."""

    path: Path
    name: str
    line_no: int
    content_len: int


def scan_markers(root: Path) -> list[MarkerHit]:
    """Scan ``**/*.md`` under *root* for SIMBA markers, skipping excluded dirs.

    Returns a list of :class:`MarkerHit` tuples sorted by (path, line_no).
    """
    hits: list[MarkerHit] = []
    for md_file in sorted(root.rglob("*.md")):
        # Skip files under excluded directories.
        parts = md_file.relative_to(root).parts
        if any(p in _EXCLUDE_DIRS for p in parts):
            continue

        try:
            content = md_file.read_text()
        except OSError:
            continue

        for line_no, line in enumerate(content.splitlines(), start=1):
            m = _MARKER_RE.search(line)
            if m:
                name = m.group(1)
                blocks = simba.markers.extract_blocks(content, name)
                content_len = sum(len(b) for b in blocks)
                hits.append(MarkerHit(md_file, name, line_no, content_len))
    return hits


def cmd_list(root: Path) -> int:
    """Print a table of all markers found under *root*."""
    hits = scan_markers(root)
    if not hits:
        print("No SIMBA markers found.")
        return 0

    print(f"{'File':<50s} {'Section':<25s} {'Line':>5s} {'Len':>6s}")
    print(f"{'─' * 50} {'─' * 25} {'─' * 5} {'─' * 6}")
    for hit in hits:
        try:
            display = str(hit.path.relative_to(root))
        except ValueError:
            display = str(hit.path)
        print(f"{display:<50s} {hit.name:<25s} {hit.line_no:>5d} {hit.content_len:>6d}")
    print(f"\n{len(hits)} marker(s) found.")
    return 0


def cmd_audit(root: Path) -> int:
    """Compare markers on disk against MANAGED_SECTIONS, report issues."""
    managed = simba.orchestration.templates.MANAGED_SECTIONS
    hits = scan_markers(root)
    found_names = {h.name for h in hits}
    managed_names = set(managed.keys())

    unused = sorted(managed_names - found_names)
    orphaned = sorted(found_names - managed_names)

    # Detect stale: content between markers differs from template.
    stale: list[tuple[Path, str]] = []
    for hit in hits:
        if hit.name not in managed:
            continue
        try:
            content = hit.path.read_text()
        except OSError:
            continue
        blocks = simba.markers.extract_blocks(content, hit.name)
        template = managed[hit.name]
        for block in blocks:
            # update_managed_sections prepends a timestamp comment, so we
            # compare the template substring instead of exact match.
            if template.strip() not in block and block.strip() != "":
                try:
                    display = str(hit.path.relative_to(root))
                except ValueError:
                    display = str(hit.path)
                stale.append((hit.path, hit.name))
                break

    issues = 0
    if unused:
        print("Unused sections (in MANAGED_SECTIONS but no .md file):")
        for name in unused:
            print(f"  - {name}")
        print()
        issues += len(unused)

    if orphaned:
        print("Orphaned markers (in .md files but not in MANAGED_SECTIONS):")
        for name in orphaned:
            print(f"  - {name}")
        print()
        issues += len(orphaned)

    if stale:
        print("Stale markers (content differs from template):")
        for path, name in stale:
            try:
                display = str(path.relative_to(root))
            except ValueError:
                display = str(path)
            print(f"  - {display}: {name}")
        print()
        issues += len(stale)

    if issues == 0:
        print("All markers are up to date.")
    else:
        print(f"{issues} issue(s) found.")
    return 0


def cmd_update(root: Path) -> int:
    """Update all .md files with markers to current MANAGED_SECTIONS content."""
    hits = scan_markers(root)
    if not hits:
        print("No SIMBA markers found.")
        return 0

    # Group by file.
    files = sorted({h.path for h in hits})
    updated_count = 0

    for md_file in files:
        try:
            original = md_file.read_text()
        except OSError:
            continue

        result = simba.orchestration.templates.update_managed_sections(original)
        if result != original:
            md_file.write_text(result)
            try:
                display = str(md_file.relative_to(root))
            except ValueError:
                display = str(md_file)
            print(f"  updated: {display}")
            updated_count += 1

    if updated_count:
        print(f"\n{updated_count} file(s) updated.")
    else:
        print("All files already up to date.")
    return 0


def cmd_show(section: str) -> int:
    """Print the raw template content for a MANAGED_SECTIONS entry."""
    managed = simba.orchestration.templates.MANAGED_SECTIONS
    if section not in managed:
        print(f"Unknown section: {section}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(managed.keys()))}", file=sys.stderr)
        return 1
    print(managed[section])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``simba markers``."""
    parser = argparse.ArgumentParser(
        prog="simba markers",
        description="Discover, audit, and update SIMBA markers in .md files.",
    )
    sub = parser.add_subparsers(dest="subcmd")

    p_list = sub.add_parser("list", help="List all markers found in .md files")
    p_list.add_argument("--path", type=Path, default=Path.cwd(), help="Root directory")

    p_audit = sub.add_parser("audit", help="Compare markers vs MANAGED_SECTIONS")
    p_audit.add_argument("--path", type=Path, default=Path.cwd(), help="Root directory")

    p_update = sub.add_parser("update", help="Update markers with current templates")
    p_update.add_argument(
        "--path", type=Path, default=Path.cwd(), help="Root directory"
    )

    p_show = sub.add_parser("show", help="Print raw template for a section")
    p_show.add_argument("section", help="Section name from MANAGED_SECTIONS")

    args = parser.parse_args(argv)

    if args.subcmd is None:
        parser.print_help()
        return 1

    if args.subcmd == "list":
        return cmd_list(args.path)
    elif args.subcmd == "audit":
        return cmd_audit(args.path)
    elif args.subcmd == "update":
        return cmd_update(args.path)
    elif args.subcmd == "show":
        return cmd_show(args.section)
    else:
        parser.print_help()
        return 1
