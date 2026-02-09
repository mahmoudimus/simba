"""CLI for discovering, auditing, and updating SIMBA markers in .md files.

Usage:
    simba markers list [--path DIR]     List all markers found in .md files
    simba markers audit [--path DIR]    Compare found markers vs MANAGED_SECTIONS
    simba markers update [--path DIR]   Update all markers with current template content
    simba markers show <section>        Print a MANAGED_SECTIONS template by name
    simba markers migrate [--path DIR]  Convert non-SIMBA markers to SIMBA format
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
_EXCLUDE_SUBDIRS = {".claude/handoffs", ".claude/notes"}

# Default project files to scan for markers.
_PROJECT_FILES = ["CLAUDE.md", "AGENTS.md"]
_PROJECT_GLOBS = [".claude/**/*.md"]

_MARKER_RE = re.compile(r"<!--\s*BEGIN\s+SIMBA:(\w+)\s*-->")

# Matches non-SIMBA markers: <!-- BEGIN NEURON:name -->, <!-- CORE -->,
# <!-- BEGIN something -->, <!-- BEGIN ns:name --> (any namespace except SIMBA).
_FOREIGN_BEGIN_RE = re.compile(
    r"<!--\s*(?:BEGIN\s+)?(?!SIMBA:)(\w+(?::\w+)?)\s*-->"
)
_FOREIGN_BLOCK_RE = re.compile(
    r"(<!--\s*(?:BEGIN\s+)?(?!SIMBA:)(\w+(?::\w+)?)\s*-->)"
    r"(.*?)"
    r"(<!--\s*(?:END\s+|/)?\2\s*-->)",
    re.DOTALL,
)


class ForeignHit(NamedTuple):
    """A non-SIMBA marker found during scanning."""

    path: Path
    tag: str
    line_no: int


class MarkerHit(NamedTuple):
    """A single marker occurrence found during scanning."""

    path: Path
    name: str
    line_no: int
    content_len: int


def _collect_project_files(root: Path) -> list[Path]:
    """Collect project-relevant .md files: CLAUDE.md, AGENTS.md, .claude/*.md.

    Excludes .claude/handoffs/ and .claude/notes/ subdirectories.
    """
    files: list[Path] = []
    for name in _PROJECT_FILES:
        candidate = root / name
        if candidate.is_file():
            files.append(candidate)
    for pattern in _PROJECT_GLOBS:
        for candidate in sorted(root.glob(pattern)):
            if not candidate.is_file():
                continue
            rel = str(candidate.relative_to(root))
            if any(rel.startswith(exc) for exc in _EXCLUDE_SUBDIRS):
                continue
            if candidate not in files:
                files.append(candidate)
    return sorted(files)


def _is_excluded(md_file: Path, root: Path) -> bool:
    """Check if a file is under an excluded directory."""
    parts = md_file.relative_to(root).parts
    return any(p in _EXCLUDE_DIRS for p in parts)


def _scan_file(md_file: Path) -> list[MarkerHit]:
    """Extract all SIMBA markers from a single file."""
    hits: list[MarkerHit] = []
    try:
        content = md_file.read_text()
    except OSError:
        return hits
    for line_no, line in enumerate(content.splitlines(), start=1):
        m = _MARKER_RE.search(line)
        if m:
            name = m.group(1)
            blocks = simba.markers.extract_blocks(content, name)
            content_len = sum(len(b) for b in blocks)
            hits.append(MarkerHit(md_file, name, line_no, content_len))
    return hits


def scan_markers(root: Path, *, project_only: bool = False) -> list[MarkerHit]:
    """Scan .md files under *root* for SIMBA markers.

    When *project_only* is True, only scans project-relevant files
    (CLAUDE.md, AGENTS.md, .claude/*.md) excluding handoffs/notes.
    Otherwise scans all ``**/*.md`` skipping excluded dirs.

    Returns a list of :class:`MarkerHit` tuples sorted by (path, line_no).
    """
    if project_only:
        files = _collect_project_files(root)
    else:
        files = [
            f for f in sorted(root.rglob("*.md"))
            if f.is_file() and not _is_excluded(f, root)
        ]

    hits: list[MarkerHit] = []
    for md_file in files:
        hits.extend(_scan_file(md_file))
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
    """Audit markers in project files. Only checks what's actually present."""
    managed = simba.orchestration.templates.MANAGED_SECTIONS
    hits = scan_markers(root, project_only=True)
    found_names = {h.name for h in hits}

    issues = 0

    # Report markers in files that aren't in MANAGED_SECTIONS (user-defined
    # markers like "core" are fine — just informational).
    user_defined = sorted(found_names - set(managed.keys()))
    if user_defined:
        print("User-defined markers (not in MANAGED_SECTIONS):")
        for name in user_defined:
            locs = [h for h in hits if h.name == name]
            for loc in locs:
                try:
                    display = str(loc.path.relative_to(root))
                except ValueError:
                    display = str(loc.path)
                print(
                    f"  - {name} in {display}:{loc.line_no}"
                    f" ({loc.content_len} chars)"
                )
        print()

    # Detect stale: managed markers whose content differs from template.
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
            if template.strip() not in block and block.strip() != "":
                stale.append((hit.path, hit.name))
                break

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

    # Summary.
    if not hits:
        print("No SIMBA markers found in project files.")
        print(f"Scanned: {', '.join(_PROJECT_FILES)} + {', '.join(_PROJECT_GLOBS)}")
        return 0

    if issues == 0:
        marker_count = len(hits)
        file_count = len({h.path for h in hits})
        print(f"All good. {marker_count} marker(s) across {file_count} file(s).")
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


def scan_foreign_markers(root: Path) -> list[ForeignHit]:
    """Scan ``**/*.md`` for non-SIMBA markers (NEURON, CORE, bare BEGIN, etc.)."""
    hits: list[ForeignHit] = []
    for md_file in sorted(root.rglob("*.md")):
        parts = md_file.relative_to(root).parts
        if any(p in _EXCLUDE_DIRS for p in parts):
            continue
        try:
            content = md_file.read_text()
        except OSError:
            continue
        for line_no, line in enumerate(content.splitlines(), start=1):
            m = _FOREIGN_BEGIN_RE.search(line)
            if m:
                hits.append(ForeignHit(md_file, m.group(1), line_no))
    return hits


def _migrate_content(content: str) -> tuple[str, list[tuple[str, str]]]:
    """Rewrite non-SIMBA blocks in *content* to SIMBA format.

    Returns (new_content, [(old_tag, new_name), ...]) for reporting.
    """
    changes: list[tuple[str, str]] = []

    def _replacer(m: re.Match[str]) -> str:
        tag, body = m.group(2), m.group(3)
        # "NEURON:foo" → "foo", "CORE" → "core", "bar" → "bar"
        name = tag.split(":", 1)[1].lower() if ":" in tag else tag.lower()
        changes.append((tag, name))
        return (
            f"{simba.markers.begin_tag(name)}"
            f"{body}"
            f"{simba.markers.end_tag(name)}"
        )

    new_content = _FOREIGN_BLOCK_RE.sub(_replacer, content)
    return new_content, changes


def cmd_migrate(root: Path, *, dry_run: bool = False) -> int:
    """Find non-SIMBA markers and convert them to ``SIMBA:`` format."""
    foreign = scan_foreign_markers(root)
    if not foreign:
        print("No non-SIMBA markers found.")
        return 0

    # Show what was found.
    print(f"Found {len(foreign)} non-SIMBA marker(s):\n")
    for hit in foreign:
        try:
            display = str(hit.path.relative_to(root))
        except ValueError:
            display = str(hit.path)
        print(f"  {display}:{hit.line_no}  {hit.tag}")

    if dry_run:
        print("\n(dry run — no files modified)")
        return 0

    # Migrate each file.
    files = sorted({h.path for h in foreign})
    migrated = 0
    for md_file in files:
        try:
            original = md_file.read_text()
        except OSError:
            continue
        new_content, changes = _migrate_content(original)
        if new_content != original:
            md_file.write_text(new_content)
            try:
                display = str(md_file.relative_to(root))
            except ValueError:
                display = str(md_file)
            for old_tag, new_name in changes:
                print(f"  migrated: {display}  {old_tag} → SIMBA:{new_name}")
            migrated += 1

    print(f"\n{migrated} file(s) migrated.")
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

    p_migrate = sub.add_parser(
        "migrate", help="Find non-SIMBA markers and convert to SIMBA format"
    )
    p_migrate.add_argument(
        "--path", type=Path, default=Path.cwd(), help="Root directory"
    )
    p_migrate.add_argument(
        "--dry-run", action="store_true", help="Show what would be migrated"
    )

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
    elif args.subcmd == "migrate":
        return cmd_migrate(args.path, dry_run=args.dry_run)
    else:
        parser.print_help()
        return 1
