"""Export DB rows as markdown files for QMD semantic indexing.

Creates one .md file per table under `.simba/exports/`. Overwrites each
cycle to prevent unbounded growth.
"""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path

from simba.sync.text_render import render_row_markdown


def get_export_dir(cwd: str | Path) -> Path:
    """Return the export directory path (`.simba/exports/`)."""
    d = Path(cwd) / ".simba" / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def export_table(export_dir: Path, table_name: str, rows: list[dict]) -> Path | None:
    """Write rows as a single markdown file.

    Returns the written path, or ``None`` if *rows* is empty.
    """
    if not rows:
        return None
    sections = []
    for row in rows:
        md = render_row_markdown(table_name, row)
        if md:
            sections.append(md)
    if not sections:
        return None
    path = export_dir / f"{table_name}.md"
    path.write_text("\n\n---\n\n".join(sections) + "\n", encoding="utf-8")
    return path


def export_all_tables(cwd: str | Path, tables: dict[str, list[dict]]) -> list[Path]:
    """Export all tables and optionally run ``qmd embed``.

    Returns the list of paths written.
    """
    export_dir = get_export_dir(cwd)
    paths: list[Path] = []
    for table_name, rows in tables.items():
        p = export_table(export_dir, table_name, rows)
        if p is not None:
            paths.append(p)

    if paths:
        _run_qmd_embed(export_dir)

    return paths


def _run_qmd_embed(export_dir: Path) -> None:
    """Run ``qmd embed`` on the export directory if qmd is available."""
    from simba.search.qmd import is_available

    if not is_available():
        return
    with contextlib.suppress(subprocess.SubprocessError, OSError):
        subprocess.run(
            ["qmd", "embed", str(export_dir)],
            capture_output=True,
            timeout=30,
        )
