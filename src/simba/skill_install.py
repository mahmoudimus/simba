"""Install/refresh bundled skills into a target skills directory.

Replaces the old create-only, lowercase-`skill.md`-only copy loop (which never
updated an already-installed skill, so SKILL.md edits never reached
~/.claude/skills). ``sync_skills`` copies each skill's whole directory, detects
``SKILL.md`` (or legacy ``skill.md``), and rewrites only the files whose content
changed — so it's update-correct and idempotent.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import pathlib

_SKILL_MD_NAMES = ("SKILL.md", "skill.md")


def find_skill_md(skill_dir: pathlib.Path) -> pathlib.Path | None:
    """Return the skill's manifest (SKILL.md, or legacy skill.md), else None."""
    for name in _SKILL_MD_NAMES:
        candidate = skill_dir / name
        if candidate.is_file():
            return candidate
    return None


def _sync_dir(src: pathlib.Path, dest: pathlib.Path) -> bool:
    """Copy every file under src into dest, rewriting only changed ones.

    Returns True if anything was written (new or updated).
    """
    changed = False
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        target = dest / rel
        data = path.read_bytes()
        if target.exists() and target.read_bytes() == data:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        changed = True
    return changed


def sync_skills(src_dir: pathlib.Path, dest_dir: pathlib.Path) -> tuple[int, int]:
    """Sync every skill (a dir containing SKILL.md) from src_dir into dest_dir.

    Returns ``(installed, updated)``: newly created skills vs. existing skills
    whose files changed. Unchanged skills are skipped (idempotent).
    """
    installed = updated = 0
    if not src_dir.is_dir():
        return (0, 0)
    for skill_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        if find_skill_md(skill_dir) is None:
            continue
        dest = dest_dir / skill_dir.name
        existed = dest.exists()
        if _sync_dir(skill_dir, dest):
            if existed:
                updated += 1
            else:
                installed += 1
    return (installed, updated)
