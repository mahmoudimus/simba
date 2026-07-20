"""Project-scoped resolution of exported transcripts for learning extraction.

``pre_compact`` exports each session to ``~/.claude/transcripts/<session_id>/``
with a ``metadata.json`` (session_id, project_path, status, exported_at, …) and a
single global ``latest.json`` symlink. That symlink is overwritten by *whichever
session compacts last across all projects*, so resolving "what to extract" from it
cross-wires sessions (e.g. /memories-learn in project B extracting project A's
transcript).

``find_pending`` instead selects the newest transcript whose ``project_path``
matches the *current* project and whose ``status`` is still ``pending_extraction``;
``mark_extracted`` flips that status so the same transcript isn't re-extracted.
"""

from __future__ import annotations

import json
import pathlib
import typing

PENDING = "pending_extraction"
EXTRACTED = "extracted"


def default_transcripts_dir() -> pathlib.Path:
    return pathlib.Path.home() / ".claude" / "transcripts"


def _norm(p: str) -> str:
    # Collapse '.' segments + trailing/duplicate slashes so cwd vs stored
    # project_path compare equal. PurePath does this without touching the disk.
    return str(pathlib.PurePath(p)) if p else p


def find_pending(
    project_path: str, *, transcripts_dir: pathlib.Path | None = None
) -> dict[str, typing.Any] | None:
    """Newest ``pending_extraction`` transcript for ``project_path`` (or None).

    The returned dict carries an extra ``_metadata_path`` so the caller can flip
    its status after extracting.
    """
    base = transcripts_dir or default_transcripts_dir()
    target = _norm(project_path)
    candidates: list[dict[str, typing.Any]] = []
    for meta_path in base.glob("*/metadata.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            continue
        if _norm(meta.get("project_path", "")) != target:
            continue
        if meta.get("status") != PENDING:
            continue
        meta["_metadata_path"] = str(meta_path)
        candidates.append(meta)
    if not candidates:
        return None
    candidates.sort(key=lambda m: m.get("exported_at", ""), reverse=True)
    return candidates[0]


def mark_extracted(metadata_path: str | pathlib.Path) -> bool:
    """Set ``status=extracted`` in a transcript's metadata.json. False on failure."""
    p = pathlib.Path(metadata_path)
    try:
        meta = json.loads(p.read_text())
    except (OSError, ValueError):
        return False
    meta.pop("_metadata_path", None)
    meta["status"] = EXTRACTED
    try:
        p.write_text(json.dumps(meta, indent=2))
    except OSError:
        return False
    return True
