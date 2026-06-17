"""Append-only extraction ledger for Codex JSONL transcripts."""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
from typing import Any

PENDING = "pending_extraction"
EXTRACTED = "extracted"


def ledger_path(codex_home: pathlib.Path) -> pathlib.Path:
    """Return the append-only Codex extraction ledger path."""
    return codex_home / "simba" / "extractions.jsonl"


def transcript_fingerprint(path: pathlib.Path) -> dict[str, Any] | None:
    """Hash the current transcript contents without mutating the transcript."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def _iter_records(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def is_extracted(
    *,
    codex_home: pathlib.Path,
    transcript_path: str,
    session_id: str,
    project_path: str,
    fingerprint: dict[str, Any] | None = None,
) -> bool:
    """Return True when this exact transcript fingerprint has been processed."""
    if fingerprint is None:
        fingerprint = transcript_fingerprint(pathlib.Path(transcript_path))
    if not fingerprint:
        return False

    expected_sha = fingerprint.get("sha256")
    expected_size = fingerprint.get("size")
    for record in _iter_records(ledger_path(codex_home)):
        if record.get("status") != EXTRACTED:
            continue
        if record.get("transcript_path") != transcript_path:
            continue
        if record.get("session_id") != session_id:
            continue
        if record.get("project_path") != project_path:
            continue
        seen = record.get("fingerprint", {})
        if not isinstance(seen, dict):
            continue
        if seen.get("sha256") == expected_sha and seen.get("size") == expected_size:
            return True
    return False


def status_for(
    *,
    codex_home: pathlib.Path,
    transcript_path: str,
    session_id: str,
    project_path: str,
    fingerprint: dict[str, Any] | None = None,
) -> str:
    """Return extracted/pending status for the current Codex transcript state."""
    if is_extracted(
        codex_home=codex_home,
        transcript_path=transcript_path,
        session_id=session_id,
        project_path=project_path,
        fingerprint=fingerprint,
    ):
        return EXTRACTED
    return PENDING


def append_extracted(
    *,
    codex_home: pathlib.Path,
    transcript_path: str,
    session_id: str,
    project_path: str,
    fingerprint: dict[str, Any],
    candidates: int,
    stored: int,
    duplicates: int,
) -> pathlib.Path:
    """Append a successful extraction record and return the ledger path."""
    path = ledger_path(codex_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "codex",
        "status": EXTRACTED,
        "session_id": session_id,
        "project_path": project_path,
        "transcript_path": transcript_path,
        "fingerprint": fingerprint,
        "candidates": candidates,
        "stored": stored,
        "duplicates": duplicates,
    }
    with path.open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return path
