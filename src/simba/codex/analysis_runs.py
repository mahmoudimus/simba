"""JSONL trace artifacts for Codex transcript analysis runs."""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from typing import TYPE_CHECKING, Any

import simba.db

if TYPE_CHECKING:
    import pathlib


@dataclasses.dataclass(frozen=True)
class AnalysisRun:
    run_id: str
    trace_path: pathlib.Path
    session_id: str
    project_path: str
    transcript_path: str


def default_root(cwd: pathlib.Path | None = None) -> pathlib.Path:
    return simba.db.get_db_path(cwd).parent / "analysis_runs"


def start_run(
    *,
    session_id: str,
    project_path: str,
    transcript_path: str,
    root: pathlib.Path | None = None,
    cwd: pathlib.Path | None = None,
    now: float | None = None,
) -> AnalysisRun:
    """Create a run descriptor and append the opening event."""
    now = time.time() if now is None else now
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
    safe_session = (session_id or "unknown").replace("/", "_")[:48]
    run_id = f"{stamp}-{safe_session}-{uuid.uuid4().hex[:8]}"
    base = root if root is not None else default_root(cwd)
    trace_path = base / f"{run_id}.jsonl"
    run = AnalysisRun(
        run_id=run_id,
        trace_path=trace_path,
        session_id=session_id,
        project_path=project_path,
        transcript_path=transcript_path,
    )
    append_event(run, "run_started", {}, now=now)
    return run


def append_event(
    run: AnalysisRun,
    event: str,
    payload: dict[str, Any],
    *,
    now: float | None = None,
) -> None:
    """Append one JSONL event to an analysis run trace."""
    now = time.time() if now is None else now
    run.trace_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "run_id": run.run_id,
        "event": event,
        "session_id": run.session_id,
        "project_path": run.project_path,
        "transcript_path": run.transcript_path,
        "payload": payload,
    }
    with run.trace_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
