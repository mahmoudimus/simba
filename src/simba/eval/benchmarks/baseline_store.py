"""Append-only store for eval baseline results.

Each call to ``append_baseline`` writes one timestamped JSON line to
``<root>/<eval.baseline_dir>/<name>.jsonl``. History is never overwritten, so a
re-run records a new line and the trend stays auditable.
"""

from __future__ import annotations

import datetime
import json
from typing import TYPE_CHECKING

import simba.config
import simba.eval.config

if TYPE_CHECKING:
    import pathlib


def _baselines_dir(root: pathlib.Path | None) -> pathlib.Path:
    resolved = simba.config._find_root(root)
    baseline_dir = simba.config.load("eval", root=resolved).baseline_dir
    return resolved / baseline_dir


def append_baseline(
    name: str,
    report: dict,
    *,
    root: pathlib.Path | None = None,
    metadata: dict | None = None,
) -> pathlib.Path:
    """Append a timestamped baseline entry to ``<baseline_dir>/<name>.jsonl``.

    Append-only: never overwrites. Returns the path written to.
    """
    out_dir = _baselines_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    entry = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "report": report,
        "metadata": metadata or {},
    }
    with path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return path


def load_baselines(
    name: str,
    *,
    root: pathlib.Path | None = None,
) -> list[dict]:
    """Return all baseline entries for ``name`` in chronological order."""
    path = _baselines_dir(root) / f"{name}.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
