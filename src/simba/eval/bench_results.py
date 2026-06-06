"""Append-only results store for ``simba eval bench`` + leaderboard helpers.

Each bench run appends one JSON record to ``bench.results_path`` (a JSONL file).
The leaderboard reads the log back, groups runs by ``(dataset, split)``, and
diffs the latest two. The JSONL is the source of truth; ``BENCHMARKS.md`` is
derived state computed from it.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import typing

if typing.TYPE_CHECKING:
    import pathlib


def current_git_sha() -> str:
    """Return the short HEAD sha, or ``"unknown"`` on any failure."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return "unknown"
    out = proc.stdout.strip()
    return out or "unknown"


def append_result(path: pathlib.Path, record: dict[str, object]) -> None:
    """Append one JSON record as a line to the results log (never truncates)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def config_snapshot(mcfg: object, bcfg: object) -> dict[str, object]:
    """Snapshot both configs into the record so future diffs can pinpoint a change."""
    return {
        "memory": dataclasses.asdict(mcfg),
        "bench": dataclasses.asdict(bcfg),
    }


def load_results(path: pathlib.Path) -> list[dict[str, object]]:
    """Read all JSONL records from the results log; skip malformed lines."""
    if not path.exists():
        return []
    out: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def latest_two_by_group(
    records: list[dict[str, object]],
) -> dict[str, tuple[dict[str, object], dict[str, object] | None]]:
    """Group by ``(dataset, split)``; return (latest, prev|None) sorted by timestamp."""
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for r in records:
        dataset = str(r.get("dataset", ""))
        split = r.get("split") or "full"
        groups.setdefault((dataset, str(split)), []).append(r)

    out: dict[str, tuple[dict[str, object], dict[str, object] | None]] = {}
    for (dataset, split), rows in groups.items():
        rows.sort(key=lambda r: r.get("timestamp", 0.0), reverse=True)
        key = f"{dataset} ({split})"
        latest = rows[0]
        prev = rows[1] if len(rows) >= 2 else None
        out[key] = (latest, prev)
    return out
