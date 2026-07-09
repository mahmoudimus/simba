"""Append-only results store for ``simba eval bench`` + leaderboard helpers.

Each bench run appends one JSON record to ``bench.results_path`` (a JSONL file).
The leaderboard reads the log back, groups runs by ``(dataset, split)``, and
diffs the latest two. The JSONL is the source of truth; ``BENCHMARKS.md`` is
derived state computed from it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import subprocess


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


def config_snapshot(
    mcfg: object,
    bcfg: object,
    llm_cfg: object | None = None,
    judge_cfg: object | None = None,
) -> dict[str, object]:
    """Snapshot the configs into the record so future diffs can pinpoint a change.

    ``llm_cfg`` / ``judge_cfg`` (the answerer + grader) are included when given:
    QA accuracy depends on which models answered and judged, so a record is only
    attributable to a model — e.g. gpt-oss vs Qwen — if both are captured.
    """

    def _as_dict(obj: object) -> dict[str, object]:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        return dict(vars(obj)) if hasattr(obj, "__dict__") else {}

    snap: dict[str, object] = {
        "memory": _as_dict(mcfg),
        "bench": _as_dict(bcfg),
    }
    if llm_cfg is not None:
        snap["llm"] = _as_dict(llm_cfg)
    if judge_cfg is not None:
        snap["judge"] = _as_dict(judge_cfg)
    return snap


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: dict[str, object]) -> str:
    """Stable SHA256 over a config snapshot."""
    return hashlib.sha256(_stable_json(config).encode()).hexdigest()


def path_digest(path: str | pathlib.Path) -> dict[str, object]:
    """Return deterministic dataset path metadata and a SHA256 when readable.

    Files hash their bytes. Directories hash every contained file in sorted order,
    mixing each relative path before its bytes so renames change the digest.
    Missing paths are explicit rather than fatal because tests and dry-run
    configs often stub loader functions without a real benchmark file.
    """
    p = pathlib.Path(path)
    meta: dict[str, object] = {
        "path": str(p),
        "exists": p.exists(),
        "kind": "missing",
        "sha256": "",
        "file_count": 0,
    }
    if not p.exists():
        return meta
    h = hashlib.sha256()
    if p.is_file():
        meta["kind"] = "file"
        meta["file_count"] = 1
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
    elif p.is_dir():
        meta["kind"] = "directory"
        count = 0
        for child in sorted(c for c in p.rglob("*") if c.is_file()):
            rel = child.relative_to(p).as_posix()
            h.update(rel.encode())
            h.update(b"\0")
            with child.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            h.update(b"\0")
            count += 1
        meta["file_count"] = count
    else:
        meta["kind"] = "other"
    meta["sha256"] = h.hexdigest()
    return meta


def model_identity(cfg: object | None) -> dict[str, object] | None:
    """Small attribution block for answerer/judge configs."""
    if cfg is None:
        return None
    raw = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else vars(cfg)
    return {
        "provider": raw.get("provider", ""),
        "model": raw.get("model", ""),
        "base_url": raw.get("base_url", ""),
    }


def build_provenance(
    *,
    dataset_name: str,
    dataset_path: str | pathlib.Path,
    split: str | None,
    config: dict[str, object],
    git_sha: str,
    answerer_cfg: object | None = None,
    judge_cfg: object | None = None,
    excluded_count: int = 0,
    abstained_count: int = 0,
    contaminated_count: int = 0,
    judge_replay_agreement: float | None = None,
    significance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Attribution metadata for benchmark records.

    This is intentionally additive to the existing record shape. The detailed
    config snapshot remains under ``record["config"]``; this block gives quick
    machine-checkable provenance for graduation reviews.
    """
    return {
        "schema_version": 1,
        "dataset": {
            "name": dataset_name,
            "split": split,
            **path_digest(dataset_path),
        },
        "source": {"git_sha": git_sha},
        "config_hash": config_hash(config),
        "answerer": model_identity(answerer_cfg),
        "judge": model_identity(judge_cfg),
        "judge_replay_agreement": judge_replay_agreement,
        "counts": {
            "excluded": excluded_count,
            "abstained": abstained_count,
            "contaminated": contaminated_count,
        },
        "significance": significance or {},
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
