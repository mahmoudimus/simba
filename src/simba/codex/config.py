"""Configuration for Codex lifecycle integration."""

from __future__ import annotations

import dataclasses

import simba.config


@simba.config.configurable("codex")
@dataclasses.dataclass
class CodexConfig:
    # Optional fallback: analyze the newest pending raw Codex JSONL transcript
    # when `simba codex-status` runs. PreCompact is the primary automatic path,
    # so status remains diagnostic by default.
    auto_extract_on_status: bool = False
    # Max heuristic memories to store from one automatic extraction pass.
    auto_extract_max_items: int = 15
    # Optional JSONL sidecar for replaying why raw transcript extraction kept,
    # rejected, or failed candidate memories. Disabled until measured.
    extraction_trace_enabled: bool = False
    # Empty means `.simba/analysis_runs` for the transcript project.
    extraction_trace_dir: str = ""
    # Empty means `.simba/curator_runs` for the transcript project.
    curator_report_dir: str = ""
    # `markdown` or `json`; CLI flags can override it per run.
    curator_default_format: str = "markdown"
    # Keep all candidates by default; report-only filter for future review tuning.
    curator_min_candidate_score: float = 0.0
