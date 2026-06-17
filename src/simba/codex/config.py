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
