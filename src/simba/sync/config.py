"""Configuration for simba sync pipeline."""

from __future__ import annotations

import dataclasses

import simba.config


@simba.config.configurable("sync")
@dataclasses.dataclass
class SyncConfig:
    daemon_url: str = "http://localhost:8741"
    batch_limit: int = 20
    rate_limit_sec: float = 0.1
    retry_count: int = 2
    retry_backoff: float = 0.5
    page_size: int = 50
    default_interval: int = 300
    # Synchronous LLM fact extraction for memories the regex heuristics miss
    # (uses the llm provider; fail-open to regex-only). On by default
    # (experimental); runs in the background sync pipeline, not the hot path.
    llm_extract_enabled: bool = True
    llm_extract_max_triples: int = 8
    # Extraction strategy for the KG feed: "regex" (heuristics only), "llm"
    # (LLM only), or "llm+regex" (both, unioned — the primary feed). Default
    # "llm+regex" so the KG is richly fed; degrades to regex when no llm provider.
    extract_strategy: str = "llm+regex"
    # Safety cap: max memories LLM-extracted per cycle (0 = unlimited). Bounds the
    # cost of the first sweep over a large backlog; the rest are picked up next
    # cycle (the watermark only advances past processed memories).
    llm_extract_max_per_cycle: int = 100
