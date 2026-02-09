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
