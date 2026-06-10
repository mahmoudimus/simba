"""Configuration for the durable workflow engine (``simba.workflow``)."""

from __future__ import annotations

import dataclasses

import simba.config


@simba.config.configurable("workflow")
@dataclasses.dataclass
class WorkflowConfig:
    # queue.py — retry policy
    default_max_attempts: int = 3
    retry_backoff_base_seconds: float = 2.0
    retry_backoff_max_seconds: float = 300.0
    # queue.py — dead-worker recovery (reclaim_stale default window)
    stale_after_seconds: int = 3600
    # runner.py — bounded within-stage parallelism
    fan_out_max_workers: int = 8
    # runner.py — how worker_loop dispatches: "sync" (in-process) | "detached"
    worker_mode: str = "detached"
