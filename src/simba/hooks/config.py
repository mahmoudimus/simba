"""Configuration for simba hooks."""

from __future__ import annotations

import dataclasses

import simba.config


@simba.config.configurable("hooks")
@dataclasses.dataclass
class HooksConfig:
    # Session start — daemon polling
    health_timeout: float = 0.5
    poll_attempts: int = 15
    poll_interval: float = 0.3

    # Memory client
    daemon_host: str = "localhost"
    daemon_port: int = 8741
    default_max_results: int = 3
    default_timeout: float = 2.0

    # Pre-tool-use
    min_similarity: float = 0.35
    thinking_chars: int = 1500
    dedup_ttl: int = 60
    context_low_bytes: int = 4_000_000

    # Tool rules (auto-learning from failures)
    rule_check_enabled: bool = True
    rule_min_similarity: float = 0.6
    rule_max_per_session: int = 10
    auto_learn_from_failures: bool = True

    # memories-learn skill dispatch mode
    # False (default): dispatch synchronously, Claude waits for extraction to complete
    # True: dispatch in background, Claude continues immediately
    learn_async: bool = False
