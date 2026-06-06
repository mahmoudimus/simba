"""ReflectionConfig dataclass (Phase 5, Task A.2)."""

from __future__ import annotations


def test_reflection_config_defaults() -> None:
    import simba.reflection.config as rc

    cfg = rc.ReflectionConfig()
    assert cfg.enabled is True
    assert cfg.min_source_memories == 10
    assert cfg.importance_threshold == 0.6


def test_reflection_config_via_simba_config(monkeypatch) -> None:
    import simba.config
    import simba.reflection.config  # registers section

    _ = simba.reflection.config
    cfg = simba.config.load("reflection")
    assert hasattr(cfg, "scheduler_enabled")
    assert hasattr(cfg, "deduplicate_threshold")
