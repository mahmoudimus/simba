"""Configuration for cross-session reflection (L4)."""

from __future__ import annotations

import dataclasses

import simba.config


@simba.config.configurable("reflection")
@dataclasses.dataclass
class ReflectionConfig:
    # Master on/off switch.
    enabled: bool = True
    # Allow the background scheduler to trigger a reflect pass.
    scheduler_enabled: bool = True
    # Minimum non-REFLECTION/SYSTEM memories before a pass runs.
    min_source_memories: int = 10
    # Cap on memories baked into the LLM prompt.
    max_source_memories: int = 100
    # Confidence below which a candidate reflection is discarded.
    importance_threshold: float = 0.6
    # Vector similarity above which a new reflection is a duplicate.
    deduplicate_threshold: float = 0.88
    # Run reflection once every N scheduler sync cycles (0 = every cycle).
    interval_cycles: int = 5
    # Hard cap on new REFLECTION memories stored per pass.
    max_reflections_per_pass: int = 3
    # Scope reflection to current project_path (False = global).
    project_scoped: bool = True
