"""Configuration for episodic consolidation (L2)."""

from __future__ import annotations

import dataclasses
import typing

import simba.config


@simba.config.configurable("episodes")
@dataclasses.dataclass
class EpisodesConfig:
    # Master switch. Consolidation is also engine-gated: with no RLM engine
    # configured (rlm.engine = "claude"), it is a no-op regardless of this.
    enabled: bool = True
    # Minimum non-SYSTEM/non-EPISODE memories a session needs before it's worth
    # consolidating into an episode.
    min_memories: int = 5
    # Cap on member memories baked into the consolidation prompt.
    max_members: int = 50
    # Auto-consolidate eligible sessions at session end (PreCompact).
    auto_on_precompact: bool = True
    # Let the background sync scheduler consolidate eligible sessions.
    scheduler_enabled: bool = True
    # A claimed 'running' job older than this is treated as stale (the detached
    # agent died before closing it) and reclaimed, so a session is never
    # permanently locked out of re-consolidation.
    job_timeout_hours: int = 4


def load_config(**overrides: typing.Any) -> EpisodesConfig:
    """Load config from TOML files, then apply CLI/keyword overrides."""
    base = simba.config.load("episodes")
    valid = {f.name for f in dataclasses.fields(EpisodesConfig)}
    filtered = {k: v for k, v in overrides.items() if v is not None and k in valid}
    if not filtered:
        return base
    merged = dataclasses.asdict(base)
    merged.update(filtered)
    return EpisodesConfig(**merged)
