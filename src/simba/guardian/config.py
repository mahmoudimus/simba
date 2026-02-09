"""Configuration for the guardian subsystem."""

from __future__ import annotations

import dataclasses

import simba.config


@simba.config.configurable("guardian")
@dataclasses.dataclass
class GuardianConfig:
    """Guardian settings — core instructions file management."""

    # Filename for the consolidated core instructions file.
    # Default: CORE_INSTRUCTIONS.md
    # Legacy projects can set this to CORE.md via:
    #   simba config set guardian.core_filename CORE.md
    core_filename: str = "CORE_INSTRUCTIONS.md"
