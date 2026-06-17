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

    # UserPromptSubmit CORE injection shape. "full" preserves the exact extracted
    # SIMBA:core block; "capsule" compiles it into a short deterministic reminder
    # with a pointer back to the rules file. Capsule mode is intended for dogfood /
    # measured rollout because it trades some passive context for much lower token
    # tax.
    core_injection_mode: str = "full"  # full | capsule
    core_capsule_max_chars: int = 1200
    core_capsule_rule_chars: int = 140
