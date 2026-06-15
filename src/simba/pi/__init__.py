"""pi coding-agent harness integration: config section."""

from __future__ import annotations

import dataclasses

import simba.config


@simba.config.configurable("pi")
@dataclasses.dataclass
class PiConfig:
    # pi's agent home; the bridge extension is written to
    # <agent_home>/extensions/simba.ts and registered in <agent_home>/settings.json.
    # The PI_CODING_AGENT_DIR env var (pi's own convention) overrides this when set.
    agent_home: str = "~/.pi/agent"
