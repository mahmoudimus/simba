"""pi coding-agent harness integration: config section."""

from __future__ import annotations

import dataclasses

import simba.config


@simba.config.configurable("pi")
@dataclasses.dataclass
class PiConfig:
    # Whether `simba install` also wires the pi extension.
    enabled: bool = True
    # Where the bundled bridge extension is written and registered.
    extension_path: str = "~/.pi/agent/extensions/simba.ts"
    # pi's settings.json (extensions[] registration target).
    settings_path: str = "~/.pi/agent/settings.json"
