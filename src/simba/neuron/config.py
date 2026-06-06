"""Configuration for the Neuron neuro-symbolic subsystem (Phase 7).

``NeuronConfig`` is a ``@configurable`` dataclass so every knob is reachable via
``simba config get/set neuron.<key>`` — no hidden constants. ``ServerConfig`` is
kept as a backward-compat alias, and the module-level ``CONFIG`` instance (loaded
from the config system) is preserved for existing ``verify.py`` callers.
"""

from __future__ import annotations

import dataclasses
import shutil
import sys

import simba.config


@simba.config.configurable("neuron")
@dataclasses.dataclass
class NeuronConfig:
    python_cmd: str = dataclasses.field(default_factory=lambda: sys.executable)
    souffle_cmd: str = dataclasses.field(
        default_factory=lambda: shutil.which("souffle") or ""
    )
    enabled: bool = True
    derive_enabled: bool = True
    verify_enabled: bool = True
    revise_enabled: bool = True
    distill_enabled: bool = True
    induce_enabled: bool = True
    derive_max_edges: int = 500
    verify_timeout_seconds: int = 30
    induce_min_activations: int = 3
    induce_min_confidence: float = 0.7
    contradiction_sample_size: int = 200


# Backward-compat alias for existing callers that constructed ``ServerConfig``.
ServerConfig = NeuronConfig


def _load() -> NeuronConfig:
    """Load the ``neuron`` section, merging defaults → global → local TOML."""
    return simba.config.load("neuron")


# Module-level instance for existing verify.py callers. It is config-backed
# (not a hidden constant) and re-read on import.
CONFIG = _load()
