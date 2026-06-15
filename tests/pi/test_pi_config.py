from __future__ import annotations

import simba.config
import simba.pi  # registers the "pi" section


def test_pi_config_defaults():
    cfg = simba.config.load("pi")
    assert cfg.enabled is True
    assert cfg.extension_path.endswith("simba.ts")


def test_pi_section_is_registered():
    assert "pi" in simba.config.list_sections()
