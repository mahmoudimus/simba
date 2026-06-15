from __future__ import annotations

import simba.config
import simba.pi  # registers the "pi" section


def test_pi_config_defaults():
    cfg = simba.config.load("pi")
    assert cfg.agent_home.endswith(".pi/agent")


def test_pi_section_is_registered():
    assert "pi" in simba.config.list_sections()
