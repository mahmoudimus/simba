"""Tests for neuron config module -- ServerConfig."""

from __future__ import annotations

import simba.neuron.config


class TestServerConfig:
    def test_default_python_cmd_is_string(self) -> None:
        assert isinstance(simba.neuron.config.CONFIG.python_cmd, str)
        assert len(simba.neuron.config.CONFIG.python_cmd) > 0

    def test_souffle_cmd_is_string_or_none(self) -> None:
        cmd = simba.neuron.config.CONFIG.souffle_cmd
        assert cmd is None or isinstance(cmd, str)

    def test_custom_config(self) -> None:
        cfg = simba.neuron.config.ServerConfig(
            python_cmd="/usr/bin/python3",
            souffle_cmd=None,
        )
        assert cfg.python_cmd == "/usr/bin/python3"
        assert cfg.souffle_cmd is None
