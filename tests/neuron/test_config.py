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


def test_neuron_config_has_phase7_fields() -> None:
    import simba.neuron.config as nc

    cfg = nc.NeuronConfig()
    assert cfg.derive_enabled is True
    assert cfg.verify_timeout_seconds == 30
    assert cfg.induce_min_activations == 3


def test_neuron_config_via_simba_config(monkeypatch) -> None:
    import simba.config
    import simba.neuron.config  # registers section

    _ = simba.neuron.config
    cfg = simba.config.load("neuron")
    assert hasattr(cfg, "contradiction_sample_size")
