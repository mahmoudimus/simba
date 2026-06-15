from __future__ import annotations

import json
import pathlib

import simba.__main__ as cli
import simba.config


def _pi_home(tmp_path, monkeypatch) -> pathlib.Path:
    home = tmp_path / ".pi" / "agent"
    monkeypatch.setattr(cli, "_pi_agent_home", lambda: home)
    return home


def test_pi_install_writes_extension_and_registers(tmp_path, monkeypatch):
    home = _pi_home(tmp_path, monkeypatch)
    rc = cli._cmd_pi_install([])
    assert rc == 0
    ext = home / "extensions" / "simba.ts"
    assert ext.is_file()
    settings = json.loads((home / "settings.json").read_text())
    assert str(ext) in settings["extensions"]


def test_pi_install_is_idempotent(tmp_path, monkeypatch):
    home = _pi_home(tmp_path, monkeypatch)
    cli._cmd_pi_install([])
    cli._cmd_pi_install([])
    settings = json.loads((home / "settings.json").read_text())
    assert settings["extensions"].count(str(home / "extensions" / "simba.ts")) == 1


def test_pi_install_remove(tmp_path, monkeypatch):
    home = _pi_home(tmp_path, monkeypatch)
    cli._cmd_pi_install([])
    cli._cmd_pi_install(["--remove"])
    settings = json.loads((home / "settings.json").read_text())
    assert str(home / "extensions" / "simba.ts") not in settings.get("extensions", [])
    assert not (home / "extensions" / "simba.ts").exists()


def test_pi_agent_home_driven_by_config(tmp_path, monkeypatch):
    # With PI_CODING_AGENT_DIR unset, the path comes from the pi.agent_home config.
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)

    class _Cfg:
        agent_home = str(tmp_path)

    real_load = simba.config.load

    def fake_load(section, *args, **kwargs):
        if section == "pi":
            return _Cfg()
        return real_load(section, *args, **kwargs)

    monkeypatch.setattr(simba.config, "load", fake_load)
    assert cli._pi_agent_home() == tmp_path


def test_pi_agent_home_env_overrides_config(tmp_path, monkeypatch):
    # PI_CODING_AGENT_DIR (pi's own convention) takes precedence over config.
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    assert cli._pi_agent_home() == tmp_path
