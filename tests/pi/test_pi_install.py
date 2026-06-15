from __future__ import annotations

import json
import pathlib

import simba.__main__ as cli


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
