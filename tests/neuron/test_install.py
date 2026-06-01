"""Tests for `simba neuron install` (MCP registration)."""

from __future__ import annotations

import pathlib

import simba.neuron.__main__ as nm


class _Result:
    returncode = 0


def test_install_builds_claude_mcp_add(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("shutil.which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        "subprocess.run", lambda cmd, **k: calls.append(cmd) or _Result()
    )

    rc = nm._install_mcp(pathlib.Path("/proj"))

    assert rc == 0
    cmd = calls[0]
    assert cmd[:4] == ["/bin/claude", "mcp", "add", "neuron"]
    assert "--" in cmd
    # tail is the server launch command
    assert cmd[-5:] == ["/bin/simba", "neuron", "run", "--root-dir", "/proj"]
    assert "--scope" not in cmd  # default = local/project scope


def test_install_global_uses_user_scope(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("shutil.which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        "subprocess.run", lambda cmd, **k: calls.append(cmd) or _Result()
    )

    nm._install_mcp(pathlib.Path("/proj"), user_scope=True)

    assert calls[0][4:6] == ["--scope", "user"]


def test_install_remove(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("shutil.which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        "subprocess.run", lambda cmd, **k: calls.append(cmd) or _Result()
    )

    rc = nm._install_mcp(None, remove=True)

    assert rc == 0
    assert calls[0][:4] == ["/bin/claude", "mcp", "remove", "neuron"]


def test_install_no_claude_cli(monkeypatch, capsys):
    monkeypatch.setattr("shutil.which", lambda name: None)
    rc = nm._install_mcp(pathlib.Path("/proj"))
    assert rc == 1
    assert "claude" in capsys.readouterr().err.lower()
