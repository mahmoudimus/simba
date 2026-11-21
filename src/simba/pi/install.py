"""Manage Simba's read-only RLM server in Pi's native MCP configuration."""

from __future__ import annotations

import json
import os
import pathlib
import stat
import sys
import tempfile

_SERVER = "simba-rlm"


def _server(command: str) -> dict:
    return {
        "type": "stdio",
        "command": command,
        "args": ["-m", "simba", "neuron", "run", "--root-dir", ".", "--rlm-only"],
        "exposure": "direct",
        "description": "Read-only transcripts (managed by simba pi-install)",
    }


def sync_rlm_server(path: pathlib.Path, *, remove: bool) -> bool:
    """Write or remove our exact registration, retaining user-edited entries."""
    path = path.resolve()
    original = path.read_bytes() if path.exists() else None
    config = json.loads(original) if original is not None else {}
    if not isinstance(config, dict):
        raise ValueError("expected an object")
    servers = config.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError("expected mcpServers to be an object")

    if _SERVER in servers:
        current = servers[_SERVER]
        command = current.get("command") if isinstance(current, dict) else None
        if not isinstance(command, str) or current != _server(command):
            print(f"Preserved existing {_SERVER} entry in {path}", file=sys.stderr)
            return False
    elif remove:
        return False

    if remove:
        del servers[_SERVER]
    else:
        desired = _server(sys.executable)
        if servers.get(_SERVER) == desired:
            return True
        servers[_SERVER] = desired

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if original is not None else 0o600
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".simba-mcp-")
    pending = pathlib.Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")
        pending.chmod(mode)
        if (path.read_bytes() if path.exists() else None) != original:
            raise OSError("MCP configuration changed during installation")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    return True
