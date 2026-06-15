from __future__ import annotations

import json
import os
import subprocess
import sys


def _run_cli(args: list[str], stdin: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "simba", *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_hook_canonical_prompt_submit_emits_canonical_json():
    out = _run_cli(
        ["hook-canonical", "prompt_submit"],
        json.dumps({"prompt": "", "cwd": "/tmp"}),
    )
    body = json.loads(out)
    assert "additional_context" in body and "suppress_output" in body


def test_native_stop_envelope_unchanged():
    out = _run_cli(["hook", "Stop"], json.dumps({"cwd": "/tmp"}))
    assert json.loads(out) == {}


def test_dispatch_canonical_injects_process_cwd_when_absent(monkeypatch):
    """A payload with no cwd gets the process cwd before dispatch.

    Belt-and-suspenders against a missing cwd leaking to the daemon's own
    process cwd: the CLI runs in the agent's project directory, so its cwd is
    the correct one. Forces the inline path and captures what dispatch sees.
    """
    import simba.__main__ as cli
    import simba.config
    import simba.harness.core

    # Force the inline path so dispatch() is the chokepoint we can capture.
    monkeypatch.setattr(
        cli, "_hook_via_daemon", lambda event, payload: None
    )

    captured: dict = {}

    def _fake_dispatch(event: str, payload: dict):
        captured["payload"] = payload
        return simba.harness.core.CanonicalResult()

    monkeypatch.setattr(simba.harness.core, "dispatch", _fake_dispatch)
    monkeypatch.chdir("/tmp")

    cli._dispatch_canonical("prompt_submit", {"prompt": ""})

    assert captured["payload"]["cwd"] == os.getcwd()


def test_dispatch_canonical_preserves_explicit_cwd(monkeypatch):
    """An explicit cwd in the payload is not overwritten by the process cwd."""
    import simba.__main__ as cli
    import simba.harness.core

    monkeypatch.setattr(
        cli, "_hook_via_daemon", lambda event, payload: None
    )

    captured: dict = {}

    def _fake_dispatch(event: str, payload: dict):
        captured["payload"] = payload
        return simba.harness.core.CanonicalResult()

    monkeypatch.setattr(simba.harness.core, "dispatch", _fake_dispatch)

    cli._dispatch_canonical("prompt_submit", {"prompt": "", "cwd": "/explicit"})

    assert captured["payload"]["cwd"] == "/explicit"
