from __future__ import annotations

import json
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
