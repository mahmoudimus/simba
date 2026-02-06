"""Stop hook — guardian signal check + tailor error capture.

Reads stdin JSON, checks response for [✓ rules] signal marker,
runs tailor error capture pipeline on transcript.
"""

from __future__ import annotations

import json
import pathlib

import simba.guardian.check_signal
import simba.tailor.hook


def main(hook_input: dict) -> str:
    """Run the Stop hook pipeline. Returns JSON output string."""

    cwd_str = hook_input.get("cwd")
    cwd = pathlib.Path(cwd_str) if cwd_str else None

    parts: list[str] = []

    # 1. Guardian: check for [✓ rules] signal in response
    response = hook_input.get("response", "")
    if response:
        signal_result = simba.guardian.check_signal.main(response=response, cwd=cwd)
        if signal_result:
            parts.append(signal_result)

    # 2. Tailor: error capture from transcript
    simba.tailor.hook.process_hook(json.dumps(hook_input))

    # Stop hooks don't support hookSpecificOutput — only top-level fields.
    # The tailor error capture writes to disk as a side effect.
    # Return a minimal valid object.
    output: dict = {}
    combined = "\n\n".join(parts)
    if combined:
        output["stopReason"] = combined
    return json.dumps(output)
