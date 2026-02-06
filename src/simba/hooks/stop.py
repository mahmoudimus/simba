"""Stop hook — guardian signal check + tailor error capture.

Reads stdin JSON, checks response for [✓ rules] signal marker,
runs tailor error capture pipeline on transcript.
"""

from __future__ import annotations

import json
import pathlib
import sys

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

    combined = "\n\n".join(parts)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "Stop",
            "additionalContext": combined,
        }
    }
    return json.dumps(output)


if __name__ == "__main__":
    hook_data: dict = {}
    try:
        raw = sys.stdin.read()
        if raw:
            hook_data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    print(main(hook_data))
