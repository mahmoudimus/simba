"""Simba CLI — unified Claude Code plugin.

Usage:
    simba install          Register hooks in ~/.claude/settings.json
    simba install --remove Remove simba hooks from settings
    simba server [opts]    Start the memory daemon
    simba search <cmd>     Project memory operations
    simba hook <event>     Run a hook (called by Claude Code, not users)
"""

from __future__ import annotations

import json
import pathlib
import sys

_HOOK_EVENTS = {
    "SessionStart": "simba.hooks.session_start",
    "UserPromptSubmit": "simba.hooks.user_prompt_submit",
    "PreToolUse": "simba.hooks.pre_tool_use",
    "PostToolUse": "simba.hooks.post_tool_use",
    "PreCompact": "simba.hooks.pre_compact",
    "Stop": "simba.hooks.stop",
}

_HOOK_TIMEOUTS = {
    "SessionStart": 15000,
    "UserPromptSubmit": 3000,
    "PreToolUse": 3000,
    "PostToolUse": 3000,
    "PreCompact": 5000,
    "Stop": 5000,
}

_SETTINGS_PATH = pathlib.Path.home() / ".claude" / "settings.json"


def _build_hooks_config() -> dict:
    """Build the hooks section for settings.json."""
    hooks: dict = {}
    for event in _HOOK_EVENTS:
        timeout = _HOOK_TIMEOUTS[event]
        hooks[event] = [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"simba hook {event}",
                        "timeout": timeout,
                    }
                ]
            }
        ]
    return hooks


def _cmd_install(args: list[str]) -> int:
    """Register or remove simba hooks in ~/.claude/settings.json."""
    remove = "--remove" in args

    if not _SETTINGS_PATH.parent.exists():
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    settings: dict = {}
    if _SETTINGS_PATH.exists():
        settings = json.loads(_SETTINGS_PATH.read_text())

    if remove:
        if "hooks" in settings:
            del settings["hooks"]
        _SETTINGS_PATH.write_text(json.dumps(settings, indent=2) + "\n")
        print("Simba hooks removed from", _SETTINGS_PATH)
        return 0

    settings["hooks"] = _build_hooks_config()
    _SETTINGS_PATH.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Simba hooks registered in {_SETTINGS_PATH}")
    print(f"  {len(_HOOK_EVENTS)} hooks: {', '.join(_HOOK_EVENTS)}")
    return 0


def _cmd_hook(args: list[str]) -> int:
    """Dispatch a hook event. Called by Claude Code, not users."""
    if not args:
        print("Usage: simba hook <event>", file=sys.stderr)
        print(f"Events: {', '.join(_HOOK_EVENTS)}", file=sys.stderr)
        return 1

    event = args[0]
    module_name = _HOOK_EVENTS.get(event)
    if module_name is None:
        print(f"Unknown hook event: {event}", file=sys.stderr)
        return 1

    import importlib

    module = importlib.import_module(module_name)

    hook_data: dict = {}
    try:
        raw = sys.stdin.read()
        if raw:
            hook_data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    print(module.main(hook_data))
    return 0


def _cmd_server(args: list[str]) -> int:
    """Start the memory daemon."""
    # Rewrite sys.argv so argparse in server.main() sees the right args
    sys.argv = ["simba server", *args]
    import simba.memory.server

    simba.memory.server.main()
    return 0


def _cmd_search(args: list[str]) -> int:
    """Project memory operations."""
    sys.argv = ["simba search", *args]
    import simba.search.__main__

    return simba.search.__main__.main()


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    cmd = args[0]
    rest = args[1:]

    if cmd == "install":
        sys.exit(_cmd_install(rest))
    elif cmd == "hook":
        sys.exit(_cmd_hook(rest))
    elif cmd == "server":
        sys.exit(_cmd_server(rest))
    elif cmd == "search":
        sys.exit(_cmd_search(rest))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
