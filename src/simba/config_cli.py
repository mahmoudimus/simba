"""CLI for unified simba configuration.

Usage:
    simba config list                         Show all configurable sections
    simba config get <section.key>            Print effective value
    simba config set [--global] <key> <value> Write a config value
    simba config reset [--global] <key>       Remove an override
    simba config show                         Dump full effective config
    simba config edit [--global]              Open config.toml in $EDITOR
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import simba.config


def _ensure_registry() -> None:
    """Import all config modules so the registry is populated."""
    import simba.hooks.config
    import simba.memory.config
    import simba.search.config
    import simba.sync.config  # noqa: F401


def cmd_list() -> int:
    """Print all registered sections with their fields."""
    _ensure_registry()
    sections = simba.config.list_sections()
    if not sections:
        print("No configurable sections registered.")
        return 0

    for name, cls in sorted(sections.items()):
        print(f"[{name}]")
        for f in dataclasses.fields(cls):
            type_name = f.type if isinstance(f.type, str) else f.type.__name__
            print(f"  {f.name}: {type_name} = {f.default!r}")
        print()
    return 0


def cmd_get(key: str, root: Path) -> int:
    """Print the effective value for section.key."""
    _ensure_registry()
    parts = key.split(".", 1)
    if len(parts) != 2:
        print(f"Invalid key format: {key!r} (expected section.key)", file=sys.stderr)
        return 1
    section, field = parts
    try:
        value = simba.config.get_effective(section, field, root)
    except (KeyError, AttributeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(value)
    return 0


def cmd_set(key: str, value: str, *, global_flag: bool, root: Path) -> int:
    """Set a config value in the TOML file."""
    _ensure_registry()
    parts = key.split(".", 1)
    if len(parts) != 2:
        print(f"Invalid key format: {key!r} (expected section.key)", file=sys.stderr)
        return 1
    section, field = parts
    scope = "global" if global_flag else "local"
    try:
        simba.config.set_value(section, field, value, scope=scope, root=root)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Set {key} = {value} ({scope})")
    return 0


def cmd_reset(key: str, *, global_flag: bool, root: Path) -> int:
    """Remove a config override."""
    _ensure_registry()
    parts = key.split(".", 1)
    if len(parts) != 2:
        print(f"Invalid key format: {key!r} (expected section.key)", file=sys.stderr)
        return 1
    section, field = parts
    scope = "global" if global_flag else "local"
    simba.config.reset_value(section, field, scope=scope, root=root)
    print(f"Reset {key} ({scope})")
    return 0


def cmd_show(root: Path) -> int:
    """Dump the full effective config."""
    _ensure_registry()
    sections = simba.config.list_sections()
    for name in sorted(sections):
        instance = simba.config.load(name, root)
        print(f"[{name}]")
        for f in dataclasses.fields(instance):
            print(f"  {f.name} = {getattr(instance, f.name)!r}")
        print()
    return 0


def cmd_edit(*, global_flag: bool, root: Path) -> int:
    """Open the config TOML in the user's editor."""
    if global_flag:
        path = simba.config._global_path()
    else:
        path = simba.config._local_path(root)

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Simba configuration\n# See: simba config list\n")

    editor = os.environ.get("EDITOR", "vi")
    return subprocess.call([editor, str(path)])


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``simba config``."""
    parser = argparse.ArgumentParser(
        prog="simba config",
        description="Unified simba configuration.",
    )
    sub = parser.add_subparsers(dest="subcmd")

    sub.add_parser("list", help="Show all configurable sections")

    p_get = sub.add_parser("get", help="Print effective value")
    p_get.add_argument("key", help="section.key")
    p_get.add_argument("--path", type=Path, default=Path.cwd())

    p_set = sub.add_parser("set", help="Set a config value")
    p_set.add_argument("key", help="section.key")
    p_set.add_argument("value", help="New value")
    p_set.add_argument("--global", dest="global_flag", action="store_true")
    p_set.add_argument("--path", type=Path, default=Path.cwd())

    p_reset = sub.add_parser("reset", help="Remove an override")
    p_reset.add_argument("key", help="section.key")
    p_reset.add_argument("--global", dest="global_flag", action="store_true")
    p_reset.add_argument("--path", type=Path, default=Path.cwd())

    p_show = sub.add_parser("show", help="Dump full effective config")
    p_show.add_argument("--path", type=Path, default=Path.cwd())

    p_edit = sub.add_parser("edit", help="Open config.toml in $EDITOR")
    p_edit.add_argument("--global", dest="global_flag", action="store_true")
    p_edit.add_argument("--path", type=Path, default=Path.cwd())

    args = parser.parse_args(argv)

    if args.subcmd is None:
        parser.print_help()
        return 1

    if args.subcmd == "list":
        return cmd_list()
    elif args.subcmd == "get":
        return cmd_get(args.key, args.path)
    elif args.subcmd == "set":
        return cmd_set(
            args.key, args.value, global_flag=args.global_flag, root=args.path
        )
    elif args.subcmd == "reset":
        return cmd_reset(args.key, global_flag=args.global_flag, root=args.path)
    elif args.subcmd == "show":
        return cmd_show(args.path)
    elif args.subcmd == "edit":
        return cmd_edit(global_flag=args.global_flag, root=args.path)
    else:
        parser.print_help()
        return 1
