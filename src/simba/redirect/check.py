"""Entry point the PreToolUse hook calls: command -> redirect Decision | None.

Loads the hooks config + merged rules (TOML + store) and evaluates. Fail-open:
any error returns None (the command runs unchanged).
"""

from __future__ import annotations

import pathlib
import typing

if typing.TYPE_CHECKING:
    from simba.redirect.rules import Decision


def check_command(command: str, cwd: str | None) -> Decision | None:
    """Return a redirect Decision for ``command``, or None to leave it alone."""
    if not command or not command.strip():
        return None
    try:
        import simba.config
        import simba.db
        import simba.hooks.config  # registers "hooks"
        import simba.redirect.rules as rules
        import simba.redirect.store as store

        _ = simba.hooks.config
        cfg = simba.config.load("hooks")
        if not getattr(cfg, "redirect_enabled", False):
            return None

        cwd_path = pathlib.Path(cwd) if cwd else pathlib.Path.cwd()
        project_id = simba.db.resolve_project_id(cwd_path)
        ruleset = store.load_rules(cwd_path, project_path=project_id)
        if not ruleset:
            return None
        return rules.evaluate(
            command, ruleset, mode=getattr(cfg, "redirect_mode", "deny")
        )
    except Exception:
        return None
