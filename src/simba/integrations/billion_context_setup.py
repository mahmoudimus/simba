"""Configure Billion Context (bili) for simba, and undo it.

``simba bili setup`` (and SessionStart, automatically, the first time bili is
detected) turns on bili's rules feature (``compress.rules: true``), which
simba uses to keep CORE rules through folds as ``acp_rule`` entries.

simba needs no model credentials here: it extracts learnings from the
summaries bili already stores, with each host's own CLI and login. Who writes
those summaries (bili's ``compress.externalSummary``) is bili tuning, left to
the user; simba never copies API keys between tools.

Every key simba changes is recorded with its previous value in
``~/.cache/simba/bili/setup.json``; ``simba bili restore`` puts exactly those
keys back. After a restore, or once setup has run, SessionStart never
re-applies it, so the user's later changes stand. A config bili accepts but
simba cannot round-trip (JSONC with comments) is never edited. See
docs/adr/2026-11-23-billion-context-adapter.md.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import time
import typing

import simba.integrations.billion_context as _bili

RESTORE_HINT = "run `simba bili restore` to restore settings"
_ABSENT: dict[str, bool] = {"absent": True}


def state_path() -> pathlib.Path:
    return _bili.state_dir() / "setup.json"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def load_state() -> dict[str, typing.Any]:
    try:
        state = json.loads(state_path().read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(state, dict):
        return {}
    if "rules_enabled_at" in state and "applied_at" not in state:
        # First release recorded only that it set compress.rules.
        state = {
            "applied_at": state["rules_enabled_at"],
            "changes": [{"key": "compress.rules", "before": _ABSENT}],
        }
    return state


def _save_state(state: dict[str, typing.Any]) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _get(config: dict, key: str) -> typing.Any:
    node: typing.Any = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return _ABSENT
        node = node[part]
    return node


def _set(config: dict, key: str, value: typing.Any) -> None:
    *parents, leaf = key.split(".")
    node = config
    for part in parents:
        node = node.setdefault(part, {})
    if value is _ABSENT or value == _ABSENT:
        node.pop(leaf, None)
    else:
        node[leaf] = value


def _read_config(path: pathlib.Path) -> tuple[dict | None, str, str | None]:
    """(config, raw text, problem). ``config`` is None when it must not be edited."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, "", None
    except OSError as exc:
        return None, "", f"unreadable: {exc}"
    try:
        config = json.loads(raw) if raw.strip() else {}
    except ValueError:
        return None, raw, "not plain JSON (JSONC?); edit it by hand"
    if not isinstance(config, dict):
        return None, raw, "unexpected config shape; edit it by hand"
    for section in ("compress", "providers"):
        if section in config and not isinstance(config[section], dict):
            return None, raw, f"unexpected {section} section; edit it by hand"
    return config, raw, None


def _write_config(path: pathlib.Path, raw: str, config: dict) -> str | None:
    """Write *config* atomically; back up the old text. Returns the backup path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if raw:
        backup_path = path.with_name(f"{path.name}.bak-simba-{int(time.time())}")
        backup_path.write_text(raw, encoding="utf-8")
        backup = str(backup_path)
    tmp = path.with_name(path.name + ".simba-tmp")
    tmp.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return backup


# ---------------------------------------------------------------------------
# Setup and restore
# ---------------------------------------------------------------------------


def apply(
    *,
    auto: bool = False,
    dry_run: bool = False,
    home: pathlib.Path | None = None,
) -> dict[str, typing.Any]:
    """Configure bili for simba; returns what happened and a user notice."""
    state = load_state()
    path = _bili.bili_config_path()
    agents = _bili.detect_agents(home)
    result: dict[str, typing.Any] = {"agents": agents, "config": str(path)}
    if auto and (state.get("applied_at") or state.get("restored_at")):
        return {**result, "status": "already handled", "notice": ""}
    if not agents:
        status = "bili not used by any installed agent"
        return {**result, "status": status, "notice": ""}
    config, raw, problem = _read_config(path)
    if config is None:
        return {**result, "status": problem, "notice": ""}

    wanted: list[tuple[str, typing.Any]] = [("compress.rules", True)]
    changes = [
        {"key": key, "before": _get(config, key), "after": value}
        for key, value in wanted
        if _get(config, key) != value
    ]
    result["changes"] = [c["key"] for c in changes]
    if dry_run:
        status = (
            "would change " + ", ".join(result["changes"])
            if changes
            else "nothing to change"
        )
        return {**result, "status": status, "notice": ""}

    if changes:
        for change in changes:
            _set(config, change["key"], change["after"])
        result["backup"] = _write_config(path, raw, config)
    recorded = {c["key"]: c for c in state.get("changes", [])}
    for change in changes:  # keep the earliest "before" for each key
        recorded.setdefault(
            change["key"], {"key": change["key"], "before": change["before"]}
        )
    state.update(
        applied_at=time.time(),
        agents=agents,
        config=str(path),
        changes=list(recorded.values()),
    )
    state.pop("restored_at", None)
    _save_state(state)

    applied = ", ".join(result["changes"]) or "nothing; already configured"
    status = f"changed: {applied}"
    notice = ""
    if auto and changes:
        notice = (
            f"Detected Billion Context ({', '.join(agents)}): ran `simba bili setup` "
            f"automatically ({applied}); {RESTORE_HINT}."
        )
    return {**result, "status": status, "notice": notice}


def restore(*, dry_run: bool = False) -> dict[str, typing.Any]:
    """Undo exactly the keys simba changed."""
    state = load_state()
    changes = state.get("changes") or []
    path = pathlib.Path(state.get("config") or _bili.bili_config_path())
    result: dict[str, typing.Any] = {
        "config": str(path),
        "changes": [c["key"] for c in changes],
    }
    if not changes:
        return {**result, "status": "nothing to restore"}
    config, raw, problem = _read_config(path)
    if config is None:
        return {**result, "status": problem}
    if dry_run:
        return {**result, "status": "would restore " + ", ".join(result["changes"])}
    for change in reversed(changes):
        _set(config, change["key"], change["before"])
    result["backup"] = _write_config(path, raw, config)
    _save_state({"restored_at": time.time(), "config": str(path)})
    return {**result, "status": "restored " + ", ".join(result["changes"])}


def auto_setup() -> str:
    """SessionStart entry: set up bili once; return a notice for the user."""
    with contextlib.suppress(Exception):
        return str(apply(auto=True).get("notice") or "")
    return ""
