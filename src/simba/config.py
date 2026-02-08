"""Unified configuration registry with TOML-backed persistence.

Provides a ``@configurable`` decorator that registers dataclass config
models into a global registry, and a ``load()`` function that merges
code defaults → global TOML → local TOML into a populated instance.

Config files:
    ~/.config/simba/config.toml     global (user-wide)
    .simba/config.toml              local  (project-specific)
"""

from __future__ import annotations

import dataclasses
import pathlib
import tomllib
from typing import Any, TypeVar

T = TypeVar("T")

_REGISTRY: dict[str, type] = {}


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def configurable(section: str):
    """Class decorator — register a dataclass as a configurable section."""

    def decorator(cls: type[T]) -> type[T]:
        _REGISTRY[section] = cls
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _global_path() -> pathlib.Path:
    return pathlib.Path.home() / ".config" / "simba" / "config.toml"


def _local_path(root: pathlib.Path) -> pathlib.Path:
    return root / ".simba" / "config.toml"


def _find_root(root: pathlib.Path | None = None) -> pathlib.Path:
    """Locate the project root (repo root or cwd)."""
    if root is not None:
        return root
    import simba.db

    found = simba.db.find_repo_root(pathlib.Path.cwd())
    return found if found is not None else pathlib.Path.cwd()


# ---------------------------------------------------------------------------
# TOML I/O
# ---------------------------------------------------------------------------

def _load_toml(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _write_toml(path: pathlib.Path, data: dict[str, Any]) -> None:
    import tomli_w

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tomli_w.dumps(data).encode())


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------

def _coerce(value: str, target_type: type) -> Any:
    """Coerce a CLI string to *target_type*."""
    if target_type is bool:
        return value.lower() in ("true", "1", "yes")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def _field_type(cls: type, field_name: str) -> type:
    """Return the concrete type for a dataclass field."""
    for f in dataclasses.fields(cls):
        if f.name == field_name:
            t = f.type
            # Handle string annotations
            if isinstance(t, str):
                mapping = {"int": int, "float": float, "bool": bool, "str": str}
                return mapping.get(t, str)
            return t
    raise KeyError(field_name)


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def list_sections() -> dict[str, type]:
    """Return a copy of the registry."""
    return dict(_REGISTRY)


def load(section: str, root: pathlib.Path | None = None) -> Any:
    """Load a config section, merging defaults → global → local."""
    cls = _REGISTRY.get(section)
    if cls is None:
        raise KeyError(f"Unknown config section: {section}")

    root = _find_root(root)

    # Collect overrides from TOML files
    global_data = _load_toml(_global_path()).get(section, {})
    local_data = _load_toml(_local_path(root)).get(section, {})

    # Merge: global overrides defaults, local overrides global
    merged = {**global_data, **local_data}

    # Filter to valid field names and coerce types
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in merged.items():
        if k in valid_fields:
            kwargs[k] = v

    return cls(**kwargs)


def get_effective(
    section: str,
    key: str,
    root: pathlib.Path | None = None,
) -> Any:
    """Get the effective value for a single config key."""
    instance = load(section, root)
    return getattr(instance, key)


def set_value(
    section: str,
    key: str,
    value: Any,
    *,
    scope: str = "local",
    root: pathlib.Path | None = None,
) -> None:
    """Write a config value to the appropriate TOML file."""
    # Validate section and key
    cls = _REGISTRY.get(section)
    if cls is None:
        raise KeyError(f"Unknown config section: {section}")
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    if key not in valid_fields:
        raise KeyError(f"Unknown key: {section}.{key}")

    # Coerce string values
    if isinstance(value, str):
        value = _coerce(value, _field_type(cls, key))

    root = _find_root(root)
    path = _global_path() if scope == "global" else _local_path(root)
    data = _load_toml(path)
    data.setdefault(section, {})[key] = value
    _write_toml(path, data)


def reset_value(
    section: str,
    key: str,
    *,
    scope: str = "local",
    root: pathlib.Path | None = None,
) -> None:
    """Remove a config override from the TOML file."""
    root = _find_root(root)
    path = _global_path() if scope == "global" else _local_path(root)
    data = _load_toml(path)
    sec = data.get(section, {})
    if key in sec:
        del sec[key]
        if not sec:
            del data[section]
        _write_toml(path, data)
