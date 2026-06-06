"""REFLECTION memory-type registration (Phase 5, Task A.1)."""

from __future__ import annotations


def test_reflection_in_routes_valid_types() -> None:
    import simba.memory.routes as routes

    assert "REFLECTION" in routes.VALID_TYPES


def test_reflection_in_cli_valid_types() -> None:
    import simba.__main__ as cli

    assert "REFLECTION" in cli._VALID_MEMORY_TYPES
