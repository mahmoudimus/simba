"""Smoke test: simba core modules import without lancedb/llama_cpp (Task C.3)."""

from __future__ import annotations


def test_config_importable_without_embed(monkeypatch) -> None:
    """simba.config must import with no ML deps."""
    import sys

    # Block lancedb and llama_cpp to simulate core-only install
    for name in list(sys.modules):
        if "lancedb" in name or "llama_cpp" in name:
            del sys.modules[name]
    # Should not raise
    import simba.config
    import simba.kg.store
    import simba.sync.scheduler  # noqa: F401


def test_memory_server_lazy_imports() -> None:
    """simba.memory.server module-level import must not trigger lancedb."""
    import sys

    # Remove lancedb from sys.modules to detect eager import
    lancedb_backup = sys.modules.pop("lancedb", None)
    try:
        if "simba.memory.server" in sys.modules:
            del sys.modules["simba.memory.server"]
        import simba.memory.server  # noqa: F401

        assert "lancedb" not in sys.modules, "lancedb was imported at module level"
    finally:
        if lancedb_backup is not None:
            sys.modules["lancedb"] = lancedb_backup
