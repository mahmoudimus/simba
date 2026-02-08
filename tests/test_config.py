"""Tests for simba.config — registry, TOML loading, merge precedence."""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

import simba.config


@simba.config.configurable("test_section")
@dataclasses.dataclass
class _TestConfig:
    port: int = 9999
    name: str = "default"
    ratio: float = 0.5
    enabled: bool = True


class TestConfigurable:
    def test_registers_section(self) -> None:
        assert "test_section" in simba.config.list_sections()
        assert simba.config.list_sections()["test_section"] is _TestConfig

    def test_load_defaults(self, tmp_path: pathlib.Path) -> None:
        cfg = simba.config.load("test_section", root=tmp_path)
        assert cfg.port == 9999
        assert cfg.name == "default"
        assert cfg.ratio == 0.5
        assert cfg.enabled is True

    def test_unknown_section(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(KeyError, match="Unknown config section"):
            simba.config.load("nonexistent_section_xyz", root=tmp_path)


class TestTomlMerge:
    def test_local_overrides_default(self, tmp_path: pathlib.Path) -> None:
        toml_path = tmp_path / ".simba" / "config.toml"
        toml_path.parent.mkdir(parents=True)
        toml_path.write_text("[test_section]\nport = 1234\n")
        cfg = simba.config.load("test_section", root=tmp_path)
        assert cfg.port == 1234
        assert cfg.name == "default"  # unchanged

    def test_global_overrides_default(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_dir = tmp_path / "global_config"
        global_dir.mkdir()
        global_toml = global_dir / "config.toml"
        global_toml.write_text("[test_section]\nname = 'global_name'\n")
        monkeypatch.setattr(
            simba.config, "_global_path", lambda: global_toml
        )
        cfg = simba.config.load("test_section", root=tmp_path)
        assert cfg.name == "global_name"

    def test_local_overrides_global(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Global sets port=1111
        global_dir = tmp_path / "global_config"
        global_dir.mkdir()
        global_toml = global_dir / "config.toml"
        global_toml.write_text("[test_section]\nport = 1111\n")
        monkeypatch.setattr(
            simba.config, "_global_path", lambda: global_toml
        )
        # Local sets port=2222
        local_toml = tmp_path / ".simba" / "config.toml"
        local_toml.parent.mkdir(parents=True)
        local_toml.write_text("[test_section]\nport = 2222\n")
        cfg = simba.config.load("test_section", root=tmp_path)
        assert cfg.port == 2222


class TestSetAndReset:
    def test_set_local(self, tmp_path: pathlib.Path) -> None:
        simba.config.set_value(
            "test_section", "port", "5555", scope="local", root=tmp_path
        )
        val = simba.config.get_effective("test_section", "port", root=tmp_path)
        assert val == 5555

    def test_set_global(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_toml = tmp_path / "global" / "config.toml"
        monkeypatch.setattr(
            simba.config, "_global_path", lambda: global_toml
        )
        simba.config.set_value(
            "test_section", "name", "custom", scope="global", root=tmp_path
        )
        val = simba.config.get_effective("test_section", "name", root=tmp_path)
        assert val == "custom"

    def test_reset_local(self, tmp_path: pathlib.Path) -> None:
        simba.config.set_value(
            "test_section", "port", "7777", scope="local", root=tmp_path
        )
        assert (
            simba.config.get_effective("test_section", "port", root=tmp_path)
            == 7777
        )
        simba.config.reset_value(
            "test_section", "port", scope="local", root=tmp_path
        )
        assert (
            simba.config.get_effective("test_section", "port", root=tmp_path)
            == 9999
        )

    def test_set_unknown_key(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(KeyError, match="Unknown key"):
            simba.config.set_value(
                "test_section", "bogus", "x", root=tmp_path
            )

    def test_set_unknown_section(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(KeyError, match="Unknown config section"):
            simba.config.set_value(
                "no_such_section", "x", "y", root=tmp_path
            )


class TestTypeCoercion:
    def test_coerce_int(self) -> None:
        assert simba.config._coerce("42", int) == 42

    def test_coerce_float(self) -> None:
        assert simba.config._coerce("3.14", float) == pytest.approx(3.14)

    def test_coerce_bool_true(self) -> None:
        assert simba.config._coerce("true", bool) is True
        assert simba.config._coerce("1", bool) is True
        assert simba.config._coerce("yes", bool) is True

    def test_coerce_bool_false(self) -> None:
        assert simba.config._coerce("false", bool) is False
        assert simba.config._coerce("no", bool) is False

    def test_coerce_str(self) -> None:
        assert simba.config._coerce("hello", str) == "hello"
