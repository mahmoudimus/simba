"""Tests for search.qmd_update — `simba update-qmd`."""

from __future__ import annotations

import pathlib
import subprocess

import pytest

import simba.search.qmd_update as qu


class FakeTools:
    """Stand-ins for `qmd --version`, `npm view` and the installer."""

    def __init__(self, before: str | None, latest: str | None, after: str | None):
        self.version = before
        self.latest = latest
        self.after = after
        self.calls: list[list[str]] = []
        self.install_code = 0

    def run(self, cmd: list[str], timeout: float = 600.0):
        self.calls.append(cmd)
        if cmd[:2] == ["qmd", "--version"]:
            out = f"qmd {self.version} (abc)" if self.version else ""
            return subprocess.CompletedProcess(cmd, 0, out, "")
        if cmd[:2] == ["npm", "view"]:
            return subprocess.CompletedProcess(cmd, 0, f"{self.latest}\n", "")
        if "install" in cmd:
            self.version = self.after
            return subprocess.CompletedProcess(cmd, self.install_code, "", "gyp err")
        raise AssertionError(cmd)


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    def make(before, latest, after, have=("qmd", "npm", "bun")):
        fake = FakeTools(before, latest, after)
        monkeypatch.setattr(qu, "_run", fake.run)
        monkeypatch.setattr(
            qu.shutil, "which", lambda name: f"/bin/{name}" if name in have else None
        )
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        index = tmp_path / "qmd" / "index.sqlite"
        index.parent.mkdir()
        index.write_text("db")
        (tmp_path / "qmd" / "index.sqlite-wal").write_text("wal")
        return fake

    return make


def test_major_upgrade_backs_up_the_index_first(tools, tmp_path) -> None:
    fake = tools("1.0.0", "2.9.0", "2.9.0")
    result = qu.update()
    assert result.ok and (result.before, result.after) == ("1.0.0", "2.9.0")
    assert result.backup == tmp_path / "qmd" / "index.sqlite.bak-qmd-1.0.0"
    assert result.backup.read_text() == "db"
    assert (tmp_path / "qmd" / "index.sqlite.bak-qmd-1.0.0-wal").read_text() == "wal"
    assert ["bun", "install", "-g", "@tobilu/qmd@latest"] in fake.calls


def test_minor_upgrade_skips_backup(tools) -> None:
    tools("2.8.3", "2.9.0", "2.9.0")
    result = qu.update()
    assert result.ok and result.backup is None


def test_already_latest_does_nothing(tools) -> None:
    fake = tools("2.9.0", "2.9.0", "2.9.0")
    result = qu.update()
    assert result.ok and "already the latest" in result.message
    assert not any("install" in c for c in fake.calls)


def test_failed_native_build_still_succeeds_when_qmd_runs(tools) -> None:
    fake = tools("2.8.3", "2.9.0", "2.9.0")
    fake.install_code = 1
    result = qu.update()
    assert result.ok and "but qmd 2.9.0 runs" in result.message


def test_install_that_leaves_old_version_fails(tools) -> None:
    tools("2.8.3", "2.9.0", "2.8.3")
    assert not qu.update().ok


def test_npm_fallback_and_fresh_install(tools) -> None:
    fake = tools(None, "2.9.0", "2.9.0", have=("npm",))
    fake.version = None
    result = qu.update()
    assert ["npm", "install", "-g", "@tobilu/qmd@latest"] in fake.calls
    assert result.before is None and result.backup is None


def test_dry_run_changes_nothing(tools, tmp_path) -> None:
    fake = tools("1.0.0", "2.9.0", "2.9.0")
    result = qu.update(dry_run=True)
    assert result.ok and "would run: bun install" in result.message
    assert not any("install" in c for c in fake.calls)
    assert not (tmp_path / "qmd" / "index.sqlite.bak-qmd-1.0.0").exists()
