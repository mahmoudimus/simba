"""Tests for search.qmd_index — lazy background qmd collection creation."""

from __future__ import annotations

import pathlib
import subprocess
import unittest.mock

import pytest

import simba.search.qmd
import simba.search.qmd_index as qi
import simba.search.rag_context


@pytest.fixture
def repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    home = tmp_path / "home"
    root = home / "src" / "proj"
    (root / "pkg").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    monkeypatch.setattr(qi, "mounts", lambda: [("/", "apfs")])
    monkeypatch.setattr(simba.search.qmd, "is_available", lambda: True)
    return root.resolve()


def _schedule(cwd: pathlib.Path, config: pathlib.Path) -> tuple[bool, list]:
    with unittest.mock.patch.object(qi, "_spawn") as spawn:
        started = qi.schedule_index(cwd, config)
    return started, [call.args[0] for call in spawn.call_args_list]


def test_indexes_the_git_root_from_a_subdirectory(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    started, cmds = _schedule(repo / "pkg", tmp_path / "absent.yml")
    assert started
    cmd = cmds[0]
    assert cmd[cmd.index("qmd") :] == [
        "qmd", "collection", "add", str(repo), "--name", "proj", "--mask", "**/*.md",
    ]  # fmt: skip


def test_attempts_each_root_once_per_retry_window(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    config = tmp_path / "absent.yml"
    assert _schedule(repo, config)[0]
    assert _schedule(repo / "pkg", config) == (False, [])


def test_skips_a_project_that_already_has_a_collection(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    config = tmp_path / "index.yml"
    config.write_text(f"collections:\n  proj:\n    path: {repo}\n")
    assert _schedule(repo, config) == (False, [])


def test_skips_non_git_directories(repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    plain = tmp_path / "home" / "notes"
    plain.mkdir()
    assert _schedule(plain, tmp_path / "absent.yml") == (False, [])


def test_never_indexes_network_filesystems(
    repo: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qi, "mounts", lambda: [("/", "apfs"), (str(repo), "smbfs")])
    assert _schedule(repo, tmp_path / "absent.yml") == (False, [])


def test_unknown_mounts_count_as_network(
    repo: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> list[tuple[str, str]]:
        raise OSError("mount failed")

    monkeypatch.setattr(qi, "mounts", broken)
    assert _schedule(repo, tmp_path / "absent.yml") == (False, [])


def test_never_indexes_home_or_its_ancestors(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    with unittest.mock.patch.object(pathlib.Path, "home", return_value=home):
        assert not qi.eligible(home.resolve())
        assert not qi.eligible(tmp_path.resolve())


def test_parses_macos_mount_output() -> None:
    text = (
        "/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)\n"
        "//u@erdos.local/public on /Volumes/public (smbfs, nodev, nosuid)\n"
    )
    run = unittest.mock.Mock(return_value=unittest.mock.Mock(stdout=text))
    with (
        unittest.mock.patch.object(qi.sys, "platform", "darwin"),
        unittest.mock.patch.object(qi.subprocess, "run", run),
    ):
        assert qi.mounts() == [("/", "apfs"), ("/Volumes/public", "smbfs")]
        assert qi.is_network_path(pathlib.Path("/Volumes/public/wow/x"))
        assert not qi.is_network_path(pathlib.Path("/Volumes/publicity"))


def test_name_collision_gets_a_suffix(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "a" / "proj"
    root.mkdir(parents=True)
    other = tmp_path / "b" / "proj"
    assert qi.collection_name(root.resolve(), {}) == "proj"
    name = qi.collection_name(root.resolve(), {"proj": str(other)})
    assert name.startswith("proj-") and len(name) == len("proj-") + 6


def test_rag_context_schedules_indexing_without_a_collection(
    tmp_path: pathlib.Path,
) -> None:
    with (
        unittest.mock.patch("simba.search.qmd.is_available", return_value=True),
        unittest.mock.patch("simba.search.qmd.collection_for", return_value=None),
        unittest.mock.patch("simba.search.qmd_index.schedule_index") as schedule,
        unittest.mock.patch("simba.search.qmd.search") as search,
    ):
        simba.search.rag_context.build_context(
            "how does the authentication module work", tmp_path
        )
    schedule.assert_called_once_with(tmp_path)
    search.assert_not_called()


def test_auto_index_can_be_turned_off(tmp_path: pathlib.Path) -> None:
    cfg = simba.search.rag_context._search_cfg()
    off = unittest.mock.Mock(wraps=cfg, qmd_auto_index=False)
    for name in ("min_query_length", "memory_token_budget", "max_code_results"):
        setattr(off, name, getattr(cfg, name))
    with (
        unittest.mock.patch.object(
            simba.search.rag_context, "_search_cfg", return_value=off
        ),
        unittest.mock.patch("simba.search.qmd.is_available", return_value=True),
        unittest.mock.patch("simba.search.qmd.collection_for", return_value=None),
        unittest.mock.patch("simba.search.qmd_index.schedule_index") as schedule,
    ):
        simba.search.rag_context.build_context(
            "how does the authentication module work", tmp_path
        )
    schedule.assert_not_called()
