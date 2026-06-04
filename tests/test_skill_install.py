"""Tests for skill installation sync (fixes: create-only + wrong filename case).

Old `_install_skills` looked for lowercase `skill.md` and skipped any skill that
already existed (`if dest_file.exists(): continue`), so edits never propagated to
installed copies — which is why a fixed SKILL.md never reached ~/.claude/skills.
`sync_skills` copies the whole skill dir, detects `SKILL.md`, and *updates*
changed files.
"""

from __future__ import annotations

import pathlib

import simba.skill_install as si


def _skill(d: pathlib.Path, name: str, body: str, *, md: str = "SKILL.md") -> None:
    sk = d / name
    sk.mkdir(parents=True, exist_ok=True)
    (sk / md).write_text(body)


def test_fresh_install_counts_and_copies(tmp_path: pathlib.Path) -> None:
    src, dest = tmp_path / "src", tmp_path / "dest"
    _skill(src, "memories-learn", "use simba transcript pending")
    installed, updated = si.sync_skills(src, dest)
    assert (installed, updated) == (1, 0)
    assert (dest / "memories-learn" / "SKILL.md").read_text() == (
        "use simba transcript pending"
    )


def test_updates_changed_skill(tmp_path: pathlib.Path) -> None:
    src, dest = tmp_path / "src", tmp_path / "dest"
    _skill(src, "memories-learn", "NEW: simba transcript pending")
    _skill(dest, "memories-learn", "OLD: read latest.json")  # stale installed copy
    installed, updated = si.sync_skills(src, dest)
    assert (installed, updated) == (0, 1)  # updated, not skipped
    assert "transcript pending" in (dest / "memories-learn" / "SKILL.md").read_text()


def test_idempotent_second_run_is_noop(tmp_path: pathlib.Path) -> None:
    src, dest = tmp_path / "src", tmp_path / "dest"
    _skill(src, "qmd", "search")
    si.sync_skills(src, dest)
    assert si.sync_skills(src, dest) == (0, 0)


def test_ignores_dirs_without_a_skill_md(tmp_path: pathlib.Path) -> None:
    src, dest = tmp_path / "src", tmp_path / "dest"
    (src / "not-a-skill").mkdir(parents=True)
    (src / "not-a-skill" / "readme.txt").write_text("x")
    assert si.sync_skills(src, dest) == (0, 0)


def test_detects_lowercase_skill_md(tmp_path: pathlib.Path) -> None:
    src, dest = tmp_path / "src", tmp_path / "dest"
    _skill(src, "legacy", "x", md="skill.md")
    installed, _ = si.sync_skills(src, dest)
    assert installed == 1


def test_copies_helper_files_not_just_the_md(tmp_path: pathlib.Path) -> None:
    src, dest = tmp_path / "src", tmp_path / "dest"
    _skill(src, "turbo", "do it")
    (src / "turbo" / "scripts").mkdir()
    (src / "turbo" / "scripts" / "run.sh").write_text("echo hi")
    si.sync_skills(src, dest)
    assert (dest / "turbo" / "scripts" / "run.sh").read_text() == "echo hi"
