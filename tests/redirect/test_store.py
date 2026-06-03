"""Tests for redirect rule storage: TOML file + CLI-managed DB store, merged."""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.redirect.store as store


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def test_toml_loading(tmp_path: pathlib.Path) -> None:
    toml = tmp_path / ".simba" / "redirects.toml"
    toml.parent.mkdir(parents=True, exist_ok=True)
    toml.write_text(
        '[[redirect]]\nprogram = "cargo"\nreplacement = "soldr cargo"\n'
        'reason = "pinned toolchain"\n\n'
        '[[redirect]]\nprogram = "python"\nreplacement = "uv run python"\n'
    )
    rules = store.load_toml(toml)
    assert {r.program for r in rules} == {"cargo", "python"}
    assert rules[0].source == "toml"
    cargo = next(r for r in rules if r.program == "cargo")
    assert cargo.replacement == "soldr cargo"


def test_store_add_list_remove() -> None:
    store.add("cargo", "soldr cargo", reason="pinned", project_path="p")
    rules = store.list_rules(project_path="p")
    assert [(r.program, r.replacement) for r in rules] == [("cargo", "soldr cargo")]
    assert rules[0].source == "store"
    assert store.remove("cargo", project_path="p") == 1
    assert store.list_rules(project_path="p") == []


def test_store_is_project_scoped() -> None:
    store.add("cargo", "soldr cargo", project_path="p1")
    assert store.list_rules(project_path="p2") == []


def test_load_rules_merges_toml_and_store(tmp_path: pathlib.Path) -> None:
    toml = tmp_path / ".simba" / "redirects.toml"
    toml.parent.mkdir(parents=True, exist_ok=True)
    toml.write_text('[[redirect]]\nprogram = "python"\nreplacement = "uv run python"\n')
    store.add("cargo", "soldr cargo", project_path="proj")
    merged = store.load_rules(tmp_path, project_path="proj")
    progs = {r.program for r in merged}
    assert progs == {"python", "cargo"}


def test_load_rules_no_toml_ok(tmp_path: pathlib.Path) -> None:
    store.add("cargo", "soldr cargo", project_path="proj")
    merged = store.load_rules(tmp_path, project_path="proj")
    assert {r.program for r in merged} == {"cargo"}
