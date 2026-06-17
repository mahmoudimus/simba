"""Tests for the doctrine-triggers store (spec 28 Phase A)."""

from __future__ import annotations

import pathlib

import simba.doctrine.store as store


class TestDoctrineStore:
    def test_add_then_list_roundtrips(self, tmp_path: pathlib.Path) -> None:
        store.add(
            doctrine="Use the worktree skill for PR review.",
            triggers=["review PR", "PR review"],
            trigger_embeddings=[[1.0, 0.0], [0.9, 0.1]],
            risk_tier=True,
            applicable_rules=["redirect:git show pr-N"],
            project_path="/repo",
            cwd=tmp_path,
        )
        rows = store.list_doctrines(project_path="/repo", cwd=tmp_path)
        assert len(rows) == 1
        d = rows[0]
        assert d.doctrine.startswith("Use the worktree skill")
        assert d.triggers == ["review PR", "PR review"]
        assert d.trigger_embeddings == [[1.0, 0.0], [0.9, 0.1]]
        assert d.risk_tier is True
        assert d.applicable_rules == ["redirect:git show pr-N"]
        assert d.project_path == "/repo"

    def test_list_scopes_by_project(self, tmp_path: pathlib.Path) -> None:
        store.add(
            doctrine="A",
            triggers=["a"],
            trigger_embeddings=[[1.0]],
            project_path="/repo-a",
            cwd=tmp_path,
        )
        store.add(
            doctrine="B",
            triggers=["b"],
            trigger_embeddings=[[1.0]],
            project_path="/repo-b",
            cwd=tmp_path,
        )
        a = store.list_doctrines(project_path="/repo-a", cwd=tmp_path)
        assert [d.doctrine for d in a] == ["A"]

    def test_remove_deletes(self, tmp_path: pathlib.Path) -> None:
        store.add(
            doctrine="gone soon",
            triggers=["x"],
            trigger_embeddings=[[1.0]],
            project_path="/repo",
            cwd=tmp_path,
        )
        rows = store.list_doctrines(project_path="/repo", cwd=tmp_path)
        n = store.remove(rows[0].id, project_path="/repo", cwd=tmp_path)
        assert n == 1
        assert store.list_doctrines(project_path="/repo", cwd=tmp_path) == []

    def test_defaults_non_risk_no_rules(self, tmp_path: pathlib.Path) -> None:
        store.add(
            doctrine="plain note",
            triggers=["note"],
            trigger_embeddings=[[1.0]],
            project_path="/repo",
            cwd=tmp_path,
        )
        d = store.list_doctrines(project_path="/repo", cwd=tmp_path)[0]
        assert d.risk_tier is False
        assert d.applicable_rules == []
