"""Tests for `simba rule list` scoping and `simba rule prune`."""

from __future__ import annotations

import simba.db
import simba.rules_cli as rc


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = ""

    def json(self):
        return self._payload


def _rule(mid, project, content="Bash: boom", created="2026-05-30T00:00:00Z"):
    return {
        "id": mid,
        "content": content,
        "projectPath": project,
        "createdAt": created,
        "context": "{}",
    }


def _patch_list(monkeypatch, memories):
    monkeypatch.setattr(
        rc.httpx, "get", lambda *a, **k: _Resp({"memories": memories})
    )


class TestListScoping:
    def test_list_defaults_to_current_project(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(simba.db, "resolve_project_id", lambda p=None: "proj-A")
        _patch_list(monkeypatch, [_rule("m1", "proj-A"), _rule("m2", "proj-B")])

        rc.main(["list"])
        out = capsys.readouterr().out
        assert "m1" in out
        assert "m2" not in out

    def test_list_all_shows_every_project(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(simba.db, "resolve_project_id", lambda p=None: "proj-A")
        _patch_list(monkeypatch, [_rule("m1", "proj-A"), _rule("m2", "proj-B")])

        rc.main(["list", "--all"])
        out = capsys.readouterr().out
        assert "m1" in out
        assert "m2" in out


class TestPrune:
    def test_dry_run_lists_without_deleting(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(simba.db, "resolve_project_id", lambda p=None: "proj-A")
        _patch_list(monkeypatch, [_rule("m1", "proj-A"), _rule("m2", "proj-A")])

        def _no_delete(*a, **k):
            raise AssertionError("delete must not be called on --dry-run")

        monkeypatch.setattr(rc.httpx, "delete", _no_delete)

        rc.main(["prune", "--dry-run"])
        out = capsys.readouterr().out
        assert "Would prune 2" in out

    def test_all_projects_deletes_everything(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(simba.db, "resolve_project_id", lambda p=None: "proj-A")
        _patch_list(
            monkeypatch, [_rule("m1", "proj-A"), _rule("m2", "proj-B")]
        )
        deleted: list[str] = []

        def _delete(url, **k):
            deleted.append(url.rsplit("/", 1)[-1])
            return _Resp({}, status=200)

        monkeypatch.setattr(rc.httpx, "delete", _delete)

        rc.main(["prune", "--all-projects"])
        out = capsys.readouterr().out
        assert set(deleted) == {"m1", "m2"}
        assert "Pruned 2" in out

    def test_scopes_to_current_project_by_default(self, monkeypatch) -> None:
        monkeypatch.setattr(simba.db, "resolve_project_id", lambda p=None: "proj-A")
        _patch_list(
            monkeypatch, [_rule("m1", "proj-A"), _rule("m2", "proj-B")]
        )
        deleted: list[str] = []
        monkeypatch.setattr(
            rc.httpx,
            "delete",
            lambda url, **k: deleted.append(url.rsplit("/", 1)[-1])
            or _Resp({}, 200),
        )

        rc.main(["prune"])
        assert deleted == ["m1"]  # proj-B left untouched

    def test_older_than_filters(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(simba.db, "resolve_project_id", lambda p=None: "proj-A")
        _patch_list(
            monkeypatch,
            [
                _rule("old", "proj-A", created="2000-01-01T00:00:00Z"),
                _rule("new", "proj-A", created="2099-01-01T00:00:00Z"),
            ],
        )
        deleted: list[str] = []
        monkeypatch.setattr(
            rc.httpx,
            "delete",
            lambda url, **k: deleted.append(url.rsplit("/", 1)[-1])
            or _Resp({}, 200),
        )

        rc.main(["prune", "--older-than", "14d"])
        assert deleted == ["old"]

    def test_invalid_older_than_errors(self, monkeypatch) -> None:
        monkeypatch.setattr(simba.db, "resolve_project_id", lambda p=None: "proj-A")
        _patch_list(monkeypatch, [_rule("m1", "proj-A")])
        assert rc.main(["prune", "--older-than", "soon"]) == 1
