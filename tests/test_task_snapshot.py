from __future__ import annotations

import json
import pathlib

import simba.__main__ as cli
import simba.db
import simba.task_snapshot as snapshots


def test_task_snapshot_latest_active_and_clear(tmp_path: pathlib.Path) -> None:
    project = str(tmp_path.resolve())
    with simba.db.connect(tmp_path):
        first = snapshots.save(
            project_path=project,
            task="implement active task snapshot",
            summary="sidecar",
            next_step="wire CLI",
            now=100.0,
        )
        assert snapshots.latest(project_path=project).id == first.id
        snapshots.clear(project_path=project, reason="done", now=200.0)
        assert snapshots.latest(project_path=project) is None
        second = snapshots.save(
            project_path=project,
            task="continue borrow list",
            files=["src/simba/task_snapshot.py"],
            blockers=[],
            next_step="run tests",
            now=300.0,
        )
        latest = snapshots.latest(project_path=project)

    assert latest is not None
    assert latest.id == second.id
    assert snapshots.to_dict(latest)["files"] == ["src/simba/task_snapshot.py"]
    assert "next_step: run tests" in snapshots.render(latest)


def test_task_snapshot_cli_save_show_clear(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)

    rc = cli._cmd_task(
        [
            "snapshot",
            "save",
            "--task",
            "ship task snapshot",
            "--summary",
            "append-only sidecar",
            "--next-step",
            "validate",
            "--file",
            "src/simba/task_snapshot.py",
            "--blocker",
            "none",
            "--json",
        ]
    )

    assert rc == 0
    saved = json.loads(capsys.readouterr().out)
    assert saved["task"] == "ship task snapshot"
    assert saved["files"] == ["src/simba/task_snapshot.py"]

    assert cli._cmd_task(["snapshot", "show", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["id"] == saved["id"]
    assert shown["next_step"] == "validate"

    assert cli._cmd_task(["snapshot", "clear", "--reason", "done", "--json"]) == 0
    cleared = json.loads(capsys.readouterr().out)
    assert cleared["status"] == snapshots.STATUS_CLEARED
    assert cli._cmd_task(["snapshot", "show", "--json"]) == 1
    assert json.loads(capsys.readouterr().out) == {}
