from __future__ import annotations

import json
import pathlib

import fastapi.testclient

import simba.memory.routes


def _client() -> fastapi.testclient.TestClient:
    app = fastapi.FastAPI()
    app.include_router(simba.memory.routes.router)
    return fastapi.testclient.TestClient(app)


def test_hook_endpoint_prompt_submit_returns_canonical_fields():
    resp = _client().post("/hook/prompt_submit", json={"prompt": "", "cwd": "/tmp"})
    assert resp.status_code == 200
    body = resp.json()
    assert "additional_context" in body
    assert "suppress_output" in body
    assert "transform" in body


def test_hook_endpoint_empty_body_ok():
    resp = _client().post("/hook/prompt_submit", json={})
    assert resp.status_code == 200


def test_hook_endpoint_unknown_event_404():
    resp = _client().post("/hook/bogus", json={})
    assert resp.status_code == 404


def _snapshot(root: pathlib.Path) -> set[pathlib.Path]:
    """Files currently under ``root`` (empty set if it doesn't exist)."""
    return set(root.rglob("*")) if root.exists() else set()


def test_stop_capture_uses_payload_cwd(tmp_path):
    # A transcript with an error so the tailor pipeline actually writes to disk;
    # the write must land under the payload cwd, never the daemon/process cwd.
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "toolUseResult": (
                    "Traceback (most recent call last): AssertionError: boom in "
                    "module foo at /some/path/x.py:12:3 — the command failed badly."
                )
            }
        )
        + "\n"
    )

    # Snapshot the daemon's own cwd so we can prove the dispatch adds nothing here.
    daemon_simba = pathlib.Path.cwd() / ".simba"
    before = _snapshot(daemon_simba)

    resp = _client().post(
        "/hook/stop",
        json={
            "cwd": str(tmp_path),
            "transcript_path": str(transcript),
            "response": "done [✓ rules]",
        },
    )
    assert resp.status_code == 200

    # The tailor capture wrote under <payload cwd>/.simba/, not the process cwd.
    assert (tmp_path / ".simba").exists()
    assert _snapshot(daemon_simba) == before
