from __future__ import annotations

import json

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
    assert "memory_count" in body
    assert isinstance(body["memory_count"], int)
    assert "transform" in body


def test_hook_endpoint_empty_body_ok():
    resp = _client().post("/hook/prompt_submit", json={})
    assert resp.status_code == 200


def test_hook_endpoint_unknown_event_404():
    resp = _client().post("/hook/bogus", json={})
    assert resp.status_code == 404


def test_hook_endpoint_null_body_fails_open():
    # The CLI path catches a decode error and continues with {}; the daemon must
    # match (fail-open, not 422).
    resp = _client().post("/hook/prompt_submit", json=None)
    assert resp.status_code == 200


def test_hook_endpoint_array_body_fails_open():
    resp = _client().post("/hook/prompt_submit", json=[])
    assert resp.status_code == 200


def test_hook_endpoint_malformed_body_fails_open():
    resp = _client().post(
        "/hook/prompt_submit",
        content="not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200


def test_hook_endpoint_transform_present_in_response():
    resp = _client().post("/hook/prompt_submit", json={"prompt": "", "cwd": "/tmp"})
    assert resp.status_code == 200
    assert "transform" in resp.json()


def test_hook_endpoint_pre_tool_returns_canonical_fields():
    resp = _client().post(
        "/hook/pre_tool",
        json={"tool_name": "Read", "tool_input": {}, "cwd": "/tmp"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "transform" in body
    assert "escalated_block" in body
    # No redirect/rule fired → no block/transform/escalation.
    assert body["block_reason"] is None
    assert body["transform"] is None
    assert body["escalated_block"] is None


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
    # A real Path.cwd() leak would write to the process cwd and never create
    # tmp_path/.simba, so this positive assertion catches the realistic leak
    # without depending on the daemon being quiescent.
    assert (tmp_path / ".simba").exists()
