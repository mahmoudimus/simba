from __future__ import annotations

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


def test_hook_endpoint_unknown_event_404():
    resp = _client().post("/hook/bogus", json={})
    assert resp.status_code == 404
