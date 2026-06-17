"""Tests for the daemon /preflight endpoint (spec 28 Phase C)."""

from __future__ import annotations

import pytest

import simba.guardian.preflight_flag as pf


class TestPreflightRoute:
    @pytest.mark.asyncio
    async def test_returns_brief_with_doctrine(self, async_client, monkeypatch):
        # Store a doctrine-shaped memory so recall surfaces it for the task.
        await async_client.post(
            "/store",
            json={
                "type": "PREFERENCE",
                "content": "regenerate init-schema via the docker script",
                "projectPath": "/repo",
            },
        )
        resp = await async_client.post(
            "/preflight",
            json={
                "task": "regenerate the init-schema",
                "projectPath": "/repo",
                "session_id": "",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "🦁☑" in data["brief"]
        assert "regenerate the init-schema" in data["brief"]
        # the doctrine memory surfaced (mock embed -> cosine 1.0 -> recalled)
        assert any("docker script" in d for d in data["doctrine"])

    @pytest.mark.asyncio
    async def test_sets_per_turn_flag(self, async_client, monkeypatch, tmp_path):
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        resp = await async_client.post(
            "/preflight",
            json={"task": "do a thing", "session_id": "sess-Z"},
        )
        assert resp.status_code == 200
        assert pf.preflight_ran("sess-Z") is True

    @pytest.mark.asyncio
    async def test_empty_task_is_400(self, async_client):
        resp = await async_client.post("/preflight", json={"task": ""})
        assert resp.status_code == 400
