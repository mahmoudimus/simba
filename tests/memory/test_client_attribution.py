"""The daemon attributes each request to its X-Simba-Client header."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

import simba.memory.fts
import simba.memory.server
from simba.harness.client import CLIENT_HEADER
from simba.memory.diagnostics import DiagnosticsTracker


@pytest_asyncio.fixture
async def app_with_diag(tmp_path, memory_config, lance_table, mock_embed):
    """App wired with a DiagnosticsTracker (report disabled) for attribution."""
    app = simba.memory.server.create_app(memory_config)
    app.state.table = lance_table
    app.state.embed = mock_embed
    app.state.embed_query = mock_embed
    app.state.db_path = None
    app.state.cwd = tmp_path
    fts_path = tmp_path / simba.memory.fts.FTS_FILENAME
    simba.memory.fts.init(fts_path, tokenize=memory_config.fts_tokenize)
    app.state.fts_path = str(fts_path)
    diag = DiagnosticsTracker(report_interval=0)  # 0 disables auto-report
    app.state.diagnostics = diag
    return app, diag


@pytest.mark.asyncio
async def test_recall_records_client_header(app_with_diag) -> None:
    app, diag = app_with_diag
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/recall", json={"query": "hello"}, headers={CLIENT_HEADER: "pi"}
        )
    assert resp.status_code == 200
    assert diag._client_hits["pi"] == 1


@pytest.mark.asyncio
async def test_recall_logs_client(app_with_diag, caplog) -> None:
    app, _diag = app_with_diag
    transport = httpx.ASGITransport(app=app)
    with caplog.at_level("INFO", logger="simba.memory"):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.post(
                "/recall", json={"query": "hello"}, headers={CLIENT_HEADER: "codex"}
            )
    assert any("client=codex" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_missing_header_records_unknown(app_with_diag) -> None:
    app, diag = app_with_diag
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        await ac.post("/recall", json={"query": "hello"})
    assert diag._client_hits["unknown"] == 1


@pytest.mark.asyncio
async def test_stats_exposes_client_hits(app_with_diag) -> None:
    app, _diag = app_with_diag
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        await ac.post("/recall", json={"query": "a"}, headers={CLIENT_HEADER: "pi"})
        await ac.post("/recall", json={"query": "b"}, headers={CLIENT_HEADER: "pi"})
        await ac.post(
            "/recall", json={"query": "c"}, headers={CLIENT_HEADER: "claude-code"}
        )
        resp = await ac.get("/stats")
    body = resp.json()
    assert resp.status_code == 200
    # The /stats GET itself also counts toward "pi"? No — it carries no header, so
    # the three recalls dominate. pi=2, claude-code=1, plus the unknown GET.
    assert body["clientHits"]["pi"] == 2
    assert body["clientHits"]["claude-code"] == 1
