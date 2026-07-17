"""Tests for memory routes -- all 6 endpoints via real LanceDB + httpx AsyncClient."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

import simba.config
import simba.db
import simba.memory.anticipated as anticipated
import simba.memory.config
import simba.memory.fts
import simba.memory.provenance as provenance
import simba.memory.server
import simba.memory.usage as usage


class TestStoreEndpoint:
    @pytest.mark.asyncio
    async def test_stores_memory(self, async_client, tmp_path):
        resp = await async_client.post(
            "/store",
            json={"type": "GOTCHA", "content": "test memory"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stored"
        assert data["id"].startswith("mem_")
        assert data["embedding_dims"] == 768
        with simba.db.connect(tmp_path):
            row = usage.get_many([data["id"]])[data["id"]]
        assert row.save_count == 1

    @pytest.mark.asyncio
    async def test_store_appends_provenance(self, async_client, tmp_path):
        resp = await async_client.post(
            "/store",
            json={
                "type": "GOTCHA",
                "content": "provenance memory",
                "sessionSource": "sess-1",
                "occurredAt": "2026-06-01",
                "sourceFile": "src/simba/example.py",
                "sourceSpan": "10-12",
                "extractionAgent": "test-agent",
                "extractionVersion": "1",
                "trustSource": "user_stated",
                "captureOrigin": "cli",
            },
        )
        memory_id = resp.json()["id"]
        with simba.db.connect(tmp_path):
            row = provenance.latest_for([memory_id])[memory_id]
        assert row.occurred_at == "2026-06-01"
        assert row.source_file == "src/simba/example.py"
        assert row.source_span == "10-12"
        assert row.extraction_agent == "test-agent"
        assert row.extraction_version == "1"
        assert row.source_session == "sess-1"
        assert row.trust_source == "user_stated"
        assert row.capture_origin == "cli"
        assert row.trust_score > 0

    @pytest.mark.asyncio
    async def test_store_appends_anticipated_queries(self, async_client, tmp_path):
        resp = await async_client.post(
            "/store",
            json={
                "type": "PATTERN",
                "content": "anticipated query memory",
                "captureOrigin": "cli",
                "anticipatedQueries": [
                    "How do I query this later?",
                    "how do i query this later?",
                    "What should recall match?",
                ],
            },
        )
        memory_id = resp.json()["id"]
        with simba.db.connect(tmp_path):
            rows = anticipated.list_for(memory_id)
        assert [row.query for row in rows] == [
            "How do I query this later?",
            "What should recall match?",
        ]
        assert rows[0].source == "cli"

    @pytest.mark.asyncio
    async def test_rejects_missing_type(self, async_client):
        resp = await async_client.post("/store", json={"content": "test"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_missing_content(self, async_client):
        resp = await async_client.post("/store", json={"type": "GOTCHA"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_invalid_type(self, async_client):
        resp = await async_client.post(
            "/store",
            json={"type": "INVALID", "content": "test"},
        )
        assert resp.status_code == 400
        assert "invalid type" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_rejects_long_content(self, async_client):
        resp = await async_client.post(
            "/store",
            json={"type": "GOTCHA", "content": "x" * 300},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "too long" in detail
        # Consistent with the CLI error: names the exact command to raise
        # the cap, pre-filled with the actual content length (300).
        assert "simba config set memory.max_content_length 300" in detail

    @pytest.mark.asyncio
    async def test_long_content_detail_tracks_configured_limit(
        self, monkeypatch, tmp_path
    ):
        """The 400 detail must derive its cap from that project's layered
        config (a local .simba/config.toml override), NOT the daemon's
        frozen boot-time MemoryConfig and NOT a hardcoded 200 -- mirrors
        the CLI-side proof in test_main_cli.py. Isolate the global config
        layer so this is deterministic regardless of the developer's own
        ~/.config/simba/config.toml."""
        monkeypatch.setattr(
            simba.config, "_global_path", lambda: tmp_path / "global.toml"
        )
        project_dir = tmp_path / "proj"
        (project_dir / ".simba").mkdir(parents=True)
        (project_dir / ".simba" / "config.toml").write_text(
            "[memory]\nmax_content_length = 60\n"
        )
        # Frozen app-level config deliberately says 200 -- if enforcement
        # were still reading THIS instead of the per-project resolver, the
        # assertions below on "60" would fail.
        cfg = simba.memory.config.MemoryConfig(max_content_length=200)
        app = simba.memory.server.create_app(cfg)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/store",
                json={
                    "type": "GOTCHA",
                    "content": "x" * 84,
                    "projectPath": str(project_dir),
                },
            )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "max 60 chars" in detail
        assert "200" not in detail
        assert "simba config set memory.max_content_length 84" in detail

    @pytest.mark.asyncio
    async def test_empty_project_path_resolves_root_as_none_not_cwd(
        self, monkeypatch, tmp_path
    ):
        """CRITICAL FOOTGUN: an empty/blank projectPath (most callers omit
        it entirely for a global-scope memory) must resolve the cap via
        root=None, never Path("") -- Path("") == Path(".") would silently
        scope the cap to wherever the DAEMON process happens to have its
        cwd. Isolate both config layers (global file + repo-root search)
        so this proves the real resolver fails open to the dataclass
        default (200) -- not the app's frozen MemoryConfig (999,
        deliberately different so a stale/wrong lookup would be visible),
        and not a crash or an unlimited cap."""
        monkeypatch.setattr(
            simba.config, "_global_path", lambda: tmp_path / "global.toml"
        )
        empty_root = tmp_path / "no_local_override_here"
        empty_root.mkdir()
        monkeypatch.setattr(simba.db, "find_repo_root", lambda cwd: empty_root)

        cfg = simba.memory.config.MemoryConfig(max_content_length=999)
        app = simba.memory.server.create_app(cfg)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/store", json={"type": "GOTCHA", "content": "x" * 300}
            )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "max 200 chars" in detail
        assert "999" not in detail

    @pytest.mark.asyncio
    async def test_per_project_cap_override_is_isolated_to_that_project(
        self, monkeypatch, tmp_path, lance_table, mock_embed
    ):
        """A project-local .simba/config.toml override on
        memory.max_content_length must raise the cap ONLY for stores
        scoped to that project -- a different project at the same content
        length still hits the base (unoverridden) cap."""
        monkeypatch.setattr(
            simba.config, "_global_path", lambda: tmp_path / "global.toml"
        )
        project_a = tmp_path / "project_a"
        (project_a / ".simba").mkdir(parents=True)
        (project_a / ".simba" / "config.toml").write_text(
            "[memory]\nmax_content_length = 500\n"
        )
        project_b = tmp_path / "project_b"
        project_b.mkdir()

        cfg = simba.memory.config.MemoryConfig(max_content_length=200)
        app = simba.memory.server.create_app(cfg)
        app.state.table = lance_table
        app.state.embed = mock_embed
        app.state.embed_query = mock_embed
        app.state.db_path = None
        app.state.cwd = tmp_path
        fts_path = tmp_path / simba.memory.fts.FTS_FILENAME
        simba.memory.fts.init(fts_path, tokenize=cfg.fts_tokenize)
        app.state.fts_path = str(fts_path)
        transport = httpx.ASGITransport(app=app)

        content = "x" * 300  # over the base 200 cap, under project_a's 500
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp_a = await ac.post(
                "/store",
                json={
                    "type": "GOTCHA",
                    "content": content,
                    "projectPath": str(project_a),
                },
            )
            resp_b = await ac.post(
                "/store",
                json={
                    "type": "GOTCHA",
                    "content": content,
                    "projectPath": str(project_b),
                },
            )
        assert resp_a.status_code == 200
        assert resp_b.status_code == 400
        assert (
            "simba config set memory.max_content_length 300" in resp_b.json()["detail"]
        )

    @pytest.mark.asyncio
    async def test_stores_with_context(self, async_client, lance_table):
        resp = await async_client.post(
            "/store",
            json={
                "type": "PATTERN",
                "content": "test pattern",
                "context": "in module X",
            },
        )
        assert resp.status_code == 200
        stored_id = resp.json()["id"]
        # Verify in the real table
        rows = await lance_table.query().where(f"id = '{stored_id}'").to_list()
        assert len(rows) == 1
        assert rows[0]["context"] == "in module X"

    @pytest.mark.asyncio
    async def test_stores_with_tags(self, async_client, lance_table):
        resp = await async_client.post(
            "/store",
            json={
                "type": "DECISION",
                "content": "chose FastAPI",
                "tags": ["python", "api"],
            },
        )
        assert resp.status_code == 200
        stored_id = resp.json()["id"]
        rows = await lance_table.query().where(f"id = '{stored_id}'").to_list()
        assert len(rows) == 1
        assert json.loads(rows[0]["tags"]) == ["python", "api"]

    @pytest.mark.asyncio
    async def test_default_confidence(self, async_client, lance_table):
        resp = await async_client.post(
            "/store",
            json={"type": "GOTCHA", "content": "test confidence"},
        )
        assert resp.status_code == 200
        stored_id = resp.json()["id"]
        rows = await lance_table.query().where(f"id = '{stored_id}'").to_list()
        assert len(rows) == 1
        assert rows[0]["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_normalizes_project_path_on_store(self, async_client, lance_table):
        """projectPath is stored as an absolute, symlink-resolved path (spec 26).

        A relative or non-normalized path is resolved so the client's ancestor
        chain (also resolved) can match it by string membership.
        """
        import pathlib

        resp = await async_client.post(
            "/store",
            json={
                "type": "PATTERN",
                "content": "scoped fact",
                "projectPath": ".",
            },
        )
        assert resp.status_code == 200
        stored_id = resp.json()["id"]
        rows = await lance_table.query().where(f"id = '{stored_id}'").to_list()
        assert len(rows) == 1
        expected = str(pathlib.Path(".").resolve())
        assert rows[0]["projectPath"] == expected
        assert pathlib.Path(rows[0]["projectPath"]).is_absolute()

    @pytest.mark.asyncio
    async def test_empty_project_path_stays_global(self, async_client, lance_table):
        """An empty projectPath (global memory) is NOT resolved to cwd."""
        resp = await async_client.post(
            "/store",
            json={"type": "PATTERN", "content": "global fact"},
        )
        assert resp.status_code == 200
        stored_id = resp.json()["id"]
        rows = await lance_table.query().where(f"id = '{stored_id}'").to_list()
        assert rows[0]["projectPath"] == ""

    @pytest.mark.asyncio
    async def test_duplicate_detection(self, async_client):
        """Storing the same content twice flags the second as duplicate."""
        resp1 = await async_client.post(
            "/store",
            json={"type": "GOTCHA", "content": "duplicate content"},
        )
        assert resp1.status_code == 200
        assert resp1.json()["status"] == "stored"

        # Second store with same embedding vector -> flagged as duplicate
        resp2 = await async_client.post(
            "/store",
            json={"type": "GOTCHA", "content": "duplicate content again"},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["status"] == "duplicate"
        assert "existing_id" in data2
        assert data2["similarity"] >= 0.92


class TestRecallEndpoint:
    @pytest.mark.asyncio
    async def test_recall_empty(self, async_client):
        """Recall with no user-added memories returns empty list."""
        resp = await async_client.post("/recall", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert "memories" in data
        assert "queryTimeMs" in data
        # Only the SYSTEM init row exists, which is filtered out
        assert len(data["memories"]) == 0

    @pytest.mark.asyncio
    async def test_recall_returns_results(self, async_client, lance_table):
        """Add a memory directly to the table, then recall it."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "mem_test1",
                    "type": "GOTCHA",
                    "content": "test memory for recall",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.85,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                }
            ]
        )
        resp = await async_client.post("/recall", json={"query": "test"})
        data = resp.json()
        assert len(data["memories"]) >= 1
        ids = [m["id"] for m in data["memories"]]
        assert "mem_test1" in ids

    @pytest.mark.asyncio
    async def test_recall_includes_session_source(self, async_client, lance_table):
        """/recall must return sessionSource so RLM can resolve a transcript."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "mem_ss",
                    "type": "GOTCHA",
                    "content": "from a session",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "sess-123",
                    "projectPath": "/p",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                }
            ]
        )
        resp = await async_client.post(
            "/recall", json={"query": "test", "projectPath": "/p"}
        )
        mems = resp.json()["memories"]
        assert any(m.get("sessionSource") == "sess-123" for m in mems)

    @pytest.mark.asyncio
    async def test_recall_with_project_path_filters(self, async_client, lance_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "mem_a",
                    "type": "GOTCHA",
                    "content": "project A memory",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "/path/to/project-a",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                },
                {
                    "id": "mem_b",
                    "type": "GOTCHA",
                    "content": "project B memory",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "/path/to/project-b",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                },
            ]
        )
        resp = await async_client.post(
            "/recall",
            json={"query": "test", "projectPath": "/path/to/project-a"},
        )
        data = resp.json()
        # Only project A memories should pass the filter
        for mem in data["memories"]:
            assert mem["id"] != "mem_b"

    @pytest.mark.asyncio
    async def test_recall_without_project_path_returns_all(
        self, async_client, lance_table
    ):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "mem_a",
                    "type": "GOTCHA",
                    "content": "project A memory",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "/path/to/project-a",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                },
                {
                    "id": "mem_b",
                    "type": "GOTCHA",
                    "content": "project B memory",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "/path/to/project-b",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                },
            ]
        )
        resp = await async_client.post("/recall", json={"query": "test"})
        data = resp.json()
        # Without projectPath filter, both memories are returned
        ids = [m["id"] for m in data["memories"]]
        assert "mem_a" in ids
        assert "mem_b" in ids

    @pytest.mark.asyncio
    async def test_recall_with_project_excludes_global(self, async_client, lance_table):
        """A project-scoped query must NOT return untagged/global memories."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "mem_scoped",
                    "type": "GOTCHA",
                    "content": "project A memory",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "/path/to/project-a",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                },
                {
                    "id": "mem_global",
                    "type": "GOTCHA",
                    "content": "untagged global memory",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                },
            ]
        )
        resp = await async_client.post(
            "/recall",
            json={"query": "test", "projectPath": "/path/to/project-a"},
        )
        ids = [m["id"] for m in resp.json()["memories"]]
        assert "mem_scoped" in ids
        assert "mem_global" not in ids  # no leak from untagged memories

    @pytest.mark.asyncio
    async def test_recall_requires_query(self, async_client):
        resp = await async_client.post("/recall", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_recall_supports_structured_tag_filter(
        self, async_client, lance_table
    ):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "mem_python",
                    "type": "PATTERN",
                    "content": "python memory",
                    "context": "",
                    "tags": '["python"]',
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                },
                {
                    "id": "mem_ruby",
                    "type": "PATTERN",
                    "content": "ruby memory",
                    "context": "",
                    "tags": '["ruby"]',
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                },
            ]
        )
        resp = await async_client.post("/recall", json={"query": "memory tag:python"})
        contents = [m["content"] for m in resp.json()["memories"]]
        assert "python memory" in contents
        assert "ruby memory" not in contents


class TestRecallAdmissionControl:
    """``memory.max_concurrent_recalls`` (0 = unlimited, default): an
    asyncio.Semaphore gating the whole /recall handler. The LLAMA lock
    already serializes native embed/rerank compute end to end, but the
    surrounding pyarrow/RRF/rerank-loop orchestration does not go through
    that lock and can stack unboundedly across concurrent requests ---
    this knob bounds that."""

    async def _build_app(self, tmp_path, lance_table, *, max_concurrent_recalls: int):
        cfg = simba.memory.config.MemoryConfig(
            max_content_length=200,
            duplicate_threshold=0.92,
            max_concurrent_recalls=max_concurrent_recalls,
            # Isolate admission control from the (unrelated) HyDE 2nd-arm
            # feature: with it on, embed_query fires twice per recall
            # (primary + hyde_text), which for a single distinctive query
            # word resolves to the SAME text and confounds the enter/exit
            # ordering these tests assert on.
            expansion_enabled=False,
        )
        app = simba.memory.server.create_app(cfg)
        app.state.table = lance_table
        app.state.db_path = None
        app.state.cwd = tmp_path
        fts_path = tmp_path / simba.memory.fts.FTS_FILENAME
        simba.memory.fts.init(fts_path, tokenize=cfg.fts_tokenize)
        app.state.fts_path = str(fts_path)
        return app

    @pytest.mark.asyncio
    async def test_default_config_builds_no_semaphore(self, tmp_path, lance_table):
        app = await self._build_app(tmp_path, lance_table, max_concurrent_recalls=0)
        assert app.state.recall_semaphore is None

    @pytest.mark.asyncio
    async def test_enabled_config_builds_a_semaphore(self, tmp_path, lance_table):
        app = await self._build_app(tmp_path, lance_table, max_concurrent_recalls=2)
        assert isinstance(app.state.recall_semaphore, asyncio.Semaphore)

    @pytest.mark.asyncio
    async def test_serializes_concurrent_recalls_when_limit_is_one(
        self, tmp_path, lance_table
    ):
        app = await self._build_app(tmp_path, lance_table, max_concurrent_recalls=1)

        order: list[str] = []
        release_first = asyncio.Event()

        async def _embed_query(text: str) -> list[float]:
            order.append(f"enter:{text}")
            if text == "first":
                await release_first.wait()
            order.append(f"exit:{text}")
            return [0.1] * 768

        app.state.embed = _embed_query
        app.state.embed_query = _embed_query

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            first_task = asyncio.create_task(
                ac.post("/recall", json={"query": "first"})
            )
            await asyncio.sleep(0.05)  # let `first` acquire + block inside embed
            second_task = asyncio.create_task(
                ac.post("/recall", json={"query": "second"})
            )
            await asyncio.sleep(0.05)  # `second` must be blocked on the semaphore
            assert order == ["enter:first"]

            release_first.set()
            resp1 = await first_task
            resp2 = await second_task

        assert order == ["enter:first", "exit:first", "enter:second", "exit:second"]
        assert resp1.status_code == 200
        assert resp2.status_code == 200

    @pytest.mark.asyncio
    async def test_default_unlimited_does_not_serialize(self, tmp_path, lance_table):
        """Byte-identical control: with the default (0 = unlimited), two
        concurrent recalls are NOT forced into strict enter/exit/enter/exit
        ordering --- the second can enter before the first exits."""
        app = await self._build_app(tmp_path, lance_table, max_concurrent_recalls=0)

        order: list[str] = []
        release_first = asyncio.Event()
        second_entered = asyncio.Event()

        async def _embed_query(text: str) -> list[float]:
            order.append(f"enter:{text}")
            if text == "first":
                await release_first.wait()
            else:
                second_entered.set()
            order.append(f"exit:{text}")
            return [0.1] * 768

        app.state.embed = _embed_query
        app.state.embed_query = _embed_query

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            first_task = asyncio.create_task(
                ac.post("/recall", json={"query": "first"})
            )
            await asyncio.sleep(0.05)
            second_task = asyncio.create_task(
                ac.post("/recall", json={"query": "second"})
            )
            # Unlike the gated test above, `second` is free to enter now.
            await asyncio.wait_for(second_entered.wait(), timeout=2.0)
            assert order[0] == "enter:first"
            assert "enter:second" in order

            release_first.set()
            resp1 = await first_task
            resp2 = await second_task

        assert resp1.status_code == 200
        assert resp2.status_code == 200


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_ok(self, async_client):
        resp = await async_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["ready"] is True
        assert data["degraded"] is False
        assert "uptime" in data
        assert "memoryCount" in data
        assert data["embeddingModel"] == "bge-large-en-v1.5"
        assert data["embeddingDims"] == 1024
        assert "requestId" in data
        assert data["components"]["vector"]["ready"] is True
        assert data["components"]["fts"]["ready"] is True
        assert data["components"]["embedder"]["ready"] is True
        assert "x-simba-request-id" in resp.headers

    @pytest.mark.asyncio
    async def test_health_rss_null_when_watchdog_disabled(self, async_client):
        """Default config (rss_soft_limit_mb=rss_hard_limit_mb=0): the response
        shape stays stable for existing tests (rssMb/rssPeakMb present but
        null), and no watchdog task needs to be running to compute this ---
        the gate is config, not task presence."""
        resp = await async_client.get("/health")
        data = resp.json()
        assert data["rssMb"] is None
        assert data["rssPeakMb"] is None

    @pytest.mark.asyncio
    async def test_health_rss_present_when_soft_limit_enabled(self):
        app = simba.memory.server.create_app(
            simba.memory.config.MemoryConfig(rss_soft_limit_mb=100_000.0)
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
        data = resp.json()
        assert isinstance(data["rssMb"], (int, float))
        assert data["rssMb"] > 0
        assert isinstance(data["rssPeakMb"], (int, float))
        assert data["rssPeakMb"] > 0

    @pytest.mark.asyncio
    async def test_health_rss_present_when_hard_limit_enabled(self):
        """Either knob alone (not just soft) gates the surfaced fields on."""
        app = simba.memory.server.create_app(
            simba.memory.config.MemoryConfig(rss_hard_limit_mb=100_000.0)
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
        data = resp.json()
        assert data["rssMb"] is not None


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_shape(self, async_client):
        resp = await async_client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "uptime_seconds" in data
        assert "latency" in data
        assert isinstance(data["latency"], dict)
        assert "total_requests" in data

    @pytest.mark.asyncio
    async def test_metrics_reports_latency_per_endpoint(self, async_client):
        # The middleware records latency for every served request.
        await async_client.get("/health")
        await async_client.get("/health")
        resp = await async_client.get("/metrics")
        latency = resp.json()["latency"]
        assert "/health" in latency
        assert "p50" in latency["/health"]
        assert "p95" in latency["/health"]

    @pytest.mark.asyncio
    async def test_health_memory_count(self, async_client, lance_table):
        """Health endpoint reports the real row count from LanceDB."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "mem_health",
                    "type": "GOTCHA",
                    "content": "health check memory",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                }
            ]
        )
        resp = await async_client.get("/health")
        # 1 SYSTEM init row + 1 added row = 2
        assert resp.json()["memoryCount"] == 2


class TestStatsEndpoint:
    @pytest.mark.asyncio
    async def test_empty_stats(self, async_client):
        """Stats endpoint with only the SYSTEM init row."""
        resp = await async_client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        # The SYSTEM init row is included in stats
        assert data["total"] == 1
        assert data["byType"]["SYSTEM"] == 1
        assert data["avgConfidence"] == 1.0

    @pytest.mark.asyncio
    async def test_stats_with_data(self, async_client, lance_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "1",
                    "type": "GOTCHA",
                    "content": "g1",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
                {
                    "id": "2",
                    "type": "GOTCHA",
                    "content": "g2",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.8,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
                {
                    "id": "3",
                    "type": "PATTERN",
                    "content": "p1",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.7,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
            ]
        )
        resp = await async_client.get("/stats")
        data = resp.json()
        # 3 added + 1 SYSTEM init row = 4 total
        assert data["total"] == 4
        assert data["byType"]["GOTCHA"] == 2
        assert data["byType"]["PATTERN"] == 1
        assert data["byType"]["SYSTEM"] == 1
        # avg confidence: (1.0 + 0.9 + 0.8 + 0.7) / 4 = 0.85
        assert data["avgConfidence"] == pytest.approx(0.85, rel=0.01)


class TestListEndpoint:
    @pytest.mark.asyncio
    async def test_empty_list(self, async_client):
        """List with only the SYSTEM init row still returns it."""
        resp = await async_client.get("/list")
        assert resp.status_code == 200
        data = resp.json()
        # The SYSTEM init row is listed
        assert data["total"] == 1
        assert data["memories"][0]["type"] == "SYSTEM"

    @pytest.mark.asyncio
    async def test_list_filtered_by_project(self, async_client, lance_table):
        """`?type=&projectPath=` scopes `total` to that project (the count the
        PreToolUse tool-rule gate reads to skip recall for ruleless projects)."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "r1",
                    "type": "TOOL_RULE",
                    "content": "rule p1",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "/p1",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
                {
                    "id": "r2",
                    "type": "TOOL_RULE",
                    "content": "rule p2",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "/p2",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
            ]
        )
        resp = await async_client.get(
            "/list", params={"type": "TOOL_RULE", "projectPath": "/p1"}
        )
        data = resp.json()
        assert data["total"] == 1
        assert data["memories"][0]["id"] == "r1"

        # A project with no TOOL_RULE rows reports zero — the skip signal.
        empty = await async_client.get(
            "/list", params={"type": "TOOL_RULE", "projectPath": "/nope"}
        )
        assert empty.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_list_with_data(self, async_client, lance_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "1",
                    "type": "GOTCHA",
                    "content": "memory one",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
                {
                    "id": "2",
                    "type": "PATTERN",
                    "content": "memory two",
                    "context": "ctx",
                    "tags": "[]",
                    "confidence": 0.8,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 1,
                    "vector": [0.0] * 768,
                },
            ]
        )
        resp = await async_client.get("/list")
        data = resp.json()
        # 2 added + 1 SYSTEM init = 3
        assert data["total"] == 3
        assert len(data["memories"]) == 3

    @pytest.mark.asyncio
    async def test_list_type_filter(self, async_client, lance_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "1",
                    "type": "GOTCHA",
                    "content": "g1",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
                {
                    "id": "2",
                    "type": "PATTERN",
                    "content": "p1",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.8,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
            ]
        )
        resp = await async_client.get("/list?type=GOTCHA")
        data = resp.json()
        assert data["total"] == 1
        assert data["memories"][0]["type"] == "GOTCHA"

    @pytest.mark.asyncio
    async def test_list_pagination(self, async_client, lance_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for i in range(5):
            await lance_table.add(
                [
                    {
                        "id": str(i),
                        "type": "GOTCHA",
                        "content": f"mem {i}",
                        "context": "",
                        "tags": "[]",
                        "confidence": 0.9,
                        "sessionSource": "",
                        "projectPath": "",
                        "createdAt": now,
                        "lastAccessedAt": now,
                        "accessCount": 0,
                        "vector": [0.0] * 768,
                    }
                ]
            )
        resp = await async_client.get("/list?limit=2&offset=1")
        data = resp.json()
        # 5 added + 1 SYSTEM init = 6 total
        assert data["total"] == 6
        assert len(data["memories"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 1

    @pytest.mark.asyncio
    async def test_list_excludes_vectors_by_default(self, async_client, lance_table):
        """Root cause of the 2026-07-10 CPU/RSS incident: /list must never
        materialize the 1024-dim `vector` column into the response unless the
        caller explicitly opts in (`include_vectors=true`)."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "v1",
                    "type": "GOTCHA",
                    "content": "has a vector",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                },
            ]
        )
        resp = await async_client.get("/list")
        data = resp.json()
        assert data["total"] == 2  # init SYSTEM row + v1
        assert len(data["memories"]) == 2
        for m in data["memories"]:
            assert "vector" not in m

    @pytest.mark.asyncio
    async def test_list_include_vectors_true_returns_vector(
        self, async_client, lance_table
    ):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "v1",
                    "type": "GOTCHA",
                    "content": "has a vector",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.5] * 768,
                },
            ]
        )
        resp = await async_client.get("/list", params={"include_vectors": "true"})
        data = resp.json()
        by_id = {m["id"]: m for m in data["memories"]}
        assert by_id["v1"]["vector"] == [0.5] * 768

    @pytest.mark.asyncio
    async def test_list_fields_returns_exactly_requested_keys(
        self, async_client, lance_table
    ):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "f1",
                    "type": "GOTCHA",
                    "content": "some content",
                    "context": "some context",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "sess-1",
                    "projectPath": "/proj",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
            ]
        )
        resp = await async_client.get("/list", params={"fields": "id,type"})
        data = resp.json()
        assert len(data["memories"]) == data["total"]
        for m in data["memories"]:
            assert set(m.keys()) == {"id", "type"}

    @pytest.mark.asyncio
    async def test_list_fields_vector_requires_include_vectors(self, async_client):
        """`fields=vector` alone (without `include_vectors=true`) must not
        smuggle the embedding back into the response --- `include_vectors` is
        the master switch, `fields` only narrows further."""
        resp = await async_client.get("/list", params={"fields": "id,vector"})
        data = resp.json()
        for m in data["memories"]:
            assert set(m.keys()) == {"id"}

    @pytest.mark.asyncio
    async def test_list_session_source_filter(self, async_client, lance_table):
        """`?sessionSource=` scopes `total`/`memories` to that session --
        the server-side filter `_fetch_session` (episodes/consolidate.py)
        relies on instead of a client-side group-by over the whole corpus."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "s1",
                    "type": "GOTCHA",
                    "content": "session one",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "sess-a",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
                {
                    "id": "s2",
                    "type": "GOTCHA",
                    "content": "session two",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "sess-b",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
            ]
        )
        resp = await async_client.get("/list", params={"sessionSource": "sess-a"})
        data = resp.json()
        assert data["total"] == 1
        assert data["memories"][0]["id"] == "s1"

        empty = await async_client.get(
            "/list", params={"sessionSource": "no-such-session"}
        )
        assert empty.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_list_since_excludes_older(self, async_client, lance_table):
        await lance_table.add(
            [
                {
                    "id": "old",
                    "type": "GOTCHA",
                    "content": "old",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": "2026-01-01T00:00:00Z",
                    "lastAccessedAt": "2026-01-01T00:00:00Z",
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
                {
                    "id": "new",
                    "type": "GOTCHA",
                    "content": "new",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": "2026-07-01T00:00:00Z",
                    "lastAccessedAt": "2026-07-01T00:00:00Z",
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
            ]
        )
        resp = await async_client.get("/list", params={"since": "2026-06-01T00:00:00Z"})
        data = resp.json()
        ids = {m["id"] for m in data["memories"]}
        # `init_0` (the fixture's SYSTEM row) is stamped with the real
        # clock at fixture setup time, which is also >= since -- only "old"
        # must be excluded.
        assert "new" in ids
        assert "old" not in ids

    @pytest.mark.asyncio
    async def test_list_since_mixed_precision_boundary_includes_later_fractional(
        self, async_client, lance_table
    ):
        """The timestamp trap: a row created at `...05.959Z` (fractional
        seconds) is chronologically LATER than `since=...05Z` (whole
        seconds) even though it sorts EARLIER lexicographically ('.' <
        'Z') -- raw string comparison would wrongly exclude it. Robust
        comparison parses both sides to datetimes."""
        await lance_table.add(
            [
                {
                    "id": "frac",
                    "type": "GOTCHA",
                    "content": "fractional, later",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": "2026-07-17T16:40:05.959Z",
                    "lastAccessedAt": "2026-07-17T16:40:05.959Z",
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
            ]
        )
        resp = await async_client.get("/list", params={"since": "2026-07-17T16:40:05Z"})
        data = resp.json()
        ids = {m["id"] for m in data["memories"]}
        assert "frac" in ids

    @pytest.mark.asyncio
    async def test_list_since_invalid_timestamp_400(self, async_client):
        resp = await async_client.get("/list", params={"since": "not-a-date"})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_list_daemon_internal_without_fields_rejected(self, async_client):
        """2026-07-10 incident rule, enforced at runtime: a caller
        self-attributed as the daemon (X-Simba-Client: daemon) MUST pass
        fields= --- an unprojected internal self-call is exactly the shape
        that materialized 45GB of vectors."""
        resp = await async_client.get("/list", headers={"X-Simba-Client": "daemon"})
        assert resp.status_code == 400
        assert "fields=" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_list_daemon_internal_nested_origin_without_fields_rejected(
        self, async_client
    ):
        """A dispatched-hook loopback nests as "<origin>.daemon" (see
        harness/client.py's detect_client) --- still daemon-internal."""
        resp = await async_client.get(
            "/list", headers={"X-Simba-Client": "claude-code.daemon"}
        )
        assert resp.status_code == 400
        assert "fields=" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_list_daemon_internal_with_fields_allowed(self, async_client):
        """The gate only requires fields= --- it doesn't forbid the daemon
        from calling /list correctly."""
        resp = await async_client.get(
            "/list",
            params={"fields": "id,type"},
            headers={"X-Simba-Client": "daemon"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_external_client_without_fields_allowed(self, async_client):
        """Non-daemon clients (claude-code, codex, pi, cli, unknown) are
        unaffected by the internal-projection gate."""
        for client_name in ("claude-code", "codex", "pi", "cli"):
            resp = await async_client.get(
                "/list", headers={"X-Simba-Client": client_name}
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_unknown_client_without_fields_allowed(self, async_client):
        """No X-Simba-Client header at all -> "unknown" -> unaffected (this
        is the existing behavior every other TestListEndpoint test relies
        on, asserted explicitly here so a regression is caught directly)."""
        resp = await async_client.get("/list")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_daemon_internal_context_without_bound_rejected(
        self, async_client
    ):
        """2026-07-17 RSS-storm rule: projection alone (rule 1) didn't stop
        the corpus-wide-context incident --- a daemon-internal caller whose
        fields= includes `context` must ALSO pass a row-bounding constraint
        (sessionSource=, projectPath=, since=, or limit<=1000)."""
        resp = await async_client.get(
            "/list",
            params={"fields": "id,type,content,context", "limit": 5000},
            headers={"X-Simba-Client": "daemon"},
        )
        assert resp.status_code == 400
        assert "context" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_list_daemon_internal_context_with_session_source_allowed(
        self, async_client
    ):
        resp = await async_client.get(
            "/list",
            params={
                "fields": "id,type,content,context",
                "sessionSource": "sess-a",
                "limit": 5000,
            },
            headers={"X-Simba-Client": "daemon"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_daemon_internal_context_with_project_path_allowed(
        self, async_client
    ):
        resp = await async_client.get(
            "/list",
            params={
                "fields": "id,type,content,context",
                "projectPath": "/proj",
                "limit": 5000,
            },
            headers={"X-Simba-Client": "daemon"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_daemon_internal_context_with_since_allowed(self, async_client):
        resp = await async_client.get(
            "/list",
            params={
                "fields": "id,type,content,context",
                "since": "2026-01-01T00:00:00Z",
                "limit": 5000,
            },
            headers={"X-Simba-Client": "daemon"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_daemon_internal_context_with_limit_under_1000_allowed(
        self, async_client
    ):
        resp = await async_client.get(
            "/list",
            params={"fields": "id,type,content,context", "limit": 1000},
            headers={"X-Simba-Client": "daemon"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_daemon_internal_context_over_1000_limit_rejected(
        self, async_client
    ):
        resp = await async_client.get(
            "/list",
            params={"fields": "id,type,content,context", "limit": 1001},
            headers={"X-Simba-Client": "daemon"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_list_daemon_internal_no_context_unaffected_by_new_gate(
        self, async_client
    ):
        """The new bound-gate only fires when `context` is projected ---
        `fields=id,type` (no bound at all, large limit) must stay 200,
        matching the pre-existing rule-1-only gate."""
        resp = await async_client.get(
            "/list",
            params={"fields": "id,type", "limit": 50000},
            headers={"X-Simba-Client": "daemon"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_external_client_context_without_bound_allowed(
        self, async_client
    ):
        """External/CLI/plain clients are unaffected by the new gate too."""
        resp = await async_client.get(
            "/list", params={"fields": "id,type,content,context", "limit": 5000}
        )
        assert resp.status_code == 200


class TestDeleteEndpoint:
    @pytest.mark.asyncio
    async def test_deletes_memory(self, async_client, lance_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "mem_abc",
                    "type": "GOTCHA",
                    "content": "to be deleted",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                }
            ]
        )
        count_before = await lance_table.count_rows()
        assert count_before == 2  # 1 SYSTEM init + 1 added

        resp = await async_client.delete("/memory/mem_abc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        assert data["id"] == "mem_abc"

        count_after = await lance_table.count_rows()
        assert count_after == 1  # Only SYSTEM init remains

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, async_client):
        resp = await async_client.delete("/memory/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
