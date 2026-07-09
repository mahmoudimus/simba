"""Route-level tests for hybrid recall wiring (/store, /recall, /delete, /patch)."""

from __future__ import annotations

import datetime

import httpx
import pytest
import pytest_asyncio

import simba.db
import simba.memory.config
import simba.memory.conflict_store
import simba.memory.fts as fts
import simba.memory.judge_log
import simba.memory.server
import simba.memory.supersession
import simba.memory.vector_db


@pytest.fixture
def memory_config():
    """Disable the recall cache for hybrid tests.

    These assert immediate freshness after store/supersede/delete mutations
    (some via direct DB calls that bypass the /store cache-invalidation), which
    the short-TTL recall cache intentionally trades away. The cache is verified
    in test_recall_cache.py.
    """
    return simba.memory.config.MemoryConfig(
        max_content_length=200,
        duplicate_threshold=0.92,
        recall_cache_ttl_seconds=0.0,
    )


@pytest_asyncio.fixture
async def hybrid_client(memory_config, lance_table, mock_embed, tmp_path):
    """Async client backed by a real LanceDB table + a real FTS mirror."""
    fts_path = tmp_path / fts.FTS_FILENAME
    fts.init(fts_path)
    app = simba.memory.server.create_app(memory_config)
    app.state.table = lance_table
    app.state.embed = mock_embed
    app.state.embed_query = mock_embed
    app.state.db_path = None
    app.state.fts_path = str(fts_path)
    app.state.cwd = tmp_path
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, str(fts_path), app


def _mirror_count(fts_path: str) -> int:
    with fts.connect(fts_path):
        return fts.count()


async def _store(ac, content, *, project="/proj-1", mtype="GOTCHA"):
    resp = await ac.post(
        "/store",
        json={"type": mtype, "content": content, "projectPath": project},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


class FakeConflictJudge:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete_json(self, prompt: str):
        self.calls += 1
        return self.reply


class TestStoreSync:
    @pytest.mark.asyncio
    async def test_store_dual_writes_to_mirror(self, hybrid_client) -> None:
        ac, fts_path, _ = hybrid_client
        await _store(ac, "ruff lints the python source tree")
        assert _mirror_count(fts_path) == 1
        with fts.connect(fts_path):
            hits = fts.search("lints", project_path="/proj-1")
            assert len(hits) == 1

    @pytest.mark.asyncio
    async def test_store_endpoint_records_logged_write_conflict(
        self, hybrid_client
    ) -> None:
        ac, _, app = hybrid_client
        app.state.config.duplicate_threshold = 1.5
        app.state.config.supersede_enabled = False
        app.state.config.conflict_detect_on_write = True
        app.state.llm_client = FakeConflictJudge(
            {"conflict": True, "description": "Alice has two cities"}
        )

        first = await _store(ac, "Alice lives in Paris.")
        second = await _store(ac, "Alice lives in Berlin.")

        assert app.state.llm_client.calls == 1
        with simba.db.connect(app.state.cwd):
            rows = simba.memory.conflict_store.conflicts_among(
                [first["id"], second["id"]],
                project_path="/proj-1",
            )
            logs = simba.memory.judge_log.recent(project_path="/proj-1")
        assert len(rows) == 1
        assert rows[0].description == "Alice has two cities"
        assert len(logs) == 1
        assert logs[0].winner_id == second["id"]


class TestRecallHybrid:
    @pytest.mark.asyncio
    async def test_keyword_arm_supplies_hit_below_vector_cutoff(
        self, hybrid_client
    ) -> None:
        ac, _, _ = hybrid_client
        await _store(ac, "the unique_zeta_marker keyword lives here")
        # minSimilarity 1.1 makes the vector arm return nothing (cosine maxes at
        # 1.0); only the keyword arm can surface the memory.
        resp = await ac.post(
            "/recall",
            json={
                "query": "unique_zeta_marker",
                "projectPath": "/proj-1",
                "minSimilarity": 1.1,
            },
        )
        assert resp.status_code == 200
        memories = resp.json()["memories"]
        contents = [m["content"] for m in memories]
        assert any("unique_zeta_marker" in c for c in contents)


class TestRecallIntentAwareFloor:
    """The daemon picks the cosine floor from query intent when none is sent."""

    @staticmethod
    def _capture_floor(monkeypatch, captured: dict) -> None:
        async def fake_hybrid(
            table,
            fts_path,
            embedding,
            query,
            *,
            min_similarity,
            max_results,
            filters,
            cfg,
            candidate_pool=None,
            extra_embedding=None,
            llm_client=None,
            rerank_cache=None,
            bg_tasks=None,
            cwd=None,
        ):
            captured["min_similarity"] = min_similarity
            return []

        monkeypatch.setattr("simba.memory.hybrid.hybrid_search", fake_hybrid)

    @pytest.mark.asyncio
    async def test_broad_query_lowers_floor(self, hybrid_client, monkeypatch) -> None:
        ac, _, app = hybrid_client
        captured: dict = {}
        self._capture_floor(monkeypatch, captured)
        resp = await ac.post(
            "/recall", json={"query": "list all the decisions", "projectPath": "p1"}
        )
        assert resp.status_code == 200
        assert captured["min_similarity"] == app.state.config.min_similarity_broad

    @pytest.mark.asyncio
    async def test_precise_query_keeps_strict_floor(
        self, hybrid_client, monkeypatch
    ) -> None:
        ac, _, app = hybrid_client
        captured: dict = {}
        self._capture_floor(monkeypatch, captured)
        resp = await ac.post(
            "/recall",
            json={"query": "what port does the daemon use", "projectPath": "p1"},
        )
        assert resp.status_code == 200
        assert captured["min_similarity"] == app.state.config.min_similarity

    @pytest.mark.asyncio
    async def test_explicit_min_similarity_overrides_intent(
        self, hybrid_client, monkeypatch
    ) -> None:
        ac, _, _ = hybrid_client
        captured: dict = {}
        self._capture_floor(monkeypatch, captured)
        resp = await ac.post(
            "/recall",
            json={
                "query": "list all the decisions",
                "projectPath": "p1",
                "minSimilarity": 0.5,
            },
        )
        assert resp.status_code == 200
        assert captured["min_similarity"] == 0.5

    @pytest.mark.asyncio
    async def test_intent_aware_disabled_uses_strict_floor(
        self, hybrid_client, monkeypatch
    ) -> None:
        ac, _, app = hybrid_client
        app.state.config.intent_aware = False
        captured: dict = {}
        self._capture_floor(monkeypatch, captured)
        resp = await ac.post(
            "/recall", json={"query": "list all the decisions", "projectPath": "p1"}
        )
        assert resp.status_code == 200
        assert captured["min_similarity"] == app.state.config.min_similarity


class TestRecallBroadWidening:
    """Broad queries widen maxResults + the RRF candidate pool (Phase 0.1)."""

    @staticmethod
    def _capture(monkeypatch, captured: dict) -> None:
        async def fake_hybrid(
            table,
            fts_path,
            embedding,
            query,
            *,
            min_similarity,
            max_results,
            filters,
            cfg,
            candidate_pool=None,
            extra_embedding=None,
            llm_client=None,
            rerank_cache=None,
            bg_tasks=None,
            cwd=None,
        ):
            captured["max_results"] = max_results
            captured["candidate_pool"] = candidate_pool
            return []

        monkeypatch.setattr("simba.memory.hybrid.hybrid_search", fake_hybrid)

    @pytest.mark.asyncio
    async def test_broad_widens_results_and_pool(
        self, hybrid_client, monkeypatch
    ) -> None:
        ac, _, app = hybrid_client
        captured: dict = {}
        self._capture(monkeypatch, captured)
        # A broad/exploration query that is NOT aggregation (so it tests the broad
        # tier in isolation — "list all" would now upgrade to aggregation breadth,
        # which is default-on as of the 2026-06-14 SoTA-lever policy).
        resp = await ac.post(
            "/recall",
            json={
                "query": "what is the history of the config schema",
                "projectPath": "p1",
            },
        )
        assert resp.status_code == 200
        assert captured["max_results"] == app.state.config.max_results_broad
        assert captured["candidate_pool"] == app.state.config.fts_candidate_pool_broad

    @pytest.mark.asyncio
    async def test_precise_uses_base_results_and_pool(
        self, hybrid_client, monkeypatch
    ) -> None:
        ac, _, app = hybrid_client
        captured: dict = {}
        self._capture(monkeypatch, captured)
        resp = await ac.post(
            "/recall",
            json={"query": "what port does the daemon use", "projectPath": "p1"},
        )
        assert resp.status_code == 200
        assert captured["max_results"] == app.state.config.max_results
        assert captured["candidate_pool"] == app.state.config.fts_candidate_pool

    @pytest.mark.asyncio
    async def test_explicit_max_results_overrides_broad(
        self, hybrid_client, monkeypatch
    ) -> None:
        ac, _, _ = hybrid_client
        captured: dict = {}
        self._capture(monkeypatch, captured)
        resp = await ac.post(
            "/recall",
            json={
                "query": "list all the decisions",
                "projectPath": "p1",
                "maxResults": 2,
            },
        )
        assert resp.status_code == 200
        assert captured["max_results"] == 2


class TestRecallExpansion:
    """expansion_enabled adds a 2nd HyDE vector arm (extra embedding)."""

    @staticmethod
    def _capture(monkeypatch, captured: dict) -> None:
        async def fake_hybrid(
            table,
            fts_path,
            embedding,
            query,
            *,
            min_similarity,
            max_results,
            filters,
            cfg,
            candidate_pool=None,
            extra_embedding=None,
            llm_client=None,
            rerank_cache=None,
            bg_tasks=None,
            cwd=None,
        ):
            captured["extra_embedding"] = extra_embedding
            return []

        monkeypatch.setattr("simba.memory.hybrid.hybrid_search", fake_hybrid)

    @pytest.mark.asyncio
    async def test_disabled_passes_no_extra_embedding(
        self, hybrid_client, monkeypatch
    ) -> None:
        ac, _, app = hybrid_client
        app.state.config.expansion_enabled = False
        captured: dict = {}
        self._capture(monkeypatch, captured)
        resp = await ac.post(
            "/recall",
            json={"query": "open hybrid_search in routes", "projectPath": "p1"},
        )
        assert resp.status_code == 200
        assert captured["extra_embedding"] is None

    @pytest.mark.asyncio
    async def test_enabled_passes_extra_embedding(
        self, hybrid_client, monkeypatch
    ) -> None:
        ac, _, app = hybrid_client
        app.state.config.expansion_enabled = True
        captured: dict = {}
        self._capture(monkeypatch, captured)
        resp = await ac.post(
            "/recall",
            json={"query": "open hybrid_search in routes", "projectPath": "p1"},
        )
        assert resp.status_code == 200
        assert captured["extra_embedding"] is not None


class TestSupersede:
    """Opt-in: a near-duplicate same-type store supersedes the older memory.

    The mock embed returns a constant vector (cosine 1.0 for everything), so the
    thresholds are tuned per test: duplicate_threshold=1.5 keeps the dup-check
    from firing, supersede_threshold=0.5 puts the constant 1.0 in the band.
    """

    @pytest.mark.asyncio
    async def test_supersede_replaces_same_type(self, hybrid_client) -> None:
        ac, fts_path, app = hybrid_client
        app.state.config.duplicate_threshold = 1.5
        app.state.config.supersede_threshold = 0.5
        app.state.config.supersede_enabled = True

        first = await _store(ac, "ruff is the linter", mtype="PATTERN")
        resp = await ac.post(
            "/store",
            json={"type": "PATTERN", "content": "ruff lints", "projectPath": "/proj-1"},
        )
        body = resp.json()
        assert body["status"] == "superseded"
        assert body["supersededId"] == first["id"]
        # Append-only: old and new rows remain in the rebuildable mirror.
        assert _mirror_count(fts_path) == 2
        with simba.db.connect(app.state.cwd):
            chain = simba.memory.supersession.chain(first["id"])
        assert len(chain) == 1
        assert chain[0].old_id == first["id"]
        assert chain[0].new_id == body["id"]
        assert chain[0].project_path == "/proj-1"

    @pytest.mark.asyncio
    async def test_superseded_recall_hit_is_demoted_and_annotated(
        self, hybrid_client
    ) -> None:
        ac, _, app = hybrid_client
        app.state.config.duplicate_threshold = 1.5
        app.state.config.supersede_threshold = 0.5
        app.state.config.supersede_enabled = True

        first = await _store(ac, "ruff is the linter", mtype="PATTERN")
        second_resp = await ac.post(
            "/store",
            json={"type": "PATTERN", "content": "ruff lints", "projectPath": "/proj-1"},
        )
        second = second_resp.json()

        recall = await ac.post(
            "/recall",
            json={"query": "ruff", "projectPath": "/proj-1", "maxResults": 5},
        )
        memories = recall.json()["memories"]
        ids = [m["id"] for m in memories]
        assert ids.index(second["id"]) < ids.index(first["id"])
        old = next(m for m in memories if m["id"] == first["id"])
        assert old["supersededBy"] == second["id"]

    @pytest.mark.asyncio
    async def test_low_trust_replacement_is_pending_until_confirmed(
        self, hybrid_client
    ) -> None:
        ac, _, app = hybrid_client
        app.state.config.duplicate_threshold = 1.5
        app.state.config.supersede_threshold = 0.5
        app.state.config.supersede_enabled = True
        app.state.config.supersede_trust_gate_enabled = True

        first_resp = await ac.post(
            "/store",
            json={
                "type": "PREFERENCE",
                "content": "use ruff for linting",
                "projectPath": "/proj-1",
                "confidence": 1.0,
                "trustSource": "user_stated",
                "captureOrigin": "cli",
            },
        )
        first = first_resp.json()
        second_resp = await ac.post(
            "/store",
            json={
                "type": "PREFERENCE",
                "content": "use pyflakes for linting",
                "projectPath": "/proj-1",
                "confidence": 1.0,
                "trustSource": "llm_extracted",
                "captureOrigin": "hook",
            },
        )
        second = second_resp.json()

        assert second["status"] == "pending_confirmation"
        assert second["supersededCandidateId"] == first["id"]
        audit_id = second["pendingSupersessionId"]

        recall = await ac.post(
            "/recall",
            json={"query": "linting", "projectPath": "/proj-1", "maxResults": 5},
        )
        old = next(m for m in recall.json()["memories"] if m["id"] == first["id"])
        assert "supersededBy" not in old
        assert old["pendingSupersededBy"] == second["id"]
        assert old["pendingSupersessionId"] == audit_id

        with simba.db.connect(app.state.cwd):
            decision = simba.memory.supersession.confirm(audit_id, now=1001.0)
        assert decision.status == simba.memory.supersession.STATUS_ACTIVE

        recall = await ac.post(
            "/recall",
            json={"query": "linting", "projectPath": "/proj-1", "maxResults": 5},
        )
        old = next(m for m in recall.json()["memories"] if m["id"] == first["id"])
        assert old["supersededBy"] == second["id"]
        assert "pendingSupersededBy" not in old

    @pytest.mark.asyncio
    async def test_rejected_pending_replacement_stays_diagnostic_only(
        self, hybrid_client
    ) -> None:
        ac, _, app = hybrid_client
        app.state.config.duplicate_threshold = 1.5
        app.state.config.supersede_threshold = 0.5
        app.state.config.supersede_enabled = True
        app.state.config.supersede_trust_gate_enabled = True

        first_resp = await ac.post(
            "/store",
            json={
                "type": "PREFERENCE",
                "content": "use ruff for linting",
                "projectPath": "/proj-1",
                "confidence": 1.0,
                "trustSource": "user_stated",
                "captureOrigin": "cli",
            },
        )
        first = first_resp.json()
        second_resp = await ac.post(
            "/store",
            json={
                "type": "PREFERENCE",
                "content": "use pyflakes for linting",
                "projectPath": "/proj-1",
                "confidence": 1.0,
                "trustSource": "llm_extracted",
                "captureOrigin": "hook",
            },
        )
        second = second_resp.json()

        with simba.db.connect(app.state.cwd):
            decision = simba.memory.supersession.reject(
                second["pendingSupersessionId"], now=1001.0
            )
        assert decision.status == simba.memory.supersession.STATUS_REJECTED

        recall = await ac.post(
            "/recall",
            json={"query": "linting", "projectPath": "/proj-1", "maxResults": 5},
        )
        old = next(m for m in recall.json()["memories"] if m["id"] == first["id"])
        assert "supersededBy" not in old
        assert "pendingSupersededBy" not in old

    @pytest.mark.asyncio
    async def test_disabled_keeps_both(self, hybrid_client) -> None:
        ac, fts_path, app = hybrid_client
        app.state.config.duplicate_threshold = 1.5
        app.state.config.supersede_enabled = False

        await _store(ac, "alpha", mtype="PATTERN")
        resp = await ac.post(
            "/store",
            json={"type": "PATTERN", "content": "beta", "projectPath": "/proj-1"},
        )
        assert resp.json()["status"] == "stored"
        assert _mirror_count(fts_path) == 2

    @pytest.mark.asyncio
    async def test_supersede_only_same_type(self, hybrid_client) -> None:
        ac, fts_path, app = hybrid_client
        app.state.config.duplicate_threshold = 1.5
        app.state.config.supersede_threshold = 0.5
        app.state.config.supersede_enabled = True

        await _store(ac, "alpha", mtype="GOTCHA")
        resp = await ac.post(
            "/store",
            json={"type": "PATTERN", "content": "beta", "projectPath": "/proj-1"},
        )
        # Different type -> no supersession; both remain.
        assert resp.json()["status"] == "stored"
        assert _mirror_count(fts_path) == 2


class TestDeleteSync:
    @pytest.mark.asyncio
    async def test_delete_removes_from_mirror(self, hybrid_client) -> None:
        ac, fts_path, _ = hybrid_client
        stored = await _store(ac, "deletable kappa memory content")
        assert _mirror_count(fts_path) == 1
        resp = await ac.delete(f"/memory/{stored['id']}")
        assert resp.status_code == 200
        assert _mirror_count(fts_path) == 0


class TestPatchSync:
    @pytest.mark.asyncio
    async def test_patch_moves_project_in_mirror(self, hybrid_client) -> None:
        ac, fts_path, _ = hybrid_client
        stored = await _store(ac, "movable lambda memory content", project="/proj-1")
        resp = await ac.patch(
            f"/memory/{stored['id']}", json={"projectPath": "/proj-2"}
        )
        assert resp.status_code == 200
        with fts.connect(fts_path):
            assert fts.search("lambda", project_path="/proj-1") == []
            moved = fts.search("lambda", project_path="/proj-2")
            assert [m["memory_id"] for m in moved] == [stored["id"]]


class TestHybridDisabled:
    @pytest.mark.asyncio
    async def test_disabled_uses_vector_path(self, hybrid_client) -> None:
        ac, _, app = hybrid_client
        app.state.config.hybrid_enabled = False
        await _store(ac, "vector only mu memory content")
        # Constant mock embeddings -> cosine 1.0 -> vector arm returns the row.
        resp = await ac.post(
            "/recall", json={"query": "anything", "projectPath": "/proj-1"}
        )
        assert resp.status_code == 200
        assert len(resp.json()["memories"]) >= 1


class TestReindex:
    @pytest.mark.asyncio
    async def test_reindex_rebuilds_from_lancedb(self, hybrid_client) -> None:
        ac, fts_path, _ = hybrid_client
        await _store(ac, "reindexable nu memory content")
        # Simulate drift by wiping the mirror out of band.
        with fts.connect(fts_path):
            fts.MemoryFTS.delete().execute()
        assert _mirror_count(fts_path) == 0

        resp = await ac.post("/reindex")
        assert resp.status_code == 200
        assert resp.json()["indexed"] == 1
        assert _mirror_count(fts_path) == 1


class TestCompact:
    @pytest.mark.asyncio
    async def test_compact_dry_run_reports_storage_snapshot(
        self, hybrid_client, tmp_path
    ) -> None:
        ac, _, app = hybrid_client
        db_path = tmp_path / "memories.lance"
        db_path.mkdir()
        (db_path / "retained-data.bin").write_bytes(b"x" * 2048)
        app.state.db_path = str(db_path)

        resp = await ac.post("/compact")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "dry_run"
        assert body["retentionSeconds"] == 86400
        assert body["deleteUnverified"] is False
        assert body["before"]["rows"] == 1
        assert body["before"]["onDiskBytes"] >= 2048
        assert body["before"]["versions"] >= 1

    @pytest.mark.asyncio
    async def test_compact_run_passes_lance_retention_options(
        self, hybrid_client, monkeypatch
    ) -> None:
        ac, _, _ = hybrid_client
        seen: dict[str, object] = {}

        async def _fake_compact(table, *, cleanup_older_than, delete_unverified):
            seen["table"] = table
            seen["cleanup_older_than"] = cleanup_older_than
            seen["delete_unverified"] = delete_unverified
            return {"ok": True}

        monkeypatch.setattr(simba.memory.vector_db, "compact_table", _fake_compact)

        resp = await ac.post(
            "/compact",
            params={
                "dry_run": "false",
                "older_than_seconds": "3600",
                "delete_unverified": "true",
            },
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "compacted"
        assert seen["cleanup_older_than"] == datetime.timedelta(hours=1)
        assert seen["delete_unverified"] is True
