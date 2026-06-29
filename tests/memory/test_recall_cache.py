"""Tests for the short-TTL recall result cache."""

from __future__ import annotations

import pytest

from simba.memory.recall_cache import RecallCache


def _key(**over) -> str:
    base = dict(
        query="q",
        project_path="/p",
        min_similarity=None,
        max_results=10,
        filters={},
        project_scopes=None,
    )
    base.update(over)
    return RecallCache.key(**base)


class TestKey:
    def test_same_inputs_same_key(self) -> None:
        assert _key() == _key()

    def test_filter_order_insensitive(self) -> None:
        assert _key(filters={"a": 1, "b": 2}) == _key(filters={"b": 2, "a": 1})

    def test_query_changes_key(self) -> None:
        assert _key(query="x") != _key(query="y")

    def test_project_changes_key(self) -> None:
        assert _key(project_path="/a") != _key(project_path="/b")


class TestGetPut:
    def test_miss_returns_none(self) -> None:
        c = RecallCache(ttl_seconds=5.0)
        assert c.get("k", now=100.0) is None

    def test_hit_within_ttl(self) -> None:
        c = RecallCache(ttl_seconds=5.0)
        c.put("k", [{"id": "m1"}], now=100.0)
        assert c.get("k", now=104.0) == [{"id": "m1"}]

    def test_expired_after_ttl(self) -> None:
        c = RecallCache(ttl_seconds=5.0)
        c.put("k", [{"id": "m1"}], now=100.0)
        assert c.get("k", now=106.0) is None

    def test_expired_entry_is_evicted(self) -> None:
        c = RecallCache(ttl_seconds=5.0)
        c.put("k", [{"id": "m1"}], now=100.0)
        c.get("k", now=106.0)  # expired -> drop
        assert "k" not in c._d

    def test_lru_eviction_at_capacity(self) -> None:
        c = RecallCache(max_entries=2, ttl_seconds=100.0)
        c.put("a", [1], now=0.0)
        c.put("b", [2], now=0.0)
        c.get("a", now=1.0)  # touch a -> b is now LRU
        c.put("c", [3], now=1.0)  # evicts b
        assert c.get("a", now=1.0) == [1]
        assert c.get("c", now=1.0) == [3]
        assert c.get("b", now=1.0) is None

    def test_disabled_ttl_zero_never_hits(self) -> None:
        c = RecallCache(ttl_seconds=0.0)
        c.put("k", [{"id": "m1"}], now=100.0)
        assert c.get("k", now=100.0) is None

    def test_clear_drops_all(self) -> None:
        c = RecallCache(ttl_seconds=5.0)
        c.put("k", [{"id": "m1"}], now=100.0)
        c.clear()
        assert c.get("k", now=100.0) is None


class TestRecallRouteCaching:
    """Identical recalls within the TTL skip the embed+search+rerank pipeline."""

    @pytest.mark.asyncio
    async def test_second_identical_recall_is_cached(
        self, tmp_path, memory_config, lance_table
    ) -> None:
        import httpx

        import simba.memory.fts
        import simba.memory.server

        calls = {"embed": 0}

        async def counting_embed(text: str) -> list[float]:
            calls["embed"] += 1
            return [0.1] * 768

        app = simba.memory.server.create_app(memory_config)
        app.state.table = lance_table
        app.state.embed = counting_embed
        app.state.embed_query = counting_embed
        app.state.db_path = None
        app.state.cwd = tmp_path
        fts_path = tmp_path / simba.memory.fts.FTS_FILENAME
        simba.memory.fts.init(fts_path, tokenize=memory_config.fts_tokenize)
        app.state.fts_path = str(fts_path)
        # create_app already built recall_cache from config (ttl default 5s).
        assert app.state.recall_cache is not None

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            r1 = await ac.post("/recall", json={"query": "same q"})
            embeds_after_r1 = calls["embed"]
            r2 = await ac.post("/recall", json={"query": "same q"})
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json().get("cached") is None  # first is a real recall
        assert r2.json().get("cached") is True  # second served from cache
        # The cached recall added zero embed calls (skipped the pipeline).
        assert embeds_after_r1 >= 1
        assert calls["embed"] == embeds_after_r1
