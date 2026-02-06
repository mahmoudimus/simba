"""Tests for memory embeddings service — queue behavior, Ollama HTTP calls (mocked)."""

from __future__ import annotations

import asyncio
import unittest.mock

import httpx
import pytest

import simba.memory.config
import simba.memory.embeddings


@pytest.fixture
def config() -> simba.memory.config.MemoryConfig:
    return simba.memory.config.MemoryConfig(
        ollama_url="http://localhost:11434", timeout_ms=5000
    )


@pytest.fixture
def mock_embedding() -> list[float]:
    return [0.1] * 768


class TestEmbeddingService:
    @pytest.mark.asyncio
    async def test_embed_returns_vector(self, config, mock_embedding):
        service = simba.memory.embeddings.EmbeddingService(config)

        mock_response = httpx.Response(
            200,
            json={"embedding": mock_embedding},
            request=httpx.Request("POST", "http://test"),
        )

        with unittest.mock.patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=unittest.mock.AsyncMock,
            return_value=mock_response,
        ):
            await service.start()
            try:
                result = await service.embed("test text")
                assert result == mock_embedding
                assert len(result) == 768
            finally:
                await service.stop()

    @pytest.mark.asyncio
    async def test_queue_processes_sequentially(self, config, mock_embedding):
        service = simba.memory.embeddings.EmbeddingService(config)
        call_order: list[str] = []

        async def tracking_embed(text: str) -> list[float]:
            call_order.append(text)
            return mock_embedding

        service._embed_direct = tracking_embed

        await service.start()
        try:
            results = await asyncio.gather(
                service.embed("first"),
                service.embed("second"),
                service.embed("third"),
            )
            assert len(results) == 3
            assert call_order == ["first", "second", "third"]
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_embed_propagates_errors(self, config):
        service = simba.memory.embeddings.EmbeddingService(config)

        async def failing_embed(text: str) -> list[float]:
            raise httpx.HTTPStatusError(
                "Ollama error: 500",
                request=httpx.Request("POST", "http://test"),
                response=httpx.Response(500),
            )

        service._embed_direct = failing_embed
        await service.start()
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await service.embed("test")
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, config):
        service = simba.memory.embeddings.EmbeddingService(config)
        await service.start()
        await service.stop()
        await service.stop()  # Should not raise
