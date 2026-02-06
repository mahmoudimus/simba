"""Tests for embedding service — queue behavior, llama-cpp-python calls (mocked)."""

from __future__ import annotations

import asyncio
import pathlib
import unittest.mock

import pytest

import simba.memory.config
import simba.memory.embeddings


@pytest.fixture
def config(tmp_path: pathlib.Path) -> simba.memory.config.MemoryConfig:
    return simba.memory.config.MemoryConfig(
        model_path=str(tmp_path / "fake.gguf"),
    )


@pytest.fixture
def mock_embedding() -> list[float]:
    return [0.1] * 768


@pytest.fixture
def mock_llama(mock_embedding: list[float]) -> unittest.mock.MagicMock:
    """Create a mock Llama instance."""
    llama = unittest.mock.MagicMock()
    llama.create_embedding.return_value = {"data": [{"embedding": mock_embedding}]}
    return llama


class TestEmbeddingService:
    @pytest.mark.asyncio
    async def test_embed_returns_vector(
        self,
        config: simba.memory.config.MemoryConfig,
        mock_embedding: list[float],
        mock_llama: unittest.mock.MagicMock,
    ) -> None:
        service = simba.memory.embeddings.EmbeddingService(config)

        with (
            unittest.mock.patch.object(
                service,
                "_resolve_model_path",
                return_value=pathlib.Path(config.model_path),
            ),
            unittest.mock.patch.object(service, "_load_model", return_value=mock_llama),
        ):
            await service.start()
            try:
                result = await service.embed("test text")
                assert result == mock_embedding
                assert len(result) == 768
                mock_llama.create_embedding.assert_called_once_with(
                    "search_document: test text"
                )
            finally:
                await service.stop()

    @pytest.mark.asyncio
    async def test_embed_query_uses_query_prefix(
        self,
        config: simba.memory.config.MemoryConfig,
        mock_embedding: list[float],
        mock_llama: unittest.mock.MagicMock,
    ) -> None:
        service = simba.memory.embeddings.EmbeddingService(config)

        with (
            unittest.mock.patch.object(
                service,
                "_resolve_model_path",
                return_value=pathlib.Path(config.model_path),
            ),
            unittest.mock.patch.object(service, "_load_model", return_value=mock_llama),
        ):
            await service.start()
            try:
                result = await service.embed(
                    "test query",
                    task=simba.memory.embeddings.TaskType.QUERY,
                )
                assert result == mock_embedding
                mock_llama.create_embedding.assert_called_once_with(
                    "search_query: test query"
                )
            finally:
                await service.stop()

    @pytest.mark.asyncio
    async def test_queue_processes_sequentially(
        self,
        config: simba.memory.config.MemoryConfig,
        mock_embedding: list[float],
        mock_llama: unittest.mock.MagicMock,
    ) -> None:
        service = simba.memory.embeddings.EmbeddingService(config)
        call_order: list[str] = []

        def tracking_embed(
            text: str, task: simba.memory.embeddings.TaskType
        ) -> list[float]:
            call_order.append(text)
            return mock_embedding

        with (
            unittest.mock.patch.object(
                service,
                "_resolve_model_path",
                return_value=pathlib.Path(config.model_path),
            ),
            unittest.mock.patch.object(service, "_load_model", return_value=mock_llama),
        ):
            await service.start()
            service._embed_sync = tracking_embed  # type: ignore[assignment]
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
    async def test_embed_propagates_errors(
        self,
        config: simba.memory.config.MemoryConfig,
        mock_llama: unittest.mock.MagicMock,
    ) -> None:
        service = simba.memory.embeddings.EmbeddingService(config)
        mock_llama.create_embedding.side_effect = RuntimeError("model error")

        with (
            unittest.mock.patch.object(
                service,
                "_resolve_model_path",
                return_value=pathlib.Path(config.model_path),
            ),
            unittest.mock.patch.object(service, "_load_model", return_value=mock_llama),
        ):
            await service.start()
            try:
                with pytest.raises(RuntimeError, match="model error"):
                    await service.embed("test")
            finally:
                await service.stop()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(
        self,
        config: simba.memory.config.MemoryConfig,
        mock_llama: unittest.mock.MagicMock,
    ) -> None:
        service = simba.memory.embeddings.EmbeddingService(config)

        with (
            unittest.mock.patch.object(
                service,
                "_resolve_model_path",
                return_value=pathlib.Path(config.model_path),
            ),
            unittest.mock.patch.object(service, "_load_model", return_value=mock_llama),
        ):
            await service.start()
            await service.stop()
            await service.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_stop_clears_model(
        self,
        config: simba.memory.config.MemoryConfig,
        mock_llama: unittest.mock.MagicMock,
    ) -> None:
        service = simba.memory.embeddings.EmbeddingService(config)

        with (
            unittest.mock.patch.object(
                service,
                "_resolve_model_path",
                return_value=pathlib.Path(config.model_path),
            ),
            unittest.mock.patch.object(service, "_load_model", return_value=mock_llama),
        ):
            await service.start()
            assert service._model is not None
            await service.stop()
            assert service._model is None
