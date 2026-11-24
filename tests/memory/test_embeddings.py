"""Tests for embedding service — queue behavior, llama-cpp-python calls (mocked)."""

from __future__ import annotations

import asyncio
import pathlib
import unittest.mock

import pytest

import simba.memory.config
import simba.memory.embeddings


# These fixtures pin nomic-style model + prefixes so the tests exercise the
# embedding *mechanism* (prefix application, model name in the request) and stay
# valid regardless of the production default embedder (now Nemotron-3-Embed-1B, 2048-d).
@pytest.fixture
def config(tmp_path: pathlib.Path) -> simba.memory.config.MemoryConfig:
    return simba.memory.config.MemoryConfig(
        model_path=str(tmp_path / "fake.gguf"),
        embedding_model="nomic-embed-text",
        embed_doc_prefix="search_document: ",
        embed_query_prefix="search_query: ",
    )


@pytest.fixture
def http_config() -> simba.memory.config.MemoryConfig:
    return simba.memory.config.MemoryConfig(
        embed_url="http://localhost:8080",
        embedding_model="nomic-embed-text",
        embed_doc_prefix="search_document: ",
        embed_query_prefix="search_query: ",
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


class TestEmbeddingServiceLocal:
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
                assert result == pytest.approx(
                    simba.memory.embeddings.unit_normalize(mock_embedding)
                )
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
                assert result == pytest.approx(
                    simba.memory.embeddings.unit_normalize(mock_embedding)
                )
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

    @pytest.mark.asyncio
    async def test_is_not_http_mode(
        self,
        config: simba.memory.config.MemoryConfig,
    ) -> None:
        service = simba.memory.embeddings.EmbeddingService(config)
        assert service.is_http_mode is False


class TestEmbeddingServiceHTTP:
    @pytest.mark.asyncio
    async def test_is_http_mode(
        self,
        http_config: simba.memory.config.MemoryConfig,
    ) -> None:
        service = simba.memory.embeddings.EmbeddingService(http_config)
        assert service.is_http_mode is True

    @pytest.mark.asyncio
    async def test_embed_calls_http_endpoint(
        self,
        http_config: simba.memory.config.MemoryConfig,
        mock_embedding: list[float],
    ) -> None:
        service = simba.memory.embeddings.EmbeddingService(http_config)

        mock_response = unittest.mock.MagicMock()
        mock_response.raise_for_status = unittest.mock.MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": mock_embedding}]}

        mock_client = unittest.mock.AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.aclose = unittest.mock.AsyncMock()

        service._http_client = mock_client
        service._task = asyncio.create_task(service._process_loop())

        try:
            result = await service.embed("test text")
            assert result == pytest.approx(
                simba.memory.embeddings.unit_normalize(mock_embedding)
            )
            mock_client.post.assert_called_once_with(
                "/v1/embeddings",
                json={
                    "input": "search_document: test text",
                    "model": "nomic-embed-text",
                },
            )
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_embed_query_uses_query_prefix_http(
        self,
        http_config: simba.memory.config.MemoryConfig,
        mock_embedding: list[float],
    ) -> None:
        service = simba.memory.embeddings.EmbeddingService(http_config)

        mock_response = unittest.mock.MagicMock()
        mock_response.raise_for_status = unittest.mock.MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": mock_embedding}]}

        mock_client = unittest.mock.AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.aclose = unittest.mock.AsyncMock()

        service._http_client = mock_client
        service._task = asyncio.create_task(service._process_loop())

        try:
            result = await service.embed(
                "test query",
                task=simba.memory.embeddings.TaskType.QUERY,
            )
            assert result == pytest.approx(
                simba.memory.embeddings.unit_normalize(mock_embedding)
            )
            mock_client.post.assert_called_once_with(
                "/v1/embeddings",
                json={
                    "input": "search_query: test query",
                    "model": "nomic-embed-text",
                },
            )
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_stop_closes_http_client(
        self,
        http_config: simba.memory.config.MemoryConfig,
    ) -> None:
        service = simba.memory.embeddings.EmbeddingService(http_config)

        mock_client = unittest.mock.AsyncMock()
        mock_client.aclose = unittest.mock.AsyncMock()

        service._http_client = mock_client
        service._task = asyncio.create_task(service._process_loop())

        await service.stop()
        mock_client.aclose.assert_called_once()
        assert service._http_client is None


class TestConfigurablePrefixes:
    def test_doc_and_query_prefixes_applied(self) -> None:
        cfg = simba.memory.config.MemoryConfig(
            embed_doc_prefix="DOC: ", embed_query_prefix="Q: "
        )
        svc = simba.memory.embeddings.EmbeddingService(cfg)
        tt = simba.memory.embeddings.TaskType
        assert svc._prefixed("hello", tt.DOCUMENT) == "DOC: hello"
        assert svc._prefixed("hello", tt.QUERY) == "Q: hello"

    def test_empty_doc_prefix_passes_text_through(self) -> None:
        cfg = simba.memory.config.MemoryConfig(embed_doc_prefix="")
        svc = simba.memory.embeddings.EmbeddingService(cfg)
        assert svc._prefixed("x", simba.memory.embeddings.TaskType.DOCUMENT) == "x"


class TestLlmCliBackend:
    def test_backend_resolves_to_llm_cli(self) -> None:
        cfg = simba.memory.config.MemoryConfig(embed_provider="llm-cli")
        svc = simba.memory.embeddings.EmbeddingService(cfg)
        assert svc._backend == "llm-cli"

    def test_embed_llm_cli_argv_and_parse(self, monkeypatch) -> None:
        import subprocess
        import types

        captured = {}

        def fake_run(argv, **kw):
            captured["argv"] = argv
            return types.SimpleNamespace(
                stdout="[0.1, 0.2, 0.3]", stderr="", returncode=0
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        cfg = simba.memory.config.MemoryConfig(
            embed_provider="llm-cli",
            embedding_model="my-embed",
            embed_query_prefix="Q: ",
        )
        svc = simba.memory.embeddings.EmbeddingService(cfg)
        vec = svc._embed_llm_cli("hi", simba.memory.embeddings.TaskType.QUERY)
        assert vec == [0.1, 0.2, 0.3]
        argv = captured["argv"]
        assert argv[0] == "llm" and "embed" in argv
        assert argv[argv.index("-m") + 1] == "my-embed"
        assert "Q: hi" in argv  # prefixed content passed via -c


class TestUnitNormalize:
    def test_scales_to_unit_length_preserving_direction(self) -> None:
        out = simba.memory.embeddings.unit_normalize([3.0, 4.0])
        assert out == pytest.approx([0.6, 0.8])

    @pytest.mark.parametrize("bad", [[0.0, 0.0], [float("nan"), 1.0], [float("inf")]])
    def test_rejects_degenerate_vectors(self, bad: list[float]) -> None:
        with pytest.raises(ValueError, match="degenerate"):
            simba.memory.embeddings.unit_normalize(bad)

    @pytest.mark.asyncio
    async def test_degenerate_output_fails_the_embed(
        self, config: simba.memory.config.MemoryConfig
    ) -> None:
        service = simba.memory.embeddings.EmbeddingService(config)
        mock_llama = unittest.mock.MagicMock()
        mock_llama.create_embedding.return_value = {"data": [{"embedding": [0.0]}]}
        with (
            unittest.mock.patch.object(service, "_resolve_model_path"),
            unittest.mock.patch.object(service, "_load_model", return_value=mock_llama),
        ):
            await service.start()
            try:
                with pytest.raises(ValueError, match="degenerate"):
                    await service.embed("x")
            finally:
                await service.stop()


class TestResolveNCtx:
    @pytest.mark.parametrize(
        ("requested", "train", "expected"),
        [
            (4096, 262144, 4096),  # Nemotron: bounded, not the 262k training ctx
            (4096, 512, 512),  # BERT-style model: never past trained positions
            (0, 512, 512),  # 0 = the training context
            (0, 262144, 262144),
            (4096, None, 4096),  # unknown training ctx: use the request
            (0, None, 512),  # ...but never an unbounded one
        ],
    )
    def test_resolve(self, requested: int, train: int | None, expected: int) -> None:
        assert simba.memory.embeddings.resolve_n_ctx(requested, train) == expected


class TestLoadModelContext:
    def test_load_uses_bounded_ctx_and_matching_batch(
        self, config: simba.memory.config.MemoryConfig
    ) -> None:
        """n_batch/n_ubatch == n_ctx so `embed_n_ctx` is the real per-text token
        budget (llama-cpp-python truncates to n_batch, default 512)."""
        fake = unittest.mock.MagicMock()
        fake.LLAMA_POOLING_TYPE_MEAN = 1
        meta = {"general.architecture": "mistral3", "mistral3.context_length": "262144"}
        calls: list[dict] = []

        def llama(**kw):
            calls.append(kw)
            m = unittest.mock.MagicMock()
            m.metadata = meta
            return m

        fake.Llama.side_effect = llama
        service = simba.memory.embeddings.EmbeddingService(config)
        with unittest.mock.patch.dict("sys.modules", {"llama_cpp": fake}):
            service._load_model(pathlib.Path("/m.gguf"))
        assert calls[0]["vocab_only"] is True  # metadata probe
        real = calls[-1]
        assert real["n_ctx"] == real["n_batch"] == real["n_ubatch"] == 4096
        assert real["embedding"] is True


class TestEmbedderConfigWarnings:
    def test_defaults_are_clean(self) -> None:
        cfg = simba.memory.config.MemoryConfig()
        assert simba.memory.embeddings.embedder_config_warnings(cfg) == []

    def test_pinned_bge_with_default_prefixes_warns(self) -> None:
        cfg = simba.memory.config.MemoryConfig(
            model_repo="CompendiumLabs/bge-large-en-v1.5-gguf",
            model_file="bge-large-en-v1.5-q8_0.gguf",
            embedding_dims=1024,
        )
        (w,) = simba.memory.embeddings.embedder_config_warnings(cfg)
        assert "embed_doc_prefix" in w

    def test_pinned_bge_with_its_own_prefixes_is_clean(self) -> None:
        cfg = simba.memory.config.MemoryConfig(
            model_repo="CompendiumLabs/bge-large-en-v1.5-gguf",
            model_file="bge-large-en-v1.5-q8_0.gguf",
            embed_doc_prefix="",
            embed_query_prefix="Represent this sentence: ",
        )
        assert simba.memory.embeddings.embedder_config_warnings(cfg) == []

    def test_http_with_default_model_name_warns(self) -> None:
        cfg = simba.memory.config.MemoryConfig(
            embed_url="http://localhost:8080", embed_doc_prefix="d: "
        )
        (w,) = simba.memory.embeddings.embedder_config_warnings(cfg)
        assert "embedding_model" in w
