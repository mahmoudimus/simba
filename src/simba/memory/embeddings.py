"""Embedding service with two backends: in-process GGUF or external HTTP server.

In-process mode (default): loads a GGUF model via llama-cpp-python.
HTTP mode (--embed-url): calls an OpenAI-compatible /v1/embeddings endpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import logging
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import llama_cpp

    import simba.memory.config

logger = logging.getLogger("simba.memory")


class TaskType(enum.Enum):
    """Nomic-embed-text task prefixes for asymmetric embedding."""

    DOCUMENT = "search_document"
    QUERY = "search_query"


class EmbeddingService:
    """Async embedding service with sequential queue.

    Two backends selected by config.embed_url:
    - empty (default): in-process GGUF via llama-cpp-python
    - set: HTTP POST to an OpenAI-compatible /v1/embeddings endpoint
    """

    def __init__(self, config: simba.memory.config.MemoryConfig) -> None:
        self.config = config
        self._queue: asyncio.Queue[
            tuple[str, TaskType, asyncio.Future[list[float]]]
        ] = asyncio.Queue()
        self._model: llama_cpp.Llama | None = None
        self._http_client: object | None = None  # httpx.AsyncClient when in HTTP mode
        self._task: asyncio.Task[None] | None = None

    @property
    def is_http_mode(self) -> bool:
        return bool(self.config.embed_url)

    async def start(self) -> None:
        """Load the model or connect to the HTTP server, then start the queue."""
        if self.is_http_mode:
            await self._start_http()
        else:
            await self._start_local()
        self._task = asyncio.create_task(self._process_loop())

    async def _start_local(self) -> None:
        """Load the GGUF model for in-process inference."""
        model_path = await asyncio.to_thread(self._resolve_model_path)
        self._model = await asyncio.to_thread(self._load_model, model_path)
        logger.info("[embed] Model loaded: %s", model_path.name)

    async def _start_http(self) -> None:
        """Create an HTTP client for the external embedding server."""
        import httpx

        self._http_client = httpx.AsyncClient(
            base_url=self.config.embed_url,
            timeout=30.0,
        )
        logger.info("[embed] Using external server: %s", self.config.embed_url)

    async def stop(self) -> None:
        """Stop the service and release resources."""
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._http_client is not None:
            await self._http_client.aclose()  # type: ignore[union-attr]
            self._http_client = None
        self._model = None

    async def embed(
        self, text: str, *, task: TaskType = TaskType.DOCUMENT
    ) -> list[float]:
        """Queue text for embedding and return the result.

        Default task is DOCUMENT (for storage). Use TaskType.QUERY for recall.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[float]] = loop.create_future()
        await self._queue.put((text, task, future))
        return await future

    async def _process_loop(self) -> None:
        """Background loop that processes embedding requests sequentially."""
        while True:
            text, task, future = await self._queue.get()
            try:
                if self.is_http_mode:
                    result = await self._embed_http(text, task)
                else:
                    result = await asyncio.to_thread(self._embed_sync, text, task)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self._queue.task_done()

    # ── Local (in-process) backend ──────────────────────────────────

    def _resolve_model_path(self) -> pathlib.Path:
        """Resolve model path: use config.model_path if set, else download."""
        if self.config.model_path:
            path = pathlib.Path(self.config.model_path)
            if not path.exists():
                msg = f"Model file not found: {path}"
                raise FileNotFoundError(msg)
            return path

        import huggingface_hub

        return pathlib.Path(
            huggingface_hub.hf_hub_download(
                repo_id=self.config.model_repo,
                filename=self.config.model_file,
            )
        )

    def _load_model(self, model_path: pathlib.Path) -> llama_cpp.Llama:
        """Load the GGUF model with embedding mode enabled."""
        import ctypes

        import llama_cpp

        # Suppress C-level ggml/llama.cpp log noise (bf16 kernel skips, etc.)
        # Store on class to prevent garbage collection of the C callback.
        if not hasattr(EmbeddingService, "_llama_log_cb"):
            EmbeddingService._llama_log_cb = llama_cpp.llama_log_callback(
                lambda *_args: None
            )
            llama_cpp.llama_log_set(EmbeddingService._llama_log_cb, ctypes.c_void_p(0))

        return llama_cpp.Llama(
            model_path=str(model_path),
            embedding=True,
            n_ctx=0,  # use model's training context (2048 for nomic-embed-text)
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_MEAN,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=False,
        )

    def _embed_sync(self, text: str, task: TaskType) -> list[float]:
        """Synchronous embedding call (runs in executor thread)."""
        assert self._model is not None
        prefixed = f"{task.value}: {text}"
        result = self._model.create_embedding(prefixed)
        embedding = result["data"][0]["embedding"]
        assert isinstance(embedding, list)
        return embedding  # type: ignore[return-value]

    # ── HTTP (external server) backend ──────────────────────────────

    async def _embed_http(self, text: str, task: TaskType) -> list[float]:
        """Call the external OpenAI-compatible embedding endpoint."""
        assert self._http_client is not None
        prefixed = f"{task.value}: {text}"
        response = await self._http_client.post(
            "/v1/embeddings",
            json={
                "input": prefixed,
                "model": self.config.embedding_model,
            },
        )
        response.raise_for_status()
        data = response.json()
        embedding = data["data"][0]["embedding"]
        assert isinstance(embedding, list)
        return embedding  # type: ignore[return-value]
