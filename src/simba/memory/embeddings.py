"""In-process embedding service using llama-cpp-python with async queue.

Replaces the Ollama HTTP-based approach with direct GGUF model inference.
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
    """Async embedding service using llama-cpp-python with sequential queue."""

    def __init__(self, config: simba.memory.config.MemoryConfig) -> None:
        self.config = config
        self._queue: asyncio.Queue[
            tuple[str, TaskType, asyncio.Future[list[float]]]
        ] = asyncio.Queue()
        self._model: llama_cpp.Llama | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Load the GGUF model and start the background queue processor."""
        model_path = await asyncio.to_thread(self._resolve_model_path)
        self._model = await asyncio.to_thread(self._load_model, model_path)
        logger.info("[embed] Model loaded: %s", model_path.name)
        self._task = asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        """Stop the service and release model resources."""
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
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
                result = await asyncio.to_thread(self._embed_sync, text, task)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self._queue.task_done()

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
        import llama_cpp

        return llama_cpp.Llama(
            model_path=str(model_path),
            embedding=True,
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
