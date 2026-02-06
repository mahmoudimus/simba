"""Ollama embedding service with async queue.

Ported from claude-memory/services/embeddings.js.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from simba.memory.config import MemoryConfig

logger = logging.getLogger("simba.memory")

_MAX_RETRIES = 3
_RETRY_DELAYS = (0.5, 1.0, 2.0)


class EmbeddingService:
    """Async embedding service that queues requests to Ollama."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._queue: asyncio.Queue[tuple[str, asyncio.Future[list[float]]]] = (
            asyncio.Queue()
        )
        self._processing = False
        self._client: httpx.AsyncClient | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background queue processor."""
        self._client = httpx.AsyncClient(timeout=self.config.timeout_ms / 1000)
        self._task = asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        """Stop the service and clean up."""
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._client:
            await self._client.aclose()

    async def embed(self, text: str) -> list[float]:
        """Queue a text for embedding and return the result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[float]] = loop.create_future()
        await self._queue.put((text, future))
        return await future

    async def _process_loop(self) -> None:
        """Background loop that processes embedding requests sequentially."""
        while True:
            text, future = await self._queue.get()
            try:
                result = await self._embed_direct(text)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self._queue.task_done()
                # Small delay to let Ollama breathe
                if not self._queue.empty():
                    await asyncio.sleep(0.1)

    async def _embed_direct(self, text: str) -> list[float]:
        """Direct embedding call to Ollama with retry on transient errors."""
        assert self._client is not None
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post(
                    f"{self.config.ollama_url}/api/embeddings",
                    json={
                        "model": self.config.embedding_model,
                        "prompt": text,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["embedding"]
            except httpx.HTTPStatusError as e:
                last_exc = e
                status = e.response.status_code
                if status >= 500 and attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_DELAYS[attempt]
                    logger.warning(
                        "[embed] Ollama %d error, retrying in %.1fs (attempt %d/%d)",
                        status,
                        delay,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                last_exc = e
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_DELAYS[attempt]
                    logger.warning(
                        "[embed] Ollama connection error, retrying in %.1fs (attempt %d/%d)",
                        delay,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
        raise last_exc  # type: ignore[misc]
