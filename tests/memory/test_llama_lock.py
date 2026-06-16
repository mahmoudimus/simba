"""The process-global llama.cpp lock: concurrent embed + rerank must serialize.

ggml corrupts shared backend buffers when two threads call into llama.cpp at
once (GGML_ASSERT 'tensor write out of bounds' -> SIGSEGV). These tests drive the
REAL embedder and reranker native-call paths concurrently through a tripwire
model that records any overlapping entry, and assert the shared lock prevents it.
"""

from __future__ import annotations

import threading
import time

import simba.memory._llama as _llama
import simba.memory.embeddings as embeddings
import simba.memory.reranker as reranker
from simba.memory.embeddings import EmbeddingService, TaskType


def _new_state() -> dict:
    return {"inside": 0, "violations": 0, "meta": threading.Lock()}


class _Tripwire:
    """Fake ``llama_cpp.Llama`` whose native calls flag concurrent entry."""

    def __init__(self, state: dict) -> None:
        self._state = state

    def _native(self) -> None:
        st = self._state
        with st["meta"]:
            st["inside"] += 1
            if st["inside"] > 1:
                st["violations"] += 1
        time.sleep(0.002)  # widen the race window
        with st["meta"]:
            st["inside"] -= 1

    def create_embedding(self, _text: str) -> dict:
        self._native()
        return {"data": [{"embedding": [0.5]}]}


def _run(workers: list, n_each: int = 8) -> None:
    threads = [threading.Thread(target=w) for w in workers for _ in range(n_each)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_reranker_and_embedder_share_one_lock():
    # Both modules must guard with the SAME lock object, or they can't serialize
    # against each other.
    assert reranker._LOCK is _llama.LLAMA_LOCK
    assert embeddings.LLAMA_LOCK is _llama.LLAMA_LOCK


def test_embed_and_rerank_never_run_concurrently():
    state = _new_state()
    tw = _Tripwire(state)

    svc = EmbeddingService.__new__(EmbeddingService)
    svc._model = tw  # type: ignore[assignment]
    svc._prefixed = lambda text, task: text  # type: ignore[assignment]
    rr = reranker._CrossEncoderReranker(tw)  # type: ignore[arg-type]

    def embed_worker() -> None:
        for _ in range(15):
            svc._embed_sync("x", TaskType.QUERY)

    def rerank_worker() -> None:
        for _ in range(15):
            rr.score("q", "d")

    _run([embed_worker, rerank_worker], n_each=4)
    assert state["violations"] == 0, (
        f"{state['violations']} concurrent llama.cpp entries — the shared lock "
        "did not serialize embed vs rerank"
    )


def test_tripwire_detects_unguarded_concurrency():
    # Sensitivity check: the tripwire MUST catch overlap when calls are NOT
    # guarded, otherwise the test above would be vacuous.
    state = _new_state()
    tw = _Tripwire(state)

    def hammer() -> None:
        for _ in range(30):
            tw.create_embedding("x")  # direct, no LLAMA_LOCK

    _run([hammer], n_each=8)
    assert state["violations"] > 0
