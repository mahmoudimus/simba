"""Process-global serialization for in-process llama.cpp.

ggml is not safe under concurrent context use: when an embedding call and a
reranker score (two separate ``llama_cpp.Llama`` instances sharing the same
process-global ggml backend) run on different worker threads at once, they
corrupt shared backend buffers — observed as
``GGML_ASSERT(... tensor write out of bounds ...)`` followed by ``SIGSEGV``
(crash report: faulting thread deep in ``libllama``).

Every native llama.cpp call in the daemon — the embedder's embed and model
load, and the reranker's load and per-pair score — acquires this single lock,
so no two threads are ever inside llama.cpp simultaneously.

Deadlock-safety: all call sites run in worker threads (``asyncio.to_thread`` /
``run_in_threadpool``), never on the event loop, and the lock is held only
around the synchronous native call (no ``await`` inside). A plain
non-reentrant ``Lock`` is correct because no call site re-acquires it while
already holding it (loads release before scoring; embed/score never nest).
"""

from __future__ import annotations

import threading

#: One process-wide lock guarding ALL in-process llama.cpp / ggml access.
LLAMA_LOCK = threading.Lock()
