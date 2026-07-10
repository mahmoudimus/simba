"""Shutdown-aware background-task registry for the memory daemon.

Live 2026-07-08 SIGTERM breach: uvicorn's graceful shutdown window was
exceeded ("Cancel 54 running task(s), timeout graceful shutdown exceeded" +
CancelledError tracebacks from in-flight ASGI stacks). Three structural
causes:

  (a) Fire-and-forget ``asyncio.create_task`` call sites (usage bumps,
      demand logging, ack handling, HyDE/rerank cache warms, ...) were
      untracked --- nothing owned them at shutdown.
  (b) Background passes self-HTTP the daemon's own endpoints
      (``_fetch_type_map``, ``_fetch_tool_rule_ids``, hygiene's list/delete
      calls, ...). Once uvicorn stops serving, those requests can never
      complete, guaranteeing the graceful-shutdown timeout is breached if a
      pass is mid-flight.
  (c) ``asyncio.to_thread`` workers blocked on the process-global LLAMA lock
      are uncancellable --- cancellation must be bounded-wait-then-abandon,
      never wait-forever.

This module addresses (a) and (c) via ``spawn``/``drain``, and gives
self-HTTP helpers (addressing (b)) a process-level flag to check via
``mark_shutting_down``/``is_shutting_down``.
"""

from __future__ import annotations

import asyncio
import os
import threading
import typing

#: Registry of in-flight fire-and-forget tasks. Module-level (mirrors the
#: pre-existing ``routes._background_tasks`` convention this replaces) since
#: the daemon is a single process with a single registry; tests reset it via
#: ``reset_for_tests`` (see tests/conftest.py's autouse fixture).
TASKS: set[asyncio.Task[typing.Any]] = set()

# threading.Event (not asyncio.Event): the self-HTTP guards this backs
# (_fetch_type_map, _fetch_tool_rule_ids, run_hygiene_pass) run inside
# asyncio.to_thread workers, i.e. plain worker threads --- is_set()/set()
# must be safe to call from there without a running event loop.
_shutting_down = threading.Event()


def spawn(coro: typing.Coroutine[typing.Any, typing.Any, typing.Any]) -> asyncio.Task:
    """Create a tracked fire-and-forget task.

    Equivalent to the ad hoc ``asyncio.create_task(coro); tasks.add(task);
    task.add_done_callback(tasks.discard)`` triple this replaces, just
    centralized so ``drain()`` can find every outstanding task at shutdown.
    Auto-discards from the registry on completion (success, exception, or
    cancellation alike) --- zero overhead beyond one set add + one discard.
    """
    task = asyncio.create_task(coro)
    TASKS.add(task)
    task.add_done_callback(TASKS.discard)
    return task


def _derive_tail_seconds(grace_seconds: float) -> float:
    """Auto-derive a bounded tail from the grace budget when none is given.

    Production calls (``drain(config.shutdown_timeout)``) get a sane
    absolute cap (2s) so a large grace doesn't also imply a large tail; tests
    that pass a tiny grace (for speed) get a proportionally tiny tail rather
    than a fixed multi-second one.
    """
    return min(2.0, max(0.1, grace_seconds * 0.2))


async def drain(grace_seconds: float, *, tail_seconds: float | None = None) -> None:
    """Wait up to ``grace_seconds`` for every tracked task, then cancel
    stragglers and wait a short bounded tail before giving up on them.

    Never hangs on an uncancellable ``asyncio.to_thread`` worker (concern
    (c) above, e.g. one blocked on the process-global LLAMA lock): calling
    ``.cancel()`` on such a task cannot interrupt the underlying OS thread,
    and a task that swallows ``CancelledError`` may never actually finish.
    Either way, after the bounded tail this function ABANDONS whatever is
    still running and returns --- the process is exiting regardless, so
    there is nothing to gain from waiting longer.
    """
    pending = {t for t in TASKS if not t.done()}
    if not pending:
        return

    grace = max(0.0, grace_seconds)
    if tail_seconds is None:
        tail_seconds = _derive_tail_seconds(grace)
    tail = max(0.0, tail_seconds)

    if grace > 0:
        _done, pending = await asyncio.wait(pending, timeout=grace)
    if not pending:
        return

    for task in pending:
        task.cancel()

    if tail > 0:
        await asyncio.wait(pending, timeout=tail)
    # Anything still not done after the tail is deliberately abandoned here.


def mark_shutting_down() -> None:
    """Flag the process as shutting down. Set at the START of app shutdown,
    before anything else, so a self-HTTP helper already mid-flight sees it
    on its very next check."""
    _shutting_down.set()


def is_shutting_down() -> bool:
    """True once ``mark_shutting_down`` has been called. Thread-safe."""
    return _shutting_down.is_set()


def reset_for_tests() -> None:
    """Test-only: clear the shutdown flag and forget any leftover tasks.

    The registry and flag are process-global module state (mirroring
    ``simba.db``'s shared proxy); call this between tests so one test's
    shutdown simulation never bleeds into the next.
    """
    _shutting_down.clear()
    TASKS.clear()


# --- self-restart exec seam (POST /restart) --------------------------------
#
# A thin indirection over ``os.execv``: routes.py's ``/restart`` handler
# calls ``reexec`` after draining background tasks and flushing stdio, and
# tests monkeypatch this module attribute to assert ordering without ever
# exec'ing the real test process.
_execv = os.execv


def reexec(argv: list[str]) -> None:
    """Replace the current process image with ``argv`` (POSIX self-restart).

    ``argv[0]`` is conventionally the executable path --- always
    ``sys.executable`` in practice, which is guaranteed absolute, so no PATH
    search is needed (``os.execv``, not ``os.execvp``). The full list becomes
    the new process's ``sys.argv``.

    Every open file descriptor --- stdout/stderr included --- survives
    ``execv`` on POSIX; only sockets opened non-inheritable (PEP 446) are
    closed at exec, which is exactly the uvicorn listener, and the new image
    rebinds it (uvicorn sets ``SO_REUSEADDR``). The PID is unchanged, so a
    foreground terminal keeps streaming output from the new image under
    intact shell job control. Threads and locks die with the replaced image
    --- which is why the caller must already have drained background work
    and flushed stdio: nothing Python-level survives this call to clean up
    afterward.
    """
    _execv(argv[0], argv)
