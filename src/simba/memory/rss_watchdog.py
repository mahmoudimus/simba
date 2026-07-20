"""RSS watchdog: self-monitoring + escalation for the memory daemon.

macOS has no enforceable RSS rlimit: ``RLIMIT_RSS`` is a documented no-op
under Darwin's XNU kernel (``setrlimit(2)`` accepts it and silently does
nothing), and the alternative that IS enforced --- ``RLIMIT_AS`` --- would
abort LanceDB's mmap-based table reads well below any sane budget (a memory
map counts fully against ``RLIMIT_AS`` regardless of how much of it is
actually resident). So the hard cap this module implements is a SELF
watchdog, not a kernel-enforced one: poll current RSS on an interval, ask
the allocator to relieve freed-but-unmapped pages back to the OS once a
soft budget is crossed, and self-restart --- reusing the existing,
battle-tested ``os.execv`` exec seam (PR #89/#91:
``simba.memory.routes._run_restart_sequence`` /
``simba.memory.background.reexec``) --- once a hard budget is crossed AND
the process is old enough that a genuine leak, not startup cost, is the
likely explanation.

``current_rss_mb()`` reads a CURRENT (not high-water-mark) memory metric. On
macOS that metric is PHYSICAL FOOTPRINT (``ri_phys_footprint``), not resident
size (``ri_resident_size``): macOS's memory compressor pages ballooning
memory OUT of residency, so resident size structurally under-reports leaks.
The motivating incident: a wedged daemon reached a 52GB physical footprint
while showing only ~0.36GB resident --- a resident-gated hard limit never
tripped. ``footprint(1)`` and Activity Monitor's "Memory" column report
footprint (dirty + compressed + swapped pages included) for the same reason;
this module gates on the same metric. Linux/``ps`` fall back to resident size
(no compressor to dodge there, and it's the portable best-effort figure).

``resource.getrusage(...).ru_maxrss`` is a HIGH-WATER MARK that only ever
grows across the process's lifetime, so it cannot answer "are we over budget
RIGHT NOW" --- a single transient spike (e.g. one large recall candidate
pool) would trip a ru_maxrss-based gate forever after, even once the freed
pages were returned to the OS. It remains a useful annotation on log lines
(bounded, cheap to read), so ``peak_rss_mb()`` is exposed for that purpose
only --- never for the soft/hard comparison itself.

psutil is NOT a dependency of this project (checked via ``rg psutil
pyproject.toml uv.lock`` --- no hits), so ``current_rss_mb`` reads through
the platform-native syscall instead: macOS via ``libproc.proc_pid_rusage``,
Linux via ``/proc/<pid>/statm``, with a ``ps -o rss=`` subprocess as the
final portable fallback.
"""

from __future__ import annotations

import asyncio
import collections
import ctypes
import ctypes.util
import gc
import logging
import os
import subprocess
import sys
import time
import typing

logger = logging.getLogger("simba.memory")


# --- current RSS: portable, best-effort, never raises ------------------------


class _RusageInfoV2(ctypes.Structure):
    """``RUSAGE_INFO_V2`` --- the stable C ABI from ``<sys/resource.h>``
    (fields through ``ri_diskio_byteswritten`` are the V2 shape; later SDKs
    only ever APPEND fields for newer flavors, never reorder existing ones,
    so reading through a V2-shaped struct with the V2 flavor constant is
    safe across macOS versions). Module-level (not nested in a function) so
    tests can construct and populate it directly against the pure helpers
    below without going through a real ``proc_pid_rusage`` syscall.
    """

    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


def _watchdog_metric_mb(info: _RusageInfoV2) -> float:
    """The soft/hard gating metric, in MB: ``ri_phys_footprint`` (physical
    footprint), never ``ri_resident_size``. Pure --- takes an already-
    populated struct, does no I/O. See the module docstring for why
    footprint and not resident size: the compressor pages ballooning memory
    OUT of residency, so resident size structurally under-reports leaks.
    """
    return float(info.ri_phys_footprint) / (1024 * 1024)


def _rusage_resident_mb(info: _RusageInfoV2) -> float:
    """Resident size, in MB --- forensic annotation only, never the gating
    metric (see ``_watchdog_metric_mb``). Pure, same shape as its sibling."""
    return float(info.ri_resident_size) / (1024 * 1024)


def _libproc_rusage(pid: int) -> _RusageInfoV2 | None:
    """macOS: fetch a populated ``RUSAGE_INFO_V2`` via
    ``libproc.proc_pid_rusage``. None off-darwin or on failure. Empirically
    verified against ``ps -o rss=`` and ``resource.getrusage().ru_maxrss`` at
    implementation time.
    """
    if sys.platform != "darwin":
        return None

    rusage_info_v2 = 2
    libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
    buf = _RusageInfoV2()
    rc = libproc.proc_pid_rusage(
        ctypes.c_int(pid), ctypes.c_int(rusage_info_v2), ctypes.byref(buf)
    )
    if rc != 0:
        return None
    return buf


def _rss_via_libproc(pid: int) -> float | None:
    """macOS: the watchdog gating metric (physical footprint, in MB) via
    ``libproc.proc_pid_rusage``. None off-darwin or on failure."""
    info = _libproc_rusage(pid)
    if info is None:
        return None
    return _watchdog_metric_mb(info)


def _resident_mb(pid: int | None = None) -> float | None:
    """Best-effort resident size, in MB, for ``pid`` (default: self) ---
    forensic log annotation only (see the module docstring); NEVER the
    soft/hard gating metric, which is ``current_rss_mb()`` /
    ``_watchdog_metric_mb``. macOS only; None on every other platform or on
    any read failure (reuses the same struct/syscall as ``_rss_via_libproc``,
    just reading a different field).
    """
    target = os.getpid() if pid is None else pid
    info = _libproc_rusage(target)
    if info is None:
        return None
    return _rusage_resident_mb(info)


def _rss_via_procfs(pid: int) -> float | None:
    """Linux: ``/proc/<pid>/statm`` field 2 (resident, in pages)."""
    if sys.platform != "linux":
        return None
    with open(f"/proc/{pid}/statm") as fh:
        fields = fh.readline().split()
    resident_pages = int(fields[1])
    page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    return resident_pages * page_size / (1024 * 1024)


def _rss_via_ps(pid: int) -> float | None:
    """Final fallback: shell out to ``ps`` (works on any POSIX platform)."""
    proc = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=2.0,
        check=False,
    )
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    if not text:
        return None
    return int(text) / 1024  # ps reports KB


_STRATEGIES: tuple[typing.Callable[[int], float | None], ...] = (
    _rss_via_libproc,
    _rss_via_procfs,
    _rss_via_ps,
)


def current_rss_mb(pid: int | None = None) -> float | None:
    """Best-effort CURRENT memory-pressure metric in MB for ``pid`` (default:
    self) --- the soft/hard watchdog gating metric.

    On macOS this is PHYSICAL FOOTPRINT (``ri_phys_footprint``), not resident
    size: resident size is compressible (the memory compressor pages
    ballooning memory OUT of residency) and structurally under-reports leaks
    --- the motivating incident was a process at 52GB physical footprint
    showing ~0.36GB resident. On Linux (``/proc/<pid>/statm``) and the final
    ``ps`` fallback this is resident size, unchanged --- there is no
    compressor to dodge there, and no cheaper portable alternative.

    Tries the native platform reader first (macOS libproc / Linux procfs),
    then a ``ps`` subprocess. Returns ``None`` if every strategy fails ---
    fail-open: callers must treat ``None`` as "skip this check", never as
    zero (zero would read like an OOM-adjacent process and could trigger
    nonsensical relief/restart behavior).
    """
    target = os.getpid() if pid is None else pid
    for strategy in _STRATEGIES:
        try:
            value = strategy(target)
        except Exception:
            value = None
        if value is not None and value >= 0:
            return value
    return None


def peak_rss_mb() -> float | None:
    """Best-effort PEAK resident set size in MB (``resource.getrusage().ru_maxrss``).

    A high-water mark that only ever grows --- useful as a log annotation,
    never as the live soft/hard comparison (see module docstring). Units
    differ by platform: Linux reports KB, macOS/BSD report bytes.
    """
    try:
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    if raw <= 0:
        return None
    divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
    return raw / divisor


# --- allocator pressure relief (soft-limit response) --------------------------


def _release_free_memory_to_os() -> None:
    """Best-effort: ask macOS's allocator to return freed pages to the OS.

    ``malloc_zone_pressure_relief(NULL, 0)`` (libmalloc) asks every
    registered zone to relieve as many freed-but-unmapped pages as it can
    right now --- objects freed by ``gc.collect()`` don't necessarily shrink
    the process's RSS on their own, since malloc implementations commonly
    retain freed pages in per-size-class caches rather than returning them
    to the kernel immediately. Guarded by ``hasattr``/``try`` --- a no-op on
    every other platform, and fail-soft on macOS itself.
    """
    if sys.platform != "darwin":
        return
    try:
        lib_path = ctypes.util.find_library("c") or "/usr/lib/libSystem.B.dylib"
        libc = ctypes.CDLL(lib_path)
        if not hasattr(libc, "malloc_zone_pressure_relief"):
            return
        libc.malloc_zone_pressure_relief.restype = ctypes.c_size_t
        libc.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.malloc_zone_pressure_relief(None, 0)
    except Exception:
        logger.debug("[rss-watchdog] malloc_zone_pressure_relief failed", exc_info=True)


# --- formatting helpers (log lines only) --------------------------------------


def _fmt_mb(value: float | None) -> str:
    return f"{value:.1f}MB" if value is not None else "unknown"


def _fmt_limit(value: float) -> str:
    return f"{value:.0f}MB" if value > 0 else "off"


def _fmt_resident(value: float | None) -> str:
    """The resident-size forensic annotation for a log line, e.g.
    ``" resident=361.2MB"`` --- empty string (no token at all) when
    unavailable, so off-darwin/failure log lines are byte-identical to
    before this annotation existed."""
    return f" resident={value:.1f}MB" if value is not None else ""


# --- the watchdog loop ---------------------------------------------------------


class RssWatchdog:
    """Poll process RSS on an interval; relieve pressure, then self-restart.

    Soft and hard limits are independent (either may be 0/disabled on its
    own). A soft crossing fires ``gc.collect()`` + the platform relief hook
    + one WARNING log line, exactly once per crossing (an "edge", not a
    "level": staying above soft on a later check is silent; dropping back
    below and re-crossing fires again). A hard crossing always logs
    CRITICAL; it additionally triggers the injected ``restart`` coroutine
    only when a boot argv is on record AND the process has been up for at
    least ``min_uptime_seconds`` (anti-flap --- never restart a process
    young enough that the crossing might be ordinary startup cost rather
    than a leak).

    ``rss_reader`` and ``restart`` are injectable seams (constructor
    parameters, not module-level lookups) so unit tests can drive this
    class synchronously via ``check_once()`` without a real process or a
    real ``os.execv``.

    ``history_maxlen`` (default 0 --- disabled) bounds a ring buffer of
    ``(t, rssMb)`` samples appended on every successful ``check_once()``,
    independent of the soft/hard limits (a disabled-limits watchdog started
    only for ``memory.rss_history_samples`` still samples). Surfaced via
    ``history()`` as ``GET /health``'s ``rssHistory`` for forensics --- the
    2026-07-17 incident's timeline was otherwise guesswork. ``0`` reproduces
    ``collections.deque(maxlen=0)``'s natural "keep nothing" behavior, so no
    special-casing is needed for the disabled case.
    """

    def __init__(
        self,
        *,
        soft_limit_mb: float,
        hard_limit_mb: float,
        interval_seconds: float,
        min_uptime_seconds: float,
        start_time: float,
        boot_argv: typing.Sequence[str] | None,
        restart: typing.Callable[[list[str]], typing.Awaitable[None]],
        rss_reader: typing.Callable[[], float | None] = current_rss_mb,
        history_maxlen: int = 0,
    ) -> None:
        self.soft_limit_mb = soft_limit_mb
        self.hard_limit_mb = hard_limit_mb
        self.interval_seconds = interval_seconds
        self.min_uptime_seconds = min_uptime_seconds
        self.start_time = start_time
        self.boot_argv = boot_argv
        self._restart = restart
        self._rss_reader = rss_reader
        self._stop_event = asyncio.Event()
        self._soft_tripped = False
        self.last_rss_mb: float | None = None
        self._history: collections.deque[tuple[float, float]] = collections.deque(
            maxlen=max(history_maxlen, 0)
        )

    async def _wait(self, seconds: float) -> bool:
        """Wait up to ``seconds``; True when ``stop()`` fired during the wait."""
        if seconds <= 0:
            return self._stop_event.is_set()
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
            return True
        except TimeoutError:
            return False

    async def check_once(self) -> None:
        """Read RSS once and act on it. Safe to call directly (e.g. from tests)."""
        rss = self._rss_reader()
        if rss is None:
            return
        self.last_rss_mb = rss
        self._history.append((time.time(), rss))

        if self.soft_limit_mb > 0:
            if rss > self.soft_limit_mb:
                self._handle_soft(rss)
            else:
                self._soft_tripped = False  # reset: a future crossing fires again

        if self.hard_limit_mb > 0 and rss > self.hard_limit_mb:
            await self._handle_hard(rss)

    def _handle_soft(self, rss: float) -> None:
        if self._soft_tripped:
            return
        self._soft_tripped = True
        gc.collect()
        _release_free_memory_to_os()
        logger.warning(
            "[rss-watchdog] soft limit crossed: rss=%s%s soft=%s hard=%s",
            _fmt_mb(rss),
            _fmt_resident(_resident_mb()),
            _fmt_limit(self.soft_limit_mb),
            _fmt_limit(self.hard_limit_mb),
        )

    async def _handle_hard(self, rss: float) -> None:
        peak = peak_rss_mb()
        resident = _fmt_resident(_resident_mb())
        if not self.boot_argv:
            logger.critical(
                "[rss-watchdog] HARD limit exceeded: rss=%s%s hard=%s peak=%s -- "
                "no boot argv on record, cannot self-restart",
                _fmt_mb(rss),
                resident,
                _fmt_limit(self.hard_limit_mb),
                _fmt_mb(peak),
            )
            return

        uptime = time.time() - self.start_time
        if uptime < self.min_uptime_seconds:
            logger.critical(
                "[rss-watchdog] HARD limit exceeded: rss=%s%s hard=%s peak=%s "
                "uptime=%.0fs -- younger than min_uptime=%.0fs, refusing to "
                "restart (anti-flap)",
                _fmt_mb(rss),
                resident,
                _fmt_limit(self.hard_limit_mb),
                _fmt_mb(peak),
                uptime,
                self.min_uptime_seconds,
            )
            return

        logger.critical(
            "[rss-watchdog] HARD limit exceeded: rss=%s%s hard=%s peak=%s "
            "uptime=%.0fs -- triggering self-restart",
            _fmt_mb(rss),
            resident,
            _fmt_limit(self.hard_limit_mb),
            _fmt_mb(peak),
            uptime,
        )
        try:
            await self._restart(list(self.boot_argv))
        except Exception:
            logger.critical("[rss-watchdog] restart trigger failed", exc_info=True)

    async def run_forever(self) -> None:
        """Check every ``interval_seconds`` until ``stop()``.

        The stop event is deliberately NOT cleared on entry (mirrors
        ``MaintenanceScheduler.run_forever``): a ``stop()`` issued between
        task creation and the first wait must not be lost.
        """
        while not self._stop_event.is_set():
            if await self._wait(self.interval_seconds):
                return
            try:
                await self.check_once()
            except Exception:
                logger.debug("[rss-watchdog] check failed", exc_info=True)

    def stop(self) -> None:
        """Signal the loop to stop (interrupts the interval wait)."""
        self._stop_event.set()

    def history(self) -> list[dict[str, float]]:
        """The RSS sample ring buffer, oldest first: ``{"t": <unix seconds
        int>, "rssMb": <float, 1 decimal>}``. Empty when history is disabled
        (``history_maxlen=0``) or no successful sample has been taken yet."""
        return [{"t": int(ts), "rssMb": round(rss, 1)} for ts, rss in self._history]
