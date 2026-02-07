"""Hot-reload proxy (Einhorn-style) for the Orchestration MCP server.

Sits between Claude Code and the actual MCP backend, enabling backend
restarts without dropping the frontend connection.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path


class MCPProxy:
    """Async proxy between Claude Code and the MCP backend.

    Usage::

        simba orchestration proxy --root-dir .

    Hot-reload::

        kill -HUP <proxy_pid>
    """

    def __init__(
        self,
        backend_cmd: list[str],
        pid_file: Path,
        root_dir: Path,
    ) -> None:
        self.backend_cmd = backend_cmd
        self.pid_file = pid_file
        self.root_dir = root_dir
        self.backend: asyncio.subprocess.Process | None = None
        self.running = True
        self._reload_event = asyncio.Event()

    async def start_backend(self) -> None:
        """Start (or restart) the backend MCP server process."""
        if self.backend and self.backend.returncode is None:
            self.backend.terminate()
            try:
                await asyncio.wait_for(self.backend.wait(), timeout=5.0)
            except TimeoutError:
                self.backend.kill()
                await self.backend.wait()

        self.backend = await asyncio.create_subprocess_exec(
            *self.backend_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr,
            cwd=str(self.root_dir),
        )
        sys.stderr.write(f"Backend started (PID: {self.backend.pid})\n")
        sys.stderr.flush()

    def _handle_sighup(self) -> None:
        """Signal handler for SIGHUP — triggers backend reload."""
        sys.stderr.write("SIGHUP received, scheduling reload...\n")
        sys.stderr.flush()
        self._reload_event.set()

    async def _reload_watcher(self) -> None:
        """Background task that watches for reload signals."""
        while self.running:
            await self._reload_event.wait()
            self._reload_event.clear()
            await self.start_backend()

    async def forward_stdin_to_backend(self) -> None:
        """Forward stdin from Claude Code to the backend server."""
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while self.running:
            try:
                line = await reader.readline()
                if not line:
                    break
                if self.backend and self.backend.stdin:
                    self.backend.stdin.write(line)
                    await self.backend.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception as exc:
                sys.stderr.write(f"stdin forward error: {exc}\n")
                break

        self.running = False

    async def forward_backend_to_stdout(self) -> None:
        """Forward stdout from backend to Claude Code."""
        while self.running:
            try:
                if not self.backend or not self.backend.stdout:
                    await asyncio.sleep(0.1)
                    continue

                line = await self.backend.stdout.readline()
                if not line:
                    sys.stderr.write("Backend stdout closed, waiting for reload...\n")
                    await asyncio.sleep(0.5)
                    continue

                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception as exc:
                sys.stderr.write(f"stdout forward error: {exc}\n")
                await asyncio.sleep(0.1)

    async def run(self) -> None:
        """Main proxy loop."""
        if hasattr(signal, "SIGHUP"):
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(signal.SIGHUP, self._handle_sighup)

        await self.start_backend()

        tasks = [
            asyncio.create_task(self._reload_watcher()),
            asyncio.create_task(self.forward_stdin_to_backend()),
            asyncio.create_task(self.forward_backend_to_stdout()),
        ]

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self.running = False
            if self.backend and self.backend.returncode is None:
                self.backend.terminate()
                await self.backend.wait()
            sys.stderr.write("Proxy shutdown complete\n")

    def __enter__(self) -> MCPProxy:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        if self.pid_file.exists():
            self.pid_file.unlink()
        return False


def reload_server() -> str:
    """Hot-reload the MCP server backend (requires proxy mode).

    Sends SIGHUP to the proxy process, which restarts the backend
    while maintaining the connection to Claude Code.
    """
    pid_file = Path(".claude/proxy.pid")

    if not pid_file.exists():
        return (
            "Proxy not running (no PID file found).\n"
            "   Hot-reload requires proxy mode. Reinstall with:\n"
            "   simba orchestration install --proxy"
        )

    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError) as exc:
        return f"Failed to read proxy PID: {exc}"

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return f"Proxy process (PID {pid}) not found. Stale PID file removed."
    except PermissionError:
        return f"No permission to signal proxy (PID {pid})."

    try:
        os.kill(pid, signal.SIGHUP)
        return (
            f"Reload signal sent to proxy (PID {pid}).\n"
            "   Backend is restarting with updated code."
        )
    except Exception as exc:
        return f"Failed to send reload signal: {exc}"


def run_proxy(pid_file: Path, root_dir: Path) -> None:
    """Entry point for proxy mode."""
    root_dir = root_dir.resolve()
    os.chdir(root_dir)

    backend_cmd = [
        sys.executable,
        "-m",
        "simba.orchestration",
        "run",
        "--root-dir",
        str(root_dir),
    ]

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

    sys.stderr.write("MCP Proxy starting...\n")
    sys.stderr.write(f"   Root dir: {root_dir}\n")
    sys.stderr.write(f"   Backend: {' '.join(backend_cmd)}\n")
    sys.stderr.write(f"   Hot-reload: kill -HUP {os.getpid()}\n")
    sys.stderr.write(f"   PID file: {pid_file}\n")
    sys.stderr.flush()

    with MCPProxy(backend_cmd, pid_file=pid_file, root_dir=root_dir) as proxy:
        asyncio.run(proxy.run())
