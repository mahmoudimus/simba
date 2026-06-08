"""Persistent local MLX model server (the eval latency unlock).

Wraps ``mlx_lm.server`` (OpenAI-compatible). The model is loaded **once** and
serves many completions, so LLM-judged eval (HaluMem QA, IRCoT, ablations) stops
paying a per-call model reload — the real fix for the local-eval latency trap.

``ensure_server(model, port)`` is the entry point: returns the base_url if a server
is already up, otherwise spawns one (detached) and polls until ready. Fail-open:
returns ``None`` if it can't be started, so callers fall back gracefully.
"""

from __future__ import annotations

import logging
import subprocess
import time
import typing
import urllib.parse

import httpx

logger = logging.getLogger("simba.llm")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8082


def base_url_for(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{port}"


def build_serve_cmd(model: str, port: int, host: str = DEFAULT_HOST) -> list[str]:
    """The argv to launch ``mlx_lm.server`` for ``model`` on ``host:port``."""
    return [
        "mlx_lm.server",
        "--model", model,
        "--host", host,
        "--port", str(port),
    ]


def is_up(base_url: str, *, timeout: float = 2.0) -> bool:
    """True if an OpenAI-compatible server is responding at ``base_url``."""
    try:
        resp = httpx.get(base_url.rstrip("/") + "/v1/models", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def ensure_server(
    model: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    ready_timeout: float = 240.0,
    poll_interval: float = 2.0,
) -> str | None:
    """Return a ready server's base_url, starting ``mlx_lm.server`` if needed.

    Idempotent: if a server is already up at host:port, returns immediately (does
    NOT verify it's serving the same model — the caller owns that). Fail-open:
    returns ``None`` if the server can't be launched or never becomes ready.
    """
    url = base_url_for(host, port)
    if is_up(url):
        return url
    try:
        subprocess.Popen(
            build_serve_cmd(model, port, host),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (FileNotFoundError, OSError) as exc:
        logger.warning("mlx server: could not launch mlx_lm.server: %s", exc)
        return None

    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        if is_up(url):
            logger.info("mlx server: ready at %s (model=%s)", url, model)
            return url
    logger.warning("mlx server: %s did not become ready in %.0fs", url, ready_timeout)
    return None


def ensure_for_config(cfg: typing.Any) -> str | None:
    """If ``cfg.provider == "mlx-server"``, ensure its server is up; else no-op.

    Parses host/port from ``cfg.base_url`` and starts ``mlx_lm.server`` for the
    configured model if needed. Returns the base_url (ready) or None (fail-open /
    not an mlx-server config).
    """
    if getattr(cfg, "provider", "") != "mlx-server":
        return None
    parsed = urllib.parse.urlparse(getattr(cfg, "base_url", "") or base_url_for())
    host = parsed.hostname or DEFAULT_HOST
    port = parsed.port or DEFAULT_PORT
    model = getattr(cfg, "model", "") or getattr(cfg, "model_path", "")
    return ensure_server(model, host=host, port=port)
