"""Persistent local OpenAI-compatible model server (the eval latency unlock).

Auto-spawns a local inference server that loads the model **once** and serves
many completions, so LLM-judged eval (HaluMem QA, IRCoT, ablations) and the live
reranker stop paying a per-call model reload — the real fix for the local-eval
latency trap (vs the ``llama-cli``/``mlx-lm`` subprocess providers, which reload
every call).

Engine-agnostic: the launch command is a **preset per provider** (``mlx-server``
→ ``mlx_lm.server`` on Apple Silicon; ``llama-server`` → llama.cpp's
``llama-server`` everywhere else), or a fully custom ``serve_cmd`` template
(e.g. vLLM). All speak the OpenAI ``/v1`` API, so the client transport is the
same regardless. ``ensure_for_config(cfg)`` is the entry point: returns the
base_url if a server is already up, spawns one if the endpoint is local, and is a
no-op for the non-spawning ``openai-http`` provider (a server you run yourself).
Fail-open throughout: returns ``None`` if it can't start, so callers fall back.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
import typing
import urllib.parse

import httpx

logger = logging.getLogger("simba.llm")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8082

# provider -> serve-command template. ``{model}``/``{host}``/``{port}`` are
# substituted; a non-empty ``cfg.serve_cmd`` overrides the preset (e.g. vLLM).
SERVE_PRESETS: dict[str, str] = {
    "mlx-server": "mlx_lm.server --model {model} --host {host} --port {port}",
    "llama-server": "llama-server -m {model} --host {host} --port {port}",
}

_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


def base_url_for(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{port}"


def build_serve_cmd(template: str, model: str, host: str, port: int) -> list[str]:
    """Render a serve-command template into an argv list."""
    return shlex.split(template.format(model=model, host=host, port=port))


def is_up(base_url: str, *, timeout: float = 2.0) -> bool:
    """True if an OpenAI-compatible server is responding at ``base_url``."""
    try:
        resp = httpx.get(base_url.rstrip("/") + "/v1/models", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def ensure_server(
    serve_cmd: list[str],
    *,
    base_url: str,
    ready_timeout: float = 240.0,
    poll_interval: float = 2.0,
) -> str | None:
    """Return a ready server's base_url, spawning ``serve_cmd`` if needed.

    Idempotent: if a server is already up at ``base_url``, returns immediately
    (does NOT verify it serves the same model — the caller owns that). Fail-open:
    returns ``None`` if the server can't be launched or never becomes ready.
    """
    if is_up(base_url):
        return base_url
    try:
        subprocess.Popen(
            serve_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (FileNotFoundError, OSError) as exc:
        logger.warning("local-server: could not launch %s: %s", serve_cmd[:1], exc)
        return None

    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        if is_up(base_url):
            logger.info("local-server: ready at %s", base_url)
            return base_url
    logger.warning("local-server: %s did not become ready in %.0fs", base_url,
                   ready_timeout)
    return None


def ensure_for_config(cfg: typing.Any) -> str | None:
    """Ensure the local server for an auto-spawn provider is up; else no-op.

    Spawns for ``mlx-server`` / ``llama-server`` (or any provider in
    ``SERVE_PRESETS``) when the endpoint is **local**; for a remote base_url we
    only check reachability (you run the server on that host). ``openai-http`` and
    the CLI providers are no-ops. Returns the ready base_url or None (fail-open).
    """
    provider = getattr(cfg, "provider", "")
    if provider not in SERVE_PRESETS:
        return None
    parsed = urllib.parse.urlparse(getattr(cfg, "base_url", "") or base_url_for())
    host = parsed.hostname or DEFAULT_HOST
    port = parsed.port or DEFAULT_PORT
    url = base_url_for(host, port)
    if is_up(url):
        return url
    if host not in _LOCAL_HOSTS:
        logger.warning(
            "local-server: %s is remote and down — start the server on that host",
            url,
        )
        return None
    model = getattr(cfg, "model", "") or getattr(cfg, "model_path", "")
    template = getattr(cfg, "serve_cmd", "") or SERVE_PRESETS[provider]
    return ensure_server(build_serve_cmd(template, model, host, port), base_url=url)
