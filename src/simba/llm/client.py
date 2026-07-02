"""Synchronous, CLI-backed LLM client — fail-open by design.

``complete`` shells out to the configured CLI (``claude`` or ``llm``), waits
with a timeout, and returns the model's text (empty string on any failure, so
callers transparently fall back to their non-LLM path). ``complete_json`` pulls
the first JSON value out of the reply. No SDK dependency; the model/endpoint is
config-driven so a DeepSeek-style backend works via ``base_url``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shlex
import subprocess
import typing

logger = logging.getLogger("simba.llm")

# Providers the client can actually run (must match the branches in ``_argv``).
# An unknown provider (e.g. the *vision* runtime "mlx-vlm", which this client does
# not implement) must report unavailable rather than silently returning "" — that
# footgun once skipped an entire eval run with no error.
# Providers that talk to an OpenAI-compatible HTTP endpoint (``_complete_http``).
# ``mlx-server`` (Apple Silicon) and ``llama-server`` (llama.cpp, cross-platform)
# may be auto-spawned locally (``local_server.ensure_for_config``); ``openai-http``
# is a generic endpoint (Ollama / llama.cpp / vLLM on, say, a CUDA box) that this
# client never starts — you run it yourself. See docs/eval-remote-gpu.md.
_HTTP_PROVIDERS = frozenset({"mlx-server", "llama-server", "openai-http"})
_KNOWN_PROVIDERS = frozenset(
    {"claude-cli", "llm-cli", "llama-cli", "mlx-lm", "mlx", *_HTTP_PROVIDERS}
)
_warned_providers: set[str] = set()

# Set on any LLM subprocess simba spawns; a nested LlmClient (a hook fired inside
# that subprocess) then reports unavailable, breaking the hook -> LLM -> hook
# recursion universally. See LlmClient.available()/_env().
REENTRY_ENV = "SIMBA_INTERNAL_LLM"


def _extract_json(text: str) -> typing.Any | None:
    """Best-effort: parse the first JSON value embedded in ``text``."""
    if not text:
        return None
    stripped = text.strip()
    # Strip a ```json … ``` (or plain ```) fence if present.
    if stripped.startswith("```"):
        parts = stripped.split("```", 2)
        stripped = parts[1] if len(parts) > 1 else ""
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    with contextlib.suppress(ValueError, TypeError):
        return json.loads(stripped)
    # Otherwise scan for the first balanced [...] or {...}.
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    with contextlib.suppress(ValueError, TypeError):
                        return json.loads(text[start : i + 1])
                    break
    return None


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|<\|start\|>|\Z)",
    re.DOTALL,
)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*?\|>")


def _strip_reasoning(text: str) -> str:
    """Return just the final answer from a reasoning model's output.

    Handles two formats so a thinking answerer can stand in cleanly:
    - gpt-oss **harmony** channels — the answer is ONLY the ``final`` channel's
      message; ``analysis`` channels are reasoning and are dropped (no final
      channel ⇒ truncated reasoning ⇒ "").
    - Qwen-style ``<think>…</think>`` blocks — dropped; an unterminated ``<think>``
      means reasoning ran out of room before an answer ⇒ "".

    A no-op for plain instruct output, so it is safe to apply to every
    ``mlx-server`` completion (answerer or judge).
    """
    if not text:
        return ""
    if "<|channel|>" in text or "<|message|>" in text:
        finals = _HARMONY_FINAL_RE.findall(text)
        text = finals[-1] if finals else ""
    text = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    elif "<think>" in text:
        return ""
    text = _SPECIAL_TOKEN_RE.sub("", text)
    return text.strip()


def _strip_mlx(text: str) -> str:
    """Drop mlx_lm.generate's ``====`` separators and trailing stats lines."""
    keep = []
    for line in text.splitlines():
        s = line.strip()
        if not s or set(s) == {"="}:
            continue
        if s.startswith(("Prompt:", "Generation:", "Peak memory:")):
            continue
        keep.append(line)
    return "\n".join(keep).strip()


class LlmClient:
    def __init__(self, cfg: typing.Any) -> None:
        self._cfg = cfg

    def available(self) -> bool:
        # Re-entrancy backstop: when simba spawns its own LLM subprocess it stamps
        # REENTRY_ENV on that process (and its children inherit it). A hook fired
        # INSIDE that subprocess therefore sees the client as unavailable, so
        # conflict/reasoning/pitfall detection no-ops and never spawns another
        # `claude -p` -> recursion is impossible, regardless of provider or which
        # settings scope the hooks live in (the 2026-07-01 fork-bomb backstop).
        if os.environ.get(REENTRY_ENV):
            return False
        provider = self._cfg.provider
        if provider in _HTTP_PROVIDERS:
            # OpenAI-compatible HTTP endpoint (mlx_lm.server / remote Ollama / …) —
            # needs a base_url to reach it.
            return bool(self._cfg.base_url)
        if provider in _KNOWN_PROVIDERS:
            return True
        if provider not in ("", "none") and provider not in _warned_providers:
            _warned_providers.add(provider)
            logger.warning(
                "llm: unknown provider %r — no inference will run (known: %s). "
                "For local MLX text models use 'mlx-lm'; 'mlx-vlm' is unsupported.",
                provider,
                ", ".join(sorted(_KNOWN_PROVIDERS)),
            )
        return False

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        # Mark this subprocess as simba-internal so any hook it fires no-ops its
        # own LLM calls (see available()). Children inherit it.
        env[REENTRY_ENV] = "1"
        if self._cfg.base_url:
            env["ANTHROPIC_BASE_URL"] = self._cfg.base_url
            env["ANTHROPIC_AUTH_TOKEN"] = os.environ.get(self._cfg.api_key_env, "")
        return env

    def _extra(self) -> list[str]:
        return shlex.split(self._cfg.extra_args) if self._cfg.extra_args else []

    def _local_model(self) -> str:
        return self._cfg.model_path or self._cfg.model

    def _argv(self, prompt: str) -> list[str]:
        provider = self._cfg.provider
        if provider == "claude-cli":
            return [
                "claude", "-p", prompt,
                "--model", self._cfg.model,
                "--output-format", "json",
                "--max-turns", "1",
                # Load ONLY user settings (not project/local) so simba's own
                # hooks — registered at project/local scope — don't fire when
                # simba spawns its internal `claude -p`, which would re-enter the
                # UserPromptSubmit hook -> conflict detection -> another
                # `claude -p` -> infinite recursion (the 2026-07-01 fork bomb).
                # Preferred over --bare because it keeps keychain/OAuth auth
                # (--bare skips keychain -> "Not logged in"). Caveat: if simba
                # hooks are installed at USER scope this won't exclude them.
                "--setting-sources", "user",
                *self._extra(),
            ]
        if provider == "llm-cli":
            argv = ["llm", "-m", self._cfg.model]
            if self._cfg.thinking:
                # best-effort reasoning-effort hint; harmless if the model
                # ignores it, fail-open if it rejects it.
                argv += ["-o", "reasoning_effort", self._cfg.thinking]
            argv += [*self._extra(), prompt]
            return argv
        if provider == "llama-cli":
            # llama.cpp's CLI — fully local GGUF inference.
            return [
                "llama-cli",
                "-m", self._local_model(),
                "-p", prompt,
                "-n", str(self._cfg.max_tokens),
                "-no-cnv",
                "--no-display-prompt",
                *self._extra(),
            ]
        if provider in ("mlx-lm", "mlx"):
            # Apple MLX local inference (mlx_lm.generate).
            return [
                "mlx_lm.generate",
                "--model", self._local_model(),
                "--prompt", prompt,
                "--max-tokens", str(self._cfg.max_tokens),
                *self._extra(),
            ]
        return []

    def _complete_http(self, prompt: str) -> str:
        """OpenAI-compatible chat completion against a persistent local/remote server.

        Works with any OpenAI ``/v1`` server — mlx_lm.server, llama.cpp's
        llama-server, Ollama, vLLM. The model is loaded once by the server, so each
        call is just inference — this is what makes local LLM-judged eval affordable
        (no per-call reload). Fail-open: any error (server down, bad response)
        returns "".
        """
        import httpx

        url = self._cfg.base_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": self._cfg.model or self._local_model(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._cfg.max_tokens,
            "temperature": 0.0,
        }
        try:
            resp = httpx.post(url, json=payload, timeout=self._cfg.timeout_seconds)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            return ""
        # Strip reasoning scaffolding (gpt-oss harmony / <think>) so a thinking
        # answerer surfaces only its final answer; no-op for instruct models.
        return _strip_reasoning(content) if isinstance(content, str) else ""

    def complete(self, prompt: str) -> str:
        """Run the prompt and return the model's text, or "" on any failure."""
        if not self.available():
            return ""
        if self._cfg.provider in _HTTP_PROVIDERS:
            return self._complete_http(prompt)
        argv = self._argv(prompt)
        if not argv:
            return ""
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self._cfg.timeout_seconds,
                env=self._env(),
            )
        except Exception:
            return ""
        if proc.returncode != 0:
            return ""
        out = (proc.stdout or "").strip()
        if self._cfg.provider == "claude-cli":
            try:
                data = json.loads(out)
            except (ValueError, TypeError):
                return ""
            if data.get("is_error"):
                return ""
            result = data.get("result", "")
            return result.strip() if isinstance(result, str) else ""
        if self._cfg.provider in ("mlx-lm", "mlx"):
            return _strip_mlx(out)
        return out

    def complete_json(self, prompt: str) -> typing.Any | None:
        """Run the prompt and return the first JSON value in the reply, or None."""
        return _extract_json(self.complete(prompt))


def get_client(cfg: typing.Any | None = None) -> LlmClient:
    """Return an ``LlmClient`` for the given (or loaded) llm config."""
    if cfg is None:
        import simba.config
        import simba.llm.config  # registers the "llm" section

        _ = simba.llm.config
        cfg = simba.config.load("llm")
    return LlmClient(cfg)
