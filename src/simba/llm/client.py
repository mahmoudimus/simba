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
import os
import shlex
import subprocess
import typing


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
        return self._cfg.provider not in ("", "none")

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
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

    def complete(self, prompt: str) -> str:
        """Run the prompt and return the model's text, or "" on any failure."""
        if not self.available():
            return ""
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
