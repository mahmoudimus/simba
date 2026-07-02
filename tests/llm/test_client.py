"""Tests for the synchronous LLM client (CLI-backed, fail-open)."""

from __future__ import annotations

import subprocess
import types

import simba.llm.client as llm
import simba.llm.config as llmcfg


def _cfg(**kw):
    cfg = llmcfg.LlmConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _completed(stdout: str, code: int = 0):
    return types.SimpleNamespace(stdout=stdout, stderr="", returncode=code)


class TestAvailable:
    def test_none_provider_unavailable(self) -> None:
        assert llm.LlmClient(_cfg(provider="none")).available() is False

    def test_claude_cli_available(self) -> None:
        assert llm.LlmClient(_cfg(provider="claude-cli")).available() is True

    def test_mlx_lm_available(self) -> None:
        assert llm.LlmClient(_cfg(provider="mlx-lm")).available() is True

    def test_unknown_provider_unavailable(self) -> None:
        # An unsupported provider (e.g. the VLM runtime "mlx-vlm") must report
        # unavailable, not silently produce "" — that footgun skipped a whole
        # eval run with no error. See docs/plans/10.
        assert llm.LlmClient(_cfg(provider="mlx-vlm")).available() is False

    def test_mlx_server_available_needs_base_url(self) -> None:
        assert llm.LlmClient(_cfg(provider="mlx-server")).available() is False
        cfg = _cfg(provider="mlx-server", base_url="http://127.0.0.1:8082")
        assert llm.LlmClient(cfg).available() is True

    def test_llama_server_available_needs_base_url(self) -> None:
        # Cross-platform llama.cpp llama-server — same HTTP path as mlx-server,
        # auto-spawnable via local_server.
        assert llm.LlmClient(_cfg(provider="llama-server")).available() is False
        cfg = _cfg(provider="llama-server", base_url="http://127.0.0.1:8080")
        assert llm.LlmClient(cfg).available() is True

    def test_openai_http_available_needs_base_url(self) -> None:
        # Generic OpenAI-compatible endpoint (e.g. a remote Ollama / llama.cpp /
        # vLLM box) — same HTTP path as mlx-server, but never spawns anything.
        assert llm.LlmClient(_cfg(provider="openai-http")).available() is False
        cfg = _cfg(provider="openai-http", base_url="http://192.168.1.50:11434/v1")
        assert llm.LlmClient(cfg).available() is True

    def test_openai_http_completes_via_http(self, monkeypatch) -> None:
        import httpx

        cfg = _cfg(
            provider="openai-http",
            base_url="http://192.168.1.50:11434/v1",
            model="gpt-oss:20b",
        )
        monkeypatch.setattr(
            httpx,
            "post",
            lambda url, json, timeout: types.SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"choices": [{"message": {"content": " Paris "}}]},
            ),
        )
        assert llm.LlmClient(cfg).complete("hi") == "Paris"


class TestMlxServer:
    def test_complete_via_http(self, monkeypatch) -> None:
        import httpx

        cfg = _cfg(
            provider="mlx-server", base_url="http://127.0.0.1:8082", model="m"
        )

        def _post(url, json, timeout):
            assert url.endswith("/v1/chat/completions")
            assert json["messages"][0]["content"] == "hi"
            return types.SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"choices": [{"message": {"content": " Paris "}}]},
            )

        monkeypatch.setattr(httpx, "post", _post)
        assert llm.LlmClient(cfg).complete("hi") == "Paris"

    def test_complete_failopen_on_error(self, monkeypatch) -> None:
        import httpx

        def _boom(*a, **k):
            raise ConnectionError("server down")

        monkeypatch.setattr(httpx, "post", _boom)
        cfg = _cfg(provider="mlx-server", base_url="http://127.0.0.1:8082")
        assert llm.LlmClient(cfg).complete("hi") == ""

    def test_complete_http_strips_reasoning(self, monkeypatch) -> None:
        # A reasoning answerer (gpt-oss harmony) must surface only the final
        # answer; the analysis channel is reasoning, not the prediction.
        import httpx

        raw = (
            "<|channel|>analysis<|message|>think about the capital<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Paris<|return|>"
        )
        monkeypatch.setattr(
            httpx,
            "post",
            lambda url, json, timeout: types.SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"choices": [{"message": {"content": raw}}]},
            ),
        )
        cfg = _cfg(provider="mlx-server", base_url="http://127.0.0.1:8082")
        assert llm.LlmClient(cfg).complete("hi") == "Paris"


class TestStripReasoning:
    def test_passthrough_plain(self) -> None:
        assert llm._strip_reasoning("Paris") == "Paris"

    def test_strips_qwen_think_block(self) -> None:
        assert llm._strip_reasoning("<think>the capital is...</think>Paris") == "Paris"

    def test_harmony_keeps_final_channel(self) -> None:
        raw = (
            "<|channel|>analysis<|message|>reasoning<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Paris<|return|>"
        )
        assert llm._strip_reasoning(raw) == "Paris"

    def test_harmony_without_final_is_empty(self) -> None:
        # Truncated before the final channel -> reasoning only, not an answer.
        assert llm._strip_reasoning("<|channel|>analysis<|message|>thinking") == ""

    def test_truncated_think_is_empty(self) -> None:
        assert llm._strip_reasoning("<think>still reasoning, no answer") == ""

    def test_strips_stray_special_tokens(self) -> None:
        assert llm._strip_reasoning("Paris<|return|>") == "Paris"


class TestCompleteClaudeCli:
    def test_parses_result_field(self, monkeypatch) -> None:
        captured = {}

        def fake_run(argv, **kw):
            captured["argv"] = argv
            captured["env"] = kw.get("env")
            return _completed('{"type":"result","is_error":false,"result":"hello"}')

        monkeypatch.setattr(subprocess, "run", fake_run)
        out = llm.LlmClient(_cfg(provider="claude-cli", model="haiku")).complete("hi")
        assert out == "hello"
        argv = captured["argv"]
        assert argv[0] == "claude" and "-p" in argv
        assert argv[argv.index("--model") + 1] == "haiku"
        # Load ONLY user settings so simba's own project/local-scoped hooks don't
        # fire when simba spawns its internal `claude -p` -> conflict-detect ->
        # claude -p recursion (the 2026-07-01 fork bomb). Unlike --bare this keeps
        # keychain/OAuth auth.
        assert argv[argv.index("--setting-sources") + 1] == "user"

    def test_is_error_returns_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, **kw: _completed('{"is_error":true,"result":"nope"}'),
        )
        assert llm.LlmClient(_cfg(provider="claude-cli")).complete("hi") == ""

    def test_base_url_sets_proxy_env(self, monkeypatch) -> None:
        captured = {}
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, **kw: captured.update(env=kw.get("env"))
            or _completed('{"is_error":false,"result":"x"}'),
        )
        monkeypatch.setenv("DS_KEY", "secret")
        cfg = _cfg(
            provider="claude-cli", base_url="http://ds:1234", api_key_env="DS_KEY"
        )
        llm.LlmClient(cfg).complete("hi")
        assert captured["env"]["ANTHROPIC_BASE_URL"] == "http://ds:1234"
        assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "secret"


class TestCompleteLlmCli:
    def test_returns_raw_stdout(self, monkeypatch) -> None:
        captured = {}

        def fake_run(argv, **kw):
            captured["argv"] = argv
            return _completed("plain text answer\n")

        monkeypatch.setattr(subprocess, "run", fake_run)
        out = llm.LlmClient(_cfg(provider="llm-cli", model="deepseek")).complete("hi")
        assert out == "plain text answer"
        argv = captured["argv"]
        assert argv[0] == "llm"
        assert argv[argv.index("-m") + 1] == "deepseek"


class TestFailOpen:
    def test_timeout_returns_empty(self, monkeypatch) -> None:
        def boom(argv, **kw):
            raise subprocess.TimeoutExpired(argv, 1)

        monkeypatch.setattr(subprocess, "run", boom)
        assert llm.LlmClient(_cfg(provider="claude-cli")).complete("hi") == ""

    def test_nonzero_exit_returns_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(
            subprocess, "run", lambda argv, **kw: _completed("", code=1)
        )
        assert llm.LlmClient(_cfg(provider="llm-cli")).complete("hi") == ""

    def test_none_provider_returns_empty(self) -> None:
        assert llm.LlmClient(_cfg(provider="none")).complete("hi") == ""


class TestCompleteJson:
    def test_extracts_json_array_from_noise(self, monkeypatch) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, **kw: _completed(
                '{"is_error":false,"result":"here you go:\\n```json\\n[1,2,3]\\n```"}'
            ),
        )
        out = llm.LlmClient(_cfg(provider="claude-cli")).complete_json("hi")
        assert out == [1, 2, 3]

    def test_bad_json_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, **kw: _completed('{"is_error":false,"result":"no json here"}'),
        )
        assert llm.LlmClient(_cfg(provider="claude-cli")).complete_json("hi") is None


class TestLocalProviders:
    def test_llama_cli_argv_and_text(self, monkeypatch) -> None:
        captured = {}

        def fake_run(argv, **kw):
            captured["argv"] = argv
            return _completed("the local completion\n")

        monkeypatch.setattr(subprocess, "run", fake_run)
        cfg = _cfg(provider="llama-cli", model_path="/models/q.gguf", max_tokens=128)
        out = llm.LlmClient(cfg).complete("hi")
        assert out == "the local completion"
        argv = captured["argv"]
        assert argv[0] == "llama-cli"
        assert argv[argv.index("-m") + 1] == "/models/q.gguf"
        assert "-p" in argv

    def test_llama_cli_falls_back_to_model_when_no_path(self, monkeypatch) -> None:
        captured = {}
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, **kw: captured.update(argv=argv) or _completed("x"),
        )
        llm.LlmClient(_cfg(provider="llama-cli", model="/m.gguf")).complete("hi")
        assert "/m.gguf" in captured["argv"]

    def test_mlx_argv_and_strips_markers(self, monkeypatch) -> None:
        captured = {}
        out_text = "==========\nthe answer here\n==========\nPrompt: 5 tokens\n"

        def fake_run(argv, **kw):
            captured["argv"] = argv
            return _completed(out_text)

        monkeypatch.setattr(subprocess, "run", fake_run)
        cfg = _cfg(provider="mlx-lm", model_path="mlx-community/x")
        out = llm.LlmClient(cfg).complete("hi")
        assert out == "the answer here"
        argv = captured["argv"]
        assert argv[0] == "mlx_lm.generate"
        assert argv[argv.index("--model") + 1] == "mlx-community/x"

    def test_extra_args_appended(self, monkeypatch) -> None:
        captured = {}
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, **kw: captured.update(argv=argv) or _completed("x"),
        )
        cfg = _cfg(
            provider="llama-cli",
            model_path="/m.gguf",
            extra_args="--n-gpu-layers 99 --temp 0",
        )
        llm.LlmClient(cfg).complete("hi")
        assert "--n-gpu-layers" in captured["argv"]
        assert "99" in captured["argv"]
        assert "--temp" in captured["argv"]
