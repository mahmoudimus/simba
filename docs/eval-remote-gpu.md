# Running simba evals against a remote GPU box

The eval program is bottlenecked by the **answerer + judge LLM calls** (gpt-oss
reasoning ≈ 5 s/call on Apple-Silicon MLX). An NVIDIA box (e.g. RTX 4090, 24 GB)
runs those faster and, with vLLM, can **batch** the many independent calls — and it
frees your MacBook.

The catch: **MLX is Apple-Silicon-only**, so `mlx_lm.server` doesn't run on the
NVIDIA box. We don't need it — simba's LLM client speaks the **OpenAI HTTP API**
(`POST /v1/chat/completions`), so it works against *any* OpenAI-compatible server.
Use the **`openai-http`** provider (added for exactly this; it never tries to spawn a
local server — you run the server yourself).

## Topology (recommended: hybrid)

```
  MacBook (eval orchestrator + bge-large embedder)
        │  HTTP, kilobytes/call  (a few MB per whole run)
        ▼
  Windows RTX 4090  (Ollama / llama.cpp / vLLM serving gpt-oss-20b + a judge model)
```

Embedding stays on the Mac (cheap + content-cached). Only the LLM calls cross the
wire — they're tiny text payloads, so **any LAN (even WiFi) is plenty**; the GPU
inference time dominates, not the network. Don't bother with Thunderbolt bridging.

> Full offload (run the whole eval on the Windows box) is also possible — clone the
> repo, `uv sync`, run the bge-large GGUF embedder via CUDA llama-cpp-python — but
> it's more setup. Start hybrid.

## Windows 4090 — stand up a server (pick one)

### Option A — Ollama (easiest)

1. Install Ollama for Windows (https://ollama.com/download).
2. **Bind to the LAN** (not just localhost) so the Mac can reach it. Set a system
   env var and restart Ollama:
   ```
   setx OLLAMA_HOST "0.0.0.0:11434"
   ```
3. Pull the models:
   ```
   ollama pull gpt-oss:20b          # answerer (reasoning)
   ollama pull qwen2.5:3b-instruct  # judge (small, fast) — or any small instruct model
   ```
4. **Allow inbound** TCP 11434 through Windows Defender Firewall (Private network):
   ```
   netsh advfirewall firewall add rule name="Ollama 11434" dir=in action=allow protocol=TCP localport=11434
   ```
5. Note the box's LAN IP (`ipconfig` → IPv4 Address, e.g. `192.168.1.50`).
   OpenAI base URL is then `http://192.168.1.50:11434/v1`.

### Option B — llama.cpp `llama-server` (GGUF, CUDA, native Windows)

```
llama-server -m gpt-oss-20b.gguf --host 0.0.0.0 --port 8080 -ngl 999
```
Base URL: `http://192.168.1.50:8080/v1`. (Run a second instance on another port for
the judge model, or use one model for both — see the self-grading caveat below.)

### Option C — vLLM (best throughput; needs WSL2 on Windows)

```
vllm serve openai/gpt-oss-20b --host 0.0.0.0 --port 8000
```
Base URL: `http://192.168.1.50:8000/v1`. vLLM batches concurrent requests — the
biggest win for a full run.

## Which provider?

- **`openai-http`** — talk to a server you started yourself (the hybrid case here:
  Ollama/llama.cpp/vLLM on the GPU box, eval on the Mac). Never auto-spawns.
- **`llama-server`** — llama.cpp, **auto-spawned** by simba for a *local* endpoint.
  Use this on the **full-offload** path (eval + server both on the GPU box): set
  `llm.model` to a GGUF path and simba starts `llama-server` for you. Cross-platform
  parity with `mlx-server` (Apple Silicon).
- **`mlx-server`** — Apple-Silicon `mlx_lm.server`, auto-spawned locally.
- Custom server (e.g. vLLM auto-spawn): set `llm.serve_cmd` to a template, e.g.
  `vllm serve {model} --host {host} --port {port}`.

> Auto-spawn (`mlx-server`/`llama-server`) only fires for a **local** base_url. A
> remote base_url is check-only — start the server on that host yourself (or use
> `openai-http`, which is the same thing made explicit).

## Mac — point simba at the box (hybrid)

```bash
# Answerer
simba config set llm.provider   openai-http
simba config set llm.base_url   http://192.168.1.50:11434/v1   # your box IP + port
simba config set llm.model      gpt-oss:20b                    # the server's model id
simba config set llm.model_path ""                             # clear (model_path shadows model)

# Judge (separate model so it isn't grading its own answer)
simba config set judge.provider openai-http
simba config set judge.base_url http://192.168.1.50:11434/v1
simba config set judge.model    qwen2.5:3b-instruct

# verify
for k in llm.provider llm.base_url llm.model judge.provider judge.base_url judge.model; do
  printf '%-18s = %s\n' "$k" "$(simba config get $k)"
done
```

> **zsh note:** set each key with its own quoted `simba config set` call (zsh doesn't
> word-split unquoted vars; a `for kv in "llm.provider openai-http"` loop silently
> passes the whole string as one arg). Always `config get` to verify.

Sanity check the link before a long run:
```bash
curl http://192.168.1.50:11434/v1/models           # server reachable from the Mac?
simba eval halumem --user-num 1                     # one user end-to-end
```

## Run

```bash
simba eval halumem --user-num 5
simba eval bench locomo      --qa --per 30 --k 10
simba eval bench longmemeval --qa --per 30 --k 10 --abstention
simba eval leaderboard            # refresh BENCHMARKS.md
```

## Notes / gotchas

- **Model ids differ per server.** Ollama: `gpt-oss:20b`. llama.cpp/vLLM: the path or
  HF id you launched with. Set `llm.model` to exactly what the server lists at
  `/v1/models`.
- **Reasoning output is handled.** `client._strip_reasoning` extracts the gpt-oss
  harmony `final` channel / strips `<think>` — and is a no-op if the server already
  returns clean content. So Ollama (clean) and a raw harmony server both work.
- **One server vs two.** One model serving both answerer + judge re-introduces
  self-grading bias. For HaluMem the judge grades memory correctness (weak loop) so
  one model is fine; for plain QA, use a *different* judge model (a 2nd `ollama pull`
  or a 2nd port). See the `eval-judge-one-server-default` note.
- **`openai-http` never auto-spawns.** Unlike `mlx-server`, the client won't try to
  start anything — make sure the remote server is up first (the eval fails open to
  empty answers if it's unreachable, which shows up as a high skip count).
- **Bandwidth is a non-issue**; if a run is slow it's GPU inference, not the network.
  Check `nvidia-smi` on the box to confirm the GPU is the thing working.
