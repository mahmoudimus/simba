"""RLM autonomous engine — pluggable, cheap-by-default execution.

Two engine families share one ``RlmEngine`` contract:

- **ClaudeCliEngine** (``rlm.engine = "claude-cli"``, the default): *agentic*.
  Spawns a detached, cheap ``claude -p`` that navigates the transcript via the
  rlm_* tools and runs ``simba memory store`` itself.
- **LlmCliEngine** (``rlm.engine = "llm-cli"``): *completion-driven*. Sends a
  single eval'd extraction prompt to ``llm -m <engine_model>`` (simonw's CLI,
  e.g. deepseek-v4-flash), parses a JSON array of memories from the reply, and
  stores each itself. Cheaper than an agentic loop and the basis for using simba
  as a personal-assistant memory (swap in a facts/preferences/events prompt).

``get_engine`` returns None for engine="claude" (the default agent-driven path)
and for engines not yet implemented. Both engines are fire-and-forget: the work
runs detached so a PreCompact hook is never blocked.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import subprocess
import sys
import tempfile
import typing


class RlmEngine(typing.Protocol):
    def run(self, prompt: str, *, cwd: str, session_source: str = "") -> None:
        """Dispatch a detached, cheap agent with ``prompt``. Fire-and-forget —
        the agent does the work itself and never blocks the caller.
        ``session_source`` tags stored memories with their origin session (used
        by completion engines; agentic engines bake it into the prompt)."""

    def digest(self, transcript_id: str, query: str, *, cwd: str) -> None:
        """Extract lossless memories from a transcript and store them.
        Always invoked detached/background — never blocks a hook."""


# Memory types accepted from a completion engine's JSON output. Mirrors
# ``simba.__main__._VALID_MEMORY_TYPES`` (the canonical set the store enforces).
_VALID_TYPES = frozenset(
    {
        "WORKING_SOLUTION",
        "GOTCHA",
        "PATTERN",
        "DECISION",
        "FAILURE",
        "PREFERENCE",
        "EPISODE",
        "REFLECTION",
    }
)

# Agentic default (claude-cli): the model reads the transcript via rlm_* tools
# and stores memories itself. Coding-scoped; override via rlm.digest_prompt.
_DIGEST_PROMPT = (
    "A coding session just ended; its full transcript is loaded as "
    "transcript_id '{tid}'. Use rlm_grep / rlm_peek / rlm_window to read the "
    "key decisions, gotchas, working solutions, and failures from it — read "
    "the actual regions, be lossless, do not guess. Store each as a memory:\n"
    "  simba memory store --type <TYPE> --content <<={maxlen} chars> "
    "--context <details> --project-path '{cwd}' --session-source '{tid}'\n"
    "TYPE is one of WORKING_SOLUTION, GOTCHA, PATTERN, DECISION, FAILURE, "
    "PREFERENCE. Store 5-15 high-value, specific learnings; skip generic "
    "knowledge. When finished, run: simba rlm complete '{tid}' --stored <count>."
)

# Completion default (llm-cli): a single eval'd extraction prompt. The model
# returns a JSON array; the engine stores each item. Coding-scoped by default;
# override via rlm.digest_prompt for a personal-assistant digest.
_LLM_DIGEST_PROMPT = (
    "A conversation just ended. From the transcript below, extract the durable, "
    "specific learnings worth remembering long-term. Return ONLY a JSON array; "
    'each element an object with keys "type", "content", "context". "type" is '
    "one of WORKING_SOLUTION, GOTCHA, PATTERN, DECISION, FAILURE, PREFERENCE. "
    '"content" is a specific statement of at most {maxlen} characters; "context" '
    "holds supporting detail. Capture 5-15 high-value items; skip generic "
    "knowledge. Output nothing but the JSON array.\n\nTranscript:\n{transcript}"
)


def _build_digest_prompt(
    transcript_id: str,
    cwd: str,
    *,
    template: str = _DIGEST_PROMPT,
    maxlen: int = 200,
) -> str:
    return template.format(tid=transcript_id, cwd=cwd, maxlen=maxlen)


class ClaudeCliEngine:
    def __init__(self, cfg) -> None:
        self._cfg = cfg

    def _argv(self, prompt: str) -> list[str]:
        return [
            "claude",
            "-p",
            prompt,
            "--model",
            self._cfg.engine_model,
            "--allowedTools",
            self._cfg.engine_allowed_tools,
            "--permission-mode",
            "acceptEdits",
            "--max-turns",
            str(self._cfg.engine_max_turns),
            "--output-format",
            "json",
        ]

    def _env(self) -> dict:
        env = os.environ.copy()
        if self._cfg.engine_base_url:
            env["ANTHROPIC_BASE_URL"] = self._cfg.engine_base_url
            env["ANTHROPIC_AUTH_TOKEN"] = os.environ.get(
                self._cfg.engine_api_key_env, ""
            )
        return env

    def run(self, prompt: str, *, cwd: str, session_source: str = "") -> None:
        """Spawn a detached, cheap ``claude -p`` with ``prompt`` (no blocking).

        ``session_source`` is ignored: the agentic prompt already embeds the
        provenance in its ``simba memory store --session-source`` instruction."""
        subprocess.Popen(
            self._argv(prompt),
            env=self._env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=cwd,
        )

    def digest(self, transcript_id: str, query: str, *, cwd: str) -> None:
        import simba.memory.config

        maxlen = simba.memory.config.resolve_max_content_length()
        template = getattr(self._cfg, "digest_prompt", "") or _DIGEST_PROMPT
        self.run(
            _build_digest_prompt(
                transcript_id, cwd, template=template, maxlen=maxlen
            ),
            cwd=cwd,
        )


class LlmCliEngine:
    """Completion-driven engine: a single ``llm -m <model>`` extraction call."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg

    def run(self, prompt: str, *, cwd: str, session_source: str = "") -> None:
        _spawn_worker(prompt, cwd=cwd, session_source=session_source, mark_rlm=False)

    def digest(self, transcript_id: str, query: str, *, cwd: str) -> None:
        import simba.memory.config

        maxlen = simba.memory.config.resolve_max_content_length()
        text = _load_transcript_text(transcript_id)
        template = getattr(self._cfg, "digest_prompt", "") or _LLM_DIGEST_PROMPT
        prompt = template.format(
            transcript=text, cwd=cwd, tid=transcript_id, maxlen=maxlen
        )
        _spawn_worker(prompt, cwd=cwd, session_source=transcript_id, mark_rlm=True)


def get_engine(cfg) -> RlmEngine | None:
    """Return the configured engine, or None for the agent-driven default."""
    if cfg.engine == "claude-cli":
        return ClaudeCliEngine(cfg)
    if cfg.engine == "llm-cli":
        return LlmCliEngine(cfg)
    return None  # "claude" (default) + api/local-gguf (later phases)


# --- completion-engine worker -----------------------------------------------
# The detached worker (`simba rlm run-llm`) calls run_completion_from_file ->
# run_completion_worker, which is engine-binary-agnostic and dependency-injected
# (client / store_fn / complete_job) so it is unit-testable without a real LLM.


def _load_transcript_text(transcript_id: str) -> str:
    """Read the markdown transcript PreCompact exported for ``transcript_id``."""
    path = (
        pathlib.Path.home()
        / ".claude"
        / "transcripts"
        / transcript_id
        / "transcript.md"
    )
    try:
        return path.read_text()
    except OSError:
        return ""


def _parse_memories(text: str) -> list[dict]:
    """Parse a JSON array of memory objects from an LLM reply.

    Keeps only well-formed objects whose ``type`` is a known memory type and
    whose ``content`` is non-empty; normalizes ``type`` to upper-case. Anything
    that is not a JSON array of objects yields ``[]`` (fail-open)."""
    import simba.llm.client as lc

    data = lc._extract_json(text)
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        mtype = str(item.get("type", "")).strip().upper()
        content = str(item.get("content", "")).strip()
        if mtype in _VALID_TYPES and content:
            out.append(
                {
                    "type": mtype,
                    "content": content,
                    "context": str(item.get("context", "")).strip(),
                }
            )
    return out


def _build_llm_client(cfg):
    """Build an LlmClient that runs ``llm -m <cfg.engine_model>`` (llm-cli)."""
    import dataclasses

    import simba.config
    import simba.llm.client
    import simba.llm.config  # registers the "llm" section

    _ = simba.llm.config
    base = simba.config.load("llm")
    spec = dataclasses.replace(base, provider="llm-cli", model=cfg.engine_model)
    return simba.llm.client.LlmClient(spec)


def _store_memory(mem: dict, *, cwd: str, session_source: str) -> bool:
    """Store one parsed memory via the same path the agentic engine uses
    (``simba memory store`` — reuses type/length validation + daemon POST +
    duplicate detection). Returns True on a 0 exit code (stored or duplicate)."""
    argv = [
        sys.executable,
        "-m",
        "simba",
        "memory",
        "store",
        "--type",
        mem["type"],
        "--content",
        mem["content"],
        "--context",
        mem.get("context", ""),
        "--project-path",
        cwd,
    ]
    if session_source:
        argv += ["--session-source", session_source]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except Exception:
        return False
    return proc.returncode == 0


def _complete_rlm_job(transcript_id: str, cwd: str, n_stored: int) -> None:
    import simba.rlm.jobs

    with contextlib.suppress(Exception):
        simba.rlm.jobs.complete(transcript_id, cwd, n_stored)


def run_completion_worker(
    prompt: str,
    *,
    cwd: str,
    session_source: str = "",
    mark_rlm: bool = False,
    cfg=None,
    client=None,
    store_fn=None,
    complete_job=None,
) -> int:
    """Complete ``prompt``, parse a JSON array of memories, store each.

    Returns the number of memories successfully stored. When ``mark_rlm`` is set
    (a digest run), marks the rlm_jobs row done with the stored count. All
    collaborators are injectable so the orchestration is testable without an LLM,
    a daemon, or a database."""
    if client is None:
        if cfg is None:
            import simba.config
            import simba.rlm.config  # registers the "rlm" section

            _ = simba.rlm.config
            cfg = simba.config.load("rlm")
        client = _build_llm_client(cfg)
    store = store_fn or _store_memory

    val_enabled = getattr(cfg, "extraction_validation_enabled", False) if cfg else False
    val_min_support = getattr(cfg, "extraction_validation_min_support", 0.5) if cfg \
        else 0.5

    reply = client.complete(prompt) or ""
    n = 0
    for mem in _parse_memories(reply):
        if val_enabled:
            # Source = the prompt's embedded transcript. Polarity parity is
            # unreliable over a long source, so gate on hard-value + support only;
            # an ungrounded claim (hallucinated number / unsupported) is dropped.
            import simba.memory.extraction_validation as _ev

            if not _ev.validate_extraction(
                mem["content"], prompt,
                min_support=val_min_support, check_polarity=False,
            ).ok:
                continue
        if store(mem, cwd=cwd, session_source=session_source):
            n += 1
    if mark_rlm and session_source:
        (complete_job or _complete_rlm_job)(session_source, cwd, n)
    return n


def run_completion_from_file(
    prompt_file: str,
    *,
    cwd: str,
    session_source: str = "",
    mark_rlm: bool = False,
) -> int:
    """Read the prompt the engine wrote, run the digest, then unlink the file."""
    path = pathlib.Path(prompt_file)
    try:
        prompt = path.read_text()
    except OSError:
        with contextlib.suppress(OSError):
            path.unlink()
        return 0
    with contextlib.suppress(OSError):
        path.unlink()
    return run_completion_worker(
        prompt, cwd=cwd, session_source=session_source, mark_rlm=mark_rlm
    )


def _spawn_worker(
    prompt: str, *, cwd: str, session_source: str, mark_rlm: bool
) -> None:
    """Detach a ``simba rlm run-llm`` worker to run the (blocking) completion +
    storage out of the caller's process. The prompt is handed over via a temp
    file (it can be large), which the worker reads and deletes."""
    try:
        fd, path = tempfile.mkstemp(prefix="simba-rlm-llm-", suffix=".txt")
        with os.fdopen(fd, "w") as fh:
            fh.write(prompt)
    except OSError:
        return
    argv = [
        sys.executable,
        "-m",
        "simba",
        "rlm",
        "run-llm",
        "--prompt-file",
        path,
        "--cwd",
        cwd,
        "--session-source",
        session_source,
    ]
    if mark_rlm:
        argv.append("--mark-rlm-complete")
    subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        cwd=cwd,
    )
