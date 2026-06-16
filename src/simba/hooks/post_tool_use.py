"""PostToolUse hook — activity tracking + auto-learning from failures.

Tracks tool usage in the activity log. When a tool call fails (non-zero
exit code, error patterns in output), automatically generates a TOOL_RULE
memory so PreToolUse can warn before similar mistakes in the future.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib
import re
import time

import simba.config
import simba.hooks._io
import simba.hooks._memory_client
import simba.search.activity_tracker

# Error patterns that indicate a tool failure worth learning from.
_ERROR_PATTERNS = re.compile(
    r"(?:"
    r"ImportError|ModuleNotFoundError|FileNotFoundError"
    r"|PermissionError|Permission denied"
    r"|command not found|No such file or directory"
    r"|SyntaxError|IndentationError"
    r"|ConnectionRefusedError|ConnectionError"
    r"|OSError: \[Errno"
    r")",
    re.IGNORECASE,
)

# "Not found" errors that are normal for read-only discovery probes — a
# narrower subset of _ERROR_PATTERNS.  (NB: "command not found" is a real
# missing-binary mistake and is intentionally excluded here.)
_NOT_FOUND_RE = re.compile(
    r"No such file or directory|FileNotFoundError", re.IGNORECASE
)

# Patterns to normalize commands for generalization.
_NORMALIZE_PATTERNS = [
    # Replace absolute paths with /PATH/
    (re.compile(r"/(?:Users|home)/\S+"), "/PATH/"),
    # Replace hex addresses
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    # Replace line:col numbers
    (re.compile(r":\d+:\d+"), ":LINE:COL"),
    # Replace UUIDs
    (re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    ), "UUID"),
]

# Session-level dedup to avoid storing duplicate rules.
_RULE_DEDUP_CACHE = pathlib.Path("/tmp/claude-rule-dedup-cache.json")


def _hooks_cfg():
    import simba.hooks.config

    return simba.config.load("hooks")


# Lines that *mention* an error word but are source/doc, not an actual failure:
# Python source (except/import/from/raise), REPL echoes, comments, and lines
# carrying markdown backticks or a `->` arrow (risk-register / doc prose).
# Keeping `raise` here means a real traceback's clean `ImportError: ...` line is
# preferred over the preceding `raise ImportError(...)` source echo; the niche
# cost is missing a bare shell `raise: command not found` (raise is not a real
# command), which is an acceptable trade.
_NOISE_LINE_RE = re.compile(r"^(#|>>>|\.\.\.|except\b|import\b|from\b|raise\b)")

# Keys a Bash tool_response may use to report the process exit code.
_EXIT_CODE_KEYS = ("exit_code", "exitCode", "returncode", "return_code", "code")


def _has_error_pattern(text: str) -> bool:
    """Check if text contains a recognizable error pattern."""
    return bool(_ERROR_PATTERNS.search(text))


def _is_noise_line(line: str) -> bool:
    """True if an error-word line is really source/doc, not a failure."""
    s = line.strip()
    if _NOISE_LINE_RE.match(s):
        return True
    return "`" in s or " -> " in s


def _exit_code(tool_response: dict) -> int | None:
    """Return the reported exit code, or None when the response omits one."""
    for key in _EXIT_CODE_KEYS:
        if key in tool_response:
            try:
                return int(tool_response[key])
            except (TypeError, ValueError):
                return None
    return None


def _extract_error_line(text: str) -> str:
    """Return the first genuine error line, skipping source/doc mentions.

    Only lines matching an error pattern are considered; noise lines (source,
    comments, doc prose) are skipped.  Returns ``""`` when every error-word line
    is noise — the caller treats that as "nothing worth learning".
    """
    for line in text.split("\n"):
        line = line.strip()
        if _ERROR_PATTERNS.search(line) and not _is_noise_line(line):
            return line[:200]
    return ""


def _leading_verb(command: str) -> str:
    """Return the leading command verb, skipping ``VAR=val`` env assignments.

    ``/usr/bin/find`` collapses to ``find``.  Only inspects the first segment,
    so a probe buried in a ``&&`` chain is not detected (acceptable: the common
    case is a direct ``ls``/``bfs``/``find`` probe).
    """
    for tok in command.strip().split():
        name = tok.split("=", 1)[0]
        if "=" in tok and name.isidentifier():
            continue  # leading environment assignment, e.g. FOO=bar
        return tok.rsplit("/", 1)[-1]
    return ""


def _normalize_command(command: str) -> str:
    """Normalize a command for pattern matching (strip specifics)."""
    result = command
    for pattern, replacement in _NORMALIZE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def _check_rule_dedup(error_hash: str) -> bool:
    """Return True if this error was already stored this session."""
    try:
        cache = json.loads(_RULE_DEDUP_CACHE.read_text())
        hashes = cache.get("hashes", [])
        return error_hash in hashes
    except (json.JSONDecodeError, OSError):
        return False


def _save_rule_dedup(error_hash: str) -> None:
    """Record that we stored a rule for this error pattern."""
    hashes: list[str] = []
    with contextlib.suppress(json.JSONDecodeError, OSError):
        cache = json.loads(_RULE_DEDUP_CACHE.read_text())
        hashes = cache.get("hashes", [])

    hashes.append(error_hash)
    # Keep only last N entries (from config)
    max_rules = _hooks_cfg().rule_max_per_session
    hashes = hashes[-max_rules:]

    with contextlib.suppress(OSError):
        _RULE_DEDUP_CACHE.write_text(
            json.dumps({"hashes": hashes, "timestamp": time.time()})
        )


def _detect_failure(
    tool_name: str,
    tool_input: dict,
    tool_response: dict,
    *,
    skip_probe_not_found: bool = False,
    probe_verbs: frozenset[str] = frozenset(),
    reader_verbs: frozenset[str] = frozenset(),
    require_nonzero_exit: bool = True,
) -> dict | None:
    """Return failure info if the tool call genuinely failed, else None.

    Guards, in order:
    - **Exit/stderr gate:** when the response reports an exit code, only a
      non-zero one is a failure (``require_nonzero_exit``); when it omits one,
      trust ``stderr`` only — stdout often merely *mentions* error words.
    - **Reader/echo skip:** commands whose leading verb is in ``reader_verbs``
      emit file/echoed content, so error words there are not their own failure.
    - **Probe not-found skip:** a "no such file" from an ``ls``/``find``-style
      probe (``probe_verbs``) is a normal discovery miss.
    - **Line-shape:** the captured line must be a real error, not source/doc.
    """
    if tool_name != "Bash":
        return None

    stdout = tool_response.get("stdout", "")
    stderr = tool_response.get("stderr", "")
    output = tool_response.get("output", "")

    code = _exit_code(tool_response)
    if code is not None:
        if require_nonzero_exit and code == 0:
            return None
        error_text = output or f"{stdout}\n{stderr}"
    else:
        # No exit code reported: trust stderr (fall back to a merged-only field).
        error_text = stderr or (output if not stdout and not stderr else "")

    if not error_text or not _has_error_pattern(error_text):
        return None

    command = tool_input.get("command", "")
    verb = _leading_verb(command)
    if verb in reader_verbs:
        return None
    if (
        skip_probe_not_found
        and _NOT_FOUND_RE.search(error_text)
        and verb in probe_verbs
    ):
        return None

    error_line = _extract_error_line(error_text)
    if not error_line:
        return None

    return {
        "tool": tool_name,
        "command": command[:200],
        "error": error_line,
    }


def _store_failure_rule(failure: dict, cwd: str) -> None:
    """Store a TOOL_RULE memory from a detected failure."""
    tool = failure["tool"]
    command = failure["command"]
    error = failure["error"]

    # Dedup: hash the normalized command + error
    normalized = _normalize_command(command)
    error_hash = hashlib.md5(
        f"{tool}:{normalized}:{error}".encode()
    ).hexdigest()

    if _check_rule_dedup(error_hash):
        return

    from simba.memory.config import resolve_max_content_length

    content = f"{tool}: {error}"[: resolve_max_content_length()]
    context_data = {
        "tool": tool,
        "pattern": normalized[:200],
        "error_source": error[:200],
        "correction": "",
    }

    import simba.db

    simba.hooks._memory_client.store_memory(
        memory_type="TOOL_RULE",
        content=content,
        context=json.dumps(context_data),
        tags=[tool],
        confidence=0.85,
        # Opaque, worktree-robust project id (matches recall scoping) — never the
        # raw cwd, so rules don't leak across projects and DO share across a
        # repo's worktrees.
        project_path=simba.db.resolve_project_id(pathlib.Path(cwd) if cwd else None),
    )

    _save_rule_dedup(error_hash)


def main(hook_input: dict) -> str:
    """Track tool usage and learn from failures."""
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    tool_response = hook_input.get("tool_response", {})
    cwd_str = hook_input.get("cwd")
    cwd = pathlib.Path(cwd_str) if cwd_str else pathlib.Path.cwd()

    if not tool_name:
        return simba.hooks._io.empty("PostToolUse")

    # --- Activity tracking (existing behavior) ---
    detail = ""
    if tool_name in ("Read", "Edit", "Write"):
        detail = tool_input.get("file_path", "")
    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")
        detail = cmd[:100]
    elif tool_name in ("Glob", "Grep"):
        detail = tool_input.get("pattern", "")
    elif tool_name == "Task":
        agent = tool_input.get("subagent_type", "")
        desc = tool_input.get("description", "")
        detail = f"{agent}: {desc}" if agent else desc

    with contextlib.suppress(Exception):
        simba.search.activity_tracker.log_activity(cwd, tool_name, detail)

    # --- Auto-learn from failures ---
    cfg = _hooks_cfg()
    if cfg.auto_learn_from_failures and tool_response:
        with contextlib.suppress(Exception):
            probe_verbs = frozenset(
                v.strip()
                for v in cfg.learn_probe_commands.split(",")
                if v.strip()
            )
            reader_verbs = frozenset(
                v.strip()
                for v in cfg.learn_reader_commands.split(",")
                if v.strip()
            )
            failure = _detect_failure(
                tool_name,
                tool_input,
                tool_response,
                skip_probe_not_found=cfg.learn_skip_probe_not_found,
                probe_verbs=probe_verbs,
                reader_verbs=reader_verbs,
                require_nonzero_exit=cfg.learn_require_nonzero_exit,
            )
            if failure:
                _store_failure_rule(failure, str(cwd))

    return simba.hooks._io.empty("PostToolUse")
