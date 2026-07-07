"""Shell-command parsing for the tool-redirect feature.

Extracts the *effective* program invocations from a Bash command string —
splitting on `;`/`|`/`&&`/`||`/newlines (quote-aware), stripping `env VAR=…`
prefixes, recursing into nested `bash -c "…"` / `cmd /c …`, and resolving
`uv run <tool>` to the tool itself. Pure + dependency-free. Adapted from the
user's PreToolUse guard.
"""

from __future__ import annotations

import dataclasses
import re
import shlex

_SHELL_WRAPPERS = {"cmd", "powershell", "pwsh", "bash", "sh", "zsh"}
_UV_RUN_OPTS_WITH_VALUE = {
    "--python",
    "--with",
    "--with-requirements",
    "--with-editable",
    "--directory",
    "--project",
    "--index",
    "--extra",
    "--group",
    "--no-group",
    "--only-group",
    "--config-file",
    "--env-file",
    "--python-platform",
    "--resolution",
    "--index-strategy",
    "--keyring-provider",
    "--link-mode",
    "--refresh-package",
    "--upgrade-package",
    "--no-binary-package",
    "--no-build-package",
}


@dataclasses.dataclass
class Invocation:
    program: str  # normalized effective program name (basename, lowercased)
    words: list[str]  # the segment's tokens, program first


def program_name(word: str) -> str:
    """Basename of a program token: strip ./, dirs, exe suffixes; lowercase."""
    cleaned = word.strip().strip("'\"").replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    base = cleaned.rsplit("/", 1)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat", ".ps1"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _is_env_assignment(word: str) -> bool:
    return re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", word) is not None


def split_segments(command: str) -> list[str]:
    """Split a command on ; | && || and newlines, respecting quotes."""
    segments, buf, quote, i = [], [], None, 0
    while i < len(command):
        ch = command[i]
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            buf.append(ch)
            i += 1
            continue
        dbl_amp = ch == "&" and command[i + 1 : i + 2] == "&"
        dbl_pipe = ch == "|" and command[i + 1 : i + 2] == "|"
        if ch in {";", "|", "\r", "\n"} or dbl_amp:
            seg = "".join(buf).strip()
            if seg:
                segments.append(seg)
            buf = []
            i += 2 if (dbl_amp or dbl_pipe) else 1
            continue
        buf.append(ch)
        i += 1
    seg = "".join(buf).strip()
    if seg:
        segments.append(seg)
    return segments


def tokenize(segment: str) -> list[str]:
    """Split a segment into words — quote/escape-aware via shlex.

    Uses ``shlex.split`` (POSIX) for correct handling of quotes and escapes.
    Falls back to a lenient char-split on malformed input (unbalanced quotes)
    so the hook can never crash on a weird command.
    """
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return _lenient_tokenize(segment)


def _lenient_tokenize(segment: str) -> list[str]:
    """Fallback tokenizer that never raises (best-effort on malformed quotes)."""
    words, buf, quote = [], [], None
    for ch in segment:
        if quote is not None:
            if ch == quote:
                quote = None
            else:
                buf.append(ch)
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch.isspace():
            if buf:
                words.append("".join(buf))
                buf = []
            continue
        buf.append(ch)
    if buf:
        words.append("".join(buf))
    return words


def command_words(segment: str) -> list[str]:
    """Tokens of a segment with leading wrappers + env assignments stripped."""
    words = tokenize(segment)
    while words and words[0] in {"&", "call", "exec", "command"}:
        words = words[1:]
    if words and program_name(words[0]) == "env":
        words = words[1:]
    while words and _is_env_assignment(words[0]):
        words = words[1:]
    return words


def nested_shell_command(words: list[str]) -> str | None:
    """If words is a shell wrapper (bash -c "…"), return the inner command."""
    if not words:
        return None
    first = program_name(words[0])
    if first not in _SHELL_WRAPPERS:
        return None
    if first == "cmd":
        for i, w in enumerate(words[1:], start=1):
            if w.lower() in {"/c", "/r"} and i + 1 < len(words):
                return " ".join(words[i + 1 :])
        return None
    if first in {"powershell", "pwsh"}:
        for i, w in enumerate(words[1:], start=1):
            if w.lower() in {"-command", "-c", "/c"} and i + 1 < len(words):
                return " ".join(words[i + 1 :])
        return None
    for i, w in enumerate(words[1:], start=1):
        if "c" in w.lower().lstrip("-") and i + 1 < len(words):
            return " ".join(words[i + 1 :])
    return None


def resolve_uv_run_tool(words: list[str]) -> str | None:
    """For `uv run [opts] <tool> …`, return <tool> (else None)."""
    if len(words) < 3 or program_name(words[0]) != "uv" or words[1] != "run":
        return None
    i = 2
    while i < len(words):
        w = words[i]
        if w == "--":
            i += 1
            break
        if not w.startswith("-"):
            break
        if "=" not in w and w in _UV_RUN_OPTS_WITH_VALUE:
            i += 2
        else:
            i += 1
    return words[i] if i < len(words) else None


def invoked_programs(command: str, _depth: int = 0) -> list[Invocation]:
    """Return the effective program invocations in a command string."""
    if _depth > 4:
        return []
    out: list[Invocation] = []
    for seg in split_segments(command):
        words = command_words(seg)
        if not words:
            continue
        nested = nested_shell_command(words)
        if nested is not None:
            out.extend(invoked_programs(nested, _depth + 1))
            continue
        if program_name(words[0]) == "uv" and len(words) > 1 and words[1] == "run":
            tool = resolve_uv_run_tool(words)
            if tool is not None:
                out.append(Invocation(program=program_name(tool), words=words))
                continue
        out.append(Invocation(program=program_name(words[0]), words=words))
    return out
