"""Error capture pipeline.

Reads hook input, parses transcript, detects errors, stores reflection in SQLite.
Ported from claude-tailor/src/hook.js. Uses stdlib only (no external deps).
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import string
import sys
import time

import simba._vendor.peewee as pw
import simba.config
import simba.db
import simba.hooks._tail


def _init_reflections_schema(conn) -> None:
    """Create the reflections table and indexes if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reflections (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            error_type TEXT NOT NULL,
            snippet TEXT NOT NULL DEFAULT '',
            context TEXT NOT NULL DEFAULT '{}',
            signature TEXT NOT NULL DEFAULT '',
            session_id TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_reflections_ts ON reflections(ts DESC);
        CREATE INDEX IF NOT EXISTS idx_reflections_type ON reflections(error_type);
    """)
    # Migration for DBs created before session_id existed (spec 33 v2 rule
    # R3): the reflections-ledger reader needs session identity to tell a
    # REPEAT failure (recurring across >=2 distinct sessions) from the same
    # session logging the same error more than once.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(reflections)")}
    if "session_id" not in existing:
        conn.execute(
            "ALTER TABLE reflections ADD COLUMN session_id TEXT NOT NULL DEFAULT ''"
        )


simba.db.register_schema(_init_reflections_schema)


def _hooks_cfg():
    """Load the hooks config section (registers it on first access).

    ``tailor.hook`` already depends on ``simba.db``, which itself imports
    ``simba.config`` -- so this adds no new dependency edge, just a direct
    import instead of a transitive one (``simba.config`` has no import of
    ``simba.tailor``/``simba.db``, so there is no cycle).
    """
    import simba.hooks.config

    _ = simba.hooks.config  # ensure the "hooks" section is registered
    return simba.config.load("hooks")


class Reflection(simba.db.BaseModel):
    id = pw.TextField(primary_key=True)
    ts = pw.TextField()
    error_type = pw.TextField()
    snippet = pw.TextField(default="")
    context = pw.TextField(default="{}")
    signature = pw.TextField(default="")
    # Hook payload's session_id (spec 33 v2 rule R3): the reflections-ledger
    # reader clusters by signature and needs this to count DISTINCT sessions.
    session_id = pw.TextField(default="")

    class Meta:
        table_name = "reflections"


ERROR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Error:", re.IGNORECASE),
    re.compile(r"TypeError:", re.IGNORECASE),
    re.compile(r"ReferenceError:", re.IGNORECASE),
    re.compile(r"SyntaxError:", re.IGNORECASE),
    re.compile(r"AssertionError:", re.IGNORECASE),
    re.compile(r"failed", re.IGNORECASE),
    re.compile(r"ENOENT", re.IGNORECASE),
    re.compile(r"EACCES", re.IGNORECASE),
    re.compile(r"Cannot find module", re.IGNORECASE),
    re.compile(r"Cannot read properties", re.IGNORECASE),
    re.compile(r"Uncaught", re.IGNORECASE),
    re.compile(r"Exception", re.IGNORECASE),
]


def detect_error(content: str) -> bool:
    """Return True if content matches any error pattern."""
    return any(p.search(content) for p in ERROR_PATTERNS)


def extract_error_type(content: str) -> str:
    """Extract the error type string from content (lowercase, no colon)."""
    for pattern in ERROR_PATTERNS:
        match = pattern.search(content)
        if match:
            return match.group(0).lower().replace(":", "").replace(" ", "")
    return "unknown"


def extract_snippet(content: str) -> str:
    """Extract a snippet around the first error match (100 chars before, 500 after)."""
    for pattern in ERROR_PATTERNS:
        match = pattern.search(content)
        if match:
            start = max(0, match.start() - 100)
            return content[start : match.start() + 500]
    return ""


def extract_context(snippet: str) -> dict[str, str]:
    """Extract file, operation, and module context from a snippet."""
    context: dict[str, str] = {}

    file_match = re.search(
        r"(?:at |in |file://|from\s+)([^\s:]+\.(?:jsx|tsx|js|ts))", snippet
    )
    op_match = re.search(r"(?:at\s+)?(\w+)\s*\(", snippet)
    module_match = re.search(
        r'(?:module|package|from\s+[\'"]@?)?([a-z0-9\-]+)["\']?', snippet, re.IGNORECASE
    )

    if file_match and "node_modules" not in file_match.group(1):
        context["file"] = file_match.group(1)
    if op_match:
        context["operation"] = op_match.group(1)
    if module_match:
        context["module"] = module_match.group(1)

    return context


def normalize_snippet(snippet: str) -> str:
    """Normalize a snippet for clustering.

    Replace line numbers, paths, hex addresses, large numbers.
    """
    result = snippet
    result = re.sub(r":\d+:\d+", ":LINE:COL", result)
    result = re.sub(r"/[\w\-/]+/", "/PATH/", result)
    result = re.sub(r"0x[0-9a-f]+", "0xADDR", result)
    result = re.sub(r"\d{10,}", "NUM", result)
    return result


def generate_signature(error_type: str, normalized_snippet: str) -> str:
    """Generate a signature from error type and normalized snippet."""
    signature = error_type
    sig_match = re.search(r"(\w+(?:\s+\w+)?)", normalized_snippet)
    if sig_match:
        signature += f"-{sig_match.group(1)}"
    return signature


def create_reflection_entry(
    error_type: str, snippet: str, context: dict[str, str]
) -> dict:
    """Create a reflection entry dict."""
    normalized = normalize_snippet(snippet)
    signature = generate_signature(error_type, normalized)
    random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))

    return {
        "id": f"nano-{int(time.time() * 1000)}-{random_suffix}",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "error_type": error_type,
        "snippet": snippet.strip(),
        "context": context,
        "signature": signature,
    }


def parse_transcript_content(lines: list[str]) -> str:
    """Parse transcript JSONL lines and collect content that might contain errors."""
    parts: list[str] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue

        if "toolUseResult" in entry:
            val = entry["toolUseResult"]
            parts.append(val if isinstance(val, str) else json.dumps(val))

        message = entry.get("message", {})
        if isinstance(message, dict):
            content = message.get("content", [])
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "tool_result"
                        and item.get("content")
                    ):
                        val = item["content"]
                        parts.append(val if isinstance(val, str) else json.dumps(val))

    return "\n".join(parts)


def process_hook(input_str: str) -> None:
    """Main hook processing pipeline.

    Transcript reads are bounded to a tail window (``hooks.stop_tail_mb``,
    2026-07-20): this used to ``read_text()`` the WHOLE transcript
    unconditionally on EVERY Stop hook fire -- a co-culprit in a live 30GB
    daemon RSS balloon (Stop fires every turn, unlike PreToolUse/PreCompact,
    whose reads were bounded separately). Error detection now only sees the
    most recent window: an error line that has scrolled out of the tail is no
    longer (re-)detected on THIS pass, but it was already reflected by an
    EARLIER turn's pass over the transcript -- Stop fires every turn, so
    nothing is permanently missed, only already-seen.
    """
    if not input_str:
        return

    try:
        hook_data = json.loads(input_str)
    except (json.JSONDecodeError, ValueError):
        return

    transcript_path_str = hook_data.get("transcript_path")
    if not transcript_path_str:
        return
    transcript_path = pathlib.Path(transcript_path_str)
    if not transcript_path.exists():
        return

    cap_bytes = int(_hooks_cfg().stop_tail_mb * 1_000_000)
    try:
        tail, _ = simba.hooks._tail.read_tail_bytes(transcript_path, cap_bytes)
        transcript_content = tail.decode("utf-8", errors="replace")
        transcript_lines = [
            line for line in transcript_content.strip().split("\n") if line
        ]
    except OSError:
        return

    full_content = parse_transcript_content(transcript_lines)
    if not full_content or len(full_content) < 50:
        return

    if not detect_error(full_content):
        return

    error_type = extract_error_type(full_content)
    snippet = extract_snippet(full_content)
    context = extract_context(snippet)
    context["signature"] = generate_signature(error_type, normalize_snippet(snippet))

    reflection = create_reflection_entry(error_type, snippet, context)

    cwd = hook_data.get("cwd", ".")
    session_id = hook_data.get("session_id", "")
    try:
        with simba.db.connect(pathlib.Path(cwd)):
            Reflection.create(
                id=reflection["id"],
                ts=reflection["ts"],
                error_type=reflection["error_type"],
                snippet=reflection["snippet"],
                context=json.dumps(reflection["context"]),
                signature=reflection["signature"],
                session_id=session_id,
            )
    except Exception:
        pass


if __name__ == "__main__":
    stdin_input = sys.stdin.read()
    process_hook(stdin_input)
