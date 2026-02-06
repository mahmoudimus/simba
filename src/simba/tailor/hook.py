"""Error capture pipeline.

Reads hook input, parses transcript, detects errors, appends reflection.
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

        if "toolUseResult" in entry:
            parts.append(entry["toolUseResult"])

        message = entry.get("message", {})
        if isinstance(message, dict):
            for item in message.get("content", []):
                if (
                    isinstance(item, dict)
                    and item.get("type") == "tool_result"
                    and item.get("content")
                ):
                    parts.append(item["content"])

    return "\n".join(parts)


def process_hook(input_str: str) -> None:
    """Main hook processing pipeline."""
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

    try:
        transcript_content = transcript_path.read_text()
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
    memory_path = pathlib.Path(cwd) / ".claude-tailor" / "memory" / "reflections.jsonl"
    try:
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(memory_path, "a") as f:
            f.write(json.dumps(reflection) + "\n")
    except OSError:
        pass


if __name__ == "__main__":
    stdin_input = sys.stdin.read()
    process_hook(stdin_input)
