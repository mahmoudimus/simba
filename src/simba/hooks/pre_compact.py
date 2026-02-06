"""PreCompact hook — export transcript as JSONL + markdown.

Reads stdin JSON with session_id and transcript_path, parses the
transcript into markdown, writes to ~/.claude/transcripts/{sessionId}/,
and creates metadata for later learning extraction.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import shutil
import sys
import time

import simba.hooks._memory_client


def _parse_transcript_to_markdown(lines: list[str]) -> tuple[str, int]:
    """Parse JSONL transcript lines into markdown. Returns (md, msg_count)."""
    messages: list[str] = []
    session_id = ""
    cwd = ""
    msg_count = 0

    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        if not session_id:
            session_id = entry.get("session_id", "")
        if not cwd:
            cwd = entry.get("cwd", "")

        message = entry.get("message", {})
        if not isinstance(message, dict):
            # Check for top-level content/text
            if "toolUseResult" in entry:
                continue
            text = entry.get("text") or entry.get("content")
            if text and isinstance(text, str):
                messages.append(f"<user>\n{text}\n</user>")
                msg_count += 1
            continue

        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            if isinstance(content, str):
                user_text = content
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        parts.append(item.get("text", "") or item.get("content", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                user_text = "\n".join(p for p in parts if p)
            else:
                continue
            if user_text:
                messages.append(f"<user>\n{user_text}\n</user>")
                msg_count += 1

        elif role == "assistant":
            thinking_parts: list[str] = []
            response_parts: list[str] = []

            if isinstance(content, str):
                response_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "thinking":
                        thinking_parts.append(item.get("thinking", ""))
                    elif item.get("type") == "text":
                        response_parts.append(item.get("text", ""))

            # Also check top-level thinking field
            if entry.get("thinking"):
                thinking_parts.append(entry["thinking"])

            assistant_block = "<assistant>\n"
            if thinking_parts:
                thinking_text = "\n".join(t for t in thinking_parts if t)
                assistant_block += f"<thinking>\n{thinking_text}\n</thinking>\n"
            response_text = "\n".join(r for r in response_parts if r)
            if response_text:
                assistant_block += f"<response>\n{response_text}\n</response>\n"
            assistant_block += "</assistant>"
            messages.append(assistant_block)
            msg_count += 1

    # Build final markdown
    md_parts = [
        "<session-transcript>",
        "<metadata>",
        f"  <session-id>{session_id}</session-id>",
        f"  <project-path>{cwd}</project-path>",
        "</metadata>",
        "",
    ]
    md_parts.extend(messages)
    md_parts.append("</session-transcript>")
    return "\n\n".join(md_parts), msg_count


def main(hook_input: dict) -> str:
    """Run the PreCompact hook pipeline. Returns JSON output string."""
    session_id = hook_input.get("session_id") or hook_input.get("sessionId") or ""
    transcript_path_str = (
        hook_input.get("transcript_path") or hook_input.get("transcriptPath") or ""
    )
    cwd_str = hook_input.get("cwd", "")

    if not session_id or not transcript_path_str:
        return json.dumps({"suppressOutput": True})

    transcript_path = pathlib.Path(transcript_path_str)
    if not transcript_path.exists():
        return json.dumps({"suppressOutput": True})

    try:
        transcript_lines = transcript_path.read_text().strip().split("\n")
    except OSError:
        return json.dumps({"suppressOutput": True})

    # Export directory
    transcripts_dir = pathlib.Path.home() / ".claude" / "transcripts"
    session_dir = transcripts_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy original JSONL
    dest_jsonl = session_dir / "transcript.jsonl"
    with contextlib.suppress(OSError):
        shutil.copy2(transcript_path, dest_jsonl)

    # 2. Convert to markdown
    markdown, msg_count = _parse_transcript_to_markdown(transcript_lines)
    dest_md = session_dir / "transcript.md"
    with contextlib.suppress(OSError):
        dest_md.write_text(markdown)

    # 3. Write metadata
    metadata = {
        "session_id": session_id,
        "project_path": cwd_str,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "transcript_path": str(dest_md),
        "daemon_url": simba.hooks._memory_client.daemon_url(),
        "status": "pending_extraction",
    }
    metadata_path = session_dir / "metadata.json"
    with contextlib.suppress(OSError):
        metadata_path.write_text(json.dumps(metadata, indent=2))

    # 4. Symlink latest
    latest = transcripts_dir / "latest.json"
    try:
        latest.unlink(missing_ok=True)
        latest.symlink_to(metadata_path)
    except OSError:
        pass

    # Log to stderr
    print(f"[pre-compact] Exported transcript ({msg_count} messages)", file=sys.stderr)
    print(f"[pre-compact] Transcript: {dest_md}", file=sys.stderr)

    return json.dumps({"suppressOutput": True})
