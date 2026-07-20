"""Searchable sidecar index for raw session transcript messages.

This is a rebuildable SQLite/FTS index over transcript messages. It is separate
from LanceDB semantic memories: indexing a transcript makes exact raw session
text queryable without injecting it into normal hook context.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import re
import time
import typing

import simba.config
import simba.db

logger = logging.getLogger("simba.memory")

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_FILE_REF_RE = re.compile(
    r"(?:(?:/[\w.@+-][\w.@+/-]*[\w.@+-])|(?:[\w.@+-]+/)+[\w.@+-]+)(?::\d+)?"
)
_TAG_BLOCK_RE = re.compile(r"<(user|assistant|system|tool)>\s*(.*?)\s*</\1>", re.S)


@simba.config.configurable("sessions")
@dataclasses.dataclass
class SessionsConfig:
    search_limit: int = 20
    # Skip parsing any transcript source larger than this (MB). 2026-07-20:
    # `run_index` in the daemon's executor slurped legacy multi-GB
    # transcript.jsonl copies via read_text()+splitlines -> 32GB MALLOC_LARGE
    # (malloc_history-attributed). Parsing also streams now, but the cap
    # bounds per-cycle CPU on pathological legacy artifacts. Default sits
    # above hooks.pre_compact_max_transcript_mb (256) so every capped raw
    # copy still indexes; 0 disables.
    max_parse_mb: float = 384.0


@dataclasses.dataclass(frozen=True)
class ParsedMessage:
    session_id: str
    project_path: str
    transcript_path: str
    source: str
    message_index: int
    role: str
    text: str
    tool_refs: list[str] = dataclasses.field(default_factory=list)
    file_refs: list[str] = dataclasses.field(default_factory=list)
    parent_session_id: str = ""


@dataclasses.dataclass(frozen=True)
class IndexResult:
    session_id: str
    project_path: str
    transcript_path: str
    source: str
    message_count: int


def _init_schema(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS session_messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT NOT NULL, "
        "project_path TEXT NOT NULL DEFAULT '', "
        "transcript_path TEXT NOT NULL, "
        "source TEXT NOT NULL DEFAULT '', "
        "message_index INTEGER NOT NULL, "
        "role TEXT NOT NULL DEFAULT '', "
        "text TEXT NOT NULL DEFAULT '', "
        "tool_refs TEXT NOT NULL DEFAULT '[]', "
        "file_refs TEXT NOT NULL DEFAULT '[]', "
        "parent_session_id TEXT NOT NULL DEFAULT '', "
        "indexed_at REAL NOT NULL DEFAULT 0.0, "
        "UNIQUE(transcript_path, message_index))"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_messages_session "
        "ON session_messages(session_id, message_index)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_messages_project "
        "ON session_messages(project_path, session_id)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS session_messages_fts USING fts5("
        "message_id UNINDEXED, session_id UNINDEXED, project_path UNINDEXED, "
        "transcript_path UNINDEXED, role UNINDEXED, text, tokenize='unicode61')"
    )


simba.db.register_schema(_init_schema)


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _content_text(value: typing.Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_content_text(item) for item in value if item is not None)
    if not isinstance(value, dict):
        return ""
    parts: list[str] = []
    for key in ("text", "input_text", "output", "result", "thinking", "summary"):
        val = value.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    content = value.get("content")
    if content is not None:
        txt = _content_text(content)
        if txt.strip():
            parts.append(txt)
    return "\n".join(parts)


def _json_tool_refs(entry: dict[str, typing.Any]) -> list[str]:
    refs: set[str] = set()
    payload = entry.get("payload")
    for obj in (entry, payload if isinstance(payload, dict) else {}):
        for key in ("tool_name", "toolName", "name"):
            val = obj.get(key)
            if isinstance(val, str) and val:
                refs.add(val)
    return sorted(refs)


def _file_refs(text: str) -> list[str]:
    refs: set[str] = set()
    for match in _FILE_REF_RE.finditer(text):
        token = match.group(0).strip(".,);]")
        if "/" in token and len(token) > 2:
            refs.add(token)
    return sorted(refs)


def _parent_session_id(payload: dict[str, typing.Any]) -> str:
    parent = payload.get("parent_thread_id")
    if isinstance(parent, str) and parent:
        return parent
    source = payload.get("source")
    if isinstance(source, dict):
        sub = source.get("subagent")
        if isinstance(sub, dict):
            spawn = sub.get("thread_spawn")
            if isinstance(spawn, dict):
                parent = spawn.get("parent_thread_id")
                if isinstance(parent, str):
                    return parent
    return ""


def _message_from_json_entry(
    entry: dict[str, typing.Any],
) -> tuple[str, str, list[str]] | None:
    payload = entry.get("payload")
    if isinstance(payload, dict):
        ptype = payload.get("type")
        if ptype == "message":
            text = _content_text(payload.get("content"))
            return (str(payload.get("role") or ""), text, _json_tool_refs(entry))
        if ptype == "function_call_output":
            text = _content_text(payload.get("output") or payload.get("content"))
            return ("tool", text, _json_tool_refs(entry))
        if ptype == "function_call":
            name = str(payload.get("name") or "")
            args = _content_text(payload.get("arguments"))
            text = f"{name}\n{args}".strip()
            return ("tool_call", text, _json_tool_refs(entry))

    message = entry.get("message")
    if isinstance(message, dict):
        text = _content_text(message.get("content"))
        return (str(message.get("role") or ""), text, _json_tool_refs(entry))

    if entry.get("type") == "event_msg" and isinstance(payload, dict):
        text = _content_text(payload.get("message") or payload.get("text"))
        return ("event", text, _json_tool_refs(entry))

    text = _content_text(entry.get("text") or entry.get("content"))
    if text:
        return (str(entry.get("role") or "event"), text, _json_tool_refs(entry))
    return None


def _source_over_cap(path: pathlib.Path) -> bool:
    """True (with one WARNING) when ``path`` exceeds ``sessions.max_parse_mb``."""
    try:
        cap_mb = float(simba.config.load("sessions").max_parse_mb)
        size = path.stat().st_size
    except OSError:
        return False
    if cap_mb <= 0:
        return False
    if size <= cap_mb * 1024 * 1024:
        return False
    logger.warning(
        "[sessions] skipping %s: %.1fMB exceeds sessions.max_parse_mb=%.0fMB",
        path,
        size / (1024 * 1024),
        cap_mb,
    )
    return True


def _iter_stripped_lines(path: pathlib.Path) -> typing.Iterator[str]:
    """Yield newline-stripped lines, bounded by one line of memory at a time."""
    try:
        with path.open(errors="replace") as fh:
            for raw_line in fh:
                yield raw_line.rstrip("\n")
    except OSError:
        return


def _parse_jsonl(
    path: pathlib.Path,
    *,
    default_project_path: str = "",
    default_session_id: str = "",
    source: str = "codex",
) -> list[ParsedMessage]:
    session_id = default_session_id or path.stem
    project_path = default_project_path
    parent_session_id = ""
    messages: list[ParsedMessage] = []
    if _source_over_cap(path):
        return []
    # Stream, never slurp: read_text()+splitlines on a multi-GB transcript
    # was the 2026-07-20 32GB-heap driver (see SessionsConfig.max_parse_mb).
    for line in _iter_stripped_lines(path):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "session_meta":
            payload = entry.get("payload")
            if isinstance(payload, dict):
                sid = payload.get("id")
                cwd = payload.get("cwd")
                if isinstance(sid, str) and sid:
                    session_id = sid
                if isinstance(cwd, str) and cwd:
                    project_path = cwd
                parent_session_id = _parent_session_id(payload)
            continue
        parsed = _message_from_json_entry(entry)
        if parsed is None:
            continue
        role, text, tool_refs = parsed
        text = _clean_text(text)
        if not text:
            continue
        messages.append(
            ParsedMessage(
                session_id=session_id,
                project_path=project_path,
                transcript_path=str(path),
                source=source,
                message_index=len(messages),
                role=role or "unknown",
                text=text,
                tool_refs=tool_refs,
                file_refs=_file_refs(text),
                parent_session_id=parent_session_id,
            )
        )
    return messages


def _metadata_for_markdown(path: pathlib.Path) -> dict[str, typing.Any]:
    meta_path = path.parent / "metadata.json"
    if not meta_path.is_file():
        return {}
    try:
        data = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _parse_markdown(
    path: pathlib.Path,
    *,
    default_project_path: str = "",
    default_session_id: str = "",
    source: str = "claude",
) -> list[ParsedMessage]:
    if _source_over_cap(path):
        return []
    try:
        raw = path.read_text(errors="replace")
    except OSError:
        return []
    meta = _metadata_for_markdown(path)
    session_id = (
        str(meta.get("session_id") or "")
        or default_session_id
        or _tag_value(raw, "session-id")
        or path.parent.name
    )
    project_path = (
        str(meta.get("project_path") or "")
        or default_project_path
        or _tag_value(raw, "project-path")
    )
    messages: list[ParsedMessage] = []
    for match in _TAG_BLOCK_RE.finditer(raw):
        role = match.group(1)
        text = _clean_text(match.group(2))
        if not text:
            continue
        messages.append(
            ParsedMessage(
                session_id=session_id,
                project_path=project_path,
                transcript_path=str(path),
                source=source,
                message_index=len(messages),
                role=role,
                text=text,
                file_refs=_file_refs(text),
            )
        )
    return messages


def _tag_value(raw: str, tag: str) -> str:
    match = re.search(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", raw, re.S)
    return _clean_text(match.group(1)) if match else ""


def parse_transcript(
    path: str | pathlib.Path,
    *,
    project_path: str = "",
    session_id: str = "",
    source: str = "",
) -> list[ParsedMessage]:
    p = pathlib.Path(path).expanduser()
    inferred_source = source or ("codex" if p.suffix == ".jsonl" else "claude")
    if p.suffix == ".jsonl":
        return _parse_jsonl(
            p,
            default_project_path=project_path,
            default_session_id=session_id,
            source=inferred_source,
        )
    return _parse_markdown(
        p,
        default_project_path=project_path,
        default_session_id=session_id,
        source=inferred_source,
    )


def index_transcript(
    path: str | pathlib.Path,
    *,
    project_path: str = "",
    session_id: str = "",
    source: str = "",
    cwd: pathlib.Path | None = None,
) -> IndexResult:
    """Replace the derived message index for one transcript."""
    p = pathlib.Path(path).expanduser()
    messages = parse_transcript(
        p, project_path=project_path, session_id=session_id, source=source
    )
    transcript_path = str(p)
    indexed_at = time.time()
    with simba.db.get_db(cwd) as conn:
        old_ids = [
            row["id"]
            for row in conn.execute(
                "SELECT id FROM session_messages WHERE transcript_path = ?",
                (transcript_path,),
            )
        ]
        if old_ids:
            conn.executemany(
                "DELETE FROM session_messages_fts WHERE message_id = ?",
                [(str(mid),) for mid in old_ids],
            )
        conn.execute(
            "DELETE FROM session_messages WHERE transcript_path = ?",
            (transcript_path,),
        )
        for msg in messages:
            cur = conn.execute(
                "INSERT INTO session_messages ("
                "session_id, project_path, transcript_path, source, message_index, "
                "role, text, tool_refs, file_refs, parent_session_id, indexed_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    msg.session_id,
                    msg.project_path,
                    msg.transcript_path,
                    msg.source,
                    msg.message_index,
                    msg.role,
                    msg.text,
                    json.dumps(msg.tool_refs),
                    json.dumps(msg.file_refs),
                    msg.parent_session_id,
                    indexed_at,
                ),
            )
            message_id = str(cur.lastrowid)
            conn.execute(
                "INSERT INTO session_messages_fts ("
                "message_id, session_id, project_path, transcript_path, role, text"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    msg.session_id,
                    msg.project_path,
                    msg.transcript_path,
                    msg.role,
                    msg.text,
                ),
            )
        conn.commit()
    first = messages[0] if messages else None
    return IndexResult(
        session_id=(first.session_id if first else session_id or p.stem),
        project_path=(first.project_path if first else project_path),
        transcript_path=transcript_path,
        source=(
            first.source
            if first
            else source or ("codex" if p.suffix == ".jsonl" else "claude")
        ),
        message_count=len(messages),
    )


def _match_query(query: str) -> str:
    seen: set[str] = set()
    terms: list[str] = []
    for match in _TOKEN_RE.finditer(query or ""):
        token = match.group(0)
        if len(token) < 2:
            continue
        low = token.lower()
        if low in seen:
            continue
        seen.add(low)
        terms.append('"' + token.replace('"', '""') + '"')
    return " AND ".join(terms)


def _row_to_dict(row) -> dict[str, typing.Any]:
    message_index = row["message_index"]
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "project_path": row["project_path"],
        "transcript_path": row["transcript_path"],
        "source": row["source"],
        "message_index": message_index,
        "message_span": [message_index, message_index],
        "role": row["role"],
        "text": row["text"],
        "tool_refs": json.loads(row["tool_refs"] or "[]"),
        "file_refs": json.loads(row["file_refs"] or "[]"),
        "parent_session_id": row["parent_session_id"],
    }


def search(
    query: str,
    *,
    project_path: str = "",
    limit: int = 20,
    cwd: pathlib.Path | None = None,
) -> list[dict[str, typing.Any]]:
    """Search indexed raw messages and return session/message spans."""
    match = _match_query(query)
    if not match:
        return []
    sql = (
        "SELECT m.* FROM session_messages_fts "
        "JOIN session_messages m "
        "ON m.id = CAST(session_messages_fts.message_id AS INTEGER) "
        "WHERE session_messages_fts MATCH ?"
    )
    params: list[typing.Any] = [match]
    if project_path:
        sql += " AND m.project_path = ?"
        params.append(project_path)
    sql += " ORDER BY bm25(session_messages_fts), m.indexed_at DESC LIMIT ?"
    params.append(max(1, int(limit)))
    with simba.db.get_db(cwd) as conn:
        try:
            rows = list(conn.execute(sql, params))
        except Exception:
            like = f"%{query}%"
            fallback_sql = (
                "SELECT * FROM session_messages WHERE text LIKE ? "
                "ORDER BY indexed_at DESC LIMIT ?"
            )
            fallback_params: list[typing.Any] = [like, max(1, int(limit))]
            if project_path:
                fallback_sql = (
                    "SELECT * FROM session_messages WHERE text LIKE ? "
                    "AND project_path = ? ORDER BY indexed_at DESC LIMIT ?"
                )
                fallback_params = [like, project_path, max(1, int(limit))]
            rows = list(conn.execute(fallback_sql, fallback_params))
    return [_row_to_dict(row) for row in rows]


def indexed_count(*, cwd: pathlib.Path | None = None) -> int:
    with simba.db.get_db(cwd) as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM session_messages").fetchone()
    return int(row["n"] if row is not None else 0)


def default_search_limit(*, cwd: pathlib.Path | None = None) -> int:
    cfg = simba.config.load("sessions", root=cwd)
    return max(1, int(cfg.search_limit))
