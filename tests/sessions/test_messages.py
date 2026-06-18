from __future__ import annotations

import json
import pathlib

import simba.sessions.messages as messages


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_index_codex_jsonl_and_search_exact_error(tmp_path: pathlib.Path) -> None:
    transcript = tmp_path / "rollout.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "session_meta",
                "payload": {"id": "codex-session-1", "cwd": "/repo/project"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "debug the test"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "output": "Traceback: ValueError: bad state in src/app.py:42",
                },
            },
        ],
    )

    result = messages.index_transcript(transcript, cwd=tmp_path)

    assert result.message_count == 2
    hits = messages.search(
        "ValueError bad state src/app.py:42",
        project_path="/repo/project",
        cwd=tmp_path,
    )
    assert len(hits) == 1
    assert hits[0]["session_id"] == "codex-session-1"
    assert hits[0]["role"] == "tool"
    assert hits[0]["message_span"] == [1, 1]
    assert "src/app.py:42" in hits[0]["file_refs"]


def test_index_markdown_transcript_and_project_filter(tmp_path: pathlib.Path) -> None:
    session_dir = tmp_path / "claude-session"
    session_dir.mkdir()
    transcript = session_dir / "transcript.md"
    transcript.write_text(
        "<user>Remember the release flag lives in docs/release.md:7.</user>\n"
        "<assistant>Confirmed.</assistant>\n"
    )
    (session_dir / "metadata.json").write_text(
        json.dumps(
            {
                "session_id": "claude-session",
                "project_path": "/repo/claude",
                "transcript_path": str(transcript),
            }
        )
    )

    result = messages.index_transcript(transcript, cwd=tmp_path)

    assert result.session_id == "claude-session"
    assert result.message_count == 2
    assert (
        messages.search("release flag", project_path="/repo/other", cwd=tmp_path)
        == []
    )
    hits = messages.search("release flag docs release md 7", cwd=tmp_path)
    assert hits[0]["project_path"] == "/repo/claude"
    assert "docs/release.md:7" in hits[0]["file_refs"]


def test_reindex_replaces_transcript_rows(tmp_path: pathlib.Path) -> None:
    transcript = tmp_path / "rollout.jsonl"
    _write_jsonl(
        transcript,
        [
            {"type": "session_meta", "payload": {"id": "s1", "cwd": "/repo"}},
            {"message": {"role": "user", "content": "ORIGINAL_TOKEN"}},
        ],
    )
    messages.index_transcript(transcript, cwd=tmp_path)

    _write_jsonl(
        transcript,
        [
            {"type": "session_meta", "payload": {"id": "s1", "cwd": "/repo"}},
            {"message": {"role": "user", "content": "REPLACEMENT_TOKEN"}},
        ],
    )
    messages.index_transcript(transcript, cwd=tmp_path)

    assert messages.search("ORIGINAL_TOKEN", cwd=tmp_path) == []
    assert messages.search("REPLACEMENT_TOKEN", cwd=tmp_path)[0]["session_id"] == "s1"
    assert messages.indexed_count(cwd=tmp_path) == 1
