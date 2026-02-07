"""Tests for the PreCompact hook module."""

from __future__ import annotations

import json
import unittest.mock

import simba.hooks.pre_compact


class TestParseTranscriptToMarkdown:
    def test_parses_user_message(self):
        lines = [json.dumps({"message": {"role": "user", "content": "Hello world"}})]
        md, count = simba.hooks.pre_compact._parse_transcript_to_markdown(lines)
        assert "Hello world" in md
        assert count == 1

    def test_parses_assistant_message(self):
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Response here"}],
                    }
                }
            )
        ]
        md, count = simba.hooks.pre_compact._parse_transcript_to_markdown(lines)
        assert "Response here" in md
        assert count == 1

    def test_parses_thinking_block(self):
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Let me think..."},
                            {"type": "text", "text": "Answer"},
                        ],
                    }
                }
            )
        ]
        md, _count = simba.hooks.pre_compact._parse_transcript_to_markdown(lines)
        assert "Let me think..." in md
        assert "<thinking>" in md
        assert "Answer" in md

    def test_skips_invalid_json(self):
        lines = [
            "{bad json}",
            json.dumps({"message": {"role": "user", "content": "ok"}}),
        ]
        _md, count = simba.hooks.pre_compact._parse_transcript_to_markdown(lines)
        assert count == 1

    def test_user_content_with_nested_list(self):
        """Content items may have a 'content' field that is a list, not a string."""
        lines = [
            json.dumps(
                {
                    "message": {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "normal text"},
                            {
                                "type": "tool_result",
                                "content": [
                                    {"type": "text", "text": "nested"}
                                ],
                            },
                        ],
                    }
                }
            )
        ]
        md, count = simba.hooks.pre_compact._parse_transcript_to_markdown(lines)
        assert "normal text" in md
        assert count == 1

    def test_empty_lines(self):
        _md, count = simba.hooks.pre_compact._parse_transcript_to_markdown([])
        assert count == 0


class TestPreCompactMain:
    def test_requires_session_id(self):
        result = json.loads(
            simba.hooks.pre_compact.main({"transcript_path": "/tmp/t.jsonl"})
        )
        assert result.get("suppressOutput") is True

    def test_requires_transcript_path(self):
        result = json.loads(simba.hooks.pre_compact.main({"session_id": "abc"}))
        assert result.get("suppressOutput") is True

    def test_exports_transcript(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text(
            json.dumps({"message": {"role": "user", "content": "test"}}) + "\n"
        )

        fake_home = tmp_path / "home"
        fake_home.mkdir()

        with unittest.mock.patch("pathlib.Path.home", return_value=fake_home):
            result = json.loads(
                simba.hooks.pre_compact.main(
                    {
                        "session_id": "test-session",
                        "transcript_path": str(transcript),
                        "cwd": str(tmp_path),
                    }
                )
            )

        assert result.get("suppressOutput") is True

        session_dir = fake_home / ".claude" / "transcripts" / "test-session"
        assert (session_dir / "transcript.jsonl").exists()
        assert (session_dir / "transcript.md").exists()
        assert (session_dir / "metadata.json").exists()

        metadata = json.loads((session_dir / "metadata.json").read_text())
        assert metadata["session_id"] == "test-session"
        assert metadata["status"] == "pending_extraction"

    def test_nonexistent_transcript(self, tmp_path):
        result = json.loads(
            simba.hooks.pre_compact.main(
                {
                    "session_id": "abc",
                    "transcript_path": str(tmp_path / "no.jsonl"),
                }
            )
        )
        assert result.get("suppressOutput") is True
