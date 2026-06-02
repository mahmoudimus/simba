"""Tests for the shared _memory_client module."""

from __future__ import annotations

import unittest.mock

import simba.hooks._memory_client


class TestDaemonUrl:
    def test_returns_expected_url(self):
        url = simba.hooks._memory_client.daemon_url()
        assert url == "http://localhost:8741"

    def test_uses_config_defaults(self):
        import simba.config
        import simba.hooks.config

        cfg = simba.config.load("hooks")
        assert cfg.daemon_host == "localhost"
        assert cfg.daemon_port == 8741


class TestRecallMemories:
    def test_success(self):
        mock_resp = unittest.mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "memories": [{"type": "GOTCHA", "content": "test", "similarity": 0.9}]
        }

        with unittest.mock.patch("httpx.post", return_value=mock_resp) as mock_post:
            result = simba.hooks._memory_client.recall_memories("auth module")

        assert len(result) == 1
        assert result[0]["type"] == "GOTCHA"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["json"]["query"] == "auth module"
        # No floor passed -> omitted so the daemon picks it via query intent.
        assert "minSimilarity" not in call_kwargs.kwargs["json"]
        assert call_kwargs.kwargs["json"]["maxResults"] == 3

    def test_with_project_path(self):
        mock_resp = unittest.mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"memories": []}

        with unittest.mock.patch("httpx.post", return_value=mock_resp) as mock_post:
            simba.hooks._memory_client.recall_memories(
                "query", project_path="/my/project"
            )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["projectPath"] == "/my/project"

    def test_custom_similarity(self):
        mock_resp = unittest.mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"memories": []}

        with unittest.mock.patch("httpx.post", return_value=mock_resp) as mock_post:
            simba.hooks._memory_client.recall_memories("query", min_similarity=0.45)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["minSimilarity"] == 0.45

    def test_connection_error_returns_empty(self):
        import httpx

        with unittest.mock.patch(
            "httpx.post", side_effect=httpx.ConnectError("refused")
        ):
            result = simba.hooks._memory_client.recall_memories("query")

        assert result == []

    def test_non_200_returns_empty(self):
        mock_resp = unittest.mock.MagicMock()
        mock_resp.status_code = 500

        with unittest.mock.patch("httpx.post", return_value=mock_resp):
            result = simba.hooks._memory_client.recall_memories("query")

        assert result == []


class TestFormatMemories:
    def test_empty_returns_empty_string(self):
        assert simba.hooks._memory_client.format_memories([], source="test") == ""

    def test_xml_structure(self):
        memories = [
            {"type": "GOTCHA", "content": "watch out for X", "similarity": 0.85},
            {"type": "PATTERN", "content": "always do Y", "similarity": 0.72},
        ]
        result = simba.hooks._memory_client.format_memories(
            memories, source="user-prompt"
        )

        assert 'source="user-prompt"' in result
        assert 'type="GOTCHA"' in result
        assert 'type="PATTERN"' in result
        assert "watch out for X" in result
        assert "always do Y" in result
        assert 'similarity="0.85"' in result
        assert 'similarity="0.72"' in result
        assert result.startswith("[Recalled 2 memories")
        assert "</recalled-memories>" in result

    def test_source_parameterized(self):
        memories = [{"type": "DECISION", "content": "use uv", "similarity": 0.5}]

        result_a = simba.hooks._memory_client.format_memories(
            memories, source="thinking-block"
        )
        result_b = simba.hooks._memory_client.format_memories(
            memories, source="user-prompt"
        )

        assert 'source="thinking-block"' in result_a
        assert 'source="user-prompt"' in result_b

    def test_annotates_created_date_and_marks_newest(self):
        # Relevance order is preserved (older hit ranks first by similarity),
        # but the most-recently-created memory is flagged so the model can
        # prefer fresher info when two memories conflict.
        memories = [
            {
                "type": "DECISION",
                "content": "older call",
                "similarity": 0.9,
                "createdAt": "2026-01-01T10:00:00Z",
            },
            {
                "type": "DECISION",
                "content": "newer call",
                "similarity": 0.6,
                "createdAt": "2026-05-30T10:00:00Z",
            },
        ]
        result = simba.hooks._memory_client.format_memories(
            memories, source="user-prompt"
        )
        assert 'created="2026-01-01"' in result
        assert 'created="2026-05-30"' in result
        # exactly one newest marker, on the more-recently-created memory
        assert result.count('recency="newest"') == 1
        seg = result.split('recency="newest"')[1].split("</memory>")[0]
        assert "newer call" in seg
        # order preserved (relevance first): older (top hit) before newer
        assert result.index("older call") < result.index("newer call")

    def test_omits_recency_annotations_when_no_dates(self):
        memories = [{"type": "GOTCHA", "content": "x", "similarity": 0.5}]
        result = simba.hooks._memory_client.format_memories(memories, source="t")
        assert "created=" not in result
        assert "recency=" not in result
