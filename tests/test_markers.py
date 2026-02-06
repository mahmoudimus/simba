"""Tests for simba.markers -- shared SIMBA marker utilities."""

from __future__ import annotations

import simba.markers


class TestBeginTag:
    def test_correct_string_output(self) -> None:
        assert simba.markers.begin_tag("core") == "<!-- BEGIN SIMBA:core -->"

    def test_namespace_is_embedded(self) -> None:
        tag = simba.markers.begin_tag("managed")
        assert "SIMBA" in tag
        assert "SIMBA:managed" in tag


class TestEndTag:
    def test_correct_string_output(self) -> None:
        assert simba.markers.end_tag("core") == "<!-- END SIMBA:core -->"

    def test_namespace_is_embedded(self) -> None:
        tag = simba.markers.end_tag("managed")
        assert "SIMBA" in tag
        assert "SIMBA:managed" in tag


class TestExtractBlocks:
    def test_single_block_extraction(self) -> None:
        content = (
            "# Header\n"
            "<!-- BEGIN SIMBA:core -->\n"
            "rule one\n"
            "<!-- END SIMBA:core -->\n"
            "footer\n"
        )
        blocks = simba.markers.extract_blocks(content, "core")
        assert len(blocks) == 1
        assert "rule one\n" in blocks[0]

    def test_multiple_blocks_same_name(self) -> None:
        content = (
            "## Section A\n"
            "<!-- BEGIN SIMBA:core -->\n"
            "- Block A rule\n"
            "<!-- END SIMBA:core -->\n"
            "\n"
            "## Section B\n"
            "<!-- BEGIN SIMBA:core -->\n"
            "- Block B rule\n"
            "<!-- END SIMBA:core -->\n"
        )
        blocks = simba.markers.extract_blocks(content, "core")
        assert len(blocks) == 2
        assert "Block A rule" in blocks[0]
        assert "Block B rule" in blocks[1]

    def test_no_matches_returns_empty_list(self) -> None:
        content = "# Just some markdown\nNo markers here.\n"
        blocks = simba.markers.extract_blocks(content, "core")
        assert blocks == []

    def test_empty_content_returns_empty_list(self) -> None:
        blocks = simba.markers.extract_blocks("", "core")
        assert blocks == []

    def test_empty_block(self) -> None:
        content = "<!-- BEGIN SIMBA:core -->\n<!-- END SIMBA:core -->\n"
        blocks = simba.markers.extract_blocks(content, "core")
        assert len(blocks) == 1
        assert blocks[0] == ""

    def test_preserves_inner_formatting(self) -> None:
        content = (
            "<!-- BEGIN SIMBA:core -->\n"
            "- Rule with **bold**\n"
            "- Rule with `code`\n"
            "  - Nested item\n"
            "<!-- END SIMBA:core -->\n"
        )
        blocks = simba.markers.extract_blocks(content, "core")
        assert len(blocks) == 1
        assert "**bold**" in blocks[0]
        assert "`code`" in blocks[0]
        assert "  - Nested item" in blocks[0]

    def test_excludes_markers_from_output(self) -> None:
        content = (
            "<!-- BEGIN SIMBA:core -->\n- Important rule\n<!-- END SIMBA:core -->\n"
        )
        blocks = simba.markers.extract_blocks(content, "core")
        assert len(blocks) == 1
        assert "<!-- BEGIN" not in blocks[0]
        assert "<!-- END" not in blocks[0]

    def test_tolerates_extra_whitespace_in_markers(self) -> None:
        content = (
            "<!--  BEGIN  SIMBA:core  -->\n- Spaced rule\n<!--  END  SIMBA:core  -->\n"
        )
        blocks = simba.markers.extract_blocks(content, "core")
        assert len(blocks) == 1
        assert "Spaced rule" in blocks[0]

    def test_unclosed_marker_returns_empty_list(self) -> None:
        content = "<!-- BEGIN SIMBA:core -->\n- Unclosed rule\n"
        blocks = simba.markers.extract_blocks(content, "core")
        assert blocks == []

    def test_opening_only_marker_returns_empty_list(self) -> None:
        content = "<!-- BEGIN SIMBA:core -->\nsome text"
        blocks = simba.markers.extract_blocks(content, "core")
        assert blocks == []

    def test_different_name_blocks_not_extracted(self) -> None:
        content = (
            "<!-- BEGIN SIMBA:managed -->\n"
            "managed content\n"
            "<!-- END SIMBA:managed -->\n"
        )
        blocks = simba.markers.extract_blocks(content, "core")
        assert blocks == []

    def test_mixed_blocks_extract_only_named(self) -> None:
        content = (
            "<!-- BEGIN SIMBA:core -->\n"
            "core content\n"
            "<!-- END SIMBA:core -->\n"
            "<!-- BEGIN SIMBA:managed -->\n"
            "managed content\n"
            "<!-- END SIMBA:managed -->\n"
            "<!-- BEGIN SIMBA:core -->\n"
            "more core\n"
            "<!-- END SIMBA:core -->\n"
        )
        blocks = simba.markers.extract_blocks(content, "core")
        assert len(blocks) == 2
        assert "core content" in blocks[0]
        assert "more core" in blocks[1]

        managed = simba.markers.extract_blocks(content, "managed")
        assert len(managed) == 1
        assert "managed content" in managed[0]


class TestUpdateBlocks:
    def test_updates_single_section(self) -> None:
        content = (
            "# Title\n"
            "<!-- BEGIN SIMBA:managed -->\n"
            "old content\n"
            "<!-- END SIMBA:managed -->\n"
            "footer\n"
        )
        result = simba.markers.update_blocks(content, {"managed": "new content\n"})
        assert "new content" in result
        assert "old content" not in result
        assert "# Title" in result
        assert "footer" in result

    def test_updates_multiple_sections(self) -> None:
        content = (
            "<!-- BEGIN SIMBA:alpha -->\n"
            "old alpha\n"
            "<!-- END SIMBA:alpha -->\n"
            "<!-- BEGIN SIMBA:beta -->\n"
            "old beta\n"
            "<!-- END SIMBA:beta -->\n"
        )
        result = simba.markers.update_blocks(
            content, {"alpha": "new alpha\n", "beta": "new beta\n"}
        )
        assert "new alpha" in result
        assert "new beta" in result
        assert "old alpha" not in result
        assert "old beta" not in result

    def test_leaves_non_updated_blocks_untouched(self) -> None:
        content = (
            "<!-- BEGIN SIMBA:core -->\n"
            "core rules\n"
            "<!-- END SIMBA:core -->\n"
            "<!-- BEGIN SIMBA:managed -->\n"
            "old managed\n"
            "<!-- END SIMBA:managed -->\n"
        )
        result = simba.markers.update_blocks(content, {"managed": "new managed\n"})
        assert "core rules" in result
        assert "new managed" in result
        assert "old managed" not in result

    def test_empty_updates_returns_content_unchanged(self) -> None:
        content = "<!-- BEGIN SIMBA:core -->\ncontent\n<!-- END SIMBA:core -->\n"
        result = simba.markers.update_blocks(content, {})
        assert result == content

    def test_does_not_create_new_markers(self) -> None:
        content = "# No markers here\nJust text.\n"
        result = simba.markers.update_blocks(content, {"core": "injected\n"})
        assert result == content
        assert "injected" not in result
        assert "SIMBA" not in result

    def test_content_with_no_markers_returned_unchanged(self) -> None:
        content = "plain text with no markers at all"
        result = simba.markers.update_blocks(content, {"anything": "value\n"})
        assert result == content


class TestHasMarker:
    def test_true_when_marker_exists(self) -> None:
        content = (
            "some text\n<!-- BEGIN SIMBA:core -->\ncontent\n<!-- END SIMBA:core -->\n"
        )
        assert simba.markers.has_marker(content, "core") is True

    def test_false_when_marker_missing(self) -> None:
        content = "# No markers\nJust text.\n"
        assert simba.markers.has_marker(content, "core") is False


class TestMakeEmptyBlock:
    def test_correct_format(self) -> None:
        block = simba.markers.make_empty_block("managed")
        assert block == ("<!-- BEGIN SIMBA:managed -->\n<!-- END SIMBA:managed -->")

    def test_round_trip_extract_produces_empty_item(self) -> None:
        block = simba.markers.make_empty_block("managed")
        extracted = simba.markers.extract_blocks(block, "managed")
        assert len(extracted) == 1
        assert extracted[0] == ""
