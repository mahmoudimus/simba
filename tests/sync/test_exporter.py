from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from simba.sync.exporter import export_all_tables, export_table, get_export_dir


class TestGetExportDir:
    def test_creates_directory(self, tmp_path: Path) -> None:
        d = get_export_dir(tmp_path)
        assert d.exists()
        assert d == tmp_path / ".simba" / "exports"

    def test_idempotent(self, tmp_path: Path) -> None:
        d1 = get_export_dir(tmp_path)
        d2 = get_export_dir(tmp_path)
        assert d1 == d2


class TestExportTable:
    def test_writes_markdown(self, tmp_path: Path) -> None:
        rows = [
            {"error_type": "TypeError", "snippet": "bad arg", "signature": "abc"},
        ]
        p = export_table(tmp_path, "reflections", rows)
        assert p is not None
        assert p.exists()
        text = p.read_text()
        assert "TypeError" in text
        assert "bad arg" in text

    def test_empty_rows_returns_none(self, tmp_path: Path) -> None:
        assert export_table(tmp_path, "reflections", []) is None

    def test_unknown_table_returns_none(self, tmp_path: Path) -> None:
        rows = [{"key": "val"}]
        assert export_table(tmp_path, "nonexistent", rows) is None

    def test_multiple_rows_separated(self, tmp_path: Path) -> None:
        rows = [
            {"error_type": "A", "snippet": "s1", "signature": "x"},
            {"error_type": "B", "snippet": "s2", "signature": "y"},
        ]
        p = export_table(tmp_path, "reflections", rows)
        assert p is not None
        text = p.read_text()
        assert "---" in text


class TestExportAllTables:
    @patch("simba.sync.exporter._run_qmd_embed")
    def test_exports_multiple_tables(self, mock_qmd: object, tmp_path: Path) -> None:
        tables = {
            "reflections": [
                {"error_type": "E", "snippet": "s", "signature": "sig"},
            ],
            "facts": [
                {"category": "dep", "fact": "uses ruff"},
            ],
        }
        paths = export_all_tables(tmp_path, tables)
        assert len(paths) == 2
        names = {p.name for p in paths}
        assert "reflections.md" in names
        assert "facts.md" in names

    @patch("simba.sync.exporter._run_qmd_embed")
    def test_skips_empty_tables(self, mock_qmd: object, tmp_path: Path) -> None:
        tables = {
            "reflections": [],
            "facts": [{"category": "x", "fact": "y"}],
        }
        paths = export_all_tables(tmp_path, tables)
        assert len(paths) == 1

    @patch("simba.sync.exporter._run_qmd_embed")
    def test_no_qmd_when_no_exports(self, mock_qmd: object, tmp_path: Path) -> None:
        paths = export_all_tables(tmp_path, {"reflections": []})
        assert paths == []
        mock_qmd.assert_not_called()
