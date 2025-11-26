"""Ensure adapter surfaces delegate queries through QueryService."""

from __future__ import annotations

from pathlib import Path

import pytest

ADAPTER_DIRS = (
    Path("src/codeintel/server"),
    Path("src/codeintel/mcp"),
    Path("src/codeintel/cli"),
    Path("src/codeintel/docs_export"),
)
ALLOW_RAW_SQL = {
    Path("src/codeintel/server/query_service.py"),
    Path("src/codeintel/server/datasets.py"),
    Path("src/codeintel/mcp/query_service.py"),
    Path("src/codeintel/docs_export/export_jsonl.py"),
    Path("src/codeintel/docs_export/export_parquet.py"),
    Path("src/codeintel/storage/views.py"),
}
FORBIDDEN_MARKERS = ("FROM docs.", "FROM analytics.", "FROM graph.")


def test_adapters_do_not_embed_sql() -> None:
    """Verify adapter layers avoid raw SQL in favor of QueryService."""
    adapter_files: list[Path] = []
    for base in ADAPTER_DIRS:
        adapter_files.extend(base.rglob("*.py"))

    for path in sorted(adapter_files):
        if path in ALLOW_RAW_SQL:
            continue
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "con.execute" in line and "SELECT 1" not in line:
                message = f"{path} executes unexpected SQL: {line.strip()}"
                pytest.fail(message)
        for marker in FORBIDDEN_MARKERS:
            if marker in text:
                message = f"{path} should not embed SQL marker {marker!r}"
                pytest.fail(message)
        if "duckdb.connect" in text:
            message = f"{path} should not open DuckDB directly"
            pytest.fail(message)
