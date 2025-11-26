"""Ensure DuckDB imports stay confined to the storage gateway module."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_duckdb_usage_is_localized() -> None:
    """Verify duckdb imports or types are not used outside storage/."""
    root = Path("src")
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if "storage" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "import duckdb" in text or "DuckDBPyConnection" in text or "DuckDBPyRelation" in text:
            violations.append(str(path))
    if violations:
        pytest.fail(f"duckdb usage not allowed outside storage/: {violations}")
