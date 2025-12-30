"""Tests for the SQLAlchemy-free DuckDB Iceberg catalog."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
from pyiceberg.exceptions import CommitFailedException

from codeintel.storage.iceberg.duckdb_catalog import DuckDBCatalog

pytestmark = pytest.mark.no_runtime_env


def _catalog(tmp_path: Path) -> DuckDBCatalog:
    catalog_path = tmp_path / "catalog.duckdb"
    warehouse_path = tmp_path / "warehouse"
    warehouse_path.mkdir(parents=True, exist_ok=True)
    return DuckDBCatalog(
        "test",
        uri=f"duckdb:///{catalog_path}",
        warehouse=str(warehouse_path),
    )


def test_duckdb_catalog_namespace_roundtrip(tmp_path: Path) -> None:
    """DuckDBCatalog should create and update namespaces."""
    with _catalog(tmp_path) as catalog:
        catalog.create_namespace("docs")
        namespaces = catalog.list_namespaces()
        assert ("docs",) in namespaces
        props = catalog.load_namespace_properties("docs")
        assert props.get("exists") == "true"
        summary = catalog.update_namespace_properties("docs", updates={"owner": "ci"})
        assert "owner" in summary.updated


def test_duckdb_catalog_table_roundtrip(tmp_path: Path) -> None:
    """DuckDBCatalog should create and load tables."""
    with _catalog(tmp_path) as catalog:
        catalog.create_namespace("docs")
        schema = pa.schema([("id", pa.int64())])
        table = catalog.create_table("docs.sample", schema)
        loaded = catalog.load_table("docs.sample")
        assert table.name() == ("docs", "sample")
        assert loaded.name() == ("docs", "sample")
        assert catalog.table_exists("docs.sample")
        assert catalog.list_tables("docs") == [("docs", "sample")]


def test_duckdb_catalog_commit_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent commits should raise CommitFailedException."""
    with _catalog(tmp_path) as catalog:
        catalog.create_namespace("docs")
        schema = pa.schema([("id", pa.int64())])
        catalog.create_table("docs.sample", schema)
        table = catalog.load_table("docs.sample")
        stale = catalog.load_table("docs.sample")

        table.transaction().set_properties(owner="primary").commit_transaction()

        monkeypatch.setattr(catalog, "load_table", lambda _identifier: stale)

        with pytest.raises(CommitFailedException):
            stale.transaction().set_properties(owner="stale").commit_transaction()
