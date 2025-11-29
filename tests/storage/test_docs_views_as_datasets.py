"""Docs views should behave like first-class datasets."""

from __future__ import annotations

import duckdb
import pytest

from codeintel.storage.datasets import load_dataset_registry
from codeintel.storage.gateway import open_memory_gateway
from codeintel.storage.metadata_bootstrap import bootstrap_metadata_datasets
from codeintel.storage.repositories.datasets import DatasetReadRepository
from codeintel.storage.schemas import apply_all_schemas
from codeintel.storage.views import DERIVED_DOCS_VIEWS, create_all_views


def _fresh_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    apply_all_schemas(con)
    create_all_views(con)
    bootstrap_metadata_datasets(con)
    return con


def test_docs_views_registered_in_metadata() -> None:
    """Derived docs views should be registered as views in metadata.datasets."""
    con = _fresh_connection()
    rows = con.execute(
        "SELECT table_key, is_view FROM metadata.datasets WHERE table_key LIKE 'docs.%'"
    ).fetchall()
    table_keys = {row[0] for row in rows}
    missing = set(DERIVED_DOCS_VIEWS) - table_keys
    if missing:
        pytest.fail(f"Missing docs views in metadata.datasets: {sorted(missing)}")
    if not all(row[1] for row in rows):
        pytest.fail("Expected all docs entries in metadata.datasets to be marked as views")


def test_docs_view_readable_via_dataset_rows() -> None:
    """Docs views remain readable through metadata.dataset_rows slices."""
    gateway = open_memory_gateway(ensure_views=True, validate_schema=False)
    repo = DatasetReadRepository(gateway=gateway, repo="demo/repo", commit="deadbeef")
    rows = repo.read_dataset_rows("docs.v_function_summary", limit=5, offset=0)
    if not isinstance(rows, list):
        pytest.fail("Expected list from dataset_rows")


def test_docs_views_expose_capabilities() -> None:
    """Docs views and caches surface docs/read-only capability flags."""
    con = _fresh_connection()
    registry = load_dataset_registry(con)
    profile_view = registry.by_name["v_subsystem_profile"]
    profile_caps = profile_view.capabilities()
    if not profile_caps["docs_view"]:
        pytest.fail("Expected docs views to be marked with docs_view capability")
    if not profile_caps["read_only"]:
        pytest.fail("Expected docs views to be flagged read_only")
    cache_ds = registry.by_name["subsystem_profile_cache"]
    cache_caps = cache_ds.capabilities()
    if cache_caps["docs_view"]:
        pytest.fail("Cache tables should not be marked as docs views")
    if not cache_caps["can_validate"]:
        pytest.fail("Cache tables should be validation-capable via JSON Schema")
    if cache_caps["read_only"]:
        pytest.fail("Cache tables should allow writes for refreshes")
