"""Tests for DuckDB-backed dataset metadata bootstrap."""

from __future__ import annotations

import pytest

from codeintel.storage.datasets import load_dataset_registry
from codeintel.storage.gateway import open_memory_gateway


def _require(condition: object, message: str) -> None:
    """Raise a pytest failure when the condition is false."""
    if not condition:
        pytest.fail(message)


def test_metadata_bootstrap_populates_catalog() -> None:
    """Bootstrap should create catalog rows and expose registry mappings."""
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=False)
    con = gateway.con

    count_row = con.execute("SELECT COUNT(*) FROM metadata.datasets").fetchone()
    if count_row is None:
        pytest.fail("metadata.datasets count missing")
        return
    _require(int(count_row[0]) > 0, "metadata.datasets is empty")

    registry = load_dataset_registry(con)
    dataset = registry.by_name.get("function_validation")
    if dataset is None:
        pytest.fail("function_validation dataset missing from registry")
        return
    _require(
        dataset.table_key == "analytics.function_validation",
        f"Unexpected table key: {dataset.table_key}",
    )
    _require(not dataset.is_view, "function_validation should not be a view")
    _require(dataset.schema is not None, "function_validation schema missing")
    _require(dataset.family == "analytics", f"Unexpected family: {dataset.family}")
    filename = registry.jsonl_datasets.get("analytics.function_validation")
    _require(filename == "function_validation.jsonl", f"Unexpected JSONL filename: {filename}")

    view_dataset = registry.by_name.get("v_function_summary")
    if view_dataset is None:
        pytest.fail("v_function_summary missing from registry")
        return
    _require(view_dataset.is_view, "v_function_summary should be a view")
    _require(view_dataset.schema is None, "v_function_summary should not include a TableSchema")
    _require(view_dataset.family == "docs", f"Unexpected docs family: {view_dataset.family}")

    gateway.close()
