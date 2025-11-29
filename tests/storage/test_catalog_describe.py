"""Catalog describe helpers should warn on missing schemas."""

from __future__ import annotations

import pytest

from codeintel.storage.catalog import describe_dataset_for_catalog
from codeintel.storage.datasets import Dataset


def test_describe_dataset_warns_on_missing_schema() -> None:
    """Fallback description should emit a warning when schema is absent."""
    dataset = Dataset(
        table_key="unknown.table",
        name="missing",
        schema=None,
        family="docs",
    )
    warnings: list[str] = []

    result = describe_dataset_for_catalog(dataset, warn=warnings.append)

    if result != "DuckDB table/view unknown.table":
        pytest.fail("Fallback description not returned for missing schema")
    if not warnings:
        pytest.fail("Missing schema should produce a warning")
    if "unknown.table" not in warnings[0]:
        pytest.fail("Warning should mention missing table key")
