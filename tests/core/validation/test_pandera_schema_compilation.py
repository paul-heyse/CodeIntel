"""Pandera schema compilation coverage tests."""

from __future__ import annotations

import pytest

from codeintel.core.schemas.output_registry import OUTPUT_TABLE_SCHEMAS
from codeintel.core.validation.pandera_schema import pandera_available, pandera_schema_for_table


def test_pandera_schema_compiles_for_output_tables() -> None:
    """Pandera schemas should compile for registered output tables."""
    if not pandera_available():
        pytest.skip("Pandera + Polars required for schema compilation.")
    for table_key, table_schema in OUTPUT_TABLE_SCHEMAS.items():
        schema = pandera_schema_for_table(
            table_schema,
            observation=None,
            validation_profile="data-light",
        )
        if schema is None:
            pytest.fail(f"Pandera schema missing for {table_key}")
