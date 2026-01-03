"""Tests for export validation defaults derived from the dataset contract."""

from __future__ import annotations

import pytest

from codeintel.build.exports import default_validation_schemas
from codeintel.build.schemas import iter_contracts
from codeintel.build.schemas.json_schema_registry import get_json_schema


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def test_default_validation_schemas_match_dataset_contract() -> None:
    """Default validation schemas should mirror the dataset contract mapping."""
    expected = sorted(
        contract.table_key for contract in iter_contracts() if contract.schema is not None
    )
    dynamic = sorted(default_validation_schemas())
    _require(
        condition=dynamic == expected,
        message=f"default_validation_schemas mismatch: {dynamic} != {expected}",
    )


def test_generated_schemas_available_for_contracts() -> None:
    """Ensure generated JSON Schemas are available for all contracted non-view datasets.

    Views are excluded because they don't have TableSchema definitions - their
    schemas are derived from the underlying tables.
    """
    # Get all non-view datasets with json_schema_id
    non_view_with_schema = [
        c.table_key for c in iter_contracts() if c.json_schema_id is not None and not c.is_view
    ]
    missing = []
    for table_key in non_view_with_schema:
        try:
            get_json_schema(table_key)
        except KeyError:
            missing.append(table_key)
    _require(
        condition=not missing,
        message=f"Missing generated schemas for datasets: {missing}",
    )
