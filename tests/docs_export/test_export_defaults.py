"""Tests for export validation defaults derived from the dataset contract."""

from __future__ import annotations

import pytest

from codeintel.build.exports import default_validation_schemas
from codeintel.build.schemas import get_contract_provider
from codeintel.build.schemas.json_schema_registry import get_json_schema_for_dataset_name


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def test_default_validation_schemas_match_dataset_contract() -> None:
    """Default validation schemas should mirror the dataset contract mapping."""
    json_schema_mapping = get_contract_provider().json_schema_by_dataset_name
    expected = sorted(json_schema_mapping.keys())
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
    from codeintel.build.schemas import iter_contracts  # noqa: PLC0415

    # Get all non-view datasets with json_schema_id
    non_view_with_schema = [
        c.name for c in iter_contracts() if c.json_schema_id is not None and not c.is_view
    ]
    missing = []
    for dataset_name in non_view_with_schema:
        schema = get_json_schema_for_dataset_name(dataset_name)
        if schema is None:
            missing.append(dataset_name)
    _require(
        condition=not missing,
        message=f"Missing generated schemas for datasets: {missing}",
    )
