"""Tests for export validation defaults derived from the dataset contract."""

from __future__ import annotations

import pytest

from codeintel.pipeline.export import DEFAULT_VALIDATION_SCHEMAS, default_validation_schemas
from codeintel.storage.datasets import JSON_SCHEMA_BY_DATASET_NAME


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def test_default_validation_schemas_match_dataset_contract() -> None:
    """Default validation schemas should mirror the dataset contract mapping."""
    expected = sorted(JSON_SCHEMA_BY_DATASET_NAME.keys())
    dynamic = sorted(default_validation_schemas())
    _require(
        condition=dynamic == expected,
        message=f"default_validation_schemas mismatch: {dynamic} != {expected}",
    )
    constant = sorted(DEFAULT_VALIDATION_SCHEMAS)
    _require(
        condition=constant == expected,
        message=f"DEFAULT_VALIDATION_SCHEMAS mismatch: {constant} != {expected}",
    )
