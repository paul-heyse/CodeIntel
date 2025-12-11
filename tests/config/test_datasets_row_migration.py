"""Tests for row model migration utilities.

Tests get_row_model, validate_row_model_compatibility, and related functions.
"""

from __future__ import annotations

from typing import TypedDict

import pandera as pa
import pytest

from codeintel.config.datasets.row_migration import (
    MigrationStatus,
    RowModelMigrationResult,
    get_row_model,
    validate_all_row_models,
    validate_row_model_compatibility,
)
from codeintel.config.datasets.schema import DatasetSchema
from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


def _expect_equal(actual: object, expected: object, label: str) -> None:
    """Check equality with clear failure message."""
    if actual != expected:
        pytest.fail(f"{label}: expected {expected!r}, got {actual!r}")


# ------------------------------------------------------------------
# MigrationStatus tests
# ------------------------------------------------------------------


def test_migration_status_compatible() -> None:
    """Create status for compatible models."""
    status = MigrationStatus(
        table_key="test.table",
        has_manual_model=True,
        has_schema_model=True,
        compatible=True,
        differences=[],
    )

    _expect_equal(status.table_key, "test.table", "table_key")
    _require(condition=status.has_manual_model, message="has_manual_model should be True")
    _require(condition=status.has_schema_model, message="has_schema_model should be True")
    _require(condition=status.compatible, message="compatible should be True")
    _expect_equal(len(status.differences), 0, "differences count")


def test_migration_status_incompatible() -> None:
    """Create status for incompatible models."""
    status = MigrationStatus(
        table_key="test.table",
        has_manual_model=True,
        has_schema_model=True,
        compatible=False,
        differences=["Field 'extra' in manual but not generated"],
    )

    _require(condition=not status.compatible, message="compatible should be False")
    _expect_equal(len(status.differences), 1, "differences count")


def test_migration_status_missing_schema() -> None:
    """Create status when schema is missing."""
    status = MigrationStatus(
        table_key="test.table",
        has_manual_model=True,
        has_schema_model=False,
        compatible=False,
        differences=["No schema registered for this dataset"],
    )

    _require(condition=status.has_manual_model, message="has_manual_model should be True")
    _require(condition=not status.has_schema_model, message="has_schema_model should be False")
    _require(condition=not status.compatible, message="compatible should be False")


# ------------------------------------------------------------------
# RowModelMigrationResult tests
# ------------------------------------------------------------------


def test_migration_result_ready() -> None:
    """Check is_ready_for_migration when all compatible."""
    result = RowModelMigrationResult(
        total_datasets=5,
        compatible_count=5,
        incompatible_count=0,
        missing_schema_count=0,
        statuses=[],
    )

    _require(
        condition=result.is_ready_for_migration(),
        message="should be ready for migration",
    )


def test_migration_result_not_ready_incompatible() -> None:
    """Check is_ready_for_migration when some incompatible."""
    result = RowModelMigrationResult(
        total_datasets=5,
        compatible_count=3,
        incompatible_count=2,
        missing_schema_count=0,
        statuses=[],
    )

    _require(
        condition=not result.is_ready_for_migration(),
        message="should not be ready due to incompatible",
    )


def test_migration_result_not_ready_missing() -> None:
    """Check is_ready_for_migration when some missing schema."""
    result = RowModelMigrationResult(
        total_datasets=5,
        compatible_count=3,
        incompatible_count=0,
        missing_schema_count=2,
        statuses=[],
    )

    _require(
        condition=not result.is_ready_for_migration(),
        message="should not be ready due to missing schemas",
    )


# ------------------------------------------------------------------
# get_row_model tests
# ------------------------------------------------------------------


def test_get_row_model_registered() -> None:
    """Get row model for a registered dataset."""
    all_keys = SCHEMA_REGISTRY.all()
    if not all_keys:
        pytest.skip("No schemas registered")

    table_key = next(iter(all_keys))
    model = get_row_model(table_key)

    # Should return a type (class)
    _require(condition=isinstance(model, type), message="model should be a type")

    # Should have __annotations__
    _require(
        condition=hasattr(model, "__annotations__"),
        message="model should have __annotations__",
    )


def test_get_row_model_unregistered() -> None:
    """Raise KeyError for unregistered dataset without manual model."""
    with pytest.raises(KeyError, match="No row model available"):
        get_row_model("completely.nonexistent.table")


# ------------------------------------------------------------------
# validate_row_model_compatibility tests
# ------------------------------------------------------------------


def test_validate_registered_dataset() -> None:
    """Validate compatibility for a registered dataset."""
    all_keys = SCHEMA_REGISTRY.all()
    if not all_keys:
        pytest.skip("No schemas registered")

    table_key = next(iter(all_keys))
    status = validate_row_model_compatibility(table_key)

    _expect_equal(status.table_key, table_key, "table_key")
    _require(condition=status.has_schema_model, message="has_schema_model should be True")


def test_validate_unregistered_dataset() -> None:
    """Validate returns not compatible for unregistered dataset."""
    status = validate_row_model_compatibility("nonexistent.table")

    _require(condition=not status.has_schema_model, message="has_schema_model should be False")
    _require(condition=not status.compatible, message="compatible should be False")
    _require(
        condition=any("No schema" in d for d in status.differences),
        message="should mention missing schema in differences",
    )


# ------------------------------------------------------------------
# validate_all_row_models tests
# ------------------------------------------------------------------


def test_validate_all_returns_result() -> None:
    """Validate all returns proper result structure."""
    result = validate_all_row_models()

    _require(
        condition=isinstance(result, RowModelMigrationResult),
        message="result should be RowModelMigrationResult",
    )
    _require(condition=result.total_datasets >= 0, message="total_datasets should be >= 0")
    _require(condition=result.compatible_count >= 0, message="compatible_count should be >= 0")
    _require(condition=result.incompatible_count >= 0, message="incompatible_count should be >= 0")
    _require(
        condition=result.missing_schema_count >= 0,
        message="missing_schema_count should be >= 0",
    )

    # Verify sum matches total
    total = result.compatible_count + result.incompatible_count + result.missing_schema_count
    _expect_equal(total, result.total_datasets, "counts sum")


def test_validate_all_statuses_match_counts() -> None:
    """Verify status counts match computed counts."""
    result = validate_all_row_models()

    _expect_equal(len(result.statuses), result.total_datasets, "statuses count")

    compatible = sum(1 for s in result.statuses if s.compatible and s.has_schema_model)
    incompatible = sum(1 for s in result.statuses if not s.compatible and s.has_schema_model)
    missing = sum(1 for s in result.statuses if not s.has_schema_model)

    _expect_equal(compatible, result.compatible_count, "compatible count")
    _expect_equal(incompatible, result.incompatible_count, "incompatible count")
    _expect_equal(missing, result.missing_schema_count, "missing count")


# ------------------------------------------------------------------
# DatasetSchema.get_row_model tests
# ------------------------------------------------------------------


def test_schema_get_row_model_generates_typed_dict() -> None:
    """DatasetSchema generates TypedDict from Pandera schema."""
    schema = DatasetSchema(
        name="test.generated",
        pandera_schema=pa.DataFrameSchema(
            {
                "id": pa.Column(int),
                "name": pa.Column(str),
            }
        ),
    )

    model = schema.get_row_model()

    _require(condition=isinstance(model, type), message="model should be a type")
    annotations = getattr(model, "__annotations__", {})
    _require(condition="id" in annotations, message="id should be in annotations")
    _require(condition="name" in annotations, message="name should be in annotations")


def test_schema_get_row_model_uses_precomputed() -> None:
    """DatasetSchema uses pre-computed row model if provided."""

    class PrecomputedRow(TypedDict):
        custom_field: str

    schema = DatasetSchema(
        name="test.precomputed",
        pandera_schema=pa.DataFrameSchema({"col": pa.Column(int)}),
        row_model=PrecomputedRow,
    )

    model = schema.get_row_model()

    _require(condition=model is PrecomputedRow, message="model should be PrecomputedRow")


def test_schema_get_row_model_cached() -> None:
    """DatasetSchema caches generated row models."""
    schema = DatasetSchema(
        name="test.cached",
        pandera_schema=pa.DataFrameSchema({"col": pa.Column(int)}),
    )

    model1 = schema.get_row_model()
    model2 = schema.get_row_model()

    # Should return the same object from cache
    _require(condition=model1 is model2, message="cached models should be identical")
