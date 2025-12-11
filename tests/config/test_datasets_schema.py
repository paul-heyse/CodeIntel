"""Tests for codeintel.config.datasets.schema module."""

from __future__ import annotations

from typing import TypedDict

import pandas as pd
import pytest
from pandera import Column, DataFrameSchema
from pandera.errors import SchemaErrors

from codeintel.config.datasets.schema import (
    DatasetMetadata,
    DatasetSchema,
)


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


# ------------------------------------------------------------------
# DatasetMetadata tests
# ------------------------------------------------------------------


def test_metadata_default_values() -> None:
    """Create metadata with default values."""
    metadata = DatasetMetadata()

    _require(condition=metadata.description is None, message="description should be None")
    _require(condition=metadata.owner is None, message="owner should be None")
    _require(condition=metadata.family is None, message="family should be None")
    _require(condition=metadata.freshness_sla is None, message="freshness_sla should be None")
    _require(condition=metadata.retention_policy is None, message="retention_policy should be None")
    _require(
        condition=metadata.upstream_dependencies == (),
        message="upstream_dependencies should be empty",
    )
    _require(
        condition=metadata.downstream_consumers == (),
        message="downstream_consumers should be empty",
    )
    _require(condition=metadata.tags == frozenset(), message="tags should be empty frozenset")
    _require(condition=metadata.deprecated is False, message="deprecated should be False")
    _require(
        condition=metadata.deprecation_message is None, message="deprecation_message should be None"
    )


def test_metadata_with_values() -> None:
    """Create metadata with specific values."""
    metadata = DatasetMetadata(
        description="Test dataset",
        owner="analytics",
        family="analytics",
        freshness_sla="daily",
        retention_policy="90d",
        upstream_dependencies=("core.goids",),
        tags=frozenset({"production", "metrics"}),
    )

    _require(condition=metadata.description == "Test dataset", message="description mismatch")
    _require(condition=metadata.owner == "analytics", message="owner mismatch")
    _require(condition=metadata.family == "analytics", message="family mismatch")
    _require(condition=metadata.freshness_sla == "daily", message="freshness_sla mismatch")
    _require(condition=metadata.retention_policy == "90d", message="retention_policy mismatch")
    _require(
        condition=metadata.upstream_dependencies == ("core.goids",),
        message="upstream_dependencies mismatch",
    )
    _require(
        condition=metadata.tags == frozenset({"production", "metrics"}), message="tags mismatch"
    )


def test_metadata_immutable() -> None:
    """Verify DatasetMetadata is immutable."""
    metadata = DatasetMetadata(description="Test")

    with pytest.raises(AttributeError):
        metadata.description = "Changed"  # type: ignore[misc]


# ------------------------------------------------------------------
# DatasetSchema tests
# ------------------------------------------------------------------


@pytest.fixture
def simple_pandera_schema() -> DataFrameSchema:
    """Create a simple Pandera schema for testing.

    Returns
    -------
    DataFrameSchema
        A simple schema with repo, commit, and loc columns.
    """
    return DataFrameSchema(
        {
            "repo": Column(str),
            "commit": Column(str),
            "loc": Column(int, nullable=True),
        },
        strict=True,
    )


def test_schema_create_minimal(simple_pandera_schema: DataFrameSchema) -> None:
    """Create a DatasetSchema with minimal parameters."""
    ds = DatasetSchema(
        name="test.example",
        pandera_schema=simple_pandera_schema,
    )

    _require(condition=ds.name == "test.example", message="name mismatch")
    _require(
        condition=ds.pandera_schema is simple_pandera_schema, message="pandera_schema mismatch"
    )
    _require(condition=ds.row_model is None, message="row_model should be None")
    _require(condition=ds.ddl_schema is None, message="ddl_schema should be None")
    _require(condition=ds.composition is None, message="composition should be None")


def test_schema_column_names(simple_pandera_schema: DataFrameSchema) -> None:
    """Get column names in definition order."""
    ds = DatasetSchema(
        name="test.example",
        pandera_schema=simple_pandera_schema,
    )

    columns = ds.column_names()

    _require(condition=columns == ("repo", "commit", "loc"), message=f"columns mismatch: {columns}")


def test_schema_table_key_property(simple_pandera_schema: DataFrameSchema) -> None:
    """Verify table_key property returns name."""
    ds = DatasetSchema(
        name="analytics.function_metrics",
        pandera_schema=simple_pandera_schema,
    )

    _require(condition=ds.table_key == "analytics.function_metrics", message="table_key mismatch")


def test_schema_has_composition_false(simple_pandera_schema: DataFrameSchema) -> None:
    """Check has_composition returns False when no composition."""
    ds = DatasetSchema(
        name="test.example",
        pandera_schema=simple_pandera_schema,
    )

    _require(condition=ds.has_composition() is False, message="has_composition should be False")


def test_schema_validate_valid_dataframe(simple_pandera_schema: DataFrameSchema) -> None:
    """Validate a valid DataFrame."""
    ds = DatasetSchema(
        name="test.example",
        pandera_schema=simple_pandera_schema,
    )

    df = pd.DataFrame(
        {
            "repo": ["test-repo"],
            "commit": ["abc123"],
            "loc": [100],
        }
    )

    result = ds.validate(df)

    _require(condition=len(result) == 1, message="result length mismatch")
    _require(condition=result["repo"].iloc[0] == "test-repo", message="repo value mismatch")


def test_schema_validate_invalid_dataframe(simple_pandera_schema: DataFrameSchema) -> None:
    """Validate an invalid DataFrame raises error."""
    ds = DatasetSchema(
        name="test.example",
        pandera_schema=simple_pandera_schema,
    )

    df = pd.DataFrame(
        {
            "repo": ["test-repo"],
            # Missing required column "commit"
        }
    )

    with pytest.raises(SchemaErrors):
        ds.validate(df)


def test_schema_json_schema_produces_valid_output(simple_pandera_schema: DataFrameSchema) -> None:
    """Generate JSON Schema from Pandera schema."""
    ds = DatasetSchema(
        name="test.example",
        pandera_schema=simple_pandera_schema,
    )

    json_schema = ds.json_schema()

    _require(
        condition=json_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema",
        message="$schema mismatch",
    )
    _require(condition=json_schema["type"] == "object", message="type mismatch")
    _require(condition="properties" in json_schema, message="properties missing")
    _require(condition="repo" in json_schema["properties"], message="repo missing in properties")
    _require(
        condition="commit" in json_schema["properties"], message="commit missing in properties"
    )
    _require(condition="loc" in json_schema["properties"], message="loc missing in properties")


def test_schema_get_row_model_generates_typeddict(simple_pandera_schema: DataFrameSchema) -> None:
    """Generate TypedDict row model from schema."""
    ds = DatasetSchema(
        name="test.example",
        pandera_schema=simple_pandera_schema,
    )

    row_model = ds.get_row_model()

    # Verify it's a TypedDict-like class
    _require(condition=hasattr(row_model, "__annotations__"), message="missing __annotations__")
    annotations = row_model.__annotations__
    _require(condition="repo" in annotations, message="repo missing in annotations")
    _require(condition="commit" in annotations, message="commit missing in annotations")
    _require(condition="loc" in annotations, message="loc missing in annotations")


def test_schema_get_row_model_uses_provided_model(simple_pandera_schema: DataFrameSchema) -> None:
    """Use provided row model instead of generating one."""

    class CustomRow(TypedDict):
        repo: str
        commit: str
        loc: int | None

    ds = DatasetSchema(
        name="test.example",
        pandera_schema=simple_pandera_schema,
        row_model=CustomRow,
    )

    result = ds.get_row_model()

    _require(condition=result is CustomRow, message="should return provided row_model")


def test_schema_immutable(simple_pandera_schema: DataFrameSchema) -> None:
    """Verify DatasetSchema is immutable."""
    ds = DatasetSchema(
        name="test.example",
        pandera_schema=simple_pandera_schema,
    )

    with pytest.raises(AttributeError):
        ds.name = "changed"  # type: ignore[misc]
