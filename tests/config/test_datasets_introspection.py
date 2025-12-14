"""Tests for the dataset introspection service.

Tests the DatasetIntrospection dataclass and introspect_dataset function.
"""

from __future__ import annotations

import pandera as pa
import pytest

from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
from codeintel.build.hamilton.contracts.schemas.constraints import (
    Constraint,
    ConstraintKind,
    ConstraintSet,
)
from codeintel.build.hamilton.contracts.schemas.introspection import (
    DatasetIntrospection,
    introspect_dataset,
)
from codeintel.build.hamilton.contracts.schemas.schema import DatasetMetadata, DatasetSchema


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


def _expect_equal(actual: object, expected: object, label: str) -> None:
    """Check equality with clear failure message."""
    if actual != expected:
        pytest.fail(f"{label}: expected {expected!r}, got {actual!r}")


def _expect_contains(text: str, substring: str, label: str) -> None:
    """Check that text contains substring."""
    if substring not in text:
        pytest.fail(f"{label}: expected {substring!r} to be in {text!r}")


def _create_sample_schema() -> DatasetSchema:
    """Create a sample DatasetSchema for testing.

    Returns
    -------
    DatasetSchema
        Sample schema with id and name columns.
    """
    return DatasetSchema(
        name="test.sample",
        pandera_schema=pa.DataFrameSchema(
            {
                "id": pa.Column(int),
                "name": pa.Column(str),
            }
        ),
        metadata=DatasetMetadata(
            description="Sample test dataset",
            owner="test-team",
        ),
    )


def _create_sample_constraints() -> ConstraintSet:
    """Create sample constraints for testing.

    Returns
    -------
    ConstraintSet
        Sample constraint set with TYPE constraints.
    """
    cs = ConstraintSet(table_key="test.sample")
    cs.add(
        Constraint(
            kind=ConstraintKind.TYPE,
            column="id",
            expression="id: int",
            source="pandera.column.dtype",
        )
    )
    cs.add(
        Constraint(
            kind=ConstraintKind.TYPE,
            column="name",
            expression="name: str",
            source="pandera.column.dtype",
        )
    )
    return cs


# ------------------------------------------------------------------
# DatasetIntrospection tests
# ------------------------------------------------------------------


def test_introspection_creation() -> None:
    """Create DatasetIntrospection with all fields."""
    sample_schema = _create_sample_schema()
    sample_constraints = _create_sample_constraints()

    intro = DatasetIntrospection(
        schema=sample_schema,
        constraints=sample_constraints,
        producers=["plugin.producer"],
        consumers=["plugin.consumer_a", "plugin.consumer_b"],
        upstream=["core.goids"],
        downstream=["analytics.profile"],
    )

    _expect_equal(intro.schema.name, "test.sample", "schema name")
    expected_constraints = 2
    _expect_equal(len(intro.constraints.constraints), expected_constraints, "constraints count")
    _expect_equal(intro.producers, ["plugin.producer"], "producers")
    expected_consumers = 2
    _expect_equal(len(intro.consumers), expected_consumers, "consumers count")
    _expect_equal(intro.upstream, ["core.goids"], "upstream")
    _expect_equal(intro.downstream, ["analytics.profile"], "downstream")


def test_summary_for_llm() -> None:
    """Generate LLM-readable summary."""
    sample_schema = _create_sample_schema()
    sample_constraints = _create_sample_constraints()

    intro = DatasetIntrospection(
        schema=sample_schema,
        constraints=sample_constraints,
        producers=["plugin.producer"],
        consumers=["plugin.consumer"],
        upstream=["core.goids"],
        downstream=[],
    )

    summary = intro.summary_for_llm()

    # Check key sections are present
    _expect_contains(summary, "# Dataset: test.sample", "summary header")
    _expect_contains(summary, "Sample test dataset", "description")
    _expect_contains(summary, "test-team", "owner")
    _expect_contains(summary, "## Columns", "columns section")
    _expect_contains(summary, "`id`", "id column")
    _expect_contains(summary, "`name`", "name column")
    _expect_contains(summary, "## Data Flow", "data flow section")
    _expect_contains(summary, "plugin.producer", "producer")
    _expect_contains(summary, "plugin.consumer", "consumer")


def test_summary_for_llm_empty_metadata() -> None:
    """Generate summary with minimal metadata."""
    schema = DatasetSchema(
        name="test.minimal",
        pandera_schema=pa.DataFrameSchema({"col": pa.Column(int)}),
    )
    cs = ConstraintSet(table_key="test.minimal")

    intro = DatasetIntrospection(
        schema=schema,
        constraints=cs,
        producers=[],
        consumers=[],
        upstream=[],
        downstream=[],
    )

    summary = intro.summary_for_llm()
    _expect_contains(summary, "No description", "default description")
    _expect_contains(summary, "Unassigned", "default owner")


def test_to_dict() -> None:
    """Convert introspection to dictionary."""
    sample_schema = _create_sample_schema()
    sample_constraints = _create_sample_constraints()

    intro = DatasetIntrospection(
        schema=sample_schema,
        constraints=sample_constraints,
        producers=["prod_a"],
        consumers=["cons_a", "cons_b"],
        upstream=["up_a"],
        downstream=["down_a"],
    )

    result = intro.to_dict()

    _expect_equal(result["name"], "test.sample", "name")
    _expect_equal(result["description"], "Sample test dataset", "description")
    _expect_equal(result["owner"], "test-team", "owner")
    _expect_equal(result["columns"], ["id", "name"], "columns")
    expected_col_count = 2
    _expect_equal(result["column_count"], expected_col_count, "column_count")
    expected_constraint_count = 2
    _expect_equal(result["constraint_count"], expected_constraint_count, "constraint_count")
    _expect_equal(result["producers"], ["prod_a"], "producers")
    _expect_equal(result["consumers"], ["cons_a", "cons_b"], "consumers")
    _expect_equal(result["upstream"], ["up_a"], "upstream")
    _expect_equal(result["downstream"], ["down_a"], "downstream")


def test_summary_with_table_level_constraints() -> None:
    """Include table-level constraints in summary."""
    schema = DatasetSchema(
        name="test.constrained",
        pandera_schema=pa.DataFrameSchema(
            {
                "start": pa.Column(int),
                "end": pa.Column(int),
            }
        ),
    )

    cs = ConstraintSet(table_key="test.constrained")
    cs.add(
        Constraint(
            kind=ConstraintKind.CROSS_COLUMN,
            column=None,
            expression="end >= start",
        )
    )

    intro = DatasetIntrospection(
        schema=schema,
        constraints=cs,
        producers=[],
        consumers=[],
        upstream=[],
        downstream=[],
    )

    summary = intro.summary_for_llm()
    _expect_contains(summary, "## Table-Level Constraints", "table constraints section")
    _expect_contains(summary, "end >= start", "constraint expression")


# ------------------------------------------------------------------
# introspect_dataset tests
# ------------------------------------------------------------------


def test_introspect_registered_dataset() -> None:
    """Introspect a registered dataset."""
    all_keys = SCHEMA_REGISTRY.all()
    if not all_keys:
        pytest.skip("No schemas registered")

    # Get first available dataset
    table_key = next(iter(all_keys))
    intro = introspect_dataset(table_key)

    _expect_equal(intro.schema.name, table_key, "schema name matches table_key")
    _require(
        condition=intro.constraints is not None,
        message="constraints should not be None",
    )
    _require(
        condition=isinstance(intro.producers, list),
        message="producers should be a list",
    )
    _require(
        condition=isinstance(intro.consumers, list),
        message="consumers should be a list",
    )


def test_introspect_unregistered_dataset() -> None:
    """Raise KeyError for unregistered dataset."""
    with pytest.raises(KeyError, match=r"nonexistent\.table"):
        introspect_dataset("nonexistent.table")


def test_introspection_extracts_constraints() -> None:
    """Verify constraints are extracted from schema."""
    all_keys = SCHEMA_REGISTRY.all()
    if not all_keys:
        pytest.skip("No schemas registered")

    table_key = next(iter(all_keys))
    intro = introspect_dataset(table_key)

    # Should have at least TYPE and NULLABILITY constraints for each column
    schema = SCHEMA_REGISTRY.require(table_key)
    num_columns = len(schema.column_names())

    # Expect at least 2 constraints per column (type + nullability)
    min_expected = num_columns * 2
    _require(
        condition=len(intro.constraints.constraints) >= min_expected,
        message=f"should have at least {min_expected} constraints",
    )
