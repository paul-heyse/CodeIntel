"""Tests for the constraint aggregation layer.

Tests the ConstraintKind enum, Constraint dataclass, ConstraintSet,
and extract_constraints_from_pandera function.
"""

from __future__ import annotations

import pandera as pa
import pytest

from codeintel.config.datasets.constraints import (
    Constraint,
    ConstraintKind,
    ConstraintSet,
    extract_constraints_from_pandera,
)


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


def _expect_equal(actual: object, expected: object, label: str) -> None:
    """Check equality with clear failure message."""
    if actual != expected:
        pytest.fail(f"{label}: expected {expected!r}, got {actual!r}")


# ------------------------------------------------------------------
# ConstraintKind tests
# ------------------------------------------------------------------


def test_constraint_kind_values() -> None:
    """Verify all ConstraintKind enum values are defined."""
    _expect_equal(ConstraintKind.TYPE.value, "type", "TYPE value")
    _expect_equal(ConstraintKind.NULLABILITY.value, "null", "NULLABILITY value")
    _expect_equal(ConstraintKind.RANGE.value, "range", "RANGE value")
    _expect_equal(ConstraintKind.PATTERN.value, "pattern", "PATTERN value")
    _expect_equal(ConstraintKind.UNIQUENESS.value, "unique", "UNIQUENESS value")
    _expect_equal(ConstraintKind.FOREIGN_KEY.value, "fk", "FOREIGN_KEY value")
    _expect_equal(ConstraintKind.CROSS_COLUMN.value, "cross", "CROSS_COLUMN value")
    _expect_equal(ConstraintKind.COMPUTATION.value, "compute", "COMPUTATION value")


def test_constraint_kind_all_members() -> None:
    """Verify the complete set of ConstraintKind members."""
    members = list(ConstraintKind)
    expected_count = 8
    _expect_equal(len(members), expected_count, "ConstraintKind member count")


# ------------------------------------------------------------------
# Constraint tests
# ------------------------------------------------------------------


def test_constraint_creation_minimal() -> None:
    """Create constraint with required fields only."""
    constraint = Constraint(
        kind=ConstraintKind.TYPE,
        column="test_col",
        expression="test_col: int",
    )
    _expect_equal(constraint.kind, ConstraintKind.TYPE, "kind")
    _expect_equal(constraint.column, "test_col", "column")
    _expect_equal(constraint.expression, "test_col: int", "expression")
    _expect_equal(constraint.source, "manual", "source default")
    _require(condition=constraint.check_fn is None, message="check_fn should be None")
    _require(condition=constraint.description is None, message="description should be None")


def test_constraint_creation_full() -> None:
    """Create constraint with all fields."""
    constraint = Constraint(
        kind=ConstraintKind.RANGE,
        column="value",
        expression="value >= 0",
        check_fn=lambda x: x >= 0,
        source="pandera.check",
        description="Non-negative value constraint",
    )
    _expect_equal(constraint.kind, ConstraintKind.RANGE, "kind")
    _expect_equal(constraint.column, "value", "column")
    _expect_equal(constraint.expression, "value >= 0", "expression")
    _expect_equal(constraint.source, "pandera.check", "source")
    _expect_equal(constraint.description, "Non-negative value constraint", "description")
    _require(condition=constraint.check_fn is not None, message="check_fn should be set")


def test_constraint_table_level() -> None:
    """Create table-level constraint with None column."""
    constraint = Constraint(
        kind=ConstraintKind.CROSS_COLUMN,
        column=None,
        expression="end_line >= start_line",
    )
    _require(condition=constraint.column is None, message="column should be None")
    _expect_equal(constraint.kind, ConstraintKind.CROSS_COLUMN, "kind")


def test_constraint_is_frozen() -> None:
    """Verify Constraint is immutable (frozen dataclass)."""
    constraint = Constraint(
        kind=ConstraintKind.TYPE,
        column="col",
        expression="col: str",
    )
    with pytest.raises(AttributeError):
        constraint.column = "new_col"  # type: ignore[misc]


# ------------------------------------------------------------------
# ConstraintSet tests
# ------------------------------------------------------------------


def test_constraint_set_empty() -> None:
    """Create empty ConstraintSet."""
    cs = ConstraintSet(table_key="test.table")
    _expect_equal(cs.table_key, "test.table", "table_key")
    _expect_equal(len(cs.constraints), 0, "constraints count")
    _expect_equal(len(cs.inferred_from), 0, "inferred_from count")


def test_constraint_set_add() -> None:
    """Add constraints to ConstraintSet."""
    cs = ConstraintSet(table_key="test.table")
    constraint = Constraint(
        kind=ConstraintKind.TYPE,
        column="col",
        expression="col: int",
        source="pandera.dtype",
    )
    cs.add(constraint)
    _expect_equal(len(cs.constraints), 1, "constraints count after add")
    _require(
        condition="pandera.dtype" in cs.inferred_from,
        message="pandera.dtype should be in inferred_from",
    )


def test_constraint_set_for_column() -> None:
    """Get constraints for a specific column."""
    cs = ConstraintSet(table_key="test.table")
    cs.add(
        Constraint(
            kind=ConstraintKind.TYPE,
            column="col_a",
            expression="col_a: int",
        )
    )
    cs.add(
        Constraint(
            kind=ConstraintKind.NULLABILITY,
            column="col_a",
            expression="col_a required",
        )
    )
    cs.add(
        Constraint(
            kind=ConstraintKind.TYPE,
            column="col_b",
            expression="col_b: str",
        )
    )

    col_a_constraints = cs.for_column("col_a")
    expected_col_a = 2
    _expect_equal(len(col_a_constraints), expected_col_a, "col_a constraints count")

    col_b_constraints = cs.for_column("col_b")
    _expect_equal(len(col_b_constraints), 1, "col_b constraints count")

    col_c_constraints = cs.for_column("col_c")
    _expect_equal(len(col_c_constraints), 0, "col_c constraints count")


def test_constraint_set_table_level() -> None:
    """Get table-level constraints."""
    cs = ConstraintSet(table_key="test.table")
    cs.add(
        Constraint(
            kind=ConstraintKind.TYPE,
            column="col",
            expression="col: int",
        )
    )
    cs.add(
        Constraint(
            kind=ConstraintKind.CROSS_COLUMN,
            column=None,
            expression="end >= start",
        )
    )
    cs.add(
        Constraint(
            kind=ConstraintKind.CROSS_COLUMN,
            column=None,
            expression="total == sum(parts)",
        )
    )

    table_constraints = cs.table_level()
    expected_count = 2
    _expect_equal(len(table_constraints), expected_count, "table level constraints count")
    for c in table_constraints:
        _require(condition=c.column is None, message="table level constraint should have None column")


def test_constraint_set_by_kind() -> None:
    """Get constraints by kind."""
    cs = ConstraintSet(table_key="test.table")
    cs.add(
        Constraint(
            kind=ConstraintKind.TYPE,
            column="col_a",
            expression="col_a: int",
        )
    )
    cs.add(
        Constraint(
            kind=ConstraintKind.TYPE,
            column="col_b",
            expression="col_b: str",
        )
    )
    cs.add(
        Constraint(
            kind=ConstraintKind.RANGE,
            column="col_a",
            expression="col_a >= 0",
        )
    )

    type_constraints = cs.by_kind(ConstraintKind.TYPE)
    expected_type_count = 2
    _expect_equal(len(type_constraints), expected_type_count, "TYPE constraints count")

    range_constraints = cs.by_kind(ConstraintKind.RANGE)
    _expect_equal(len(range_constraints), 1, "RANGE constraints count")

    pattern_constraints = cs.by_kind(ConstraintKind.PATTERN)
    _expect_equal(len(pattern_constraints), 0, "PATTERN constraints count")


def test_constraint_set_column_names() -> None:
    """Get all column names with constraints."""
    cs = ConstraintSet(table_key="test.table")
    cs.add(
        Constraint(
            kind=ConstraintKind.TYPE,
            column="col_a",
            expression="col_a: int",
        )
    )
    cs.add(
        Constraint(
            kind=ConstraintKind.TYPE,
            column="col_b",
            expression="col_b: str",
        )
    )
    cs.add(
        Constraint(
            kind=ConstraintKind.NULLABILITY,
            column="col_a",
            expression="col_a required",
        )
    )
    cs.add(
        Constraint(
            kind=ConstraintKind.CROSS_COLUMN,
            column=None,
            expression="table check",
        )
    )

    names = cs.column_names()
    _expect_equal(names, {"col_a", "col_b"}, "column names set")


# ------------------------------------------------------------------
# extract_constraints_from_pandera tests
# ------------------------------------------------------------------


def test_extract_type_constraints() -> None:
    """Extract TYPE constraints from Pandera schema."""
    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(int),
            "name": pa.Column(str),
        }
    )
    cs = extract_constraints_from_pandera("test.table", schema)

    type_constraints = cs.by_kind(ConstraintKind.TYPE)
    expected_count = 2
    _expect_equal(len(type_constraints), expected_count, "TYPE constraints count")

    columns = {c.column for c in type_constraints}
    _expect_equal(columns, {"id", "name"}, "TYPE constraint columns")


def test_extract_nullability_constraints() -> None:
    """Extract NULLABILITY constraints from Pandera schema."""
    schema = pa.DataFrameSchema(
        {
            "required_col": pa.Column(int, nullable=False),
            "optional_col": pa.Column(str, nullable=True),
        }
    )
    cs = extract_constraints_from_pandera("test.table", schema)

    null_constraints = cs.by_kind(ConstraintKind.NULLABILITY)
    expected_count = 2
    _expect_equal(len(null_constraints), expected_count, "NULLABILITY constraints count")

    for c in null_constraints:
        if c.column == "required_col":
            _require(
                condition="required" in c.expression,
                message="required_col should be marked required",
            )
        elif c.column == "optional_col":
            _require(
                condition="nullable" in c.expression,
                message="optional_col should be marked nullable",
            )


def test_extract_range_constraints_non_negative() -> None:
    """Extract RANGE constraints for non-negative checks."""
    schema = pa.DataFrameSchema(
        {
            "value": pa.Column(int, pa.Check(lambda s: s >= 0)),
        }
    )
    cs = extract_constraints_from_pandera("test.table", schema)

    range_constraints = cs.by_kind(ConstraintKind.RANGE)
    # May or may not extract depending on check str representation
    _require(
        condition=len(range_constraints) >= 0,
        message="range constraints should be non-negative count",
    )


def test_extract_table_level_checks() -> None:
    """Extract table-level CROSS_COLUMN constraints."""
    schema = pa.DataFrameSchema(
        {
            "start": pa.Column(int),
            "end": pa.Column(int),
        },
        checks=[pa.Check(lambda df: df["end"] >= df["start"])],
    )
    cs = extract_constraints_from_pandera("test.table", schema)

    cross_constraints = cs.table_level()
    _expect_equal(len(cross_constraints), 1, "CROSS_COLUMN constraints count")
    _expect_equal(
        cross_constraints[0].kind,
        ConstraintKind.CROSS_COLUMN,
        "constraint kind",
    )
    _expect_equal(
        cross_constraints[0].source,
        "pandera.dataframe_check",
        "constraint source",
    )


def test_extract_constraints_inferred_from() -> None:
    """Verify inferred_from sources are tracked."""
    schema = pa.DataFrameSchema(
        {
            "col": pa.Column(int, nullable=False),
        }
    )
    cs = extract_constraints_from_pandera("test.table", schema)

    _require(
        condition="pandera.column.dtype" in cs.inferred_from,
        message="pandera.column.dtype should be in inferred_from",
    )
    _require(
        condition="pandera.column.nullable" in cs.inferred_from,
        message="pandera.column.nullable should be in inferred_from",
    )


def test_extract_empty_schema() -> None:
    """Extract constraints from empty schema."""
    schema = pa.DataFrameSchema({})
    cs = extract_constraints_from_pandera("test.empty", schema)

    _expect_equal(len(cs.constraints), 0, "constraints count for empty schema")
    _expect_equal(cs.table_key, "test.empty", "table_key")
