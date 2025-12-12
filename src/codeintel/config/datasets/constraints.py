"""Constraint aggregation layer for dataset schema introspection.

This module provides infrastructure for extracting, aggregating, and querying
constraints that define a dataset's structure. Constraints are collected from
multiple sources including Pandera schemas, DuckDB DDL, and plugin metadata.

The Constraint Aggregation Layer is a key enabler for the logic framework:
once all constraints are in one queryable structure, behavior can be inferred
from dependencies rather than declared explicitly.

Architecture Reference: Section 3 - Constraint Aggregation Layer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandera import DataFrameSchema

__all__ = [
    "Constraint",
    "ConstraintKind",
    "ConstraintSet",
    "extract_constraints_from_pandera",
]


class ConstraintKind(Enum):
    """Classification of constraint types.

    Each kind represents a different category of constraint that can be
    extracted from schema definitions, DDL, or runtime checks.
    """

    TYPE = "type"
    """Column type constraint (e.g., INTEGER, VARCHAR)."""

    NULLABILITY = "null"
    """Nullable/required constraint."""

    RANGE = "range"
    """Numeric range constraint (min/max bounds)."""

    PATTERN = "pattern"
    """String pattern/regex constraint."""

    UNIQUENESS = "unique"
    """Uniqueness constraint (primary key, unique index)."""

    FOREIGN_KEY = "fk"
    """References another dataset (foreign key relationship)."""

    CROSS_COLUMN = "cross"
    """Multi-column check (e.g., end_line >= start_line)."""

    COMPUTATION = "compute"
    """Derived from calculation dependency."""


@dataclass(frozen=True)
class Constraint:
    """A single constraint on a column or table.

    Constraints are immutable records that describe one aspect of a
    dataset's structure or validation rules.

    Parameters
    ----------
    kind
        Type of constraint.
    column
        Column name (None for table-level constraints).
    expression
        Human-readable constraint expression.
    check_fn
        Optional callable for runtime validation.
    source
        Where this constraint was inferred from.
    description
        Optional extended description of the constraint.

    Examples
    --------
    >>> c = Constraint(
    ...     kind=ConstraintKind.RANGE,
    ...     column="loc",
    ...     expression="loc >= 0",
    ...     source="pandera.check.non_negative",
    ... )
    >>> c.kind
    <ConstraintKind.RANGE: 'range'>
    """

    kind: ConstraintKind
    column: str | None
    expression: str
    check_fn: Callable[[Any], bool] | None = None
    source: str = "manual"
    description: str | None = None


@dataclass
class ConstraintSet:
    """Aggregated constraints for a dataset.

    This collects constraints from multiple sources to provide
    a complete picture of what defines a dataset's structure.
    The constraint set is the foundation for introspection and
    constraint-driven behavior derivation.

    Parameters
    ----------
    table_key
        Fully qualified table name (e.g., "analytics.function_metrics").
    constraints
        List of all constraints.
    inferred_from
        Sources from which constraints were inferred.

    Examples
    --------
    >>> cs = ConstraintSet(table_key="analytics.test")
    >>> cs.add(
    ...     Constraint(
    ...         kind=ConstraintKind.TYPE,
    ...         column="value",
    ...         expression="value: int",
    ...         source="pandera.column.dtype",
    ...     )
    ... )
    >>> len(cs.constraints)
    1
    """

    table_key: str
    constraints: list[Constraint] = field(default_factory=list)
    inferred_from: set[str] = field(default_factory=set)

    def add(self, constraint: Constraint) -> None:
        """Add a constraint to the set.

        Parameters
        ----------
        constraint
            Constraint to add.
        """
        self.constraints.append(constraint)
        if constraint.source:
            self.inferred_from.add(constraint.source)

    def for_column(self, column: str) -> list[Constraint]:
        """Get constraints for a specific column.

        Parameters
        ----------
        column
            Column name.

        Returns
        -------
        list[Constraint]
            Constraints applying to this column.
        """
        return [c for c in self.constraints if c.column == column]

    def table_level(self) -> list[Constraint]:
        """Get table-level constraints.

        Returns
        -------
        list[Constraint]
            Constraints that span multiple columns (column is None).
        """
        return [c for c in self.constraints if c.column is None]

    def by_kind(self, kind: ConstraintKind) -> list[Constraint]:
        """Get constraints of a specific kind.

        Parameters
        ----------
        kind
            The constraint kind to filter by.

        Returns
        -------
        list[Constraint]
            Constraints of the specified kind.
        """
        return [c for c in self.constraints if c.kind == kind]

    def column_names(self) -> set[str]:
        """Get all column names that have constraints.

        Returns
        -------
        set[str]
            Unique column names with at least one constraint.
        """
        return {c.column for c in self.constraints if c.column is not None}


def extract_constraints_from_pandera(
    table_key: str,
    schema: DataFrameSchema,
) -> ConstraintSet:
    """Extract ConstraintSet from a Pandera schema.

    This function walks through the Pandera schema and extracts:
    - TYPE constraints from column dtypes
    - NULLABILITY constraints from nullable settings
    - RANGE constraints from common check patterns
    - CROSS_COLUMN constraints from DataFrame-level checks

    Parameters
    ----------
    table_key
        Dataset identifier (e.g., "analytics.function_metrics").
    schema
        Pandera DataFrameSchema to extract constraints from.

    Returns
    -------
    ConstraintSet
        Extracted constraints.

    Notes
    -----
    This is the primary mechanism for converting Pandera validation rules
    into the constraint aggregation layer. Additional constraint sources
    (DDL, plugin metadata) can be added to the returned ConstraintSet.

    Architecture Reference: Section 3.2 - ConstraintSet Model
    """
    cs = ConstraintSet(table_key=table_key)

    for col_name, column in schema.columns.items():
        cs.add(
            Constraint(
                kind=ConstraintKind.TYPE,
                column=col_name,
                expression=f"{col_name}: {column.dtype}",
                source="pandera.column.dtype",
            )
        )

        nullable_str = "nullable" if column.nullable else "required"
        cs.add(
            Constraint(
                kind=ConstraintKind.NULLABILITY,
                column=col_name,
                expression=f"{col_name} {nullable_str}",
                source="pandera.column.nullable",
            )
        )

        if column.checks:
            for check in column.checks:
                _extract_check_constraint(cs, col_name, check)

    if schema.checks:
        for check in schema.checks:
            cs.add(
                Constraint(
                    kind=ConstraintKind.CROSS_COLUMN,
                    column=None,
                    expression=str(check),
                    source="pandera.dataframe_check",
                )
            )

    return cs


def _extract_check_constraint(
    cs: ConstraintSet,
    col_name: str,
    check: object,
) -> None:
    """Extract constraint from a Pandera column check.

    Parameters
    ----------
    cs
        ConstraintSet to add constraints to.
    col_name
        Column name the check applies to.
    check
        Pandera Check object.
    """
    check_str = str(check)

    if ">= 0" in check_str or "(s >= 0)" in check_str:
        cs.add(
            Constraint(
                kind=ConstraintKind.RANGE,
                column=col_name,
                expression=f"{col_name} >= 0",
                source="pandera.check.non_negative",
            )
        )
        return

    if ">= 1" in check_str or "(s >= 1)" in check_str:
        cs.add(
            Constraint(
                kind=ConstraintKind.RANGE,
                column=col_name,
                expression=f"{col_name} >= 1",
                source="pandera.check.positive",
            )
        )
        return

    if "<= 1" in check_str and ">= 0" in check_str:
        cs.add(
            Constraint(
                kind=ConstraintKind.RANGE,
                column=col_name,
                expression=f"0 <= {col_name} <= 1",
                source="pandera.check.ratio",
            )
        )
        return
