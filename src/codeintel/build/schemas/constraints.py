"""Constraint aggregation helpers for schema introspection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandera.pandas import DataFrameSchema

__all__ = [
    "Constraint",
    "ConstraintKind",
    "ConstraintSet",
    "extract_constraints_from_pandera",
]


class ConstraintKind(Enum):
    """Classification of constraint types."""

    TYPE = "type"
    NULLABILITY = "null"
    RANGE = "range"
    PATTERN = "pattern"
    UNIQUENESS = "unique"
    FOREIGN_KEY = "fk"
    CROSS_COLUMN = "cross"
    COMPUTATION = "compute"


@dataclass(frozen=True)
class Constraint:
    """A single constraint on a column or table."""

    kind: ConstraintKind
    column: str | None
    expression: str
    check_fn: Callable[[Any], bool] | None = None
    source: str = "manual"
    description: str | None = None


@dataclass
class ConstraintSet:
    """Aggregated constraints for a dataset."""

    table_key: str
    constraints: list[Constraint] = field(default_factory=list)
    inferred_from: set[str] = field(default_factory=set)

    def add(self, constraint: Constraint) -> None:
        """Add a constraint to the set."""
        self.constraints.append(constraint)
        if constraint.source:
            self.inferred_from.add(constraint.source)

    def for_column(self, column: str) -> list[Constraint]:
        """Get constraints for a specific column.

        Returns
        -------
        list[Constraint]
            Constraints applied to the requested column.
        """
        return [c for c in self.constraints if c.column == column]

    def table_level(self) -> list[Constraint]:
        """Get table-level constraints.

        Returns
        -------
        list[Constraint]
            Constraints that apply to the table overall.
        """
        return [c for c in self.constraints if c.column is None]

    def by_kind(self, kind: ConstraintKind) -> list[Constraint]:
        """Get constraints of a specific kind.

        Returns
        -------
        list[Constraint]
            Constraints matching the requested kind.
        """
        return [c for c in self.constraints if c.kind == kind]

    def column_names(self) -> set[str]:
        """Get all column names that have constraints.

        Returns
        -------
        set[str]
            Column names with at least one constraint.
        """
        return {c.column for c in self.constraints if c.column is not None}


def extract_constraints_from_pandera(
    table_key: str,
    schema: DataFrameSchema,
) -> ConstraintSet:
    """Extract ConstraintSet from a Pandera schema.

    Returns
    -------
    ConstraintSet
        Aggregated constraints inferred from the schema.
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

        if not column.checks:
            continue

        for check in column.checks:
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
            elif ">= 1" in check_str or "(s >= 1)" in check_str:
                cs.add(
                    Constraint(
                        kind=ConstraintKind.RANGE,
                        column=col_name,
                        expression=f"{col_name} >= 1",
                        source="pandera.check.min_value",
                    )
                )
            elif "<= 1" in check_str and ">= 0" in check_str:
                cs.add(
                    Constraint(
                        kind=ConstraintKind.RANGE,
                        column=col_name,
                        expression=f"0 <= {col_name} <= 1",
                        source="pandera.check.unit_interval",
                    )
                )

    if schema.checks:
        for check in schema.checks:
            cs.add(
                Constraint(
                    kind=ConstraintKind.CROSS_COLUMN,
                    column=None,
                    expression=str(check),
                    source="pandera.check.dataframe",
                )
            )

    return cs
