"""Constraint aggregation helpers for schema introspection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.core.schemas.primitives import TableSchema

__all__ = [
    "Constraint",
    "ConstraintKind",
    "ConstraintSet",
    "extract_constraints_from_table_schema",
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


def extract_constraints_from_table_schema(
    table_schema: TableSchema,
) -> ConstraintSet:
    """Extract a ConstraintSet from a TableSchema definition.

    Returns
    -------
    ConstraintSet
        Aggregated constraints inferred from the schema.
    """
    cs = ConstraintSet(table_key=table_schema.table_key)

    for column in table_schema.columns:
        cs.add(
            Constraint(
                kind=ConstraintKind.TYPE,
                column=column.name,
                expression=f"{column.name}: {column.type}",
                source="table_schema.column.type",
            )
        )
        nullable_str = "nullable" if column.nullable else "required"
        cs.add(
            Constraint(
                kind=ConstraintKind.NULLABILITY,
                column=column.name,
                expression=f"{column.name} {nullable_str}",
                source="table_schema.column.nullable",
            )
        )

    if table_schema.primary_key:
        pk_expr = ", ".join(table_schema.primary_key)
        cs.add(
            Constraint(
                kind=ConstraintKind.UNIQUENESS,
                column=None,
                expression=f"primary key({pk_expr})",
                source="table_schema.primary_key",
            )
        )
        for column_name in table_schema.primary_key:
            cs.add(
                Constraint(
                    kind=ConstraintKind.UNIQUENESS,
                    column=column_name,
                    expression=f"{column_name} unique",
                    source="table_schema.primary_key",
                )
            )

    for index in table_schema.indexes:
        if not index.unique:
            continue
        columns = ", ".join(index.columns)
        cs.add(
            Constraint(
                kind=ConstraintKind.UNIQUENESS,
                column=None,
                expression=f"unique index {index.name}({columns})",
                source="table_schema.index.unique",
            )
        )

    return cs
