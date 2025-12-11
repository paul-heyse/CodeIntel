"""Column lineage tracing for dataset schema introspection.

This module provides infrastructure for answering "what defines this column?"
by tracing constraints, producer plugins, and upstream dependencies for
individual columns within datasets.

Architecture Reference: Section 5.4.1 - Enable querying "what defines this column?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from codeintel.config.datasets.constraints import (
    Constraint,
    ConstraintKind,
    extract_constraints_from_pandera,
)
from codeintel.config.datasets.plugin_constraints import (
    get_producer_plugins,
)
from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY

__all__ = [
    "ColumnLineage",
    "TableLineage",
    "trace_column_lineage",
    "trace_table_lineage",
]

log = logging.getLogger(__name__)


@dataclass
class ColumnLineage:
    """Lineage information for a single column.

    This dataclass aggregates all information that defines a column:
    constraints from Pandera schemas, producer plugins, and upstream
    column dependencies.

    Parameters
    ----------
    column
        Column name.
    table_key
        Fully qualified table name containing this column.
    constraints
        Constraints that apply to this column.
    producer_plugins
        Plugins that produce the table containing this column.
    upstream_columns
        Columns from upstream tables that this column depends on.

    Examples
    --------
    >>> lineage = trace_column_lineage("analytics.function_metrics", "loc")
    >>> lineage.column
    'loc'
    >>> len(lineage.constraints) > 0  # doctest: +SKIP
    True
    """

    column: str
    table_key: str
    constraints: list[Constraint] = field(default_factory=list)
    producer_plugins: list[str] = field(default_factory=list)
    upstream_columns: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_type_constraint(self) -> bool:
        """Check if column has a type constraint.

        Returns
        -------
        bool
            True if a type constraint exists.
        """
        return any(c.kind == ConstraintKind.TYPE for c in self.constraints)

    @property
    def has_range_constraint(self) -> bool:
        """Check if column has a range constraint.

        Returns
        -------
        bool
            True if a range constraint exists.
        """
        return any(c.kind == ConstraintKind.RANGE for c in self.constraints)

    @property
    def is_nullable(self) -> bool | None:
        """Check if column is nullable.

        Returns
        -------
        bool | None
            True if nullable, False if required, None if unknown.
        """
        for c in self.constraints:
            if c.kind == ConstraintKind.NULLABILITY:
                return "nullable" in c.expression

        return None

    def summary(self) -> str:
        """Generate a human-readable summary.

        Returns
        -------
        str
            Summary of the column lineage.
        """
        lines = [
            f"Column: {self.table_key}.{self.column}",
            "",
            "Constraints:",
        ]

        if self.constraints:
            lines.extend(f"  - [{c.kind.value}] {c.expression}" for c in self.constraints)
        else:
            lines.append("  (none)")

        lines.extend(["", "Produced by:"])

        if self.producer_plugins:
            lines.extend(f"  - {p}" for p in self.producer_plugins)
        else:
            lines.append("  (unknown)")

        if self.upstream_columns:
            lines.extend(["", "Upstream columns:"])
            lines.extend(f"  - {table}.{col}" for table, col in self.upstream_columns)

        return "\n".join(lines)


@dataclass
class TableLineage:
    """Lineage information for all columns in a table.

    Parameters
    ----------
    table_key
        Fully qualified table name.
    columns
        Column lineage for each column in the table.
    producer_plugins
        Plugins that produce this table.
    upstream_tables
        Tables that this table depends on.

    Examples
    --------
    >>> lineage = trace_table_lineage("analytics.function_metrics")
    >>> lineage.table_key
    'analytics.function_metrics'
    """

    table_key: str
    columns: dict[str, ColumnLineage] = field(default_factory=dict)
    producer_plugins: list[str] = field(default_factory=list)
    upstream_tables: list[str] = field(default_factory=list)

    @property
    def column_count(self) -> int:
        """Return number of columns with lineage.

        Returns
        -------
        int
            Number of columns.
        """
        return len(self.columns)

    def get_column(self, column: str) -> ColumnLineage | None:
        """Get lineage for a specific column.

        Parameters
        ----------
        column
            Column name.

        Returns
        -------
        ColumnLineage | None
            Column lineage if found.
        """
        return self.columns.get(column)


def trace_column_lineage(table_key: str, column: str) -> ColumnLineage:
    """Trace lineage for a specific column.

    This function answers "what defines this column?" by collecting:
    - Constraints from the Pandera schema
    - Producer plugins
    - Upstream column dependencies (when available)

    Parameters
    ----------
    table_key
        Fully qualified table name (e.g., "analytics.function_metrics").
    column
        Column name to trace.

    Returns
    -------
    ColumnLineage
        Complete lineage information for the column.

    Raises
    ------
    KeyError
        If the table is not registered in the schema registry.
    ValueError
        If the column does not exist in the table.

    Notes
    -----
    NOTE(logic-framework): Full lineage tracing requires column-level DAG
    Functional Intent: Trace all constraints and dependencies for a column
    Architecture Reference: Section 5.4.1 - Enable querying "what defines this column?"
    Activation Steps:
      1. Add column-level tracking to plugin metadata
      2. Wire column dependency extraction from plugin execution
      3. Build column-to-column dependency graph

    Examples
    --------
    >>> lineage = trace_column_lineage("analytics.function_metrics", "loc")
    >>> lineage.column
    'loc'
    """
    schema = SCHEMA_REGISTRY.get(table_key)
    if schema is None:
        msg = f"No schema registered for '{table_key}'"
        raise KeyError(msg)

    columns = schema.column_names()
    if column not in columns:
        msg = f"Column '{column}' not found in table '{table_key}'"
        raise ValueError(msg)

    # Extract constraints for this column
    all_constraints = extract_constraints_from_pandera(table_key, schema.pandera_schema)
    column_constraints = all_constraints.for_column(column)

    # Get producer plugins
    producer_metas = get_producer_plugins(table_key)
    producer_names = [m.name for m in producer_metas]

    # Get upstream columns (requires column-level dependency tracking)
    # NOTE(logic-framework): This returns empty until column-level DAG is implemented
    upstream_columns: list[tuple[str, str]] = []

    return ColumnLineage(
        column=column,
        table_key=table_key,
        constraints=column_constraints,
        producer_plugins=producer_names,
        upstream_columns=upstream_columns,
    )


def trace_table_lineage(table_key: str) -> TableLineage:
    """Trace lineage for all columns in a table.

    Parameters
    ----------
    table_key
        Fully qualified table name.

    Returns
    -------
    TableLineage
        Complete lineage information for all columns.

    Raises
    ------
    KeyError
        If the table is not registered.

    Examples
    --------
    >>> lineage = trace_table_lineage("analytics.function_metrics")
    >>> lineage.column_count > 0  # doctest: +SKIP
    True
    """
    schema = SCHEMA_REGISTRY.get(table_key)
    if schema is None:
        msg = f"No schema registered for '{table_key}'"
        raise KeyError(msg)

    columns: dict[str, ColumnLineage] = {}
    for col_name in schema.column_names():
        columns[col_name] = trace_column_lineage(table_key, col_name)

    # Get producer plugins
    producer_metas = get_producer_plugins(table_key)
    producer_names = [m.name for m in producer_metas]

    # Get upstream tables from plugin consumes_tables
    upstream_tables: list[str] = []
    for meta in producer_metas:
        if meta.consumes_tables:
            upstream_tables.extend(meta.consumes_tables)

    # Deduplicate
    upstream_tables = list(dict.fromkeys(upstream_tables))

    return TableLineage(
        table_key=table_key,
        columns=columns,
        producer_plugins=producer_names,
        upstream_tables=upstream_tables,
    )


def get_all_columns_with_constraint(
    constraint_kind: str,
) -> list[tuple[str, str]]:
    """Find all columns that have a specific constraint kind.

    Parameters
    ----------
    constraint_kind
        Constraint kind value (e.g., "type", "range", "null").

    Returns
    -------
    list[tuple[str, str]]
        List of (table_key, column) tuples.

    Examples
    --------
    >>> cols = get_all_columns_with_constraint("range")
    >>> isinstance(cols, list)
    True
    """
    try:
        kind = ConstraintKind(constraint_kind)
    except ValueError:
        return []

    result: list[tuple[str, str]] = []

    for table_key in SCHEMA_REGISTRY:
        schema = SCHEMA_REGISTRY.get(table_key)
        if schema is None:
            continue

        constraints = extract_constraints_from_pandera(table_key, schema.pandera_schema)
        result.extend(
            (table_key, c.column) for c in constraints.by_kind(kind) if c.column is not None
        )

    return result
