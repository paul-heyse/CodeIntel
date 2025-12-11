"""Dataset introspection service for LLM and tooling consumption.

This module provides complete introspection of datasets by aggregating
metadata from the unified schema registry, constraint layer, and plugin
catalog. The introspection output is suitable for programmatic querying
by code intelligence tools and LLM agents.

Architecture Reference: Section 3.3 - Constraint Introspection Service
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.config.datasets.constraints import (
    Constraint,
    ConstraintKind,
    ConstraintSet,
    extract_constraints_from_pandera,
)
from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY

if TYPE_CHECKING:
    from codeintel.config.datasets.schema import DatasetSchema

__all__ = [
    "DatasetIntrospection",
    "get_introspection_summary",
    "introspect_all_datasets",
    "introspect_dataset",
    "query_column_constraints",
    "query_tables_by_constraint_kind",
]

log = logging.getLogger(__name__)


@dataclass
class DatasetIntrospection:
    """Complete introspection of a dataset for LLM/tooling consumption.

    This aggregates all metadata about a dataset into a single,
    queryable structure suitable for code intelligence. It provides
    a unified view of schema, constraints, and data flow information.

    Parameters
    ----------
    schema
        The unified DatasetSchema.
    constraints
        Aggregated constraints from all sources.
    producers
        Plugin names that produce this dataset.
    consumers
        Plugin names that consume this dataset.
    upstream
        Datasets this one depends on.
    downstream
        Datasets that depend on this one.

    Examples
    --------
    >>> intro = introspect_dataset("analytics.function_metrics")
    >>> print(intro.summary_for_llm())  # doctest: +SKIP
    # Dataset: analytics.function_metrics
    ...
    """

    schema: DatasetSchema
    constraints: ConstraintSet
    producers: list[str]
    consumers: list[str]
    upstream: list[str]
    downstream: list[str]

    def summary_for_llm(self) -> str:
        """Generate a human/LLM-readable summary.

        Returns
        -------
        str
            Markdown summary of the dataset suitable for LLM consumption.
        """
        lines = [
            f"# Dataset: {self.schema.name}",
            "",
            f"**Description:** {self.schema.metadata.description or 'No description'}",
            f"**Owner:** {self.schema.metadata.owner or 'Unassigned'}",
            "",
            "## Columns",
            "",
        ]

        lines.extend(
            f"- `{col_name}`: {', '.join(c.expression for c in col_constraints)}"
            if (col_constraints := self.constraints.for_column(col_name))
            else f"- `{col_name}`"
            for col_name in self.schema.column_names()
        )

        lines.extend(
            [
                "",
                "## Data Flow",
                "",
                f"**Produced by:** {', '.join(self.producers) or 'Unknown'}",
                f"**Consumed by:** {', '.join(self.consumers) or 'None'}",
                f"**Depends on:** {', '.join(self.upstream) or 'None'}",
            ]
        )

        if self.downstream:
            lines.append(f"**Downstream:** {', '.join(self.downstream)}")

        # Add table-level constraints if any
        table_constraints = self.constraints.table_level()
        if table_constraints:
            lines.extend(
                [
                    "",
                    "## Table-Level Constraints",
                    "",
                ]
            )
            lines.extend(f"- {constraint.expression}" for constraint in table_constraints)

        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation suitable for JSON export.
        """
        return {
            "name": self.schema.name,
            "description": self.schema.metadata.description,
            "owner": self.schema.metadata.owner,
            "columns": list(self.schema.column_names()),
            "column_count": len(self.schema.column_names()),
            "constraint_count": len(self.constraints.constraints),
            "constraint_sources": list(self.constraints.inferred_from),
            "producers": self.producers,
            "consumers": self.consumers,
            "upstream": self.upstream,
            "downstream": self.downstream,
        }


def introspect_dataset(table_key: str) -> DatasetIntrospection:
    """Build complete introspection for a dataset.

    This is the primary entry point for dataset introspection. It
    aggregates information from the schema registry, extracts
    constraints from the Pandera schema, and queries plugin metadata
    for producer/consumer relationships.

    Parameters
    ----------
    table_key
        Fully qualified table name (e.g., "analytics.function_metrics").

    Returns
    -------
    DatasetIntrospection
        Complete dataset introspection.

    Examples
    --------
    >>> intro = introspect_dataset("analytics.function_metrics")  # doctest: +SKIP
    >>> intro.schema.name
    'analytics.function_metrics'
    """
    schema = SCHEMA_REGISTRY.require(table_key)
    constraints = extract_constraints_from_pandera(table_key, schema.pandera_schema)

    # Get producers/consumers from registry
    producers = SCHEMA_REGISTRY.producers_of(table_key)
    consumers = SCHEMA_REGISTRY.consumers_of(table_key)

    # Get dependency info from metadata
    upstream = list(schema.metadata.upstream_dependencies)
    downstream = list(schema.metadata.downstream_consumers)

    return DatasetIntrospection(
        schema=schema,
        constraints=constraints,
        producers=producers,
        consumers=consumers,
        upstream=upstream,
        downstream=downstream,
    )


def introspect_all_datasets() -> dict[str, DatasetIntrospection]:
    """Build introspection for all registered datasets.

    Returns
    -------
    dict[str, DatasetIntrospection]
        Mapping from table_key to introspection for all datasets.

    Notes
    -----
    This can be expensive for large registries. Consider using
    introspect_dataset() for individual lookups.

    Caching for repeated introspection calls could be added in the future
    with LRU cache decorator and cache invalidation on registry changes.
    See architecture Section 3.3 - Constraint Introspection Service.
    """
    result: dict[str, DatasetIntrospection] = {}
    for table_key in SCHEMA_REGISTRY.all():
        try:
            result[table_key] = introspect_dataset(table_key)
        except KeyError:
            # Skip datasets that fail introspection due to missing schema
            log.debug("Skipping introspection for %s: schema not found", table_key)
            continue

    return result


def query_column_constraints(table_key: str, column: str) -> list[Constraint]:
    """Query constraints for a specific column.

    Parameters
    ----------
    table_key
        Fully qualified table name (e.g., "analytics.function_metrics").
    column
        Column name to query.

    Returns
    -------
    list[Constraint]
        Constraints applying to this column.

    Raises
    ------
    KeyError
        If the table is not registered.
    ValueError
        If the column does not exist.

    Notes
    -----
    NOTE(logic-framework): Full constraint aggregation pending
    Functional Intent: Query all constraints for a specific column
    Architecture Reference: Section 5.4.3 - Introspection API for querying
    Activation Steps:
      1. Add plugin constraint aggregation
      2. Add DDL constraint extraction
      3. Cache results for performance

    Examples
    --------
    >>> constraints = query_column_constraints("analytics.function_metrics", "loc")
    >>> isinstance(constraints, list)
    True
    """
    schema = SCHEMA_REGISTRY.get(table_key)
    if schema is None:
        msg = f"No schema registered for '{table_key}'"
        raise KeyError(msg)

    if column not in schema.column_names():
        msg = f"Column '{column}' not found in table '{table_key}'"
        raise ValueError(msg)

    constraints = extract_constraints_from_pandera(table_key, schema.pandera_schema)
    return constraints.for_column(column)


def query_tables_by_constraint_kind(kind: ConstraintKind) -> list[str]:
    """Find all tables that have constraints of a specific kind.

    Parameters
    ----------
    kind
        The constraint kind to filter by.

    Returns
    -------
    list[str]
        Table keys that have at least one constraint of this kind.

    Notes
    -----
    NOTE(logic-framework): Full constraint aggregation pending
    Functional Intent: Find tables with specific constraint types
    Architecture Reference: Section 5.4.3 - Introspection API for querying
    Activation Steps:
      1. Add plugin constraint aggregation
      2. Index constraints by kind for faster lookup
      3. Add caching

    Examples
    --------
    >>> tables = query_tables_by_constraint_kind(ConstraintKind.RANGE)
    >>> isinstance(tables, list)
    True
    """
    result: list[str] = []

    for table_key, schema in SCHEMA_REGISTRY.items():
        constraints = extract_constraints_from_pandera(table_key, schema.pandera_schema)
        if constraints.by_kind(kind):
            result.append(table_key)

    return result


def get_introspection_summary() -> dict[str, object]:
    """Get summary of all datasets for LLM consumption.

    Returns
    -------
    dict[str, object]
        Summary statistics and metadata suitable for LLM agents.

    Notes
    -----
    NOTE(logic-framework): Full summary requires complete introspection
    Functional Intent: Provide high-level dataset catalog overview
    Architecture Reference: Section 5.4.3 - Introspection API for agent consumption
    Activation Steps:
      1. Add plugin dependency summary
      2. Add constraint coverage metrics
      3. Cache results

    Examples
    --------
    >>> summary = get_introspection_summary()
    >>> "total_datasets" in summary
    True
    """
    total_datasets = len(SCHEMA_REGISTRY)
    total_columns = 0
    total_constraints = 0
    by_family: dict[str, int] = {}
    by_owner: dict[str, int] = {}

    for table_key, schema in SCHEMA_REGISTRY.items():
        total_columns += len(schema.column_names())

        constraints = extract_constraints_from_pandera(table_key, schema.pandera_schema)
        total_constraints += len(constraints.constraints)

        # Group by family
        family_key = schema.metadata.family if schema.metadata.family else "unknown"
        by_family[family_key] = by_family.get(family_key, 0) + 1

        # Group by owner
        owner_key = schema.metadata.owner if schema.metadata.owner else "unassigned"
        by_owner[owner_key] = by_owner.get(owner_key, 0) + 1

    return {
        "total_datasets": total_datasets,
        "total_columns": total_columns,
        "total_constraints": total_constraints,
        "avg_columns_per_dataset": total_columns / total_datasets if total_datasets else 0,
        "avg_constraints_per_dataset": total_constraints / total_datasets if total_datasets else 0,
        "by_family": by_family,
        "by_owner": by_owner,
        "constraint_kinds": [kind.value for kind in ConstraintKind],
    }
