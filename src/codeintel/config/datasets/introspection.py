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

from codeintel.config.datasets.constraints import extract_constraints_from_pandera
from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY

if TYPE_CHECKING:
    from codeintel.config.datasets.constraints import ConstraintSet
    from codeintel.config.datasets.schema import DatasetSchema

__all__ = [
    "DatasetIntrospection",
    "introspect_dataset",
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
