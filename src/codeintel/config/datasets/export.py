"""JSON export service for LLM and agent consumption.

This module provides JSON export functions for dataset schemas, constraints,
and dependency graphs. The exports are designed for consumption by LLMs
and code intelligence agents.

Architecture Reference: Section 5.4.3 - Expose to LLMs
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codeintel.config.datasets.constraints import (
    ConstraintKind,
    extract_constraints_from_pandera,
)
from codeintel.config.datasets.dependency_inference import build_dependency_graph
from codeintel.config.datasets.introspection import introspect_dataset
from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY

__all__ = [
    "export_all_constraints_json",
    "export_dataset_catalog_json",
    "export_dependency_graph_json",
    "export_to_file",
]

log = logging.getLogger(__name__)


def export_all_constraints_json() -> dict[str, Any]:
    """Export all constraints for all datasets as JSON.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with all constraints.

    Notes
    -----
    NOTE(logic-framework): Full export requires all schemas registered
    Functional Intent: Provide complete constraint catalog for LLM consumption
    Architecture Reference: Section 5.4.3 - JSON export of all constraint sets
    Activation Steps:
      1. Ensure all schemas are registered at startup
      2. Add plugin constraint aggregation
      3. Add caching for performance

    Examples
    --------
    >>> result = export_all_constraints_json()
    >>> "meta" in result
    True
    >>> "datasets" in result
    True
    """
    datasets: dict[str, Any] = {}

    for table_key, schema in SCHEMA_REGISTRY.items():
        constraints = extract_constraints_from_pandera(table_key, schema.pandera_schema)

        # Group constraints by column
        by_column: dict[str, list[dict[str, Any]]] = {}
        table_level: list[dict[str, Any]] = []

        for c in constraints.constraints:
            constraint_dict = {
                "kind": c.kind.value,
                "expression": c.expression,
                "source": c.source,
            }
            if c.description:
                constraint_dict["description"] = c.description

            if c.column is None:
                table_level.append(constraint_dict)
            else:
                if c.column not in by_column:
                    by_column[c.column] = []
                by_column[c.column].append(constraint_dict)

        datasets[table_key] = {
            "columns": by_column,
            "table_level": table_level,
            "inferred_from": list(constraints.inferred_from),
            "column_count": len(schema.column_names()),
            "constraint_count": len(constraints.constraints),
        }

    return {
        "meta": {
            "export_type": "constraints",
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset_count": len(datasets),
            "version": "1.0.0",
        },
        "datasets": datasets,
    }


def export_dataset_catalog_json() -> dict[str, Any]:
    """Export complete dataset catalog as JSON.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with dataset catalog.

    Notes
    -----
    NOTE(logic-framework): Full export requires complete introspection
    Functional Intent: Provide complete dataset catalog for LLM/agent use
    Architecture Reference: Section 5.4.3 - Introspection API for agent consumption
    Activation Steps:
      1. Ensure all schemas are registered
      2. Add plugin dependency information
      3. Add JSON Schema export for each dataset

    Examples
    --------
    >>> result = export_dataset_catalog_json()
    >>> "meta" in result
    True
    >>> "datasets" in result
    True
    """
    datasets: dict[str, Any] = {}

    for table_key in SCHEMA_REGISTRY:
        try:
            intro = introspect_dataset(table_key)
        except KeyError:
            log.debug("Skipping %s: introspection failed", table_key)
            continue

        datasets[table_key] = {
            "name": intro.schema.name,
            "description": intro.schema.metadata.description,
            "owner": intro.schema.metadata.owner,
            "family": intro.schema.metadata.family,
            "columns": list(intro.schema.column_names()),
            "column_count": len(intro.schema.column_names()),
            "producers": intro.producers,
            "consumers": intro.consumers,
            "upstream": intro.upstream,
            "downstream": intro.downstream,
            "constraint_count": len(intro.constraints.constraints),
            "tags": list(intro.schema.metadata.tags),
            "deprecated": intro.schema.metadata.deprecated,
        }

    return {
        "meta": {
            "export_type": "catalog",
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset_count": len(datasets),
            "version": "1.0.0",
        },
        "datasets": datasets,
    }


def export_dependency_graph_json() -> dict[str, Any]:
    """Export dependency graph as JSON.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with dependency graph.

    Notes
    -----
    NOTE(logic-framework): Full graph requires complete plugin catalog
    Functional Intent: Provide queryable dependency graph for agents
    Architecture Reference: Section 5.4.3 - JSON export of dependency graph
    Activation Steps:
      1. Complete plugin catalog population
      2. Add edge metadata (relationship type, cardinality)
      3. Add cycle detection and reporting

    Examples
    --------
    >>> result = export_dependency_graph_json()
    >>> "meta" in result
    True
    >>> "nodes" in result
    True
    """
    graph = build_dependency_graph()

    nodes: dict[str, Any] = {}
    edges: list[dict[str, str]] = []

    for table_key, node in graph.nodes.items():
        nodes[table_key] = {
            "producer_plugins": node.producer_plugins,
            "upstream": node.upstream,
            "downstream": node.downstream,
            "is_root": node.is_root,
            "is_leaf": node.is_leaf,
        }

        # Build edge list
        edges.extend(
            {"from": upstream, "to": table_key, "type": "depends_on"} for upstream in node.upstream
        )

    # Get topological order
    topo_order = graph.topological_order()

    return {
        "meta": {
            "export_type": "dependency_graph",
            "generated_at": datetime.now(UTC).isoformat(),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "version": "1.0.0",
        },
        "nodes": nodes,
        "edges": edges,
        "topological_order": topo_order,
        "root_tables": graph.root_tables(),
        "leaf_tables": graph.leaf_tables(),
    }


def export_to_file(
    export_type: str,
    output_path: Path | str,
    *,
    indent: int = 2,
) -> int:
    """Export data to a JSON file.

    Parameters
    ----------
    export_type
        One of "constraints", "catalog", or "graph".
    output_path
        Path to write the JSON file.
    indent
        JSON indentation level.

    Returns
    -------
    int
        Number of bytes written.

    Raises
    ------
    ValueError
        If export_type is not recognized.

    Examples
    --------
    >>> from pathlib import Path
    >>> import tempfile
    >>> with tempfile.NamedTemporaryFile(suffix=".json") as f:
    ...     bytes_written = export_to_file("catalog", f.name)
    ...     bytes_written > 0
    True
    """
    export_funcs = {
        "constraints": export_all_constraints_json,
        "catalog": export_dataset_catalog_json,
        "graph": export_dependency_graph_json,
    }

    if export_type not in export_funcs:
        msg = f"Unknown export type: {export_type}. Must be one of {list(export_funcs.keys())}"
        raise ValueError(msg)

    data = export_funcs[export_type]()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(data, indent=indent, ensure_ascii=False)
    output.write_text(content, encoding="utf-8")

    return len(content.encode("utf-8"))


def get_constraint_summary() -> dict[str, Any]:
    """Get summary statistics for all constraints.

    Returns
    -------
    dict[str, Any]
        Summary statistics.

    Examples
    --------
    >>> summary = get_constraint_summary()
    >>> "total_datasets" in summary
    True
    """
    by_kind: dict[str, int] = {kind.value: 0 for kind in ConstraintKind}
    total_constraints = 0
    datasets_with_constraints = 0

    for table_key in SCHEMA_REGISTRY:
        schema = SCHEMA_REGISTRY.get(table_key)
        if schema is None:
            continue

        constraints = extract_constraints_from_pandera(table_key, schema.pandera_schema)
        if constraints.constraints:
            datasets_with_constraints += 1

        for c in constraints.constraints:
            by_kind[c.kind.value] += 1
            total_constraints += 1

    return {
        "total_datasets": len(SCHEMA_REGISTRY),
        "datasets_with_constraints": datasets_with_constraints,
        "total_constraints": total_constraints,
        "by_kind": by_kind,
    }
