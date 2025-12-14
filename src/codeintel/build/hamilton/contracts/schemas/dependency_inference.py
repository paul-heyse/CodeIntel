"""Dependency inference from plugin metadata.

This module provides automatic discovery of upstream and downstream
dependencies by analyzing plugin produces_tables and consumes_tables
declarations. It builds a dependency graph that can be used for
lineage tracing and constraint propagation.

Architecture Reference: Section 5.4.2 - Add dependency inference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from codeintel.build.hamilton.contracts.schemas.plugin_constraints import (
    get_consumer_plugins,
    get_producer_plugins,
)
from codeintel.build.hamilton.contracts.schemas.registry import SCHEMA_REGISTRY

__all__ = [
    "DependencyGraph",
    "DependencyNode",
    "build_dependency_graph",
    "infer_downstream_consumers",
    "infer_upstream_dependencies",
]

log = logging.getLogger(__name__)


@dataclass
class DependencyNode:
    """A node in the dataset dependency graph.

    Parameters
    ----------
    table_key
        Fully qualified table name.
    producer_plugins
        Plugins that produce this table.
    upstream
        Tables this one depends on.
    downstream
        Tables that depend on this one.

    Examples
    --------
    >>> node = DependencyNode(
    ...     table_key="analytics.function_metrics",
    ...     producer_plugins=["analytics.function_metrics"],
    ...     upstream=["core.goids"],
    ...     downstream=["analytics.hotspots"],
    ... )
    >>> node.has_upstream
    True
    """

    table_key: str
    producer_plugins: list[str] = field(default_factory=list)
    upstream: list[str] = field(default_factory=list)
    downstream: list[str] = field(default_factory=list)

    @property
    def has_upstream(self) -> bool:
        """Check if this table has upstream dependencies.

        Returns
        -------
        bool
            True if there are upstream tables.
        """
        return len(self.upstream) > 0

    @property
    def has_downstream(self) -> bool:
        """Check if this table has downstream consumers.

        Returns
        -------
        bool
            True if there are downstream tables.
        """
        return len(self.downstream) > 0

    @property
    def is_root(self) -> bool:
        """Check if this is a root table (no upstream dependencies).

        Returns
        -------
        bool
            True if this is a root table.
        """
        return not self.has_upstream

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf table (no downstream consumers).

        Returns
        -------
        bool
            True if this is a leaf table.
        """
        return not self.has_downstream


@dataclass
class DependencyGraph:
    """Complete dependency graph for all datasets.

    Parameters
    ----------
    nodes
        Mapping from table_key to DependencyNode.

    Examples
    --------
    >>> graph = build_dependency_graph()
    >>> isinstance(graph, DependencyGraph)
    True
    """

    nodes: dict[str, DependencyNode] = field(default_factory=dict)

    @property
    def table_count(self) -> int:
        """Return number of tables in the graph.

        Returns
        -------
        int
            Number of tables.
        """
        return len(self.nodes)

    def get(self, table_key: str) -> DependencyNode | None:
        """Get node for a table.

        Parameters
        ----------
        table_key
            Fully qualified table name.

        Returns
        -------
        DependencyNode | None
            Node if found.
        """
        return self.nodes.get(table_key)

    def root_tables(self) -> list[str]:
        """Get all root tables (no upstream dependencies).

        Returns
        -------
        list[str]
            Table keys for root tables.
        """
        return [key for key, node in self.nodes.items() if node.is_root]

    def leaf_tables(self) -> list[str]:
        """Get all leaf tables (no downstream consumers).

        Returns
        -------
        list[str]
            Table keys for leaf tables.
        """
        return [key for key, node in self.nodes.items() if node.is_leaf]

    def topological_order(self) -> list[str]:
        """Return tables in topological order (upstream before downstream).

        Returns
        -------
        list[str]
            Table keys in processing order.

        Notes
        -----
        Uses Kahn's algorithm for topological sort. Returns partial
        ordering if cycles exist (logs a warning).
        """
        in_degree: dict[str, int] = dict.fromkeys(self.nodes, 0)
        for node in self.nodes.values():
            for downstream in node.downstream:
                if downstream in in_degree:
                    in_degree[downstream] += 1

        queue = [key for key in self.nodes if in_degree[key] == 0]
        result: list[str] = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            node = self.nodes.get(current)
            if node:
                for downstream in node.downstream:
                    if downstream in in_degree:
                        in_degree[downstream] -= 1
                        if in_degree[downstream] == 0:
                            queue.append(downstream)

        if len(result) < len(self.nodes):
            log.warning(
                "Dependency graph contains cycles; partial ordering returned (%d/%d)",
                len(result),
                len(self.nodes),
            )

        return result


def infer_upstream_dependencies(table_key: str) -> list[str]:
    """Infer upstream dependencies from plugin metadata.

    This function finds all tables that the given table depends on
    by examining the consumes_tables of plugins that produce this table.

    Parameters
    ----------
    table_key
        Fully qualified table name.

    Returns
    -------
    list[str]
        Table keys of upstream dependencies.

    Notes
    -----
    NOTE(logic-framework): Full inference requires complete plugin catalog
    Functional Intent: Auto-discover upstream tables from plugin consumes_tables
    Architecture Reference: Section 5.4.2 - Add dependency inference
    Activation Steps:
      1. Ensure all producer plugins have accurate consumes_tables
      2. Add transitive dependency resolution
      3. Cache results for performance

    Examples
    --------
    >>> deps = infer_upstream_dependencies("analytics.function_metrics")
    >>> isinstance(deps, list)
    True
    """
    upstream: list[str] = []

    producers = get_producer_plugins(table_key)

    for meta in producers:
        if meta.consumes_tables:
            upstream.extend(meta.consumes_tables)

    return list(dict.fromkeys(upstream))


def infer_downstream_consumers(table_key: str) -> list[str]:
    """Infer downstream consumers from plugin metadata.

    This function finds all tables that depend on the given table
    by examining which plugins consume this table and what they produce.

    Parameters
    ----------
    table_key
        Fully qualified table name.

    Returns
    -------
    list[str]
        Table keys of downstream consumers.

    Notes
    -----
    NOTE(logic-framework): Full inference requires complete plugin catalog
    Functional Intent: Auto-discover downstream tables from plugin produces_tables
    Architecture Reference: Section 5.4.2 - Add dependency inference
    Activation Steps:
      1. Ensure all consumer plugins have accurate produces_tables
      2. Add transitive dependency resolution
      3. Cache results for performance

    Examples
    --------
    >>> consumers = infer_downstream_consumers("core.goids")
    >>> isinstance(consumers, list)
    True
    """
    downstream: list[str] = []

    consumers = get_consumer_plugins(table_key)

    for meta in consumers:
        if meta.produces_tables:
            downstream.extend(meta.produces_tables)

    return list(dict.fromkeys(downstream))


def build_dependency_graph() -> DependencyGraph:
    """Build complete dependency graph from schema registry and plugins.

    This function constructs a graph of all registered datasets with
    their upstream and downstream dependencies inferred from plugin
    metadata.

    Returns
    -------
    DependencyGraph
        Complete dependency graph.

    Notes
    -----
    NOTE(logic-framework): Full graph requires complete plugin catalog
    Functional Intent: Build queryable dependency graph for all datasets
    Architecture Reference: Section 5.4.2 - Add dependency inference
    Activation Steps:
      1. Complete plugin catalog population
      2. Add validation for missing dependencies
      3. Persist graph for incremental updates

    Examples
    --------
    >>> graph = build_dependency_graph()
    >>> graph.table_count > 0
    True
    """
    graph = DependencyGraph()

    for table_key in SCHEMA_REGISTRY:
        producer_metas = get_producer_plugins(table_key)
        producer_names = [m.name for m in producer_metas]

        upstream = infer_upstream_dependencies(table_key)
        downstream = infer_downstream_consumers(table_key)

        node = DependencyNode(
            table_key=table_key,
            producer_plugins=producer_names,
            upstream=upstream,
            downstream=downstream,
        )
        graph.nodes[table_key] = node

    return graph


def get_transitive_dependencies(table_key: str, *, include_self: bool = False) -> list[str]:
    """Get all transitive upstream dependencies.

    Parameters
    ----------
    table_key
        Starting table.
    include_self
        Whether to include the starting table in results.

    Returns
    -------
    list[str]
        All upstream dependencies (direct and transitive).

    Examples
    --------
    >>> deps = get_transitive_dependencies("analytics.hotspots")
    >>> isinstance(deps, list)
    True
    """
    visited: set[str] = set()
    result: list[str] = []

    def visit(key: str) -> None:
        if key in visited:
            return
        visited.add(key)

        for upstream in infer_upstream_dependencies(key):
            visit(upstream)

        if key != table_key or include_self:
            result.append(key)

    visit(table_key)
    return result


def get_transitive_consumers(table_key: str, *, include_self: bool = False) -> list[str]:
    """Get all transitive downstream consumers.

    Parameters
    ----------
    table_key
        Starting table.
    include_self
        Whether to include the starting table in results.

    Returns
    -------
    list[str]
        All downstream consumers (direct and transitive).

    Examples
    --------
    >>> consumers = get_transitive_consumers("core.goids")
    >>> isinstance(consumers, list)
    True
    """
    visited: set[str] = set()
    result: list[str] = []

    def visit(key: str) -> None:
        if key in visited:
            return
        visited.add(key)

        if key != table_key or include_self:
            result.append(key)

        for downstream in infer_downstream_consumers(key):
            visit(downstream)

    visit(table_key)
    return result
