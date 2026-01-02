"""Dataflow graph types and builders for dataset lineage visualization.

This module is storage-owned so config does not import storage (avoids cycles).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from codeintel.config.datasets.composites import get_composite_schemas
from codeintel.storage.contracts.provider import get_contract_for_table_key, iter_contracts

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from codeintel.core.schemas.contract_primitives import DatasetContract

NodeKind = Literal["table", "view", "operation", "graph"]
EdgeType = Literal["builds", "reads", "exposes", "depends_on"]


@dataclass(frozen=True)
class DataflowNode:
    """Node in the logical dataflow graph for CodeIntel datasets, views, and runtimes.

    Parameters
    ----------
    id
        Stable identifier for this node, e.g. "analytics.function_types".
    kind
        Node category: "table", "view", "operation", or "graph".
    family
        Optional dataset family, e.g. "core", "analytics", "docs".
    owner_package
        Optional owning package derived from the dataset contract.
    description
        Optional human-readable description.
    """

    id: str
    kind: NodeKind
    family: str | None = None
    owner_package: Literal["core", "analytics", "graphs", "qa", "docs"] | None = None
    description: str | None = None


@dataclass(frozen=True)
class DataflowEdge:
    """Directed edge in the dataflow graph.

    Parameters
    ----------
    src
        Source node id (upstream dataset, view, or runtime).
    dst
        Destination node id (downstream dataset, view, or operation).
    edge_type
        Relationship type ("builds", "reads", "exposes", "depends_on").
    """

    src: str
    dst: str
    edge_type: EdgeType


def iter_dataset_nodes() -> Iterator[DataflowNode]:
    """Iterate over dataset contracts and emit DataflowNodes.

    Yields
    ------
    DataflowNode
        Node keyed by DatasetContract.table_key for tables and views.
    """
    for contract in iter_contracts():
        kind: NodeKind = "view" if contract.is_view else "table"
        yield DataflowNode(
            id=contract.table_key,
            kind=kind,
            family=contract.family,
            owner_package=contract.owner_package,
            description=contract.description,
        )


def iter_composite_edges() -> Iterator[DataflowEdge]:
    """Yield builds edges for profile datasets defined in COMPOSITE_SCHEMAS.

    Yields
    ------
    DataflowEdge
        Edge from each composed_of source table to the profile table.
    """
    composite_schemas = get_composite_schemas()

    for table_key, composition in composite_schemas.items():
        try:
            target = get_contract_for_table_key(table_key)
        except KeyError:
            continue

        dst_id = target.table_key
        for src_table_key in composition.composed_of:
            try:
                upstream = get_contract_for_table_key(src_table_key)
            except KeyError:
                continue
            yield DataflowEdge(
                src=upstream.table_key,
                dst=dst_id,
                edge_type="builds",
            )


def iter_dependency_edges() -> Iterator[DataflowEdge]:
    """Yield builds edges derived from DatasetContract.upstream_dependencies.

    Yields
    ------
    DataflowEdge
        Edge from each declared upstream dependency to the dataset table_key.
    """
    contracts_by_name: dict[str, DatasetContract] = {}
    for contract in iter_contracts():
        contracts_by_name[contract.name] = contract

    for contract in iter_contracts():
        if not contract.upstream_dependencies:
            continue

        dst_id = contract.table_key
        for upstream_name in contract.upstream_dependencies:
            upstream = contracts_by_name.get(upstream_name)
            if upstream is None:
                continue

            yield DataflowEdge(
                src=upstream.table_key,
                dst=dst_id,
                edge_type="builds",
            )


@lru_cache(maxsize=1)
def _alias_docs_views() -> dict[str, str]:
    """Derive docs alias views for select analytics tables.

    Returns
    -------
    dict[str, str]
        Mapping from docs view key -> underlying analytics table key.
    """
    derived_views = {c.table_key for c in iter_contracts() if c.is_view}
    mapping: dict[str, str] = {}

    for contract in iter_contracts():
        if contract.is_view:
            continue
        if not contract.table_key.startswith("analytics."):
            continue
        name = contract.name
        if name.endswith("_cache"):
            continue
        if not (name.startswith("config_") or name.endswith("_profile")):
            continue
        view_key = f"docs.v_{name}"
        if view_key in derived_views:
            continue
        mapping[view_key] = contract.table_key

    return mapping


def alias_docs_views() -> dict[str, str]:
    """Return the derived docs alias view mapping.

    Returns
    -------
    dict[str, str]
        Mapping from docs view key -> underlying analytics table key.
    """
    return dict(_alias_docs_views())


def iter_docs_view_alias_edges() -> Iterator[DataflowEdge]:
    """Yield builds edges for docs views that are pure aliases.

    Yields
    ------
    DataflowEdge
        Edge from the analytics table to its docs alias view.
    """
    for view_key, table_key in _alias_docs_views().items():
        yield DataflowEdge(src=table_key, dst=view_key, edge_type="builds")


def iter_docs_view_alias_nodes() -> Iterator[DataflowNode]:
    """Yield DataflowNodes for docs views that are pure aliases.

    Yields
    ------
    DataflowNode
        Node keyed by the docs view alias.
    """
    for view_key in _alias_docs_views():
        yield DataflowNode(
            id=view_key,
            kind="view",
            family="docs",
            owner_package="docs",
            description=None,
        )


def build_contract_dataflow_graph() -> tuple[list[DataflowNode], list[DataflowEdge]]:
    """Build the dataset/docs layer of the dataflow graph from static contracts.

    Returns
    -------
    tuple[list[DataflowNode], list[DataflowEdge]]
        Nodes for all datasets/views plus deduplicated edges describing lineage.
    """
    nodes = [*iter_dataset_nodes(), *iter_docs_view_alias_nodes()]

    edges_iterables: list[Iterable[DataflowEdge]] = [
        iter_composite_edges(),
        iter_dependency_edges(),
        iter_docs_view_alias_edges(),
    ]

    seen: set[tuple[str, str, str]] = set()
    edges: list[DataflowEdge] = []
    for edges_iter in edges_iterables:
        for edge in edges_iter:
            key = (edge.src, edge.dst, edge.edge_type)
            if key in seen:
                continue
            seen.add(key)
            edges.append(edge)

    return nodes, edges


__all__ = [
    "DataflowEdge",
    "DataflowNode",
    "EdgeType",
    "NodeKind",
    "alias_docs_views",
    "build_contract_dataflow_graph",
    "iter_composite_edges",
    "iter_dataset_nodes",
    "iter_dependency_edges",
    "iter_docs_view_alias_edges",
    "iter_docs_view_alias_nodes",
]
