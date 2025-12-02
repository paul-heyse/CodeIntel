"""Unified registry for serving datasets and operations.

This module provides backward-compatible access to operation metadata.
The canonical source of truth is now `codeintel.serving.operations.catalog`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from itertools import chain

from codeintel.config.datasets import (
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
)
from codeintel.config.datasets.dataflow import (
    DataflowEdge,
    DataflowNode,
    NodeKind,
    build_contract_dataflow_graph,
)
from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.models import DatasetSpecDescriptor
from codeintel.serving.operations import (
    Operation,
    iter_operations,
)
from codeintel.serving.services.query_service import QueryService


@dataclass(frozen=True)
class DatasetMeta:
    """Dataset metadata enriched with serving limits and flags."""

    id: str
    name: str
    table_key: str
    description: str
    schema_version: str | None
    family: str | None
    is_docs_view: bool
    is_read_only: bool
    default_limit: int
    max_limit: int
    owner: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    validation_profile: str | None = None


# OperationSpec is now an alias for the canonical Operation type.
# This preserves backward compatibility for code that imports OperationSpec.
OperationSpec = Operation


def _resolve_dataset_identifier(identifier: str) -> str | None:
    """Resolve a dataset identifier used in OperationSpec into a canonical table_key.

    Returns
    -------
    str | None
        Canonical table_key when found, otherwise None.
    """
    contract = DATASET_CONTRACTS.get(identifier)
    if contract is not None:
        return contract.table_key

    contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(identifier)
    if contract is not None:
        return contract.table_key

    return None


def _build_operation_specs() -> dict[str, OperationSpec]:
    """Build operation specs from the canonical catalog.

    Patches datasets.rows with exposed_datasets from DATASET_CONTRACTS_BY_TABLE_KEY.

    Returns
    -------
    dict[str, OperationSpec]
        Mapping from operation ID to OperationSpec.
    """
    specs: dict[str, OperationSpec] = {}
    exposed_datasets_keys = tuple(DATASET_CONTRACTS_BY_TABLE_KEY.keys())

    for operation in iter_operations():
        if operation.id == "datasets.rows":
            # Patch in exposed_datasets dynamically
            patched_op = dataclasses.replace(operation, exposed_datasets=exposed_datasets_keys)
            specs[patched_op.id] = patched_op
        else:
            specs[operation.id] = operation

    return specs


# Build the operation specs dict from the canonical catalog
_OPERATION_SPECS: dict[str, OperationSpec] = _build_operation_specs()


def build_dataset_meta(service: QueryService, limits: BackendLimits) -> list[DatasetMeta]:
    """
    Build dataset metadata entries using dataset_specs and serving limits.

    Parameters
    ----------
    service
        QueryService instance (local or HTTP).
    limits
        Backend limits derived from ServingConfig.

    Returns
    -------
    list[DatasetMeta]
        One entry per dataset in the registry.
    """
    specs: list[DatasetSpecDescriptor] = service.dataset_specs()
    metas: list[DatasetMeta] = []

    for spec in specs:
        family = getattr(spec, "family", None)
        is_docs_view = bool(family == "docs" or spec.table_key.startswith("docs."))
        capabilities = getattr(spec, "capabilities", {}) or {}
        is_read_only = bool(capabilities.get("read_only", False))
        description = spec.description or f"{spec.name} ({spec.table_key})"
        metas.append(
            DatasetMeta(
                id=spec.name,
                name=spec.name,
                table_key=spec.table_key,
                description=description,
                schema_version=spec.schema_version,
                family=family,
                is_docs_view=is_docs_view,
                is_read_only=is_read_only,
                default_limit=limits.default_limit,
                max_limit=limits.max_rows_per_call,
                owner=spec.owner,
                freshness_sla=spec.freshness_sla,
                retention_policy=spec.retention_policy,
                validation_profile=spec.validation_profile,
            )
        )

    return metas


def iter_operation_specs() -> list[OperationSpec]:
    """
    Return all registered OperationSpec instances.

    Returns
    -------
    list[OperationSpec]
        Operation specifications defined in the registry.
    """
    return list(_OPERATION_SPECS.values())


def get_operation_spec(op_id: str) -> OperationSpec | None:
    """
    Return a single OperationSpec by id, or None when unknown.

    Parameters
    ----------
    op_id
        Operation identifier to look up.

    Returns
    -------
    OperationSpec | None
        Matching specification when present.
    """
    return _OPERATION_SPECS.get(op_id)


def iter_operation_nodes() -> list[DataflowNode]:
    """Return DataflowNode entries for all serving operations.

    Returns
    -------
    list[DataflowNode]
        Operation nodes keyed by OperationSpec.id.
    """
    return [
        DataflowNode(
            id=spec.id,
            kind="operation",
            family="serving",
            owner_package=None,
            description=spec.summary,
        )
        for spec in _OPERATION_SPECS.values()
    ]


def iter_graph_nodes() -> list[DataflowNode]:
    """Return DataflowNode entries for logical graph runtimes.

    Returns
    -------
    list[DataflowNode]
        Graph nodes keyed as graph.<name> for required OperationSpec graphs.
    """
    names: set[str] = set()
    for spec in _OPERATION_SPECS.values():
        for graph_name in spec.required_graphs:
            names.add(graph_name)

    return [
        DataflowNode(
            id=f"graph.{graph_name}",
            kind="graph",
            family="graph",
            owner_package="graphs",
            description=f"Logical {graph_name} graph runtime",
        )
        for graph_name in sorted(names)
    ]


def iter_operation_dataset_edges() -> list[DataflowEdge]:
    """Build edges from datasets to operations based on required/exposed datasets.

    Returns
    -------
    list[DataflowEdge]
        Reads and exposes edges from datasets/views to operations.
    """
    edges: list[DataflowEdge] = []

    for spec in _OPERATION_SPECS.values():
        edges.extend(
            DataflowEdge(src=table_key, dst=spec.id, edge_type="reads")
            for table_key in (
                _resolve_dataset_identifier(ds_identifier)
                for ds_identifier in spec.required_datasets
            )
            if table_key is not None
        )
        edges.extend(
            DataflowEdge(src=table_key, dst=spec.id, edge_type="exposes")
            for table_key in (
                _resolve_dataset_identifier(ds_identifier)
                for ds_identifier in spec.exposed_datasets
            )
            if table_key is not None
        )

    return edges


def iter_operation_graph_edges() -> list[DataflowEdge]:
    """Build edges from logical graph runtimes to operations (depends_on).

    Returns
    -------
    list[DataflowEdge]
        Edges indicating graph dependencies for each operation.
    """
    return [
        DataflowEdge(
            src=f"graph.{graph_name}",
            dst=spec.id,
            edge_type="depends_on",
        )
        for spec in _OPERATION_SPECS.values()
        for graph_name in spec.required_graphs
    ]


def build_serving_dataflow_graph() -> tuple[list[DataflowNode], list[DataflowEdge]]:
    """Build a combined dataflow graph for datasets/docs/views, operations, and graphs.

    Returns
    -------
    tuple[list[DataflowNode], list[DataflowEdge]]
        Nodes and deduplicated edges across datasets, operations, and graph runtimes.
    """
    ds_nodes, ds_edges = build_contract_dataflow_graph()
    op_nodes = iter_operation_nodes()
    graph_nodes = iter_graph_nodes()

    op_ds_edges = iter_operation_dataset_edges()
    op_graph_edges = iter_operation_graph_edges()

    node_map: dict[tuple[str, NodeKind], DataflowNode] = {}
    for node in chain(ds_nodes, op_nodes, graph_nodes):
        node_map[node.id, node.kind] = node

    nodes = list(node_map.values())

    seen_edges: set[tuple[str, str, str]] = set()
    edges: list[DataflowEdge] = []
    for edge in chain(ds_edges, op_ds_edges, op_graph_edges):
        key = (edge.src, edge.dst, edge.edge_type)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append(edge)

    return nodes, edges


__all__ = [
    "DatasetMeta",
    "OperationSpec",
    "build_dataset_meta",
    "build_serving_dataflow_graph",
    "get_operation_spec",
    "iter_graph_nodes",
    "iter_operation_dataset_edges",
    "iter_operation_graph_edges",
    "iter_operation_nodes",
    "iter_operation_specs",
]
