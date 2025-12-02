"""MCP tools for dataset and operation introspection."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from mcp.server.fastmcp import FastMCP

from codeintel.config.datasets.dataflow import DataflowEdge, DataflowNode
from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import (
    DataflowEdgePayload,
    DataflowGraphResponse,
    DataflowNodePayload,
    DatasetMetaResponse,
    OperationMetaResponse,
    ProblemDetail,
)
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.registry import (
    build_dataset_meta,
    build_serving_dataflow_graph,
    iter_operation_specs,
)
from codeintel.serving.services.query_service import QueryService


def _get_limits(backend: QueryBackendOrService) -> BackendLimits:
    limits = getattr(backend, "limits", None)
    if limits is not None:
        return limits
    query = getattr(backend, "query", None)
    query_limits = getattr(query, "limits", None)
    if query_limits is not None:
        return query_limits
    return BackendLimits()


def _get_service(backend: QueryBackendOrService) -> QueryService:
    service_obj = getattr(backend, "service", None)
    if service_obj is not None:
        return cast("QueryService", service_obj)
    return cast("QueryService", backend)


@dataclass(frozen=True)
class _MetaToolsContext:
    limits: BackendLimits
    service: QueryService
    nodes: list[DataflowNode]
    edges: list[DataflowEdge]
    incoming: dict[str, list[DataflowEdge]]
    outgoing: dict[str, list[DataflowEdge]]
    node_by_id: dict[str, DataflowNode]

    @classmethod
    def from_backend(cls, backend: QueryBackendOrService) -> _MetaToolsContext:
        limits = _get_limits(backend)
        service = _get_service(backend)
        nodes, edges = build_serving_dataflow_graph()
        node_by_id = {node.id: node for node in nodes}

        incoming: dict[str, list[DataflowEdge]] = {}
        outgoing: dict[str, list[DataflowEdge]] = {}
        for edge in edges:
            outgoing.setdefault(edge.src, []).append(edge)
            incoming.setdefault(edge.dst, []).append(edge)

        return cls(
            limits=limits,
            service=service,
            nodes=nodes,
            edges=edges,
            incoming=incoming,
            outgoing=outgoing,
            node_by_id=node_by_id,
        )


def _node_payload(node: DataflowNode) -> DataflowNodePayload:
    return DataflowNodePayload(
        id=node.id,
        kind=node.kind,
        family=node.family,
        owner_package=node.owner_package,
        description=node.description,
    )


def _edge_payload(edge: DataflowEdge) -> DataflowEdgePayload:
    return DataflowEdgePayload(src=edge.src, dst=edge.dst, edge_type=edge.edge_type)


def _shortest_path(
    outgoing: dict[str, list[DataflowEdge]], src_id: str, dst_id: str, max_hops: int
) -> dict[str, DataflowEdge | None] | None:
    """
    Compute a shortest path tree from src_id to dst_id.

    Returns
    -------
    dict[str, DataflowEdge | None] | None
        Parent mapping when a path is found; otherwise None.
    """
    queue: deque[tuple[str, int]] = deque([(src_id, 0)])
    parent: dict[str, DataflowEdge | None] = {src_id: None}
    while queue:
        current, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for edge in outgoing.get(current, []):
            if edge.dst in parent:
                continue
            parent[edge.dst] = edge
            if edge.dst == dst_id:
                return parent
            queue.append((edge.dst, depth + 1))
    return None


def _reconstruct_path(
    parent: dict[str, DataflowEdge | None],
    src_id: str,
    dst_id: str,
) -> list[DataflowEdge]:
    edges: list[DataflowEdge] = []
    node_id = dst_id
    while True:
        edge = parent.get(node_id)
        if edge is None:
            break
        edges.append(edge)
        node_id = edge.src
        if node_id == src_id:
            break
    edges.reverse()
    return edges


def _build_list_datasets_tool(context: _MetaToolsContext) -> Callable[[], object]:
    @_wrap
    def _tool() -> list[dict[str, object]] | dict[str, ProblemDetail]:
        """List dataset metadata and serving limits via MCP.

        Returns
        -------
        list[dict[str, object]] | dict[str, ProblemDetail]
            Serialized DatasetMetaResponse payloads or a ProblemDetail on error.
        """
        metas = build_dataset_meta(context.service, context.limits)
        return [
            DatasetMetaResponse(
                id=meta.id,
                name=meta.name,
                table_key=meta.table_key,
                description=meta.description,
                schema_version=meta.schema_version,
                family=meta.family,
                is_docs_view=meta.is_docs_view,
                is_read_only=meta.is_read_only,
                default_limit=meta.default_limit,
                max_limit=meta.max_limit,
            ).model_dump()
            for meta in metas
        ]

    return _tool


def _build_list_operations_tool(context: _MetaToolsContext) -> Callable[[], object]:
    @_wrap
    def _tool() -> list[dict[str, object]] | dict[str, ProblemDetail]:
        """List available operations and their characteristics via MCP.

        Returns
        -------
        list[dict[str, object]] | dict[str, ProblemDetail]
            Serialized OperationMetaResponse payloads or a ProblemDetail on error.
        """
        payloads: list[OperationMetaResponse] = []
        for spec in iter_operation_specs():
            default_limit = spec.default_limit or context.limits.default_limit
            max_limit = spec.max_limit or context.limits.max_rows_per_call
            payloads.append(
                OperationMetaResponse(
                    id=spec.id,
                    category=spec.category,
                    summary=spec.summary,
                    description=spec.description,
                    http_method=spec.http_method,
                    http_path=spec.http_path,
                    tool_name=spec.tool_name,
                    output_model=spec.output_model_name,
                    required_datasets=list(spec.required_datasets),
                    required_graphs=list(spec.required_graphs),
                    default_limit=default_limit,
                    max_limit=max_limit,
                )
            )
        return [payload.model_dump() for payload in payloads]

    return _tool


def _build_list_dataflow_graph_tool(context: _MetaToolsContext) -> Callable[[], object]:
    @_wrap
    def _tool() -> list[dict[str, object]]:
        """Return the combined dataflow graph for datasets, docs views, operations, and graphs.

        Returns
        -------
        list[dict[str, object]]
            One-item list containing the serialized DataflowGraphResponse payload.
        """
        response = DataflowGraphResponse(
            nodes=[_node_payload(node) for node in context.nodes],
            edges=[_edge_payload(edge) for edge in context.edges],
        )
        return [response.model_dump()]

    return _tool


def _build_explain_dataset_tool(context: _MetaToolsContext) -> Callable[[str], object]:
    @_wrap
    def _tool(node_id: str) -> list[dict[str, object]]:
        """Explain a dataset/docs view node in the dataflow graph.

        Parameters
        ----------
        node_id
            Dataset or docs view identifier (table_key).

        Returns
        -------
        list[dict[str, object]]
            Node payload with incoming/outgoing edges.

        Raises
        ------
        McpError
            When the node_id is unknown.
        """
        node = context.node_by_id.get(node_id)
        if node is None or node.kind not in {"table", "view"}:
            message = f"Unknown dataset/docs node_id: {node_id}"
            problem = errors.not_found(message)
            raise errors.McpError(problem.detail)

        incoming_edges = [
            _edge_payload(edge).model_dump() for edge in context.incoming.get(node.id, [])
        ]
        outgoing_edges = [
            _edge_payload(edge).model_dump() for edge in context.outgoing.get(node.id, [])
        ]

        return [
            {
                "node": _node_payload(node).model_dump(),
                "incoming_edges": incoming_edges,
                "outgoing_edges": outgoing_edges,
            }
        ]

    return _tool


def _build_explain_operation_tool(context: _MetaToolsContext) -> Callable[[str], object]:
    @_wrap
    def _tool(operation_id: str) -> list[dict[str, object]]:
        """Explain an OperationSpec node in the dataflow graph.

        Parameters
        ----------
        operation_id
            OperationSpec identifier.

        Returns
        -------
        list[dict[str, object]]
            Node payload with incoming/outgoing edges.

        Raises
        ------
        McpError
            When the operation id is unknown.
        """
        node = context.node_by_id.get(operation_id)
        if node is None or node.kind != "operation":
            message = f"Unknown operation id: {operation_id}"
            problem = errors.not_found(message)
            raise errors.McpError(problem.detail)

        incoming_edges = [
            _edge_payload(edge).model_dump() for edge in context.incoming.get(node.id, [])
        ]
        outgoing_edges = [
            _edge_payload(edge).model_dump() for edge in context.outgoing.get(node.id, [])
        ]

        return [
            {
                "node": _node_payload(node).model_dump(),
                "incoming_edges": incoming_edges,
                "outgoing_edges": outgoing_edges,
            }
        ]

    return _tool


def _build_explain_path_tool(context: _MetaToolsContext) -> Callable[[str, str, int], object]:
    @_wrap
    def _tool(src_id: str, dst_id: str, max_hops: int = 6) -> list[dict[str, object]]:
        """Return a shortest path between two dataflow nodes, if one exists.

        Returns
        -------
        list[dict[str, object]]
            Path payload with node and edge lists.

        Raises
        ------
        McpError
            When either node id is unknown.
        """
        if src_id not in context.node_by_id:
            message = f"Unknown src_id: {src_id}"
            problem = errors.not_found(message)
            raise errors.McpError(problem.detail)
        if dst_id not in context.node_by_id:
            message = f"Unknown dst_id: {dst_id}"
            problem = errors.not_found(message)
            raise errors.McpError(problem.detail)
        parent = _shortest_path(context.outgoing, src_id, dst_id, max_hops)
        if parent is None or dst_id not in parent:
            return [
                {
                    "path": [],
                    "message": f"No path from {src_id} to {dst_id} within {max_hops} hops.",
                }
            ]

        edges_in_path = _reconstruct_path(parent, src_id, dst_id)
        nodes_in_path = [_node_payload(context.node_by_id[src_id]).model_dump()]
        nodes_in_path.extend(
            _node_payload(context.node_by_id[edge.dst]).model_dump() for edge in edges_in_path
        )
        edge_payloads = [_edge_payload(edge).model_dump() for edge in edges_in_path]

        return [
            {
                "nodes": nodes_in_path,
                "edges": edge_payloads,
            }
        ]

    return _tool


def register_meta_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register meta MCP tools on the given FastMCP instance."""
    context = _MetaToolsContext.from_backend(backend)

    mcp.tool()(_build_list_datasets_tool(context))
    mcp.tool()(_build_list_operations_tool(context))
    mcp.tool()(_build_list_dataflow_graph_tool(context))
    mcp.tool()(_build_explain_dataset_tool(context))
    mcp.tool()(_build_explain_operation_tool(context))
    mcp.tool()(_build_explain_path_tool(context))


__all__ = ["register_meta_tools"]
