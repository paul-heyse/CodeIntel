"""Function query layer backed by DuckDB repositories.

This module provides the **Query Layer** implementation for function-related
operations.

Layer Hierarchy
---------------
::

    Transport Layer (MCP/HTTP backends: DuckDBBackend, HttpBackend)
         │
         ▼
    Service Layer (LocalQueryService, HttpQueryService)
         │
         ▼
    Query Layer (FunctionQueryLayer, ProfileQueryLayer, etc.) ← This module
         │
         ▼
    Repository Layer (FunctionRepository, GraphRepository)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from codeintel.serving import domain_models as dm
from codeintel.serving.backend.domain_builders import (
    build_callgraph_neighbors,
    build_function_architecture,
    build_function_profile,
    build_function_summary,
    build_graph_neighborhood,
    build_high_risk_functions,
    build_import_boundary,
    build_tests_for_function,
)
from codeintel.serving.backend.pagination import clamp_limit
from codeintel.serving.backend.query_api import FunctionQueriesApi
from codeintel.serving.mcp import errors

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.config.graph_helpers import GraphRunScope
    from codeintel.graphs.engine import GraphEngine
    from codeintel.serving.backend.core import (
        BackendContext,
        DuckDBConnection,
        DuckDBRepositories,
        GraphEngineProvider,
    )
    from codeintel.storage.repositories import FunctionRepository, GraphRepository

Message = dm.Message
ResponseMeta = dm.ResponseMeta


@dataclass
class FunctionQueryLayer(FunctionQueriesApi):
    """DuckDB-backed implementation of function query operations."""

    context: BackendContext
    repositories: DuckDBRepositories
    engine_provider: GraphEngineProvider

    @property
    def con(self) -> DuckDBConnection:
        """
        Return the active DuckDB connection.

        Returns
        -------
        DuckDBConnection
            Connection bound to the backend context.
        """
        return self.context.gateway.con

    @property
    def functions(self) -> FunctionRepository:
        """
        Return the lazily constructed function repository.

        Returns
        -------
        FunctionRepository
            Repository for function analytics tables.
        """
        return self.repositories.functions

    @property
    def graphs(self) -> GraphRepository:
        """
        Return the lazily constructed graph repository.

        Returns
        -------
        GraphRepository
            Repository for graph analytics tables.
        """
        return self.repositories.graphs

    def _require_graph_engine(self) -> GraphEngine:
        """
        Return the configured graph engine or raise when missing.

        Returns
        -------
        GraphEngine
            Graph engine configured for the backend.

        """
        return self.engine_provider.require()

    def _resolve_function_goid(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> int | None:
        """
        Resolve a function GOID from the provided identifiers.

        Returns
        -------
        int | None
            Resolved GOID when available.

        Raises
        ------
        errors.backend_failure
            If the repository raises a ValueError during resolution.
        """
        try:
            return self.functions.resolve_function_goid(
                urn=urn,
                goid_h128=goid_h128,
                rel_path=rel_path,
                qualname=qualname,
            )
        except ValueError as exc:
            message = str(exc)
            raise errors.backend_failure(message) from exc

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphRunScope | None = None,
    ) -> dm.FunctionSummaryResult:
        """
        Return a function summary for the provided identifiers.

        Returns
        -------
        dm.FunctionSummaryResult
            Summary payload for the requested function.

        Raises
        ------
        errors.invalid_argument
            If no identifying fields are provided.
        """
        _ = scope
        meta = ResponseMeta()
        if goid_h128 is None and not (urn or (rel_path and qualname)):
            message = "Must provide urn or goid_h128 or (rel_path + qualname)."
            raise errors.invalid_argument(message)
        resolved = self._resolve_function_goid(
            urn=urn,
            goid_h128=goid_h128,
            rel_path=rel_path,
            qualname=qualname,
        )
        if resolved is None:
            meta.messages.append(
                Message(
                    code="not_found",
                    severity="info",
                    detail="Function not found",
                    context={
                        "urn": urn,
                        "goid_h128": goid_h128,
                        "rel_path": rel_path,
                        "qualname": qualname,
                    },
                )
            )
            return build_function_summary(None, meta=meta)
        row = self.functions.get_function_summary_by_goid(resolved)
        if row is None:
            meta.messages.append(
                Message(
                    code="not_found",
                    severity="info",
                    detail="Function not found",
                    context={
                        "urn": urn,
                        "goid_h128": goid_h128,
                        "rel_path": rel_path,
                        "qualname": qualname,
                    },
                )
            )
            return build_function_summary(None, meta=meta)
        return build_function_summary(row, meta=meta)

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphRunScope | None = None,
    ) -> dm.HighRiskFunctionsResult:
        """
        List high-risk functions with optional limit clamping.

        Returns
        -------
        dm.HighRiskFunctionsResult
            High-risk function rows plus metadata.
        """
        _ = scope
        limit_clamp = clamp_limit(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        rows = self.functions.list_high_risk_functions(
            min_risk=min_risk,
            limit=limit_clamp.limit_or_default(self.context.limits.default_limit),
            tested_only=tested_only,
        )
        normalized_rows = [
            {"repo": self.context.repo, "commit": self.context.commit, **row} for row in rows
        ]
        meta = ResponseMeta(
            requested_limit=limit,
            applied_limit=limit_clamp.applied,
            messages=limit_clamp.messages,
        )
        return build_high_risk_functions(normalized_rows, meta=meta)

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphRunScope | None = None,
    ) -> dm.CallGraphNeighbors:
        """
        Return incoming and outgoing call graph neighbors.

        Returns
        -------
        dm.CallGraphNeighbors
            Neighbor rows plus pagination metadata.
        """
        _ = scope
        limit_clamp = clamp_limit(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        outgoing_rows: list[Mapping[str, object]] = []
        incoming_rows: list[Mapping[str, object]] = []
        default_limit = self.context.limits.default_limit
        if direction in {"outgoing", "both"}:
            outgoing = self.graphs.get_outgoing_callgraph_neighbors(
                goid_h128, limit=limit_clamp.limit_or_default(default_limit)
            )
            outgoing_rows = list(outgoing)
        if direction in {"incoming", "both"}:
            incoming = self.graphs.get_incoming_callgraph_neighbors(
                goid_h128, limit=limit_clamp.limit_or_default(default_limit)
            )
            incoming_rows = list(incoming)
        meta = ResponseMeta(
            requested_limit=limit,
            applied_limit=limit_clamp.applied,
            messages=limit_clamp.messages,
        )
        outgoing_dicts: Sequence[dict[str, object]] = [dict(row) for row in outgoing_rows]
        incoming_dicts: Sequence[dict[str, object]] = [dict(row) for row in incoming_rows]
        return build_callgraph_neighbors(outgoing_dicts, incoming_dicts, meta=meta)

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphRunScope | None = None,
    ) -> dm.TestsForFunctionResult:
        """
        Return tests linked to the given function.

        Returns
        -------
        dm.TestsForFunctionResult
            Tests plus pagination metadata.
        """
        _ = scope
        resolved = goid_h128
        if resolved is None:
            resolved = self._resolve_function_goid(urn=urn)
        if resolved is None:
            meta = ResponseMeta(
                messages=[
                    Message(
                        code="not_found",
                        severity="warning",
                        detail="Function not found",
                    )
                ]
            )
            return build_tests_for_function([], meta=meta)
        limit_clamp = clamp_limit(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        tests = self.repositories.tests.get_tests_for_function(
            resolved, limit=limit_clamp.limit_or_default(self.context.limits.default_limit)
        )
        messages = list(limit_clamp.messages)
        if not tests:
            messages.append(
                Message(
                    code="not_found",
                    severity="warning",
                    detail="Tests not found for function",
                )
            )
        meta = ResponseMeta(
            requested_limit=limit,
            applied_limit=limit_clamp.applied,
            messages=messages,
        )
        return build_tests_for_function(tests, meta=meta)

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> dm.GraphNeighborhood:
        """
        Compute a bounded ego neighborhood in the call graph.

        Parameters
        ----------
        goid_h128
            GOID of the function to center the neighborhood on.
        radius
            Hop radius for ego graph computation.
        max_nodes
            Optional node cap; when provided, truncates result set.

        Returns
        -------
        GraphNeighborhoodResponse
            Nodes and edges in the neighborhood with truncation metadata.
        """
        engine = self._require_graph_engine()
        graph = engine.call_graph()
        if goid_h128 not in graph:
            meta = ResponseMeta(
                applied_limit=max_nodes,
                truncated=max_nodes is not None and max_nodes == 0,
            )
            return build_graph_neighborhood([], [], meta=meta)
        subgraph = nx.ego_graph(graph, goid_h128, radius=radius, center=True)
        original_node_count = subgraph.number_of_nodes()
        truncated = False
        if max_nodes is not None and original_node_count > max_nodes:
            trimmed_nodes = list(subgraph.nodes)[:max_nodes]
            subgraph = subgraph.subgraph(trimmed_nodes).copy()
            truncated = True
        nodes: list[dict[str, object]] = []
        for node in subgraph.nodes:
            summary_row = self.functions.get_function_summary_by_goid(int(node))
            if summary_row is None:
                summary_row = {
                    "repo": self.context.repo,
                    "commit": self.context.commit,
                    "rel_path": "",
                    "function_goid_h128": int(node),
                }
            nodes.append(dict(summary_row))

        edges: list[dict[str, object]] = []
        for u, v, data in subgraph.edges(data=True):
            edge_data = data if isinstance(data, dict) else {}
            edges.append(
                {
                    "caller_goid_h128": int(u),
                    "caller_repo": self.context.repo,
                    "caller_commit": self.context.commit,
                    "callee_goid_h128": int(v),
                    "callee_repo": self.context.repo,
                    "callee_commit": self.context.commit,
                    "callsite_path": str(edge_data.get("path")) if edge_data else None,
                    "callsite_line": int(edge_data.get("line_number", 0)) if edge_data else None,
                    "language": str(edge_data.get("language", "python")) if edge_data else "python",
                    "kind": str(edge_data.get("edge_type", "direct")) if edge_data else "direct",
                    "confidence": float(edge_data.get("weight", 1.0)) if edge_data else None,
                }
            )
        meta = ResponseMeta(
            applied_limit=max_nodes,
            truncated=truncated or (max_nodes is not None and max_nodes == 0),
        )
        return build_graph_neighborhood(nodes, edges, meta=meta)

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> dm.ImportBoundary:
        """
        Return import edges crossing a subsystem boundary.

        Parameters
        ----------
        subsystem_id
            Subsystem identifier to find boundary edges for.
        max_edges
            Maximum edges to return; when provided, truncates result set.

        Returns
        -------
        ImportBoundaryResponse
            Boundary nodes and edges with truncation metadata.
        """
        limit_clamp = clamp_limit(
            max_edges,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        meta = ResponseMeta(
            messages=limit_clamp.messages,
            applied_limit=limit_clamp.applied,
            truncated=limit_clamp.applied == 0,
        )
        try:
            engine = self._require_graph_engine()
        except errors.McpError:
            return build_import_boundary([], [], meta=meta)
        import_graph = engine.import_graph()
        if subsystem_id not in import_graph:
            return build_import_boundary([], [], meta=meta)
        boundary_edges: list[dict[str, object]] = []
        boundary_nodes: set[str] = set()
        truncated = False
        for u, v, data in import_graph.out_edges(subsystem_id, data=True):
            if limit_clamp.applied is not None and len(boundary_edges) >= limit_clamp.applied:
                truncated = True
                break
            boundary_nodes.update({str(u), str(v)})
            boundary_edges.append(
                {
                    "source": str(u),
                    "target": str(v),
                    "weight": float(data.get("weight", 1.0)) if isinstance(data, dict) else 1.0,
                }
            )
        for u, v, data in import_graph.in_edges(subsystem_id, data=True):
            if limit_clamp.applied is not None and len(boundary_edges) >= limit_clamp.applied:
                truncated = True
                break
            boundary_nodes.update({str(u), str(v)})
            boundary_edges.append(
                {
                    "source": str(u),
                    "target": str(v),
                    "weight": float(data.get("weight", 1.0)) if isinstance(data, dict) else 1.0,
                }
            )
        nodes = sorted(boundary_nodes)
        edges = list(boundary_edges)
        meta = ResponseMeta(
            messages=limit_clamp.messages,
            applied_limit=limit_clamp.applied,
            truncated=truncated or limit_clamp.applied == 0,
        )
        return build_import_boundary(nodes, edges, meta=meta)

    def get_function_profile(self, goid_h128: int) -> dm.FunctionProfileResult:
        """
        Return a function profile by GOID.

        Returns
        -------
        dm.FunctionProfileResult
            Function profile payload.

        Raises
        ------
        errors.not_found
            If no profile exists for the GOID.
        """
        row = self.functions.get_function_profile(goid_h128)
        if row is None:
            message = f"Function profile not found: {goid_h128}"
            raise errors.not_found(message)
        return build_function_profile(row, meta=ResponseMeta())

    def get_function_architecture(self, goid_h128: int) -> dm.FunctionArchitectureResult:
        """
        Fetch function architecture metrics by GOID.

        Queries the ``docs.v_function_architecture`` view which aggregates
        graph metrics from ``analytics.graph_metrics_functions`` and related
        tables. This follows the same repository pattern as module architecture.

        Parameters
        ----------
        goid_h128
            Function global object ID (128-bit hash).

        Returns
        -------
        FunctionArchitectureResponse
            Architecture metrics including call_fan_in, call_fan_out,
            pagerank, betweenness, closeness, and other graph centrality
            measures.

        Raises
        ------
        errors.not_found
            When no architecture record exists for the given GOID.
        """
        row = self.functions.get_function_architecture(goid_h128)
        if row is None:
            message = f"Function architecture not found: {goid_h128}"
            raise errors.not_found(message)
        return build_function_architecture(row, meta=ResponseMeta())


__all__ = ["FunctionQueryLayer"]
