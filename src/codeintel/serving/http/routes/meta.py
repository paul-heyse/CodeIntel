"""Meta HTTP routes exposing dataset and operation introspection."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from codeintel.serving.auto_pipeline import build_prereq_debug_info
from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.dependencies import (
    BackendDep,
    ConfigDep,
    ServiceDep,
)
from codeintel.serving.mcp.models import (
    DataflowEdgePayload,
    DataflowGraphResponse,
    DataflowNodePayload,
    DatasetMetaResponse,
    OperationMetaResponse,
    OperationPrereqDatasetStatus,
    OperationPrereqDebugResponse,
    OperationPrereqRunSummary,
)
from codeintel.serving.operations.catalog import get_operation
from codeintel.serving.registry import (
    build_dataset_meta,
    build_serving_dataflow_graph,
    iter_registry_operations,
)
from codeintel.storage.gateway import StorageGateway

LOG_ROUTE_PREFIX = "/meta"


def build_meta_router() -> APIRouter:
    """
    Construct the router exposing meta introspection endpoints.

    Returns
    -------
    APIRouter
        Router exposing dataset and operation metadata.
    """
    router = APIRouter()

    @router.get(
        f"{LOG_ROUTE_PREFIX}/datasets",
        response_model=list[DatasetMetaResponse],
        summary="List dataset metadata and serving limits.",
    )
    def list_dataset_meta(
        service: ServiceDep,
        cfg: ConfigDep,
    ) -> list[DatasetMetaResponse]:
        limits = BackendLimits.from_config(cfg)
        metas = build_dataset_meta(service, limits)
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
            )
            for meta in metas
        ]

    @router.get(
        f"{LOG_ROUTE_PREFIX}/operations",
        response_model=list[OperationMetaResponse],
        summary="List available operations and their characteristics.",
    )
    def list_operation_meta(cfg: ConfigDep) -> list[OperationMetaResponse]:
        limits = BackendLimits.from_config(cfg)
        results: list[OperationMetaResponse] = []
        for spec in iter_registry_operations():
            default_limit = spec.default_limit or limits.default_limit
            max_limit = spec.max_limit or limits.max_rows_per_call
            results.append(
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
        return results

    @router.get(
        f"{LOG_ROUTE_PREFIX}/dataflow",
        response_model=DataflowGraphResponse,
        summary="Return a dataflow graph for datasets, docs views, operations, and graphs.",
    )
    def get_dataflow_graph() -> DataflowGraphResponse:
        """
        Return the combined dataflow graph for this deployment.

        Returns
        -------
        DataflowGraphResponse
            Payload containing nodes and edges across datasets, operations, and graphs.
        """
        nodes, edges = build_serving_dataflow_graph()

        node_payloads = [
            DataflowNodePayload(
                id=node.id,
                kind=node.kind,
                family=node.family,
                owner_package=node.owner_package,
                description=node.description,
            )
            for node in nodes
        ]
        edge_payloads = [
            DataflowEdgePayload(
                src=edge.src,
                dst=edge.dst,
                edge_type=edge.edge_type,
            )
            for edge in edges
        ]

        return DataflowGraphResponse(nodes=node_payloads, edges=edge_payloads)

    @router.get(
        f"{LOG_ROUTE_PREFIX}/debug/pipeline/prereqs",
        response_model=OperationPrereqDebugResponse,
        summary="Debug endpoint showing prerequisite satisfaction for an operation.",
    )
    def debug_pipeline_prereqs(
        cfg: ConfigDep,
        backend: BackendDep,
        op_id: str = Query(description="Operation identifier (e.g., 'function.summary')"),
        repo: str | None = Query(default=None, description="Repository slug (defaults to config)"),
        commit: str | None = Query(default=None, description="Commit SHA (defaults to config)"),
    ) -> OperationPrereqDebugResponse:
        """
        Debug endpoint providing observability into prerequisite checking.

        This endpoint returns detailed information about why prerequisites
        are or are not satisfied for a given operation, including:
        - Required datasets and their transitive expansions
        - Status of each dataset check (has_rows, errors)
        - Recent pipeline runs considered
        - Whether data-aware and run-based checks passed

        Parameters
        ----------
        cfg
            Serving configuration (injected).
        backend
            Query backend (injected).
        op_id
            Operation identifier to check prerequisites for.
        repo
            Repository slug (defaults to config.repo).
        commit
            Commit SHA (defaults to config.commit).

        Returns
        -------
        OperationPrereqDebugResponse
            Complete debug information for prerequisite checking.

        Raises
        ------
        HTTPException
            404 if the operation is not found.
            503 if gateway is not available (requires local_db mode).
        """
        # Validate operation exists
        operation = get_operation(op_id)
        if operation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown operation: {op_id}",
            )

        # Use config defaults if not provided
        effective_repo = repo or cfg.repo
        effective_commit = commit or cfg.commit

        # Get gateway from backend
        gateway: StorageGateway | None = getattr(backend, "gateway", None)
        if gateway is None:
            raise HTTPException(
                status_code=503,
                detail="Gateway not available (requires local_db mode)",
            )

        # Build debug info
        debug_info = build_prereq_debug_info(
            gateway,
            op_id,
            repo=effective_repo,
            commit=effective_commit,
        )

        # Convert to response model
        return OperationPrereqDebugResponse(
            op_id=debug_info.op_id,
            repo=debug_info.repo,
            commit=debug_info.commit,
            required_datasets=list(debug_info.required_datasets),
            expanded_datasets=list(debug_info.expanded_datasets),
            dataset_statuses=[
                OperationPrereqDatasetStatus(
                    table_key=ds.table_key,
                    name=ds.name,
                    has_rows=ds.has_rows,
                    checked=ds.checked,
                    error=ds.error,
                )
                for ds in debug_info.dataset_statuses
            ],
            runs_considered=[
                OperationPrereqRunSummary(
                    run_id=run.run_id,
                    kind=run.kind,
                    status=run.status,
                    started_at=run.started_at,
                    completed_at=run.completed_at,
                )
                for run in debug_info.runs_considered
            ],
            data_satisfied=debug_info.data_satisfied,
            run_satisfied=debug_info.run_satisfied,
            overall_satisfied=debug_info.overall_satisfied,
        )

    return router


__all__ = ["build_meta_router"]
