"""Meta HTTP routes exposing dataset and operation introspection."""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.dependencies import ConfigDep, ServiceDep
from codeintel.serving.mcp.models import DatasetMetaResponse, OperationMetaResponse
from codeintel.serving.registry import build_dataset_meta, iter_operation_specs

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
        for spec in iter_operation_specs():
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

    return router


__all__ = ["build_meta_router"]
