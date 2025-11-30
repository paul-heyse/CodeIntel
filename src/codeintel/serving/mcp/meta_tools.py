"""MCP tools for dataset and operation introspection."""

from __future__ import annotations

from typing import cast

from mcp.server.fastmcp import FastMCP

from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.models import DatasetMetaResponse, OperationMetaResponse, ProblemDetail
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.registry import build_dataset_meta, iter_operation_specs
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


def register_meta_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register meta MCP tools on the given FastMCP instance."""

    @mcp.tool()
    @_wrap
    def list_datasets_meta() -> list[dict[str, object]] | dict[str, ProblemDetail]:
        """
        List dataset metadata and serving limits via MCP.

        Returns
        -------
        list[DatasetMetaResponse]
            One entry per dataset, serialized to mappings.
        """
        limits = _get_limits(backend)
        service = _get_service(backend)
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
            ).model_dump()
            for meta in metas
        ]

    @mcp.tool()
    @_wrap
    def list_operations_meta() -> list[dict[str, object]] | dict[str, ProblemDetail]:
        """
        List available operations and their characteristics via MCP.

        Returns
        -------
        list[OperationMetaResponse]
            Operation metadata serialized to mappings.
        """
        limits = _get_limits(backend)
        payloads: list[OperationMetaResponse] = []
        for spec in iter_operation_specs():
            default_limit = spec.default_limit or limits.default_limit
            max_limit = spec.max_limit or limits.max_rows_per_call
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


__all__ = ["register_meta_tools"]
