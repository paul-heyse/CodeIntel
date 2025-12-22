"""Shared helpers for FastMCP tool implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from fastmcp import Context
from mcp import McpError

from codeintel.core.execution.ids import new_uuid_hex
from codeintel.serving.export.formats import normalize_export_format
from codeintel.serving.features import ServingFeatureSet
from codeintel.serving.mcp.models import QueryPreview
from codeintel.serving.mcp.models.requests import (
    SemanticExportToolRequest,
    SemanticQueryToolRequest,
)
from codeintel.serving.metrics import QueryMetrics, log_query_metrics
from codeintel.serving.semantic.models import SemanticExportRequest, SemanticQueryRequest

if TYPE_CHECKING:
    from codeintel.serving.export.formats import ExportFormat
    from codeintel.serving.settings import ServingSettings


READ_ONLY_LOCAL_ANNOTATIONS: Final[dict[str, bool]] = {
    "readOnlyHint": True,
    "idempotentHint": True,
    "openWorldHint": False,
}

TAG_SEMANTIC: Final[str] = "semantic"
TAG_SEARCH: Final[str] = "search"
TAG_META: Final[str] = "meta"
TAG_READ: Final[str] = "read"
TAG_EXPORT: Final[str] = "export"

PREVIEW_ROW_COUNT: Final[int] = 5


def mcp_correlation_id(ctx: Context | None) -> str:
    """Return a stable correlation identifier from the MCP context.

    Returns
    -------
    str
        Correlation identifier suitable for logging and metrics.
    """
    if ctx is None:
        return "mcp-unknown"
    session_id_obj = getattr(ctx, "session_id", None)
    if isinstance(session_id_obj, str) and session_id_obj:
        return session_id_obj
    return new_uuid_hex()


@dataclass(frozen=True, slots=True)
class McpMetricsInput:
    """Inputs for MCP query metrics logging."""

    endpoint: str
    view_id: str | None
    query: str | None
    row_count: int
    truncated: bool
    duration_ms: float
    query_hash: str | None = None
    schema_hash: str | None = None


def log_mcp_query_metrics(
    metrics: McpMetricsInput,
    *,
    ctx: Context | None,
) -> None:
    """Record structured query metrics for MCP tools."""
    log_query_metrics(
        QueryMetrics(
            endpoint=metrics.endpoint,
            view_id=metrics.view_id,
            query=metrics.query,
            row_count=metrics.row_count,
            truncated=metrics.truncated,
            duration_ms=metrics.duration_ms,
            correlation_id=mcp_correlation_id(ctx),
            query_hash=metrics.query_hash,
            schema_hash=metrics.schema_hash,
        )
    )


async def maybe_report_progress(
    ctx: Context | None,
    *,
    settings: ServingSettings,
    progress: float,
    total: float | None = None,
    message: str | None = None,
) -> None:
    """Report progress to the MCP host when enabled."""
    if ctx is None:
        return
    feature_set = ServingFeatureSet.from_settings(settings)
    if not feature_set.enable_mcp_progress:
        return
    await ctx.report_progress(progress, total, message)


async def try_sample_summary(
    ctx: Context,
    *,
    view_id: str,
    preview: QueryPreview,
    query_hash: str | None,
) -> str | None:
    """Generate a lightweight summary for a preview response.

    Returns
    -------
    str | None
        Summary text when sampling succeeds, otherwise ``None``.
    """
    payload = {
        "view_id": view_id,
        "query_hash": query_hash,
        "columns": list(preview.columns),
        "rows": list(preview.rows),
        "truncated": preview.truncated,
    }
    prompt = json.dumps(payload, indent=2, sort_keys=True, default=str)
    try:
        result = await ctx.sample(
            f"Summarize this query preview in 5 bullets (be precise, no speculation):\n\n{prompt}",
            system_prompt=(
                "You are summarizing a database query preview for an agent. "
                "Prefer actionable observations and call out truncation."
            ),
            max_tokens=250,
        )
    except (McpError, RuntimeError, ValueError):
        return None
    if isinstance(result.result, str):
        return result.result
    return result.text


class InvalidExportFormatError(ValueError):
    """Raised when an export_format value is unsupported."""

    def __init__(self, export_format: str) -> None:
        msg = f"Unsupported export_format: {export_format}"
        super().__init__(msg)
        self.export_format = export_format


def normalize_export_format_for_tool(export_format: str) -> ExportFormat:
    """Normalize and validate export_format values.

    Returns
    -------
    ExportFormat
        Normalized export format value.

    Raises
    ------
    InvalidExportFormatError
        If the input is not a supported export format.
    """
    try:
        return normalize_export_format(export_format)
    except ValueError as exc:
        raise InvalidExportFormatError(export_format) from exc


def validate_semantic_query_request(
    request: SemanticQueryRequest | SemanticQueryToolRequest | dict[str, object],
) -> SemanticQueryRequest:
    """Validate and normalize a semantic query request payload.

    Returns
    -------
    SemanticQueryRequest
        Validated semantic query request model.

    """
    if isinstance(request, SemanticQueryRequest):
        return request
    if isinstance(request, SemanticQueryToolRequest):
        return request.to_semantic_request()
    return SemanticQueryToolRequest.model_validate(request).to_semantic_request()


def validate_semantic_export_request(
    request: SemanticExportRequest | SemanticExportToolRequest | dict[str, object],
) -> SemanticExportRequest:
    """Validate and normalize a semantic export request payload.

    Returns
    -------
    SemanticExportRequest
        Validated semantic export request model.

    """
    if isinstance(request, SemanticExportRequest):
        return request
    if isinstance(request, SemanticExportToolRequest):
        return request.to_semantic_request()
    return SemanticExportToolRequest.model_validate(request).to_semantic_request()


__all__ = [
    "PREVIEW_ROW_COUNT",
    "READ_ONLY_LOCAL_ANNOTATIONS",
    "TAG_EXPORT",
    "TAG_META",
    "TAG_READ",
    "TAG_SEARCH",
    "TAG_SEMANTIC",
    "InvalidExportFormatError",
    "McpMetricsInput",
    "log_mcp_query_metrics",
    "maybe_report_progress",
    "mcp_correlation_id",
    "normalize_export_format_for_tool",
    "try_sample_summary",
    "validate_semantic_export_request",
    "validate_semantic_query_request",
]
