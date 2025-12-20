"""Shared helpers for FastMCP tool implementations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

from mcp import McpError

from codeintel.serving.export.formats import normalize_export_format
from codeintel.serving.features import ServingFeatureSet
from codeintel.serving.mcp._compat import Context
from codeintel.serving.mcp.models import QueryPreview
from codeintel.serving.semantic.models import FilterSpec, SemanticQueryRequest

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
    return "mcp-unknown"


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


def build_semantic_request(
    view_id: str,
    filters: list[dict[str, object]] | None,
    select: list[str] | None,
    order_by: list[str] | None,
    pagination: dict[str, int] | None,
) -> SemanticQueryRequest:
    """Build a SemanticQueryRequest from tool parameters.

    Returns
    -------
    SemanticQueryRequest
        Validated query request model built from tool parameters.
    """
    page = pagination or {}
    return SemanticQueryRequest(
        view_id=view_id,
        select=select,
        filters=[FilterSpec.model_validate(f) for f in (filters or [])],
        order_by=order_by or [],
        limit=page.get("limit", 200),
        offset=page.get("offset", 0),
    )


__all__ = [
    "PREVIEW_ROW_COUNT",
    "READ_ONLY_LOCAL_ANNOTATIONS",
    "TAG_EXPORT",
    "TAG_META",
    "TAG_READ",
    "TAG_SEARCH",
    "TAG_SEMANTIC",
    "InvalidExportFormatError",
    "build_semantic_request",
    "maybe_report_progress",
    "mcp_correlation_id",
    "normalize_export_format_for_tool",
    "try_sample_summary",
]
