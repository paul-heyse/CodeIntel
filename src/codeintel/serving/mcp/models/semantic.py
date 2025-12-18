"""Semantic-query tool response models for FastMCP."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from codeintel.serving.mcp.models.exports import ExportHandleResponse
from codeintel.serving.semantic.models import SemanticQueryResponse


class QueryPreview(BaseModel):
    """Small, safe preview that fits in LLM context windows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    columns: tuple[str, ...] = Field(
        default_factory=tuple, description="Column names in display order."
    )
    rows: tuple[dict[str, object], ...] = Field(
        default_factory=tuple, description="Preview rows (truncated)."
    )
    truncated: bool = Field(default=True, description="Whether preview is truncated.")


class SemanticQueryToolResponse(BaseModel):
    """Return type for semantic_query: result + optional preview/export info."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    result: SemanticQueryResponse = Field(
        ..., description="Primary query result (may be truncated)."
    )
    preview: QueryPreview | None = Field(
        default=None, description="Optional small preview for LLM-friendly output."
    )
    export: ExportHandleResponse | None = Field(
        default=None, description="If present, full results are available via export resources."
    )
    export_uri: str | None = Field(default=None, description="Shortcut to export payload URI.")
    export_meta_uri: str | None = Field(default=None, description="Shortcut to export meta URI.")
    summary: str | None = Field(
        default=None, description="Optional LLM-generated summary for large results."
    )
    sql_fingerprint: str | None = Field(
        default=None, description="SHA256 fingerprint of canonical SQL."
    )
    note: str | None = Field(default=None, description="Short, user/LLM-friendly note.")


__all__ = ["QueryPreview", "SemanticQueryToolResponse"]
