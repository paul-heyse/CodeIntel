"""Request models for FastMCP semantic tools."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from codeintel.serving.semantic.models import (
    ExportFormat,
    FilterSpec,
    SemanticExportRequest,
    SemanticQueryRequest,
)


class PaginationSpec(BaseModel):
    """Pagination inputs for MCP tool requests."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=200, ge=0, le=10_000)
    offset: int = Field(default=0, ge=0)


class SemanticQueryToolRequest(BaseModel):
    """Request envelope for semantic_query."""

    model_config = ConfigDict(extra="forbid")

    view_id: str
    select: list[str] | None = None
    filters: list[FilterSpec] = Field(default_factory=list)
    order_by: list[str] = Field(default_factory=list)
    pagination: PaginationSpec | None = None
    export_format: ExportFormat | None = None

    def to_semantic_request(self) -> SemanticQueryRequest:
        """Convert MCP query payload into the semantic request model.

        Returns
        -------
        SemanticQueryRequest
            Normalized semantic query request.
        """
        payload = self.model_dump(exclude={"pagination", "export_format"}, exclude_none=True)
        if self.pagination is not None:
            payload["limit"] = self.pagination.limit
            payload["offset"] = self.pagination.offset
        return SemanticQueryRequest.model_validate(payload)


class SemanticExplainToolRequest(SemanticQueryToolRequest):
    """Request envelope for semantic_explain."""

    model_config = ConfigDict(extra="forbid")


class SemanticExportToolRequest(BaseModel):
    """Request envelope for semantic_export."""

    model_config = ConfigDict(extra="forbid")

    view_id: str
    select: list[str] | None = None
    filters: list[FilterSpec] = Field(default_factory=list)
    order_by: list[str] = Field(default_factory=list)
    export_format: ExportFormat | None = None
    limit: int = Field(default=100_000, ge=0, le=1_000_000)
    offset: int = Field(default=0, ge=0)

    def to_semantic_request(self) -> SemanticExportRequest:
        """Convert MCP export payload into the semantic request model.

        Returns
        -------
        SemanticExportRequest
            Normalized semantic export request.
        """
        payload = self.model_dump(exclude={"export_format"}, exclude_none=True)
        if self.export_format is not None:
            payload["format"] = self.export_format
        return SemanticExportRequest.model_validate(payload)


__all__ = [
    "SemanticExplainToolRequest",
    "SemanticExportToolRequest",
    "SemanticQueryToolRequest",
]
