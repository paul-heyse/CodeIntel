"""Export-related models for FastMCP tools and resources."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from codeintel.serving.export.formats import ExportFormat
from codeintel.serving.mcp.models.primitives import SnapshotRef

ExportStatus = Literal["ready", "expired", "missing", "error"]


class ExportSnapshot(BaseModel):
    """Snapshot identity captured at export time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot: SnapshotRef = Field(
        ..., description="Serving snapshot in effect when export was created."
    )
    semantic_layer_hash: str = Field(..., description="Semantic layer hash at export time.")
    buildspec_hash: str = Field(..., description="BuildSpec hash at export time.")


class ExportURIs(BaseModel):
    """All resource URIs associated with an export."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    payload_uri: str = Field(..., description="URI to fetch the export payload.")
    meta_uri: str = Field(..., description="URI to fetch this metadata.")
    preview_uri: str | None = Field(
        default=None, description="URI to fetch a small preview (JSON)."
    )
    sql_uri: str | None = Field(default=None, description="URI to fetch compiled SQL (if stored).")
    lines_uri_template: str | None = Field(
        default=None,
        description="URI template for chunked line reads: codeintel://exports/{export_id}/lines{?offset,limit}.",
    )
    bytes_uri_template: str | None = Field(
        default=None,
        description="URI template for chunked byte reads: codeintel://exports/{export_id}/bytes{?offset,limit}.",
    )


class ExportQuerySpec(BaseModel):
    """Sanitized echo of what was exported."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    view_id: str | None = Field(
        default=None, description="Semantic view exported (if export came from semantic layer)."
    )
    select: tuple[str, ...] | None = Field(
        default=None, description="Selected columns (if specified)."
    )
    order_by: tuple[str, ...] = Field(
        default_factory=tuple, description="Ordering spec (server conventions)."
    )
    filters: tuple[dict[str, object], ...] = Field(
        default_factory=tuple, description="Filter specs (sanitized, JSON-serializable)."
    )
    limit: int | None = Field(default=None, ge=0, description="Limit used for export generation.")
    offset: int | None = Field(default=None, ge=0, description="Offset used for export generation.")
    query_hash: str | None = Field(
        default=None,
        description="Stable fingerprint of query inputs (filters/select/order/limit/offset).",
        examples=["q_7c9a2c2b0f0d6a31"],
    )


class ExportSchemaSummary(BaseModel):
    """Lightweight schema summary for the exported payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    columns: tuple[str, ...] = Field(
        default_factory=tuple, description="Column names in payload order."
    )
    types: dict[str, str] = Field(
        default_factory=dict,
        description="Column types keyed by column name.",
        examples=[{"repo": "VARCHAR", "commit": "VARCHAR", "cyclomatic_complexity": "INTEGER"}],
    )
    schema_hash: str | None = Field(default=None, description="Stable fingerprint of the schema.")


class ExportHandleResponse(BaseModel):
    """Handle returned by export tool; payload is fetched via resources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    export_id: str = Field(..., description="Opaque export token.")
    format: ExportFormat = Field(..., description="Export serialization format.")
    mime_type: str = Field(..., description="MIME type for export payload.")
    filename: str = Field(..., description="Suggested filename for clients.")

    uri: str = Field(..., description="Resource URI to fetch the export payload.")
    meta_uri: str = Field(..., description="Resource URI to fetch export metadata.")
    preview_uri: str | None = Field(
        default=None, description="Resource URI to fetch export preview."
    )
    sql_uri: str | None = Field(
        default=None, description="Resource URI to fetch compiled SQL (if available)."
    )

    created_at: datetime = Field(..., description="When this export was generated.")
    expires_at: datetime | None = Field(
        default=None, description="When this export expires (if TTL is used)."
    )

    row_count: int | None = Field(
        default=None, ge=0, description="Row count if known without reading payload."
    )
    byte_size: int | None = Field(
        default=None, ge=0, description="Byte size if known without reading payload."
    )

    snapshot: ExportSnapshot = Field(..., description="Hashes/identities captured at export time.")
    note: str | None = Field(default=None, description="Optional note for the client/LLM.")


class ExportMetaResponse(BaseModel):
    """Metadata returned by `codeintel://exports/{export_id}/meta`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    export_id: str = Field(..., description="Export identifier.")
    status: ExportStatus = Field(..., description="Current export status.")
    created_at: datetime = Field(..., description="When this export was created.")
    expires_at: datetime | None = Field(
        default=None, description="When this export expires (if TTL configured)."
    )

    format: ExportFormat = Field(..., description="Serialization format.")
    mime_type: str = Field(..., description="MIME type for payload.")
    filename: str = Field(..., description="Suggested filename for the payload.")

    row_count: int | None = Field(default=None, ge=0, description="Row count (if known).")
    byte_size: int | None = Field(default=None, ge=0, description="Byte size (if known).")
    sha256: str | None = Field(default=None, description="Hash of payload bytes (if computed).")

    snapshot: ExportSnapshot = Field(
        ..., description="Snapshot/build/semantic hashes captured at export time."
    )

    query: ExportQuerySpec | None = Field(
        default=None, description="Sanitized query spec used to generate the export."
    )
    schema_summary: ExportSchemaSummary | None = Field(
        default=None, description="Schema summary for the exported payload (if applicable)."
    )

    uris: ExportURIs = Field(..., description="Resource URIs for payload + helpers.")
    warnings: tuple[str, ...] = Field(default_factory=tuple, description="Non-fatal warnings.")
    note: str | None = Field(default=None, description="Human/LLM-friendly note.")


__all__ = [
    "ExportHandleResponse",
    "ExportMetaResponse",
    "ExportQuerySpec",
    "ExportSchemaSummary",
    "ExportSnapshot",
    "ExportStatus",
    "ExportURIs",
]
