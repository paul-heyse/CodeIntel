"""Meta/discovery models for FastMCP."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from codeintel.serving.mcp.models.primitives import ResourceTemplate, SnapshotRef


class SemanticLayerInfo(BaseModel):
    """Semantic layer identity and inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(..., description="Semantic layer version string.")
    hash: str = Field(..., description="Stable hash of semantic registry content.")
    view_count: int = Field(..., ge=0, description="Number of semantic views available.")
    schema_manifest_hash: str | None = Field(default=None, description="Hash of schema manifest (if applicable).")


class BuildSpecInfo(BaseModel):
    """BuildSpec identity exposed by serving for parity/debugging."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(..., description="BuildSpec schema/version.")
    hash: str = Field(..., description="Stable deterministic hash of BuildSpec JSON.")
    compiled_at: datetime = Field(..., description="When BuildSpec was compiled.")


class QueryLimits(BaseModel):
    """Server-enforced limits so agents do not guess."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_limit: int = Field(default=200, ge=1, description="Default row limit for query tools.")
    max_limit: int = Field(default=5_000, ge=1, description="Maximum allowed limit for query tools.")
    export_max_rows: int = Field(default=100_000, ge=1, description="Maximum rows allowed for exports.")
    export_ttl_seconds: int | None = Field(default=None, ge=1, description="Optional TTL for exports.")


class ServingMetaResponse(BaseModel):
    """High-level server + snapshot metadata for LLM discovery and debugging."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    service: Literal["codeintel"] = Field(default="codeintel", description="Service identifier.")
    server_version: str = Field(..., description="CodeIntel package/version string.")
    protocol: Literal["mcp"] = Field(default="mcp", description="Protocol identifier.")
    started_at: datetime = Field(..., description="When this server process started.")

    snapshot: SnapshotRef = Field(..., description="Currently mounted serving snapshot.")
    semantic_layer: SemanticLayerInfo = Field(..., description="Semantic layer identity + counts.")
    buildspec: BuildSpecInfo = Field(..., description="Compiled BuildSpec identity.")

    read_only: bool = Field(default=True, description="Whether serving DB connections are read-only.")
    features: dict[str, bool] = Field(default_factory=dict, description="Feature flags/capabilities.")
    limits: QueryLimits = Field(..., description="Server limits for queries and exports.")
    resource_templates: tuple[ResourceTemplate, ...] = Field(default_factory=tuple, description="Resource templates.")
    inventories: dict[str, int] = Field(default_factory=dict, description="Inventory counts for quick triage.")
    warnings: tuple[str, ...] = Field(default_factory=tuple, description="Non-fatal warnings.")


class ResourceTemplatesResponse(BaseModel):
    """Returned by `codeintel://meta/resources` for standardized discovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str = Field(default="codeintel://meta/resources", description="Canonical URI for this response.")
    generated_at: datetime = Field(..., description="When this listing was generated.")
    snapshot: SnapshotRef = Field(..., description="Currently mounted serving snapshot.")

    semantic_layer: SemanticLayerInfo | None = Field(default=None, description="Semantic layer identity.")
    buildspec: BuildSpecInfo | None = Field(default=None, description="BuildSpec identity.")

    templates: tuple[ResourceTemplate, ...] = Field(default_factory=tuple, description="Supported resource templates.")
    notes: tuple[str, ...] = Field(default_factory=tuple, description="Short usage notes for agents.")


DEFAULT_RESOURCE_TEMPLATES: tuple[ResourceTemplate, ...] = (
    ResourceTemplate(
        uri="codeintel://meta/serving",
        description="Serving metadata: snapshot, semantic layer hash, BuildSpec hash, limits.",
        mime_type="application/json",
        tags=("meta",),
    ),
    ResourceTemplate(
        uri="codeintel://meta/resources",
        description="Machine-readable inventory of all CodeIntel resources and URI templates.",
        mime_type="application/json",
        tags=("meta", "discovery"),
    ),
    ResourceTemplate(
        uri="codeintel://meta/environment",
        description="Snapshot environment metadata (tool versions) plus runtime mismatch warnings.",
        mime_type="application/json",
        tags=("meta",),
    ),
    ResourceTemplate(
        uri="codeintel://meta/views_sql",
        description="Compiled SQL for semantic views in the mounted snapshot (select-only perimeter).",
        mime_type="application/json",
        tags=("meta",),
    ),
    ResourceTemplate(
        uri="codeintel://meta/views_sql_diff",
        description="Diff summary for compiled semantic view SQL vs prior snapshot (if available).",
        mime_type="application/json",
        tags=("meta",),
    ),
    ResourceTemplate(
        uri="codeintel://semantic/views",
        description="JSON list of semantic views (catalog).",
        mime_type="application/json",
        tags=("semantic", "catalog"),
    ),
    ResourceTemplate(
        uri="codeintel://semantic/views/{view_id}",
        description="Semantic view descriptor (columns, entity, grain, defaults).",
        mime_type="application/json",
        tags=("semantic", "describe"),
    ),
    ResourceTemplate(
        uri="codeintel://exports/{export_id}",
        description="Export payload (format depends on export).",
        mime_type=None,
        tags=("exports", "payload"),
    ),
    ResourceTemplate(
        uri="codeintel://exports/{export_id}/meta",
        description="Export metadata (schema, counts, hashes, provenance).",
        mime_type="application/json",
        tags=("exports", "meta"),
    ),
    ResourceTemplate(
        uri="codeintel://exports/{export_id}/preview",
        description="Small JSON preview of export payload (LLM-friendly).",
        mime_type="application/json",
        tags=("exports", "preview"),
    ),
    ResourceTemplate(
        uri="codeintel://exports/{export_id}/sql",
        description="Compiled SQL used to generate the export (if recorded).",
        mime_type="text/plain",
        tags=("exports", "sql"),
    ),
    ResourceTemplate(
        uri="codeintel://exports/{export_id}/lines{?offset,limit}",
        description="Chunked line retrieval for ndjson exports (offset/limit lines).",
        mime_type="text/plain",
        tags=("exports", "chunk"),
    ),
    ResourceTemplate(
        uri="codeintel://exports/{export_id}/bytes{?offset,limit}",
        description="Chunked byte retrieval for parquet/arrow exports (offset/limit bytes).",
        mime_type="application/octet-stream",
        tags=("exports", "chunk"),
    ),
)


__all__ = [
    "DEFAULT_RESOURCE_TEMPLATES",
    "BuildSpecInfo",
    "QueryLimits",
    "ResourceTemplatesResponse",
    "SemanticLayerInfo",
    "ServingMetaResponse",
]
