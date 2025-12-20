"""Transport-agnostic meta/discovery models for serving."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from codeintel.serving.models.primitives import ResourceTemplate, SnapshotRef
from codeintel.serving.uris import (
    EXPORT_BYTES_URI_TEMPLATE,
    EXPORT_LINES_URI_TEMPLATE,
    EXPORT_META_URI_TEMPLATE,
    EXPORT_PREVIEW_URI_TEMPLATE,
    EXPORT_SQL_URI_TEMPLATE,
    EXPORT_URI_TEMPLATE,
    META_ENVIRONMENT_URI,
    META_RESOURCES_URI,
    META_SERVING_URI,
    META_VIEWS_SQL_DIFF_URI,
    META_VIEWS_SQL_URI,
    SEMANTIC_VIEW_URI_TEMPLATE,
    SEMANTIC_VIEWS_URI,
)


class SemanticLayerInfo(BaseModel):
    """Semantic layer identity and inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(..., description="Semantic layer version string.")
    hash: str = Field(..., description="Stable hash of semantic registry content.")
    view_count: int = Field(..., ge=0, description="Number of semantic views available.")
    schema_manifest_hash: str | None = Field(
        default=None, description="Hash of schema manifest (if applicable)."
    )


class BuildSpecInfo(BaseModel):
    """BuildSpec identity exposed by serving for parity/debugging."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(..., description="BuildSpec schema/version.")
    hash: str = Field(..., description="Stable deterministic hash of BuildSpec JSON.")
    compiled_at: datetime = Field(..., description="When BuildSpec was compiled.")


class QueryLimits(BaseModel):
    """Server-enforced limits so clients do not guess."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_limit: int = Field(default=200, ge=1, description="Default row limit for query tools.")
    max_limit: int = Field(
        default=5_000, ge=1, description="Maximum allowed limit for query tools."
    )
    export_max_rows: int = Field(
        default=100_000, ge=1, description="Maximum rows allowed for exports."
    )
    export_ttl_seconds: int | None = Field(
        default=None, ge=1, description="Optional TTL for exports."
    )


class ServingMetaResponse(BaseModel):
    """High-level server + snapshot metadata for discovery and debugging."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    service: Literal["codeintel"] = Field(default="codeintel", description="Service identifier.")
    server_version: str = Field(..., description="CodeIntel package/version string.")
    protocol: Literal["mcp"] = Field(default="mcp", description="Protocol identifier.")
    started_at: datetime = Field(..., description="When this server process started.")

    snapshot: SnapshotRef = Field(..., description="Currently mounted serving snapshot.")
    semantic_layer: SemanticLayerInfo = Field(..., description="Semantic layer identity + counts.")
    buildspec: BuildSpecInfo = Field(..., description="Compiled BuildSpec identity.")

    read_only: bool = Field(default=True, description="Whether serving DB is read-only.")
    features: dict[str, bool] = Field(default_factory=dict, description="Feature flags.")
    limits: QueryLimits = Field(..., description="Server limits for queries and exports.")
    resource_templates: tuple[ResourceTemplate, ...] = Field(
        default_factory=tuple, description="Resource templates."
    )
    inventories: dict[str, int] = Field(
        default_factory=dict, description="Inventory counts for quick triage."
    )
    warnings: tuple[str, ...] = Field(default_factory=tuple, description="Non-fatal warnings.")


class ResourceTemplatesResponse(BaseModel):
    """Returned by `codeintel://meta/resources` for standardized discovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str = Field(default=META_RESOURCES_URI, description="Canonical URI for this response.")
    generated_at: datetime = Field(..., description="When this listing was generated.")
    snapshot: SnapshotRef = Field(..., description="Currently mounted serving snapshot.")

    semantic_layer: SemanticLayerInfo | None = Field(
        default=None, description="Semantic layer identity."
    )
    buildspec: BuildSpecInfo | None = Field(default=None, description="BuildSpec identity.")

    templates: tuple[ResourceTemplate, ...] = Field(
        default_factory=tuple, description="Supported resource templates."
    )
    notes: tuple[str, ...] = Field(
        default_factory=tuple, description="Short usage notes for agents."
    )


class ServingKernelMetaResponse(BaseModel):
    """Metadata payload for the HTTP `/meta` endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str
    commit: str
    run_id: str
    published_at: datetime
    semantic_layer_version: str
    buildspec_hash: str
    buildspec_version: int
    duckdb: dict[str, object]
    environment: dict[str, object]
    semantic_views: list[dict[str, object]]
    datasets: list[dict[str, object]]
    targets: list[dict[str, object]]
    schema_inventory: dict[str, int]


DEFAULT_RESOURCE_TEMPLATES: tuple[ResourceTemplate, ...] = (
    ResourceTemplate(
        uri=META_SERVING_URI,
        description="Serving metadata: snapshot, semantic layer hash, BuildSpec hash, limits.",
        mime_type="application/json",
        tags=("meta",),
    ),
    ResourceTemplate(
        uri=META_RESOURCES_URI,
        description="Machine-readable inventory of all CodeIntel resources and URI templates.",
        mime_type="application/json",
        tags=("meta", "discovery"),
    ),
    ResourceTemplate(
        uri=META_ENVIRONMENT_URI,
        description="Snapshot environment metadata (tool versions) plus runtime mismatch warnings.",
        mime_type="application/json",
        tags=("meta",),
    ),
    ResourceTemplate(
        uri=META_VIEWS_SQL_URI,
        description="Compiled SQL for semantic views in the mounted snapshot (select-only).",
        mime_type="application/json",
        tags=("meta",),
    ),
    ResourceTemplate(
        uri=META_VIEWS_SQL_DIFF_URI,
        description="Diff summary for compiled semantic view SQL vs prior snapshot (if available).",
        mime_type="application/json",
        tags=("meta",),
    ),
    ResourceTemplate(
        uri=SEMANTIC_VIEWS_URI,
        description="JSON list of semantic views (catalog).",
        mime_type="application/json",
        tags=("semantic", "catalog"),
    ),
    ResourceTemplate(
        uri=SEMANTIC_VIEW_URI_TEMPLATE,
        description="Semantic view descriptor (columns, entity, grain, defaults).",
        mime_type="application/json",
        tags=("semantic", "describe"),
    ),
    ResourceTemplate(
        uri=EXPORT_URI_TEMPLATE,
        description="Export payload (format depends on export).",
        mime_type=None,
        tags=("exports", "payload"),
    ),
    ResourceTemplate(
        uri=EXPORT_META_URI_TEMPLATE,
        description="Export metadata (schema, counts, hashes, provenance).",
        mime_type="application/json",
        tags=("exports", "meta"),
    ),
    ResourceTemplate(
        uri=EXPORT_PREVIEW_URI_TEMPLATE,
        description="Small JSON preview of export payload (LLM-friendly).",
        mime_type="application/json",
        tags=("exports", "preview"),
    ),
    ResourceTemplate(
        uri=EXPORT_SQL_URI_TEMPLATE,
        description="Compiled SQL used to generate the export (if recorded).",
        mime_type="text/plain",
        tags=("exports", "sql"),
    ),
    ResourceTemplate(
        uri=EXPORT_LINES_URI_TEMPLATE,
        description="Chunked line retrieval for ndjson exports (offset/limit lines).",
        mime_type="text/plain",
        tags=("exports", "chunk"),
    ),
    ResourceTemplate(
        uri=EXPORT_BYTES_URI_TEMPLATE,
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
    "ServingKernelMetaResponse",
    "ServingMetaResponse",
]
