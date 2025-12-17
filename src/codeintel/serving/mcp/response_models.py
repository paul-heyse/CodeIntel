"""Pydantic response models for MCP tools and resources.

This module defines typed response models for MCP tool returns and resource payloads,
enabling structured, schema-validated outputs that LLM agents can reliably parse.

All models use:
- `extra="forbid"` to prevent undocumented fields
- `frozen=True` for immutability (best practice for response objects)
- Tuples over lists for immutable collections
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from codeintel.serving.semantic.models import SemanticQueryResponse


def _pydantic_runtime_types() -> tuple[type[object], ...]:
    """Return runtime types used in Pydantic annotations.

    This helper ensures types like ``datetime`` and ``SemanticQueryResponse`` are
    treated as runtime dependencies (not type-checking-only) so that Pydantic
    can resolve annotations without requiring explicit model rebuild calls.

    Returns
    -------
    tuple[type[object], ...]
        Runtime types referenced by model annotations.
    """
    return (datetime, SemanticQueryResponse)

# =============================================================================
# URI / ID Primitives (Annotated types with validation + documentation)
# =============================================================================

CodeIntelURI = Annotated[
    str,
    Field(
        pattern=r"^codeintel://.+",
        description="CodeIntel resource URI (codeintel://...).",
        examples=["codeintel://meta/serving", "codeintel://semantic/views/function_metrics"],
    ),
]
"""URI for CodeIntel resources, must start with `codeintel://`."""

RFC6570TemplateURI = Annotated[
    str,
    Field(
        description="RFC 6570 URI template (may include {placeholders}).",
        examples=[
            "codeintel://semantic/views/{view_id}",
            "codeintel://exports/{export_id}/meta",
        ],
    ),
]
"""RFC 6570 URI template that may contain placeholders like `{view_id}`."""

ViewId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_.-]+$",
        description="Semantic view identifier (stable).",
        examples=["function_metrics", "risk_factors", "module_profile"],
    ),
]
"""Stable semantic view identifier (alphanumeric with underscores, dots, hyphens)."""

ExportId = Annotated[
    str,
    Field(
        min_length=8,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Export identifier (opaque token).",
        examples=["01HZY9E1K8ZQ6N9J3W2K9M3A8B"],
    ),
]
"""Opaque export handle identifier (alphanumeric with underscores, hyphens)."""

Sha256Hex = Annotated[
    str,
    Field(
        pattern=r"^[a-f0-9]{64}$",
        description="SHA-256 hex digest (64 lowercase hex characters).",
        examples=["e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"],
    ),
]
"""SHA-256 hex digest (64 lowercase hex characters)."""

# =============================================================================
# Literal Types
# =============================================================================

ExportFormat = Literal["ndjson", "json", "parquet", "arrow"]
"""Export serialization format."""

ExportStatus = Literal["ready", "expired", "missing", "error"]
"""Current status of an export handle."""

# =============================================================================
# Basic Nested Models
# =============================================================================


class SnapshotRef(BaseModel):
    """Identify the immutable serving snapshot currently mounted.

    Every MCP response includes snapshot metadata so LLM agents can
    detect when data changes between calls.

    Parameters
    ----------
    repo
        Repository identifier (usually org/repo format).
    commit
        Git commit SHA (or equivalent).
    run_id
        Build run identifier (stable for the snapshot).
    published_at
        When the serving snapshot was published.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str = Field(..., description="Repository identifier (usually org/repo format).")
    commit: str = Field(..., description="Git commit SHA (or equivalent).")
    run_id: str = Field(..., description="Build run identifier (stable for the snapshot).")
    published_at: datetime = Field(..., description="When the serving snapshot was published.")


class ResourceTemplate(BaseModel):
    """Self-documenting resource discovery for LLM agents.

    Each template describes a resource endpoint that agents can call,
    with URI pattern, MIME type, and categorization tags.

    Parameters
    ----------
    uri
        Resource URI or RFC 6570 template.
    description
        Human/LLM friendly description.
    mime_type
        MIME type if fixed/known.
    tags
        Categorization tags for filtering.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str = Field(..., description="Resource URI (or RFC 6570 template).")
    description: str = Field(..., description="Human/LLM friendly description.")
    mime_type: str | None = Field(default=None, description="MIME type if fixed/known.")
    tags: tuple[str, ...] = Field(default_factory=tuple, description="Categorization tags.")


class SemanticLayerInfo(BaseModel):
    """Semantic layer identity and inventory.

    Provides versioning and hash information for the semantic registry,
    allowing agents to detect schema changes.

    Parameters
    ----------
    version
        Semantic layer version string.
    hash
        Stable hash of semantic registry content.
    view_count
        Number of semantic views available.
    schema_manifest_hash
        Hash of schema manifest backing semantic view schemas.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(..., description="Semantic layer version string.")
    hash: str = Field(..., description="Stable hash of semantic registry content.")
    view_count: int = Field(..., ge=0, description="Number of semantic views available.")
    schema_manifest_hash: str | None = Field(
        default=None,
        description="Hash of schema manifest backing semantic view schemas (if applicable).",
    )


class BuildSpecInfo(BaseModel):
    """BuildSpec identity exposed by serving for parity/debugging.

    Parameters
    ----------
    version
        BuildSpec schema/version.
    hash
        Stable deterministic hash of BuildSpec JSON.
    compiled_at
        When BuildSpec was compiled.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(..., description="BuildSpec schema/version.")
    hash: str = Field(..., description="Stable deterministic hash of BuildSpec JSON.")
    compiled_at: datetime = Field(..., description="When BuildSpec was compiled.")


class QueryLimits(BaseModel):
    """Server-enforced limits so agents do not guess.

    Parameters
    ----------
    default_limit
        Default row limit for query tools.
    max_limit
        Maximum allowed limit for query tools.
    export_max_rows
        Maximum rows allowed for exports.
    export_ttl_seconds
        Optional TTL for exports in seconds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_limit: int = Field(default=200, ge=1, description="Default row limit for query tools.")
    max_limit: int = Field(
        default=5_000, ge=1, description="Maximum allowed limit for query tools."
    )
    export_max_rows: int = Field(
        default=100_000, ge=1, description="Maximum rows allowed for exports."
    )
    export_ttl_seconds: int | None = Field(
        default=None,
        ge=1,
        description="Optional TTL for exports; if None, exports are session-scoped or manual cleanup.",
    )


# =============================================================================
# Export Models
# =============================================================================


class ExportSnapshot(BaseModel):
    """Snapshot identity captured at export time.

    Proves that export results are stable and tied to a specific build.

    Parameters
    ----------
    snapshot
        Serving snapshot in effect when export was created.
    semantic_layer_hash
        Semantic layer hash at export time.
    buildspec_hash
        BuildSpec hash at export time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot: SnapshotRef = Field(
        ..., description="Serving snapshot in effect when export was created."
    )
    semantic_layer_hash: str = Field(..., description="Semantic layer hash at export time.")
    buildspec_hash: str = Field(..., description="BuildSpec hash at export time.")


class ExportURIs(BaseModel):
    """All resource URIs associated with an export.

    Parameters
    ----------
    payload_uri
        URI to fetch the export payload.
    meta_uri
        URI to fetch this metadata.
    preview_uri
        URI to fetch a small preview (JSON).
    sql_uri
        URI to fetch compiled SQL (if stored).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    payload_uri: str = Field(..., description="URI to fetch the export payload.")
    meta_uri: str = Field(..., description="URI to fetch this metadata.")
    preview_uri: str | None = Field(
        default=None, description="URI to fetch a small preview (JSON)."
    )
    sql_uri: str | None = Field(default=None, description="URI to fetch compiled SQL (if stored).")


class ExportQuerySpec(BaseModel):
    """Sanitized echo of what was exported.

    Intentionally generic to avoid tight coupling to internal
    SemanticQueryRequest model and to keep forward-compat.

    Parameters
    ----------
    view_id
        Semantic view exported (if export came from semantic layer).
    select
        Selected columns (if specified).
    order_by
        Ordering spec (server conventions).
    filters
        Filter specs (sanitized, JSON-serializable).
    limit
        Limit used for export generation.
    offset
        Offset used for export generation.
    query_hash
        Stable fingerprint of query inputs.
    """

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
    """Lightweight schema summary for the exported payload.

    Parameters
    ----------
    columns
        Column names in payload order.
    types
        Column types keyed by column name.
    schema_hash
        Stable fingerprint of the schema.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    columns: tuple[str, ...] = Field(
        default_factory=tuple, description="Column names in payload order."
    )
    types: dict[str, str] = Field(
        default_factory=dict,
        description="Column types keyed by column name.",
        examples=[{"repo": "VARCHAR", "commit": "VARCHAR", "cyclomatic_complexity": "INTEGER"}],
    )
    schema_hash: str | None = Field(
        default=None, description="Stable fingerprint of the schema (e.g., hash(columns+types))."
    )


class ExportHandleResponse(BaseModel):
    """Handle returned by export tool; payload is fetched via resources.

    Tools return a small structured handle, and the LLM/agent fetches
    big payloads via resources (which is a core FastMCP pattern).

    Parameters
    ----------
    export_id
        Opaque export token.
    format
        Export serialization format.
    mime_type
        MIME type for export payload.
    filename
        Suggested filename for clients.
    uri
        Resource URI to fetch the export payload.
    meta_uri
        Resource URI to fetch export metadata.
    preview_uri
        Resource URI to fetch export preview.
    sql_uri
        Resource URI to fetch compiled SQL.
    created_at
        When this export was generated.
    expires_at
        When this export expires (if TTL is used).
    row_count
        Row count if known without reading payload.
    byte_size
        Byte size if known without reading payload.
    snapshot
        Hashes/identities captured at export time.
    note
        Optional note for the client/LLM.
    """

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
    note: str | None = Field(
        default=None,
        description="Optional note for the client/LLM (e.g. 'result spilled to export').",
    )


class ExportMetaResponse(BaseModel):
    """Metadata returned by `codeintel://exports/{export_id}/meta`.

    Complete enough that an agent can decide whether to fetch the actual payload.

    Parameters
    ----------
    export_id
        Export identifier.
    status
        Current export status.
    created_at
        When this export was created.
    expires_at
        When this export expires.
    format
        Serialization format.
    mime_type
        MIME type for payload.
    filename
        Suggested filename for the payload.
    row_count
        Row count (if known).
    byte_size
        Byte size (if known).
    sha256
        Hash of payload bytes (if computed).
    snapshot
        Snapshot/build/semantic hashes captured at export time.
    query
        Sanitized query spec used to generate the export.
    schema
        Schema summary for the exported payload.
    uris
        Resource URIs for payload + helpers.
    warnings
        Non-fatal warnings.
    note
        Human/LLM-friendly note.
    """

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
        default=None,
        description="Sanitized query spec used to generate the export (if applicable).",
    )
    schema_summary: ExportSchemaSummary | None = Field(
        default=None, description="Schema summary for the exported payload (if applicable)."
    )

    uris: ExportURIs = Field(..., description="Resource URIs for payload + helpers.")

    warnings: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Non-fatal warnings (e.g., 'sql unavailable', 'row_count unknown').",
    )
    note: str | None = Field(default=None, description="Human/LLM-friendly note.")


# =============================================================================
# Query Response Models
# =============================================================================


class QueryPreview(BaseModel):
    """Small, safe preview that fits in LLM context windows.

    Parameters
    ----------
    columns
        Column names in display order.
    rows
        Preview rows (truncated).
    truncated
        Whether preview is truncated.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    columns: tuple[str, ...] = Field(
        default_factory=tuple, description="Column names in display order."
    )
    rows: tuple[dict[str, object], ...] = Field(
        default_factory=tuple, description="Preview rows (truncated)."
    )
    truncated: bool = Field(default=True, description="Whether preview is truncated.")


class SemanticQueryToolResponse(BaseModel):
    """Return type for semantic_query: result + optional export spillover.

    Always returns a normal SemanticQueryResult, and optionally returns
    an export handle (and URIs) when results are large.

    Parameters
    ----------
    result
        Primary query result (may be truncated).
    preview
        Optional small preview for LLM-friendly output.
    export
        If present, full results are available via export resources.
    export_uri
        Shortcut to export payload URI.
    export_meta_uri
        Shortcut to export meta URI.
    note
        Short, user/LLM-friendly note.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    result: SemanticQueryResponse = Field(
        ..., description="Primary query result (may be truncated)."
    )
    preview: QueryPreview | None = Field(
        default=None,
        description="Optional small preview for LLM-friendly output (even if export exists).",
    )
    export: ExportHandleResponse | None = Field(
        default=None, description="If present, full results are available via export resources."
    )
    export_uri: str | None = Field(default=None, description="Shortcut to export payload URI.")
    export_meta_uri: str | None = Field(default=None, description="Shortcut to export meta URI.")
    note: str | None = Field(
        default=None,
        description="Short, user/LLM-friendly note. Example: 'Result truncated to 200 rows; use export_uri for full dataset.'",
    )


# =============================================================================
# Meta / Discovery Models
# =============================================================================


class ServingMetaResponse(BaseModel):
    """High-level server + snapshot metadata for LLM discovery and debugging.

    Makes the server self-describing, so agents can learn inventories
    and URI conventions without external docs.

    Parameters
    ----------
    service
        Service identifier.
    server_version
        CodeIntel package/version string.
    protocol
        Protocol identifier.
    started_at
        When this server process started.
    snapshot
        Currently mounted serving snapshot.
    semantic_layer
        Semantic layer identity + counts.
    buildspec
        Compiled BuildSpec identity.
    read_only
        Whether serving DB connections are read-only.
    features
        Feature flags/capabilities.
    limits
        Server limits for queries and exports.
    resource_templates
        Resource URI taxonomy templates exposed by the server.
    inventories
        Counts of available datasets/tables/exports/etc for quick triage.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    service: Literal["codeintel"] = Field(default="codeintel", description="Service identifier.")
    server_version: str = Field(..., description="CodeIntel package/version string.")
    protocol: Literal["mcp"] = Field(default="mcp", description="Protocol identifier.")
    started_at: datetime = Field(..., description="When this server process started.")

    snapshot: SnapshotRef = Field(..., description="Currently mounted serving snapshot.")
    semantic_layer: SemanticLayerInfo = Field(..., description="Semantic layer identity + counts.")
    buildspec: BuildSpecInfo = Field(..., description="Compiled BuildSpec identity.")

    read_only: bool = Field(
        default=True, description="Whether serving DB connections are read-only."
    )
    features: dict[str, bool] = Field(
        default_factory=dict,
        description=(
            "Feature flags/capabilities. Example keys: "
            "supports_explain, supports_export, supports_sampling, supports_resources."
        ),
    )

    limits: QueryLimits = Field(..., description="Server limits for queries and exports.")

    resource_templates: tuple[ResourceTemplate, ...] = Field(
        default_factory=tuple, description="Resource URI taxonomy templates exposed by the server."
    )

    inventories: dict[str, int] = Field(
        default_factory=dict,
        description="Counts of available datasets/tables/exports/etc for quick triage.",
    )


class ResourceTemplatesResponse(BaseModel):
    """Returned by `codeintel://meta/resources` for standardized discovery.

    The canonical discovery resource that an agent can call to understand
    which `codeintel://...` resources exist and their MIME types.

    Parameters
    ----------
    uri
        Canonical URI for this response.
    generated_at
        When this listing was generated.
    snapshot
        Currently mounted serving snapshot.
    semantic_layer
        Semantic layer identity (if semantic layer enabled).
    buildspec
        BuildSpec identity (if BuildSpec available at serving time).
    templates
        All supported resource templates, stable and machine-readable.
    notes
        Short usage notes for agents (limits, auth, TTLs, etc.).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str = Field(
        default="codeintel://meta/resources", description="Canonical URI for this response."
    )
    generated_at: datetime = Field(..., description="When this listing was generated.")
    snapshot: SnapshotRef = Field(..., description="Currently mounted serving snapshot.")

    semantic_layer: SemanticLayerInfo | None = Field(
        default=None, description="Semantic layer identity (if semantic layer enabled)."
    )
    buildspec: BuildSpecInfo | None = Field(
        default=None, description="BuildSpec identity (if BuildSpec available at serving time)."
    )

    templates: tuple[ResourceTemplate, ...] = Field(
        default_factory=tuple,
        description="All supported resource templates, stable and machine-readable.",
    )

    notes: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Short usage notes for agents (limits, auth, TTLs, etc.).",
    )


# =============================================================================
# Default Resource Templates
# =============================================================================

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
)
"""Default resource templates exposed by the MCP server for discovery."""


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "DEFAULT_RESOURCE_TEMPLATES",
    # Sorted alphabetically
    "BuildSpecInfo",
    "CodeIntelURI",
    "ExportFormat",
    "ExportHandleResponse",
    "ExportId",
    "ExportMetaResponse",
    "ExportQuerySpec",
    "ExportSchemaSummary",
    "ExportSnapshot",
    "ExportStatus",
    "ExportURIs",
    "QueryLimits",
    "QueryPreview",
    "RFC6570TemplateURI",
    "ResourceTemplate",
    "ResourceTemplatesResponse",
    "SemanticLayerInfo",
    "SemanticQueryToolResponse",
    "ServingMetaResponse",
    "Sha256Hex",
    "SnapshotRef",
    "ViewId",
]
