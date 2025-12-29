"""Pydantic models for semantic layer queries and responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from codeintel.serving.export.formats import ExportFormat, default_export_format
from codeintel.serving.snapshot.models import ServingSnapshotIdentity

Op = Literal["eq", "ne", "lt", "lte", "gt", "gte", "in", "contains", "startswith"]
type FilterScalar = bool | int | float | str
type FilterValue = FilterScalar | list[FilterScalar]


class SemanticViewDefaults(BaseModel):
    """Default query parameters for a semantic view."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=200, ge=0, le=10_000)
    order_by: list[str] = Field(default_factory=list)

    @field_validator("order_by")
    @classmethod
    def _validate_order_by(cls, value: list[str]) -> list[str]:
        return [item for item in value if item]


class SemanticViewSpec(BaseModel):
    """Specification for a semantic view.

    Parameters
    ----------
    id
        Stable semantic view identifier (e.g., "function.summary").
    kind
        Whether this is a "table" or "view" in DuckDB.
    table_key
        Fully qualified DuckDB object name (e.g., "docs.v_function_summary").
    entity
        Entity type this view represents (e.g., "function", "module").
    grain
        Row granularity (e.g., "per_function", "per_module").
    description
        Human-readable description.
    primary_key
        Column names forming the primary key.
    columns
        Exposed column names.
    columns_dynamic
        When True, resolve columns from the active schema inventory.
    joins
        Optional join hints for agents.
    defaults
        Default query parameters (limit, order_by).
    sensitivity
        Data sensitivity level.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["table", "view"] = "view"
    table_key: str
    entity: str
    grain: str
    description: str | None = None
    primary_key: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    columns_dynamic: bool = False
    joins: list[dict[str, object]] = Field(default_factory=list)
    defaults: SemanticViewDefaults = Field(default_factory=SemanticViewDefaults)
    sensitivity: str = "internal"


class FilterSpec(BaseModel):
    """Filter specification for semantic queries.

    Parameters
    ----------
    column
        Column name to filter on.
    op
        Filter operation.
    value
        Value to compare against.
    """

    model_config = ConfigDict(extra="forbid")

    column: str
    op: Op
    value: FilterValue


class SemanticQueryRequest(BaseModel):
    """Request for a semantic view query.

    Parameters
    ----------
    view_id
        Semantic view identifier to query.
    select
        Optional column subset (None = all columns).
    filters
        Filter conditions.
    order_by
        Column ordering (prefix with "-" for DESC).
    limit
        Maximum rows to return.
    offset
        Rows to skip.
    """

    model_config = ConfigDict(extra="forbid")

    view_id: str
    select: list[str] | None = None
    filters: list[FilterSpec] = Field(default_factory=list)
    order_by: list[str] = Field(default_factory=list)
    limit: int = Field(default=200, ge=0, le=10_000)
    offset: int = Field(default=0, ge=0)

    @field_validator("order_by")
    @classmethod
    def _validate_request_order_by(cls, value: list[str]) -> list[str]:
        return [item for item in value if item]

    @field_validator("select")
    @classmethod
    def _validate_select(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        items = [item for item in value if item]
        return items or None


class QueryScanMetrics(BaseModel):
    """Dataset scan metrics for semantic queries."""

    model_config = ConfigDict(extra="forbid")

    row_count: int | None = None
    file_count: int | None = None
    total_bytes: int | None = None


class SemanticQueryResponse(BaseModel):
    """Response from a semantic view query.

    Parameters
    ----------
    view_id
        Queried view identifier.
    columns
        Column names in result order.
    rows
        Result rows as list of dicts.
    truncated
        Whether results were truncated by limit.
    engine
        Query execution engine used for the response.
    snapshot
        Snapshot metadata (repo, commit, run_id).
    query_hash
        Stable fingerprint of validated query inputs.
    schema_hash
        Stable fingerprint of the resolved schema (when available).
    scan_metrics
        Input scan metrics derived from dataset manifests when available.
    batch_size
        Batch size used for streaming execution.
    sql_fingerprint
        Stable fingerprint of canonical SQL when compiled SQL is available.
    """

    model_config = ConfigDict(extra="forbid")

    view_id: str
    columns: list[str]
    rows: list[dict[str, object]]
    truncated: bool
    engine: str | None = None
    snapshot: ServingSnapshotIdentity
    query_hash: str | None = None
    schema_hash: str | None = None
    scan_metrics: QueryScanMetrics | None = None
    batch_size: int | None = None
    sql_fingerprint: str | None = None


class SemanticExplainResponse(BaseModel):
    """Response payload for a semantic query EXPLAIN request.

    Includes derived table/column lineage when available.
    """

    model_config = ConfigDict(extra="forbid")

    view_id: str
    sql: str
    plan: str
    snapshot: ServingSnapshotIdentity
    table_keys: list[str] = Field(default_factory=list)
    column_lineage: dict[str, list[ColumnLineageRef]] = Field(default_factory=dict)


class SemanticCatalogView(BaseModel):
    """Catalog entry for a semantic view."""

    model_config = ConfigDict(extra="forbid")

    id: str
    table_key: str
    entity: str
    grain: str
    description: str | None = None
    column_count: int = Field(default=0, ge=0)


class SemanticCatalogResponse(BaseModel):
    """Response payload for ``GET /semantic/views``."""

    model_config = ConfigDict(extra="forbid")

    version: str
    snapshot: ServingSnapshotIdentity
    views: list[SemanticCatalogView]


class ColumnLineageRef(BaseModel):
    """Reference to an upstream column in lineage metadata."""

    model_config = ConfigDict(extra="forbid")

    table_key: str
    column: str


class SemanticViewDescriptionResponse(BaseModel):
    """Response payload for ``GET /semantic/views/{view_id}``."""

    model_config = ConfigDict(extra="forbid")

    id: str
    table_key: str
    kind: Literal["table", "view"] = "view"
    entity: str
    grain: str
    description: str | None = None
    primary_key: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    column_types: dict[str, str] = Field(default_factory=dict)
    joins: list[dict[str, object]] = Field(default_factory=list)
    defaults: SemanticViewDefaults = Field(default_factory=SemanticViewDefaults)
    snapshot: ServingSnapshotIdentity
    lineage: dict[str, list[ColumnLineageRef]] = Field(default_factory=dict)


class SemanticExportRequest(BaseModel):
    """Request for streaming/export of semantic view data.

    Supports larger result sets than the standard query endpoint,
    with multiple output formats including JSONL, Parquet, and Arrow.

    Parameters
    ----------
    view_id
        Semantic view identifier to export.
    select
        Optional column subset (None = all columns).
    filters
        Filter conditions.
    order_by
        Column ordering (prefix with "-" for DESC).
    format
        Export format: jsonl, json, parquet, or arrow.
    limit
        Maximum rows to export (higher default than query).
    offset
        Rows to skip.
    """

    model_config = ConfigDict(extra="forbid")

    view_id: str
    select: list[str] | None = None
    filters: list[FilterSpec] = Field(default_factory=list)
    order_by: list[str] = Field(default_factory=list)
    format: ExportFormat = Field(default_factory=default_export_format)
    limit: int = Field(default=100_000, ge=0, le=1_000_000)
    offset: int = Field(default=0, ge=0)

    @field_validator("order_by")
    @classmethod
    def _validate_export_order_by(cls, value: list[str]) -> list[str]:
        return [item for item in value if item]

    @field_validator("select")
    @classmethod
    def _validate_export_select(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        items = [item for item in value if item]
        return items or None


__all__ = [
    "ColumnLineageRef",
    "ExportFormat",
    "FilterScalar",
    "FilterSpec",
    "FilterValue",
    "Op",
    "QueryScanMetrics",
    "SemanticCatalogResponse",
    "SemanticCatalogView",
    "SemanticExplainResponse",
    "SemanticExportRequest",
    "SemanticQueryRequest",
    "SemanticQueryResponse",
    "SemanticViewDefaults",
    "SemanticViewDescriptionResponse",
    "SemanticViewSpec",
]
