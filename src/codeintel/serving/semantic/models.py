"""Pydantic models for semantic layer queries and responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Op = Literal["eq", "ne", "lt", "lte", "gt", "gte", "in", "contains", "startswith"]


class SemanticViewDefaults(BaseModel):
    """Default query parameters for a semantic view."""

    model_config = ConfigDict(extra="forbid")

    limit: int = 200
    order_by: list[str] = Field(default_factory=list)


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
    joins
        Optional join hints for agents.
    defaults
        Default query parameters (limit, order_by).
    sensitivity
        Data sensitivity level.
    deprecated
        Whether this view is deprecated.
    replaced_by
        Successor view ID if deprecated.
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
    joins: list[dict[str, object]] = Field(default_factory=list)
    defaults: SemanticViewDefaults = Field(default_factory=SemanticViewDefaults)
    sensitivity: str = "internal"
    deprecated: bool = False
    replaced_by: str | None = None


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
    value: object


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
    limit: int = 200
    offset: int = 0


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
    snapshot
        Snapshot metadata (repo, commit, run_id).
    """

    model_config = ConfigDict(extra="forbid")

    view_id: str
    columns: list[str]
    rows: list[dict[str, object]]
    truncated: bool
    snapshot: dict[str, str]


class SemanticExplainResponse(BaseModel):
    """Response payload for a semantic query EXPLAIN request."""

    model_config = ConfigDict(extra="forbid")

    view_id: str
    sql: str
    plan: str
    snapshot: dict[str, str]


__all__ = [
    "FilterSpec",
    "Op",
    "SemanticExplainResponse",
    "SemanticQueryRequest",
    "SemanticQueryResponse",
    "SemanticViewDefaults",
    "SemanticViewSpec",
]
