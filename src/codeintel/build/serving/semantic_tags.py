"""Semantic view tagging utilities for build-time compilation.

This module provides a decorator used to attach semantic metadata to view
builder functions (e.g., Ibis view builders). The build pipeline compiles these
tags into `semantic_registry.json`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal, Required, TypedDict, TypeVar, Unpack

SEMANTIC_VIEW_TAG_ATTR = "__codeintel_semantic_view_tags__"

# Tag keys for semantic layer (persisted into the registry)
TAG_OUTPUT_KIND = "output_kind"
TAG_SEMANTIC_ID = "semantic_id"
TAG_SEMANTIC_KIND = "semantic_kind"
TAG_TABLE_KEY = "table_key"
TAG_SEMANTIC_ENTITY = "semantic_entity"
TAG_SEMANTIC_GRAIN = "semantic_grain"
TAG_SEMANTIC_PK = "semantic_primary_key"
TAG_SEMANTIC_COLS = "semantic_columns"
TAG_SEMANTIC_DESC = "semantic_description"
TAG_SEMANTIC_JOINS = "semantic_joins"
TAG_MCP_VISIBLE = "mcp_visible"
TAG_DEFAULT_ORDER = "semantic_default_order_by"
TAG_DEFAULT_LIMIT = "semantic_default_limit"
TAG_SENSITIVITY = "semantic_sensitivity"
TAG_DEPRECATED = "semantic_deprecated"
TAG_REPLACED_BY = "semantic_replaced_by"

_TFunc = TypeVar("_TFunc", bound=Callable[..., object])


def _csv(values: tuple[str, ...]) -> str:
    return ", ".join(v for v in values if v)


class SemanticViewTagSpec(TypedDict, total=False):
    """Keyword-based semantic view specification for the `@semantic_view` decorator."""

    semantic_id: Required[str]
    table_key: Required[str]
    entity: Required[str]
    grain: Required[str]
    primary_key: tuple[str, ...]
    columns: tuple[str, ...]
    description: str | None
    joins: list[dict[str, object]] | None
    default_order_by: tuple[str, ...]
    default_limit: int
    sensitivity: str
    deprecated: bool
    replaced_by: str | None
    mcp_visible: bool
    kind: Literal["table", "view"]


class SemanticViewSpecError(TypeError):
    """Raised when `semantic_view` receives an invalid specification."""

    def __init__(self) -> None:
        super().__init__("semantic_view requires semantic_id, table_key, entity, and grain")


def semantic_view(
    **spec: Unpack[SemanticViewTagSpec],
) -> Callable[[_TFunc], _TFunc]:
    """Attach semantic metadata tags to a view builder function.

    Parameters
    ----------
    spec
        Keyword-only semantic view spec (see keys below).

    Other Parameters
    ----------------
    semantic_id
        Stable semantic view identifier.
    table_key
        Fully qualified DuckDB object name (schema.table).
    entity
        Entity type represented by the view.
    grain
        Row granularity for the view.
    primary_key
        Primary key column names.
    columns
        Explicit columns to expose (empty = derive from SchemaProvider).
    description
        Human-readable description.
    joins
        Optional join hints for agents (stored as JSON).
    default_order_by
        Default ordering columns (prefix "-" for DESC).
    default_limit
        Default limit for queries.
    sensitivity
        Sensitivity label (internal/public/etc).
    deprecated
        Whether this view is deprecated.
    replaced_by
        Replacement semantic view ID.
    mcp_visible
        Whether to expose this view through MCP tools.
    kind
        "table" or "view" for describing the underlying DuckDB object.

    Returns
    -------
    Callable[[Callable[..., object]], Callable[..., object]]
        Decorator that annotates and returns the function unchanged.

    Raises
    ------
    SemanticViewSpecError
        If required spec keys are missing.
    """
    try:
        semantic_id = spec["semantic_id"]
        table_key = spec["table_key"]
        entity = spec["entity"]
        grain = spec["grain"]
    except KeyError as exc:
        raise SemanticViewSpecError from exc

    primary_key = spec.get("primary_key", ())
    columns = spec.get("columns", ())
    description = spec.get("description")
    joins = spec.get("joins")
    default_order_by = spec.get("default_order_by", ())
    default_limit = spec.get("default_limit", 200)
    sensitivity = spec.get("sensitivity", "internal")
    deprecated = spec.get("deprecated", False)
    replaced_by = spec.get("replaced_by")
    mcp_visible = spec.get("mcp_visible", True)
    kind = spec.get("kind", "view")

    def decorator(func: _TFunc) -> _TFunc:
        tags: dict[str, str] = {
            TAG_OUTPUT_KIND: "semantic",
            TAG_MCP_VISIBLE: "1" if mcp_visible else "0",
            TAG_SEMANTIC_ID: semantic_id,
            TAG_SEMANTIC_KIND: kind,
            TAG_TABLE_KEY: table_key,
            TAG_SEMANTIC_ENTITY: entity,
            TAG_SEMANTIC_GRAIN: grain,
            TAG_SEMANTIC_PK: _csv(primary_key),
            TAG_DEFAULT_ORDER: _csv(default_order_by),
            TAG_DEFAULT_LIMIT: str(default_limit),
            TAG_SENSITIVITY: sensitivity,
            TAG_DEPRECATED: "1" if deprecated else "0",
        }
        if columns:
            tags[TAG_SEMANTIC_COLS] = _csv(columns)
        if description is not None:
            tags[TAG_SEMANTIC_DESC] = description
        if joins is not None:
            tags[TAG_SEMANTIC_JOINS] = json.dumps(joins, sort_keys=True)
        if replaced_by is not None:
            tags[TAG_REPLACED_BY] = replaced_by

        setattr(func, SEMANTIC_VIEW_TAG_ATTR, tags)
        return func

    return decorator


def get_semantic_view_tags(func: object) -> dict[str, str] | None:
    """Return semantic view tags previously attached to a function.

    Returns
    -------
    dict[str, str] | None
        Attached semantic view tag mapping or None if missing.
    """
    tags = getattr(func, SEMANTIC_VIEW_TAG_ATTR, None)
    return tags if isinstance(tags, dict) else None


__all__ = [
    "SEMANTIC_VIEW_TAG_ATTR",
    "TAG_DEFAULT_LIMIT",
    "TAG_DEFAULT_ORDER",
    "TAG_DEPRECATED",
    "TAG_MCP_VISIBLE",
    "TAG_OUTPUT_KIND",
    "TAG_REPLACED_BY",
    "TAG_SEMANTIC_COLS",
    "TAG_SEMANTIC_DESC",
    "TAG_SEMANTIC_ENTITY",
    "TAG_SEMANTIC_GRAIN",
    "TAG_SEMANTIC_ID",
    "TAG_SEMANTIC_JOINS",
    "TAG_SEMANTIC_KIND",
    "TAG_SEMANTIC_PK",
    "TAG_SENSITIVITY",
    "TAG_TABLE_KEY",
    "get_semantic_view_tags",
    "semantic_view",
]
