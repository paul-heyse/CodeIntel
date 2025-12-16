"""Semantic view tagging utilities for build-time compilation.

This module provides a decorator used to attach semantic metadata to view
builder functions (e.g., Ibis view builders). The build pipeline compiles these
tags into `semantic_registry.json`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal, Required, TypedDict, TypeVar, Unpack, cast

from hamilton.function_modifiers import tag as h_tag

from codeintel.build.hamilton import tags as ht

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


def _build_semantic_tags(spec: SemanticViewTagSpec) -> dict[str, str]:
    semantic_id = spec["semantic_id"]
    table_key = spec["table_key"]
    entity = spec["entity"]
    grain = spec["grain"]

    tags: dict[str, str] = {
        TAG_OUTPUT_KIND: ht.OUTPUT_KIND_SEMANTIC_VIEW,
        TAG_DEPRECATED: "1" if spec.get("deprecated", False) else "0",
        TAG_DEFAULT_LIMIT: str(spec.get("default_limit", 200)),
        TAG_DEFAULT_ORDER: _csv(spec.get("default_order_by", ())),
        TAG_MCP_VISIBLE: "1" if spec.get("mcp_visible", True) else "0",
        TAG_SEMANTIC_ENTITY: entity,
        TAG_SEMANTIC_GRAIN: grain,
        TAG_SEMANTIC_ID: semantic_id,
        TAG_SEMANTIC_KIND: spec.get("kind", "view"),
        TAG_SEMANTIC_PK: _csv(spec.get("primary_key", ())),
        TAG_SENSITIVITY: spec.get("sensitivity", "internal"),
        TAG_TABLE_KEY: table_key,
    }
    columns = spec.get("columns", ())
    if columns:
        tags[TAG_SEMANTIC_COLS] = _csv(columns)
    description = spec.get("description")
    if description is not None:
        tags[TAG_SEMANTIC_DESC] = description
    joins = spec.get("joins")
    if joins is not None:
        tags[TAG_SEMANTIC_JOINS] = json.dumps(joins, sort_keys=True)
    replaced_by = spec.get("replaced_by")
    if replaced_by is not None:
        tags[TAG_REPLACED_BY] = replaced_by

    return tags


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
    tags = _build_semantic_tags(spec)

    def decorator(func: _TFunc) -> _TFunc:
        tagged = cast(
            "_TFunc",
            h_tag(
                output_kind=ht.OUTPUT_KIND_SEMANTIC_VIEW,
                semantic_id=semantic_id,
                table_key=table_key,
                entity=entity,
                grain=grain,
                mcp_visible=tags[TAG_MCP_VISIBLE],
                semantic_default_limit=tags[TAG_DEFAULT_LIMIT],
                semantic_default_order_by=tags[TAG_DEFAULT_ORDER],
                semantic_deprecated=tags[TAG_DEPRECATED],
                semantic_kind=tags[TAG_SEMANTIC_KIND],
                semantic_primary_key=tags[TAG_SEMANTIC_PK],
                semantic_sensitivity=tags[TAG_SENSITIVITY],
            )(func),
        )
        if TAG_REPLACED_BY in tags:
            tagged = cast("_TFunc", h_tag(semantic_replaced_by=tags[TAG_REPLACED_BY])(tagged))
        if TAG_SEMANTIC_COLS in tags:
            tagged = cast("_TFunc", h_tag(semantic_columns=tags[TAG_SEMANTIC_COLS])(tagged))
        if TAG_SEMANTIC_DESC in tags:
            tagged = cast("_TFunc", h_tag(semantic_description=tags[TAG_SEMANTIC_DESC])(tagged))
        if TAG_SEMANTIC_JOINS in tags:
            tagged = cast("_TFunc", h_tag(semantic_joins=tags[TAG_SEMANTIC_JOINS])(tagged))
        setattr(tagged, SEMANTIC_VIEW_TAG_ATTR, tags)
        return tagged

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
