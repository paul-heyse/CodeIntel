"""Semantic view tagging utilities.

This decorator attaches semantic metadata to Hamilton nodes via tags.
It is shared across layers so that storage view builders can be tagged without
importing build-owned modules.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal, Required, TypedDict, TypeVar, Unpack

from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.tagging_helpers import apply_raw_tags

SEMANTIC_VIEW_TAG_ATTR = "__codeintel_semantic_view_tags__"

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
TAG_LAYER = ht.TAG_LAYER
TAG_KIND = ht.TAG_KIND
TAG_VERSION = ht.TAG_VERSION
TAG_SCHEMA_REF = ht.TAG_SCHEMA_REF
TAG_ENTITY_KEYS = ht.TAG_ENTITY_KEYS
TAG_JOIN_KEYS = ht.TAG_JOIN_KEYS

_TFunc = TypeVar("_TFunc", bound=Callable[..., object])


def _csv(values: tuple[str, ...]) -> str:
    return ", ".join(v for v in values if v)


class SemanticViewTagSpec(TypedDict, total=False):
    """Keyword-based semantic view specification for the `@semantic_view` decorator."""

    semantic_id: Required[str]
    table_key: Required[str]
    entity: Required[str]
    grain: Required[str]
    version: str
    primary_key: tuple[str, ...]
    columns: tuple[str, ...]
    description: str | None
    joins: list[dict[str, object]] | None
    default_order_by: tuple[str, ...]
    default_limit: int
    sensitivity: str
    mcp_visible: bool
    kind: Literal["table", "view"]
    schema_ref: str | None
    entity_keys: tuple[str, ...]
    join_keys: tuple[str, ...]


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
        TAG_LAYER: "semantic",
        TAG_KIND: spec.get("kind", "view"),
        TAG_VERSION: str(spec.get("version", "1")),
        TAG_OUTPUT_KIND: ht.OUTPUT_KIND_SEMANTIC_VIEW,
        TAG_DEFAULT_LIMIT: str(spec.get("default_limit", 200)),
        TAG_DEFAULT_ORDER: _csv(spec.get("default_order_by", ())),
        TAG_MCP_VISIBLE: "1" if spec.get("mcp_visible", True) else "0",
        ht.TAG_ENTITY: entity,
        ht.TAG_GRAIN: grain,
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
    schema_ref = spec.get("schema_ref")
    if schema_ref:
        tags[TAG_SCHEMA_REF] = schema_ref
    entity_keys = spec.get("entity_keys", ())
    if entity_keys:
        tags[TAG_ENTITY_KEYS] = _csv(entity_keys)
    join_keys = spec.get("join_keys", ())
    if join_keys:
        tags[TAG_JOIN_KEYS] = _csv(join_keys)

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
        _ = (
            spec["semantic_id"],
            spec["table_key"],
            spec["entity"],
            spec["grain"],
        )
    except KeyError as exc:
        raise SemanticViewSpecError from exc
    tags = _build_semantic_tags(spec)

    def decorator(func: _TFunc) -> _TFunc:
        tagged = apply_raw_tags(func, tags=tags)
        setattr(tagged, SEMANTIC_VIEW_TAG_ATTR, tags)
        return tagged

    return decorator


def get_semantic_view_tags(func: object) -> dict[str, str] | None:
    """Return semantic view tags previously attached to a function.

    Parameters
    ----------
    func
        Callable object potentially decorated with `semantic_view`.

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
    "TAG_ENTITY_KEYS",
    "TAG_JOIN_KEYS",
    "TAG_KIND",
    "TAG_LAYER",
    "TAG_MCP_VISIBLE",
    "TAG_OUTPUT_KIND",
    "TAG_SCHEMA_REF",
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
    "TAG_VERSION",
    "SemanticViewSpecError",
    "SemanticViewTagSpec",
    "get_semantic_view_tags",
    "semantic_view",
]
