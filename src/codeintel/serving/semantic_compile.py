"""Compile semantic registry from view tag metadata.

Scans registered view builders for semantic tags and produces a deterministic
`semantic_registry.json` artifact for serving.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.semantic_tags import (
    TAG_DEFAULT_LIMIT,
    TAG_DEFAULT_ORDER,
    TAG_MCP_VISIBLE,
    TAG_OUTPUT_KIND,
    TAG_SEMANTIC_COLS,
    TAG_SEMANTIC_DESC,
    TAG_SEMANTIC_ENTITY,
    TAG_SEMANTIC_GRAIN,
    TAG_SEMANTIC_ID,
    TAG_SEMANTIC_JOINS,
    TAG_SEMANTIC_KIND,
    TAG_SEMANTIC_PK,
    TAG_SENSITIVITY,
    TAG_TABLE_KEY,
)
from codeintel.core.hamilton.tag_filters import tf_semantic_views
from codeintel.core.hamilton.tag_query import TagQuery
from codeintel.core.schemas.primitives import column_type_is_nested

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from hamilton.driver import Driver

    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider


def _split_csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _stringify_tag_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, float, int)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v is not None)
    return str(value)


def _tags_from_variable(variable: object) -> dict[str, object] | None:
    tags = getattr(variable, "tags", None)
    if isinstance(tags, dict):
        return tags
    return None


def _parse_joins(s: str | None) -> list[dict[str, object]]:
    if not s:
        return []
    value = json.loads(s)
    if not isinstance(value, list):
        msg = "Expected JSON array for semantic joins"
        raise TypeError(msg)
    joins: list[dict[str, object]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            msg = f"Expected object for semantic joins[{idx}]"
            raise TypeError(msg)
        joins.append({str(k): v for k, v in item.items()})
    return joins


_ORDER_BY_PRIORITY: tuple[str, ...] = (
    "repo",
    "commit",
    "run_id",
    "snapshot_id",
    "rel_path",
    "path",
    "module",
    "symbol",
    "symbol_id",
    "node_id",
    "cpg_node_id",
    "goid_h128",
    "id",
    "name",
    "start_byte",
    "start_line",
    "start_col",
    "end_byte",
    "end_line",
    "end_col",
    "ordinal",
)
_ORDER_BY_BLOCKLIST: frozenset[str] = frozenset(
    {
        "extras",
        "extras_kv",
        "extra",
        "metadata",
        "owners",
        "tags",
    }
)
_ORDER_BY_ID_SUFFIXES: tuple[str, ...] = ("_id", "_hash", "_sha", "_digest", "_h128")


def _looks_identifier(column: str) -> bool:
    if column in _ORDER_BY_PRIORITY:
        return True
    return column.endswith(_ORDER_BY_ID_SUFFIXES)


def _filter_order_by_candidates(
    *,
    schema: TableSchema | None,
    columns: list[str],
) -> list[str]:
    if not columns:
        return []
    type_by_name: dict[str, str] | None = None
    if schema is not None:
        type_by_name = {column.name: column.type for column in schema.columns}
    candidates: list[str] = []
    for column in columns:
        if column in _ORDER_BY_BLOCKLIST:
            continue
        if type_by_name is None:
            candidates.append(column)
            continue
        column_type = type_by_name.get(column)
        if column_type is None:
            candidates.append(column)
            continue
        if not column_type_is_nested(column_type):
            candidates.append(column)
    return candidates


def _resolve_default_order_by(
    *,
    schema: TableSchema | None,
    columns: list[str],
    primary_key: list[str],
    explicit_order: list[str],
) -> list[str]:
    if explicit_order:
        return explicit_order
    if primary_key:
        return []
    if schema is not None and schema.primary_key:
        schema_pk = [column for column in schema.primary_key if column in columns]
        if schema_pk:
            return schema_pk
    candidates = _filter_order_by_candidates(schema=schema, columns=columns)
    if candidates:
        prioritized = [column for column in _ORDER_BY_PRIORITY if column in candidates]
        if prioritized:
            return prioritized
        id_like = [column for column in candidates if _looks_identifier(column)]
        if id_like:
            return sorted(id_like)
    return []


@dataclass(frozen=True)
class CompiledSemanticRegistry:
    """Compiled semantic registry ready for serialization."""

    version: str
    views: list[dict[str, object]]

    def to_json(self) -> str:
        """Serialize to deterministic JSON.

        Returns
        -------
        str
            JSON string representation of this registry.
        """
        return json.dumps({"version": self.version, "views": self.views}, indent=2, sort_keys=True)


def compile_semantic_registry_from_views(
    *,
    schema_provider: SchemaProvider,
    view_tags: dict[str, dict[str, str]],
    version: str = "v1",
) -> CompiledSemanticRegistry:
    """Compile semantic registry from view tag metadata.

    Parameters
    ----------
    schema_provider
        Provider for resolving table schemas.
    view_tags
        Mapping from view name to tag dict.
    version
        Registry version string.

    Returns
    -------
    CompiledSemanticRegistry
        Compiled registry.
    """
    views: list[dict[str, object]] = []

    for tags in view_tags.values():
        output_kind = tags.get(TAG_OUTPUT_KIND)
        if output_kind not in {"semantic", ht.OUTPUT_KIND_SEMANTIC_VIEW}:
            continue
        if tags.get(TAG_MCP_VISIBLE, "1") != "1":
            continue

        semantic_id = tags.get(TAG_SEMANTIC_ID)
        table_key = tags.get(TAG_TABLE_KEY)

        if not semantic_id or not table_key:
            continue

        schema = schema_provider.get_table_schema(table_key)
        explicit_cols = _split_csv(tags.get(TAG_SEMANTIC_COLS))
        columns_dynamic = not explicit_cols
        cols = explicit_cols or (schema.column_names() if schema else [])

        entity = tags.get(TAG_SEMANTIC_ENTITY) or tags.get(ht.TAG_ENTITY) or "unknown"
        grain = tags.get(TAG_SEMANTIC_GRAIN) or tags.get(ht.TAG_GRAIN) or "unknown"
        primary_key = _split_csv(tags.get(TAG_SEMANTIC_PK))
        explicit_order = _split_csv(tags.get(TAG_DEFAULT_ORDER))
        resolved_order = _resolve_default_order_by(
            schema=schema,
            columns=cols,
            primary_key=primary_key,
            explicit_order=explicit_order,
        )
        view_entry: dict[str, object] = {
            "id": semantic_id,
            "kind": tags.get(TAG_SEMANTIC_KIND, "view"),
            "table_key": table_key,
            "entity": entity,
            "grain": grain,
            "description": tags.get(TAG_SEMANTIC_DESC),
            "primary_key": primary_key,
            "columns": cols,
            "columns_dynamic": columns_dynamic,
            "joins": _parse_joins(tags.get(TAG_SEMANTIC_JOINS)),
            "defaults": {
                "limit": int(tags.get(TAG_DEFAULT_LIMIT, "200")),
                "order_by": resolved_order,
            },
            "sensitivity": tags.get(TAG_SENSITIVITY, "internal"),
        }
        views.append(view_entry)

    views.sort(key=lambda v: str(v.get("id", "")))
    return CompiledSemanticRegistry(version=version, views=views)


def _compile_semantic_registry_from_variables(
    *,
    schema_provider: SchemaProvider,
    variables: Iterable[object],
    version: str = "v1",
) -> CompiledSemanticRegistry:
    """Compile semantic registry from Hamilton variables.

    Parameters
    ----------
    schema_provider
        Provider for resolving table schemas.
    variables
        Hamilton variables returned by tag-filter queries.
    version
        Registry version string.

    Returns
    -------
    CompiledSemanticRegistry
        Compiled registry.
    """
    view_tags: dict[str, dict[str, str]] = {}
    for variable in variables:
        tags = _tags_from_variable(variable)
        if not tags:
            continue
        table_key = tags.get(ht.TAG_TABLE_KEY)
        if not isinstance(table_key, str) or not table_key:
            continue
        view_tags[table_key] = {str(k): _stringify_tag_value(v) for k, v in tags.items()}

    return compile_semantic_registry_from_views(
        schema_provider=schema_provider,
        view_tags=view_tags,
        version=version,
    )


def compile_semantic_registry_from_driver(
    *,
    schema_provider: SchemaProvider,
    dr: Driver,
    version: str = "v1",
) -> CompiledSemanticRegistry:
    """Compile semantic registry from a Hamilton Driver tag filter query.

    Returns
    -------
    CompiledSemanticRegistry
        Compiled registry.
    """
    tag_query = TagQuery(dr)
    variables = tag_query.query(tf_semantic_views())
    return _compile_semantic_registry_from_variables(
        schema_provider=schema_provider,
        variables=variables,
        version=version,
    )


def compile_semantic_registry_from_tag_query(
    *,
    schema_provider: SchemaProvider,
    tag_query: TagQuery,
    version: str = "v1",
) -> CompiledSemanticRegistry:
    """Compile semantic registry from a TagQuery helper.

    Returns
    -------
    CompiledSemanticRegistry
        Compiled registry.
    """
    variables = tag_query.query(tf_semantic_views())
    return _compile_semantic_registry_from_variables(
        schema_provider=schema_provider,
        variables=variables,
        version=version,
    )


def write_semantic_registry(*, registry: CompiledSemanticRegistry, out_path: Path) -> None:
    """Write semantic registry to file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(registry.to_json() + "\n", encoding="utf-8")


__all__ = [
    "CompiledSemanticRegistry",
    "compile_semantic_registry_from_driver",
    "compile_semantic_registry_from_tag_query",
    "compile_semantic_registry_from_views",
    "write_semantic_registry",
]
