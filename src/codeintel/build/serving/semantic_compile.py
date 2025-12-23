"""Compile semantic registry from view tag metadata.

Scans registered view builders for semantic tags and produces a deterministic
`semantic_registry.json` artifact for serving.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.hamilton.tag_index import TagIndex
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

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.core.schemas.provider import SchemaProvider


def _split_csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


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

        explicit_cols = _split_csv(tags.get(TAG_SEMANTIC_COLS))
        if explicit_cols:
            cols = explicit_cols
        else:
            schema = schema_provider.get_table_schema(table_key)
            cols = schema.column_names() if schema else []

        entity = tags.get(TAG_SEMANTIC_ENTITY) or tags.get(ht.TAG_ENTITY) or "unknown"
        grain = tags.get(TAG_SEMANTIC_GRAIN) or tags.get(ht.TAG_GRAIN) or "unknown"
        view_entry: dict[str, object] = {
            "id": semantic_id,
            "kind": tags.get(TAG_SEMANTIC_KIND, "view"),
            "table_key": table_key,
            "entity": entity,
            "grain": grain,
            "description": tags.get(TAG_SEMANTIC_DESC),
            "primary_key": _split_csv(tags.get(TAG_SEMANTIC_PK)),
            "columns": cols,
            "joins": _parse_joins(tags.get(TAG_SEMANTIC_JOINS)),
            "defaults": {
                "limit": int(tags.get(TAG_DEFAULT_LIMIT, "200")),
                "order_by": _split_csv(tags.get(TAG_DEFAULT_ORDER)),
            },
            "sensitivity": tags.get(TAG_SENSITIVITY, "internal"),
        }
        views.append(view_entry)

    views.sort(key=lambda v: str(v.get("id", "")))
    return CompiledSemanticRegistry(version=version, views=views)


def compile_semantic_registry_from_tag_index(
    *,
    schema_provider: SchemaProvider,
    tag_index: TagIndex,
    version: str = "v1",
) -> CompiledSemanticRegistry:
    """Compile semantic registry from a TagIndex.

    Parameters
    ----------
    schema_provider
        Provider for resolving table schemas.
    tag_index
        TagIndex containing normalized Hamilton tags.
    version
        Registry version string.

    Returns
    -------
    CompiledSemanticRegistry
        Compiled registry.
    """
    return compile_semantic_registry_from_views(
        schema_provider=schema_provider,
        view_tags=tag_index.semantic_view_tags(),
        version=version,
    )


def write_semantic_registry(*, registry: CompiledSemanticRegistry, out_path: Path) -> None:
    """Write semantic registry to file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(registry.to_json() + "\n", encoding="utf-8")


__all__ = [
    "CompiledSemanticRegistry",
    "compile_semantic_registry_from_tag_index",
    "compile_semantic_registry_from_views",
    "write_semantic_registry",
]
