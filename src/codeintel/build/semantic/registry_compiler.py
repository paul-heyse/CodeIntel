"""Compile semantic registry artifacts from Hamilton tag metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping
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

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.core.hamilton.tag_query import TagQuery
    from codeintel.core.schemas.provider import SchemaProvider


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _stringify_tag_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, float, int)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item is not None)
    return str(value)


def _tags_from_variable(variable: object) -> Mapping[str, object] | None:
    tags = getattr(variable, "tags", None)
    if isinstance(tags, Mapping):
        return tags
    return None


def _parse_joins(value: str | None) -> list[dict[str, object]]:
    if not value:
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        msg = "Expected JSON array for semantic joins"
        raise TypeError(msg)
    joins: list[dict[str, object]] = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            msg = f"Expected object for semantic joins[{idx}]"
            raise TypeError(msg)
        joins.append({str(key): val for key, val in item.items()})
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
            JSON payload for the compiled registry.
        """
        return json.dumps({"version": self.version, "views": self.views}, indent=2, sort_keys=True)


@dataclass(frozen=True, slots=True)
class SemanticTagIssue:
    """Single semantic tag validation issue."""

    node: str
    message: str
    tag: str | None = None


class SemanticTagValidationError(ValueError):
    """Raised when semantic tag validation fails."""

    def __init__(self, issues: Iterable[SemanticTagIssue]) -> None:
        issues_tuple = tuple(issues)
        summary = ", ".join(
            f"{issue.node}:{issue.tag}" if issue.tag else issue.node for issue in issues_tuple
        )
        super().__init__(f"Semantic tag validation failed: {summary}")
        self.issues = issues_tuple


def compile_semantic_registry_from_views(
    *,
    schema_provider: SchemaProvider,
    view_tags: dict[str, dict[str, str]],
    version: str = "v1",
) -> CompiledSemanticRegistry:
    """Compile semantic registry from view tag metadata.

    Returns
    -------
    CompiledSemanticRegistry
        Compiled registry for semantic views.
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
        columns_dynamic = not explicit_cols
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
            "columns_dynamic": columns_dynamic,
            "joins": _parse_joins(tags.get(TAG_SEMANTIC_JOINS)),
            "defaults": {
                "limit": int(tags.get(TAG_DEFAULT_LIMIT, "200")),
                "order_by": _split_csv(tags.get(TAG_DEFAULT_ORDER)),
            },
            "sensitivity": tags.get(TAG_SENSITIVITY, "internal"),
        }
        views.append(view_entry)

    views.sort(key=lambda view: str(view.get("id", "")))
    return CompiledSemanticRegistry(version=version, views=views)


def _semantic_variables(tag_query: TagQuery) -> tuple[object, ...]:
    seen: set[str] = set()
    collected: list[object] = []
    for variable in tag_query.query(tf_semantic_views()):
        key = _variable_key(variable)
        if key in seen:
            continue
        seen.add(key)
        collected.append(variable)
    for variable in tag_query.query({ht.TAG_LAYER: "semantic"}):
        key = _variable_key(variable)
        if key in seen:
            continue
        seen.add(key)
        collected.append(variable)
    return tuple(collected)


def _variable_key(variable: object) -> str:
    name = getattr(variable, "name", None)
    return str(name) if name is not None else str(id(variable))


def _variable_name(variable: object) -> str:
    name = getattr(variable, "name", None)
    return str(name) if name is not None else "unknown"


def _tag_str_value(tags: Mapping[str, object], key: str) -> str | None:
    value = tags.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _semantic_tag_value(tags: Mapping[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _tag_str_value(tags, key)
        if value is not None:
            return value
    return None


def _semantic_kind(tags: Mapping[str, object]) -> str | None:
    return _semantic_tag_value(tags, (ht.TAG_KIND, TAG_SEMANTIC_KIND))


def _semantic_entity(tags: Mapping[str, object]) -> str | None:
    return _semantic_tag_value(tags, (ht.TAG_ENTITY, TAG_SEMANTIC_ENTITY))


def _semantic_grain(tags: Mapping[str, object]) -> str | None:
    return _semantic_tag_value(tags, (ht.TAG_GRAIN, TAG_SEMANTIC_GRAIN))


def _is_semantic_output(tags: Mapping[str, object]) -> bool:
    output_kind = _tag_str_value(tags, ht.TAG_OUTPUT_KIND)
    layer = _tag_str_value(tags, ht.TAG_LAYER)
    return output_kind == ht.OUTPUT_KIND_SEMANTIC_VIEW or layer == "semantic"


def _missing_semantic_tag_issue(
    *,
    name: str,
    tag: str,
    message: str,
) -> SemanticTagIssue:
    return SemanticTagIssue(node=name, tag=tag, message=message)


def _semantic_layer_issue(
    *,
    name: str,
    tags: Mapping[str, object],
) -> SemanticTagIssue | None:
    layer = _tag_str_value(tags, ht.TAG_LAYER)
    if layer == "semantic":
        return None
    detail = "missing layer tag" if layer is None else f"layer tag must be 'semantic' (got {layer})"
    return _missing_semantic_tag_issue(
        name=name,
        tag=ht.TAG_LAYER,
        message=f"Semantic output {detail}",
    )


def _semantic_required_tag_issues(
    *,
    name: str,
    tags: Mapping[str, object],
    kind: str | None,
) -> list[SemanticTagIssue]:
    required = (
        (
            ht.TAG_SEMANTIC_ID,
            _tag_str_value(tags, ht.TAG_SEMANTIC_ID),
            "Semantic output missing semantic_id tag",
        ),
        (ht.TAG_KIND, kind, "Semantic output missing kind tag"),
        (ht.TAG_ENTITY, _semantic_entity(tags), "Semantic output missing entity tag"),
        (ht.TAG_GRAIN, _semantic_grain(tags), "Semantic output missing grain tag"),
        (
            ht.TAG_VERSION,
            _tag_str_value(tags, ht.TAG_VERSION),
            "Semantic output missing version tag",
        ),
    )
    issues: list[SemanticTagIssue] = []
    for tag, value, message in required:
        if value is None:
            issues.append(_missing_semantic_tag_issue(name=name, tag=tag, message=message))
    return issues


def _semantic_table_tag_issues(
    *,
    name: str,
    tags: Mapping[str, object],
    kind: str | None,
) -> list[SemanticTagIssue]:
    if kind != "table":
        return []
    required = (
        (ht.TAG_SCHEMA_REF, "Semantic table missing schema_ref tag"),
        (ht.TAG_ENTITY_KEYS, "Semantic table missing entity_keys tag"),
        (ht.TAG_JOIN_KEYS, "Semantic table missing join_keys tag"),
    )
    issues: list[SemanticTagIssue] = []
    for tag, message in required:
        if _tag_str_value(tags, tag) is None:
            issues.append(_missing_semantic_tag_issue(name=name, tag=tag, message=message))
    return issues


def _semantic_tag_issues(
    *,
    name: str,
    tags: Mapping[str, object],
) -> list[SemanticTagIssue]:
    issues: list[SemanticTagIssue] = []
    layer_issue = _semantic_layer_issue(name=name, tags=tags)
    if layer_issue is not None:
        issues.append(layer_issue)
    kind = _semantic_kind(tags)
    issues.extend(_semantic_required_tag_issues(name=name, tags=tags, kind=kind))
    issues.extend(_semantic_table_tag_issues(name=name, tags=tags, kind=kind))
    return issues


def compile_semantic_registry(
    *,
    tag_query: TagQuery,
    schema_provider: SchemaProvider,
    version: str = "v1",
) -> CompiledSemanticRegistry:
    """Compile semantic registry from Hamilton tag metadata.

    Returns
    -------
    CompiledSemanticRegistry
        Compiled registry for semantic views.

    Raises
    ------
    SemanticTagValidationError
        If semantic tag validation fails.
    """
    variables = _semantic_variables(tag_query)
    issues: list[SemanticTagIssue] = []
    view_tags: dict[str, dict[str, str]] = {}

    for variable in variables:
        tags = _tags_from_variable(variable)
        if not tags:
            continue
        if _tag_str_value(tags, ht.TAG_MCP_VISIBLE) == "0":
            continue
        if not _is_semantic_output(tags):
            continue
        name = _variable_name(variable)
        issues.extend(_semantic_tag_issues(name=name, tags=tags))
        view_tags[name] = {str(key): _stringify_tag_value(val) for key, val in tags.items()}

    if issues:
        raise SemanticTagValidationError(issues)

    return compile_semantic_registry_from_views(
        schema_provider=schema_provider,
        view_tags=view_tags,
        version=version,
    )


__all__ = [
    "CompiledSemanticRegistry",
    "SemanticTagIssue",
    "SemanticTagValidationError",
    "compile_semantic_registry",
    "compile_semantic_registry_from_views",
]
