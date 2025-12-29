"""SQLGlot-defined view builders generated from serialized SQLGlot ASTs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from sqlglot import exp

from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.semantic_tags import SEMANTIC_VIEW_TAG_ATTR
from codeintel.core.hamilton.tagging_helpers import apply_raw_tags

if TYPE_CHECKING:
    from typing import Final


class ViewAstSpec(TypedDict):
    """Serialized view definition used for SQLGlot reconstruction."""

    node_name: str
    ast: list[object]
    tags: dict[str, str]


def _load_view_map() -> dict[str, ViewAstSpec]:
    path = Path(__file__).with_name("view_ast_map.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "view_ast_map.json must contain a mapping"
        raise TypeError(msg)
    view_map: dict[str, ViewAstSpec] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        node_name = value.get("node_name")
        ast = value.get("ast")
        tags = value.get("tags")
        if (
            not isinstance(node_name, str)
            or not isinstance(ast, list)
            or not isinstance(tags, dict)
        ):
            continue
        tag_map = {str(tag_key): str(tag_value) for tag_key, tag_value in tags.items()}
        view_map[key] = {"node_name": node_name, "ast": ast, "tags": tag_map}
    return view_map


_VIEW_AST_MAP: Final[dict[str, ViewAstSpec]] = _load_view_map()


def _apply_tags(
    builder: Callable[[], exp.Expression],
    tags: dict[str, str],
) -> Callable[[], exp.Expression]:
    bypass_value = tags.get("bypass_reserved_namespaces_")
    bypass_reserved = _parse_bool_tag(bypass_value)
    filtered = {key: value for key, value in tags.items() if key != "bypass_reserved_namespaces_"}
    tagged = apply_raw_tags(builder, tags=filtered)
    if bypass_reserved:
        tagged = apply_raw_tags(tagged, tags={"bypass_reserved_namespaces_": True})
    return tagged


def _parse_bool_tag(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "y"}


def _build_view_builder(
    *,
    table_key: str,
    spec: ViewAstSpec,
) -> Callable[[], exp.Expression]:
    ast = spec["ast"]
    tags = spec["tags"]
    node_name = spec["node_name"]

    def builder() -> exp.Expression:
        return exp.Expression.load(ast)

    builder.__name__ = node_name
    builder.__module__ = __name__
    builder.__doc__ = f"Build a SQLGlot expression for {table_key}."

    tagged = _apply_tags(builder, tags)
    if tags.get(ht.TAG_OUTPUT_KIND) == ht.OUTPUT_KIND_SEMANTIC_VIEW:
        setattr(tagged, SEMANTIC_VIEW_TAG_ATTR, dict(tags))
    return tagged


_BUILDERS: dict[str, Callable[[], exp.Expression]] = {}
fn: Callable[[], exp.Expression] | None = None
for table_key, spec in _VIEW_AST_MAP.items():
    fn = _build_view_builder(table_key=table_key, spec=spec)
    globals()[fn.__name__] = fn
    _BUILDERS[table_key] = fn
fn = None
