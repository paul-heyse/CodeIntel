"""Precompiled view builders generated from serialized SQL plan specs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.semantic_tags import SEMANTIC_VIEW_TAG_ATTR
from codeintel.core.hamilton.tagging_helpers import apply_raw_tags

if TYPE_CHECKING:
    from typing import Final


class ViewPlanSpec(TypedDict):
    """Serialized view plan definition."""

    node_name: str
    sql: str
    dependencies: list[str]
    tags: dict[str, str]


def _load_view_map() -> dict[str, ViewPlanSpec]:
    path = Path(__file__).with_name("view_plan_map.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "view_plan_map.json must contain a mapping"
        raise TypeError(msg)
    view_map: dict[str, ViewPlanSpec] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        node_name = value.get("node_name")
        sql = value.get("sql")
        dependencies = value.get("dependencies")
        tags = value.get("tags")
        if (
            not isinstance(node_name, str)
            or not isinstance(sql, str)
            or not isinstance(dependencies, list)
            or not isinstance(tags, dict)
        ):
            continue
        dep_list = [str(dep) for dep in dependencies]
        tag_map = {str(tag_key): str(tag_value) for tag_key, tag_value in tags.items()}
        view_map[key] = {
            "node_name": node_name,
            "sql": sql,
            "dependencies": dep_list,
            "tags": tag_map,
        }
    return view_map


_VIEW_PLAN_MAP: Final[dict[str, ViewPlanSpec]] = _load_view_map()


def view_plan_map() -> dict[str, ViewPlanSpec]:
    """Return a copy of the precompiled view plan map.

    Returns
    -------
    dict[str, ViewPlanSpec]
        Copy of the view plan mapping keyed by table_key.
    """
    return {
        key: {
            "node_name": spec["node_name"],
            "sql": spec["sql"],
            "dependencies": list(spec["dependencies"]),
            "tags": dict(spec["tags"]),
        }
        for key, spec in _VIEW_PLAN_MAP.items()
    }


__all__ = ["view_plan_map"]


def _apply_tags(
    builder: Callable[[], str],
    tags: dict[str, str],
) -> Callable[[], str]:
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


def _build_view_builder(*, table_key: str, spec: ViewPlanSpec) -> Callable[[], str]:
    sql = spec["sql"]
    tags = spec["tags"]
    node_name = spec["node_name"]

    def builder() -> str:
        return sql

    builder.__name__ = node_name
    builder.__module__ = __name__
    builder.__doc__ = f"Return precompiled SQL for {table_key}."

    tagged = _apply_tags(builder, tags)
    if tags.get(ht.TAG_OUTPUT_KIND) == ht.OUTPUT_KIND_SEMANTIC_VIEW:
        setattr(tagged, SEMANTIC_VIEW_TAG_ATTR, dict(tags))
    return tagged


_BUILDERS: dict[str, Callable[[], str]] = {}
fn: Callable[[], str] | None = None
for table_key, spec in _VIEW_PLAN_MAP.items():
    fn = _build_view_builder(table_key=table_key, spec=spec)
    globals()[fn.__name__] = fn
    _BUILDERS[table_key] = fn
fn = None
