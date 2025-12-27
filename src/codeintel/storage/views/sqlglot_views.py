"""SQLGlot-defined view builders generated from precompiled SQL."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from hamilton.function_modifiers import tag as h_tag
from sqlglot import exp, parse_one

from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.semantic_tags import SEMANTIC_VIEW_TAG_ATTR

if TYPE_CHECKING:
    from typing import Final


class ViewSqlSpec(TypedDict):
    """Serialized view definition used for SQLGlot reconstruction."""

    node_name: str
    sql: str
    tags: dict[str, str]


def _load_view_map() -> dict[str, ViewSqlSpec]:
    path = Path(__file__).with_name("view_sql_map.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "view_sql_map.json must contain a mapping"
        raise ValueError(msg)
    view_map: dict[str, ViewSqlSpec] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        node_name = value.get("node_name")
        sql = value.get("sql")
        tags = value.get("tags")
        if not isinstance(node_name, str) or not isinstance(sql, str) or not isinstance(tags, dict):
            continue
        tag_map = {str(tag_key): str(tag_value) for tag_key, tag_value in tags.items()}
        view_map[key] = {"node_name": node_name, "sql": sql, "tags": tag_map}
    return view_map


_VIEW_SQL_MAP: Final[dict[str, ViewSqlSpec]] = _load_view_map()


def _build_view_builder(
    *,
    table_key: str,
    spec: ViewSqlSpec,
) -> Callable[[], exp.Expression]:
    sql = spec["sql"]
    tags = spec["tags"]
    node_name = spec["node_name"]

    def builder() -> exp.Expression:
        return parse_one(sql, read="duckdb")

    builder.__name__ = node_name
    builder.__module__ = __name__
    builder.__doc__ = f"Build a SQLGlot expression for {table_key}."

    tagged = cast("Callable[[], exp.Expression]", h_tag(**tags)(builder))
    if tags.get(ht.TAG_OUTPUT_KIND) == ht.OUTPUT_KIND_SEMANTIC_VIEW:
        setattr(tagged, SEMANTIC_VIEW_TAG_ATTR, dict(tags))
    return tagged


_BUILDERS: dict[str, Callable[[], exp.Expression]] = {}
for table_key, spec in _VIEW_SQL_MAP.items():
    fn = _build_view_builder(table_key=table_key, spec=spec)
    globals()[fn.__name__] = fn
    _BUILDERS[table_key] = fn

__all__ = sorted(fn.__name__ for fn in _BUILDERS.values())
