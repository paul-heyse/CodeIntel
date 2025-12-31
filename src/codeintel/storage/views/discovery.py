"""Discovery helpers for tagged view builders.

This module replaces the legacy global view registry with deterministic
Hamilton tag discovery over a set of modules.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.tag_query import TagQuery
from codeintel.core.hamilton.tagging_helpers import extract_tag_decorator_tags

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType

    from hamilton.driver import Driver

    from codeintel.storage.views.protocol import ViewBuilder


@dataclass(frozen=True, slots=True)
class DiscoveredViewBuilder:
    """A view builder discovered from Hamilton tags."""

    table_key: str
    node_name: str
    builder: ViewBuilder
    tags: dict[str, object]


def _find_callable(modules: tuple[ModuleType, ...], name: str) -> ViewBuilder | None:
    for mod in reversed(modules):
        value = getattr(mod, name, None)
        if value is None:
            continue
        if callable(value):
            return cast("ViewBuilder", value)
    return None


def _resolve_builder(
    *,
    dr: Driver | None,
    modules: tuple[ModuleType, ...] | None,
    node_name: str,
) -> ViewBuilder | None:
    if dr is not None:
        node = dr.graph.nodes.get(node_name)
        node_callable = getattr(node, "callable", None) if node is not None else None
        if callable(node_callable):
            return cast("ViewBuilder", node_callable)
    if modules:
        return _find_callable(modules, node_name)
    return None


def _output_kind_matches(value: object, expected: str) -> bool:
    if isinstance(value, str):
        return value == expected
    if isinstance(value, list):
        return expected in value
    return False


def _tags_from_callable(fn: object) -> dict[str, object]:
    decorators = getattr(fn, "decorate_nodes", None)
    if not isinstance(decorators, list):
        return {}
    tags: dict[str, object] = {}
    for decorator in decorators:
        decorator_tags = extract_tag_decorator_tags(decorator)
        if decorator_tags is not None:
            tags.update(decorator_tags)
    return tags


def _discover_from_modules(
    *,
    modules: tuple[ModuleType, ...],
) -> list[DiscoveredViewBuilder]:
    discovered: list[DiscoveredViewBuilder] = []
    for module in modules:
        for name, value in vars(module).items():
            if not inspect.isfunction(value):
                continue
            if value.__module__ != module.__name__:
                continue
            tags = _tags_from_callable(value)
            if not tags:
                continue
            output_kind = tags.get(ht.TAG_OUTPUT_KIND)
            if not (
                _output_kind_matches(output_kind, ht.OUTPUT_KIND_VIEW)
                or _output_kind_matches(output_kind, ht.OUTPUT_KIND_SEMANTIC_VIEW)
            ):
                continue
            table_key_raw = tags.get(ht.TAG_TABLE_KEY)
            if not isinstance(table_key_raw, str) or not table_key_raw:
                continue
            discovered.append(
                DiscoveredViewBuilder(
                    table_key=table_key_raw,
                    node_name=name,
                    builder=cast("ViewBuilder", value),
                    tags=dict(tags),
                )
            )
    return discovered


def _discover_by_output_kind(
    variables: Iterable[object],
    *,
    dr: Driver | None,
    modules: tuple[ModuleType, ...] | None,
) -> list[DiscoveredViewBuilder]:
    discovered: list[DiscoveredViewBuilder] = []

    for var in variables:
        tags = getattr(var, "tags", None)
        if not isinstance(tags, dict):
            continue
        table_key_raw = tags.get(ht.TAG_TABLE_KEY)
        if not isinstance(table_key_raw, str) or not table_key_raw:
            continue
        node_name = str(getattr(var, "name", ""))
        if not node_name:
            continue
        builder = _resolve_builder(dr=dr, modules=modules, node_name=node_name)
        if builder is None:
            continue
        discovered.append(
            DiscoveredViewBuilder(
                table_key=table_key_raw,
                node_name=node_name,
                builder=builder,
                tags=dict(tags),
            )
        )

    return discovered


def discover_view_builders(
    *,
    dr: Driver | None = None,
    tag_query: TagQuery | None = None,
    modules: tuple[ModuleType, ...] | None = None,
) -> tuple[DiscoveredViewBuilder, ...]:
    """Discover view builders from Hamilton tags.

    Parameters
    ----------
    dr
        Hamilton Driver used for tag discovery and callable resolution.
    tag_query
        Optional cached tag query helper (uses the underlying Driver).
    modules
        Optional modules to scan when graph callables are not available.

    Returns
    -------
    tuple[DiscoveredViewBuilder, ...]
        Discovered builders, sorted deterministically by table_key.

    Raises
    ------
    ValueError
        If neither a Driver nor a TagQuery is provided.
    """
    if dr is None and tag_query is None:
        if not modules:
            msg = "discover_view_builders requires a Driver, TagQuery, or modules"
            raise ValueError(msg)
        discovered = _discover_from_modules(modules=modules)
        by_table: dict[str, DiscoveredViewBuilder] = {d.table_key: d for d in discovered}
        ordered = sorted(by_table.values(), key=lambda d: d.table_key)
        return tuple(ordered)

    def _list(output_kind: str) -> Iterable[object]:
        tag_filter = {ht.TAG_OUTPUT_KIND: output_kind}
        if tag_query is not None:
            return tag_query.query(tag_filter)
        if dr is None:
            return ()
        return dr.list_available_variables(tag_filter=tag_filter)

    discovered = _discover_by_output_kind(
        _list(ht.OUTPUT_KIND_VIEW),
        dr=dr,
        modules=modules,
    ) + _discover_by_output_kind(
        _list(ht.OUTPUT_KIND_SEMANTIC_VIEW),
        dr=dr,
        modules=modules,
    )
    if modules:
        discovered.extend(_discover_from_modules(modules=modules))

    # De-duplicate by table_key (prefer later modules in the input list).
    by_table: dict[str, DiscoveredViewBuilder] = {d.table_key: d for d in discovered}
    ordered = sorted(by_table.values(), key=lambda d: d.table_key)
    return tuple(ordered)


__all__ = ["DiscoveredViewBuilder", "discover_view_builders"]
