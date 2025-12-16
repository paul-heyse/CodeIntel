"""Discovery helpers for tagged Ibis view builders.

This module replaces the legacy global view registry with deterministic
Hamilton tag discovery over a set of modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from hamilton import driver

from codeintel.hamilton import tags as ht

if TYPE_CHECKING:
    from types import ModuleType

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


def _discover_by_output_kind(
    dr: driver.Driver,
    *,
    modules: tuple[ModuleType, ...],
    output_kind: str,
) -> list[DiscoveredViewBuilder]:
    variables = dr.list_available_variables(tag_filter={ht.TAG_OUTPUT_KIND: output_kind})
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
        builder = _find_callable(modules, node_name)
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
    modules: tuple[ModuleType, ...],
    config: dict[str, Any] | None = None,
) -> tuple[DiscoveredViewBuilder, ...]:
    """Discover view builders from Hamilton tags.

    Parameters
    ----------
    modules
        Modules to scan.
    config
        Optional Hamilton config (used only for graph construction).

    Returns
    -------
    tuple[DiscoveredViewBuilder, ...]
        Discovered builders, sorted deterministically by table_key.
    """
    dr = driver.Driver(config or {}, *modules)
    discovered = _discover_by_output_kind(
        dr,
        modules=modules,
        output_kind=ht.OUTPUT_KIND_VIEW,
    ) + _discover_by_output_kind(
        dr,
        modules=modules,
        output_kind=ht.OUTPUT_KIND_SEMANTIC_VIEW,
    )

    # De-duplicate by table_key (prefer later modules in the input list).
    by_table: dict[str, DiscoveredViewBuilder] = {d.table_key: d for d in discovered}
    ordered = sorted(by_table.values(), key=lambda d: d.table_key)
    return tuple(ordered)


__all__ = ["DiscoveredViewBuilder", "discover_view_builders"]
