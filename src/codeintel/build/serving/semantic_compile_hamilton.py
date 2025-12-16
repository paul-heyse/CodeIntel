"""Compile semantic registry inputs via Hamilton tag discovery.

This module discovers semantic view nodes using Hamilton tags rather than any
bespoke registry or import side effects. It is the canonical bridge between
tagged view-builder functions and the semantic registry compiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hamilton import driver

from codeintel.build.hamilton import tags as ht

if TYPE_CHECKING:
    from types import ModuleType


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


@dataclass(frozen=True, slots=True)
class DiscoveredSemanticNode:
    """Semantic node discovered from Hamilton tags."""

    node_name: str
    table_key: str
    tags: dict[str, str]


def discover_semantic_nodes(
    *,
    modules: tuple[ModuleType, ...],
    config: dict[str, Any] | None = None,
) -> tuple[DiscoveredSemanticNode, ...]:
    """Discover semantic view nodes from Hamilton tags.

    Parameters
    ----------
    modules
        Modules to scan for tagged node functions.
    config
        Optional Hamilton config dict (used only for graph construction).

    Returns
    -------
    tuple[DiscoveredSemanticNode, ...]
        Discovered semantic nodes, sorted deterministically by semantic_id then table_key.
    """
    dr = driver.Driver(config or {}, *modules)
    variables = dr.list_available_variables(
        tag_filter={ht.TAG_OUTPUT_KIND: ht.OUTPUT_KIND_SEMANTIC_VIEW},
    )

    discovered: list[DiscoveredSemanticNode] = []
    for var in variables:
        tags_raw = getattr(var, "tags", None)
        if not isinstance(tags_raw, dict):
            continue
        table_key_raw = tags_raw.get(ht.TAG_TABLE_KEY)
        if not isinstance(table_key_raw, str) or not table_key_raw:
            continue

        tags = {str(k): _stringify_tag_value(v) for k, v in tags_raw.items()}
        discovered.append(
            DiscoveredSemanticNode(
                node_name=str(getattr(var, "name", "")),
                table_key=table_key_raw,
                tags=tags,
            )
        )

    discovered.sort(
        key=lambda d: (
            d.tags.get(ht.TAG_SEMANTIC_ID, ""),
            d.table_key,
            d.node_name,
        )
    )
    return tuple(discovered)


def collect_semantic_view_tags_from_hamilton(
    *,
    modules: tuple[ModuleType, ...],
    config: dict[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    """Collect semantic tag mappings keyed by view table_key.

    This returns the same shape as the legacy `collect_semantic_view_tags()`
    helper, but uses Hamilton tag discovery instead of a view registry.

    Parameters
    ----------
    modules
        Modules to scan for tagged view-builder functions.
    config
        Optional Hamilton config dict (used only for graph construction).

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping from view table key to discovered tag values.
    """
    nodes = discover_semantic_nodes(modules=modules, config=config)
    return {n.table_key: n.tags for n in nodes}


__all__ = [
    "DiscoveredSemanticNode",
    "collect_semantic_view_tags_from_hamilton",
    "discover_semantic_nodes",
]
