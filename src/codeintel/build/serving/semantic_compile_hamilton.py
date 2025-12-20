"""Compile semantic registry inputs via Hamilton tag discovery.

This module discovers semantic view nodes using Hamilton tags rather than any
bespoke registry or import side effects. It is the canonical bridge between
tagged view-builder functions and the semantic registry compiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeintel.build.hamilton.tag_index import TagIndex
from codeintel.core.hamilton import tags as ht

if TYPE_CHECKING:
    from types import ModuleType


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
    tag_index = TagIndex.from_modules(modules, config=config)
    discovered: list[DiscoveredSemanticNode] = []
    for node_name, tags in tag_index.tags_by_node.items():
        output_kind = tags.get(ht.TAG_OUTPUT_KIND)
        if output_kind not in {ht.OUTPUT_KIND_SEMANTIC_VIEW, "semantic"}:
            continue
        table_key = tags.get(ht.TAG_TABLE_KEY)
        if not table_key:
            continue
        discovered.append(
            DiscoveredSemanticNode(
                node_name=node_name,
                table_key=table_key,
                tags=dict(tags),
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

    Collects semantic tag mappings using Hamilton tag discovery.

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
    tag_index = TagIndex.from_modules(modules, config=config)
    return tag_index.semantic_view_tags()


__all__ = [
    "DiscoveredSemanticNode",
    "collect_semantic_view_tags_from_hamilton",
    "discover_semantic_nodes",
]
