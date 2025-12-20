"""Hamilton tag index for build metadata discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hamilton.driver import Driver

from codeintel.core.hamilton import tags as ht

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from types import ModuleType

    from hamilton.node import Node

    from codeintel.build.hamilton.driver_factory import HamiltonRuntime


def _stringify_tag_value(value: object) -> str:
    """Normalize tag values to strings for storage.

    Parameters
    ----------
    value
        Raw tag value from Hamilton metadata.

    Returns
    -------
    str
        Normalized string representation.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, float, int)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v is not None)
    return str(value)


def _node_tags(node: Node) -> dict[str, str]:
    """Return normalized tags for a Hamilton node.

    Parameters
    ----------
    node
        Hamilton node instance.

    Returns
    -------
    dict[str, str]
        Normalized tag mapping.
    """
    tags_raw = node.tags
    if not isinstance(tags_raw, dict):
        return {}
    return {str(k): _stringify_tag_value(v) for k, v in tags_raw.items()}


@dataclass(frozen=True, slots=True)
class TagIndex:
    """Index of Hamilton node tags."""

    tags_by_node: Mapping[str, dict[str, str]]

    @classmethod
    def from_runtime(cls, runtime: HamiltonRuntime) -> TagIndex:
        """Build a tag index from an active Hamilton runtime.

        Parameters
        ----------
        runtime
            Hamilton runtime with an initialized graph.

        Returns
        -------
        TagIndex
            Tag index built from runtime nodes.
        """
        tags_by_node: dict[str, dict[str, str]] = {}
        for node_name, node in runtime.dr.graph.nodes.items():
            tags_by_node[node_name] = _node_tags(node)
        return cls(tags_by_node=tags_by_node)

    @classmethod
    def from_modules(
        cls,
        modules: Iterable[ModuleType],
        *,
        config: dict[str, Any] | None = None,
    ) -> TagIndex:
        """Build a tag index from Hamilton modules.

        Parameters
        ----------
        modules
            Modules containing Hamilton nodes.
        config
            Optional configuration passed into the Hamilton Driver.

        Returns
        -------
        TagIndex
            Tag index built from module nodes.
        """
        driver = Driver(config or {}, *modules)
        tags_by_node: dict[str, dict[str, str]] = {}
        for node_name, node in driver.graph.nodes.items():
            tags_by_node[node_name] = _node_tags(node)
        return cls(tags_by_node=tags_by_node)

    def semantic_view_tags(self) -> dict[str, dict[str, str]]:
        """Return semantic view tags keyed by table_key.

        Returns
        -------
        dict[str, dict[str, str]]
            Tag mappings for semantic view table keys.
        """
        result: dict[str, dict[str, str]] = {}
        for tags in self.tags_by_node.values():
            output_kind = tags.get(ht.TAG_OUTPUT_KIND)
            if output_kind not in {ht.OUTPUT_KIND_SEMANTIC_VIEW, "semantic"}:
                continue
            if tags.get(ht.TAG_MCP_VISIBLE, "1") != "1":
                continue
            table_key = tags.get(ht.TAG_TABLE_KEY)
            if not table_key:
                continue
            result[table_key] = dict(tags)
        return result

    def dataset_nodes(self) -> dict[str, dict[str, str]]:
        """Return dataset nodes keyed by node name.

        Returns
        -------
        dict[str, dict[str, str]]
            Node tag mappings for dataset nodes.
        """
        return {
            name: tags
            for name, tags in self.tags_by_node.items()
            if tags.get(ht.TAG_NODE_TYPE) == ht.NODE_TYPE_DATASET
        }

    def artifact_nodes(self) -> dict[str, dict[str, str]]:
        """Return artifact nodes keyed by node name.

        Returns
        -------
        dict[str, dict[str, str]]
            Node tag mappings for artifact nodes.
        """
        return {
            name: tags
            for name, tags in self.tags_by_node.items()
            if tags.get(ht.TAG_NODE_TYPE) == ht.NODE_TYPE_ARTIFACT
        }


__all__ = ["TagIndex"]
