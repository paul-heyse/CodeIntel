"""Internal mapping utilities for generated support modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Protocol, cast


class _SupportModule(Protocol):
    """Protocol for support modules that receive generated mapping attributes."""

    TARGET_TO_NODE: dict[str, str]
    DATASET_TO_NODE: dict[str, str]
    QUERY_TO_NODE: dict[str, str]
    DATAFRAME_TO_NODE: dict[str, str]
    ARTIFACT_TO_NODE: dict[str, str]


@dataclass
class SupportNodeMappings:
    """Collect node-name mappings for a generated support module."""

    target_to_node: dict[str, str] = field(default_factory=dict)
    dataset_to_node: dict[str, str] = field(default_factory=dict)
    query_to_node: dict[str, str] = field(default_factory=dict)
    dataframe_to_node: dict[str, str] = field(default_factory=dict)
    artifact_to_node: dict[str, str] = field(default_factory=dict)

    def attach_to(self, module: ModuleType) -> None:
        """Attach mapping dicts to the support module."""
        mod = cast("_SupportModule", module)
        mod.TARGET_TO_NODE = self.target_to_node
        mod.DATASET_TO_NODE = self.dataset_to_node
        mod.QUERY_TO_NODE = self.query_to_node
        mod.DATAFRAME_TO_NODE = self.dataframe_to_node
        mod.ARTIFACT_TO_NODE = self.artifact_to_node


__all__ = ["SupportNodeMappings"]
