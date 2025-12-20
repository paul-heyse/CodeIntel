"""Canonical target metadata service for build target access."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.hamilton.introspect import derive_target_outputs
from codeintel.build.hamilton.tag_index import TagIndex
from codeintel.build.target_system import TargetSystem, load_target_system

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.targets import OutputTarget


@dataclass(frozen=True, slots=True)
class OutputInventory:
    """Derived output inventory for build targets."""

    datasets_by_target: Mapping[str, tuple[str, ...]]
    artifacts_by_target: Mapping[str, tuple[str, ...]]

    def datasets_for(self, target_name: str) -> tuple[str, ...]:
        """Return dataset table keys for a target.

        Parameters
        ----------
        target_name
            Target name to query.

        Returns
        -------
        tuple[str, ...]
            Dataset table keys for the target.
        """
        return self.datasets_by_target.get(target_name, ())

    def artifacts_for(self, target_name: str) -> tuple[str, ...]:
        """Return artifact names for a target.

        Parameters
        ----------
        target_name
            Target name to query.

        Returns
        -------
        tuple[str, ...]
            Artifact names for the target.
        """
        return self.artifacts_by_target.get(target_name, ())

    @property
    def all_dataset_keys(self) -> frozenset[str]:
        """Return all dataset table keys across targets.

        Returns
        -------
        frozenset[str]
            Unique dataset table keys.
        """
        return frozenset(key for keys in self.datasets_by_target.values() for key in keys)

    @property
    def all_artifact_names(self) -> frozenset[str]:
        """Return all artifact names across targets.

        Returns
        -------
        frozenset[str]
            Unique artifact names.
        """
        return frozenset(name for names in self.artifacts_by_target.values() for name in names)


@dataclass(frozen=True, slots=True)
class TargetMetadataService:
    """Bundle of target system, outputs, and tag index."""

    system: TargetSystem
    outputs: OutputInventory
    tag_index: TagIndex

    def get_target(self, name: str) -> OutputTarget | None:
        """Return target metadata by name.

        Parameters
        ----------
        name
            Target name to resolve.

        Returns
        -------
        OutputTarget | None
            Target metadata if found.
        """
        return self.system.get_target(name)

    def target_for_table_key(self, table_key: str) -> OutputTarget | None:
        """Return the target that produces a dataset table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        OutputTarget | None
            Target metadata if found.
        """
        return self.system.target_for_table_key(table_key)

    def target_for_artifact(self, artifact_name: str) -> OutputTarget | None:
        """Return the target that produces an artifact name.

        Parameters
        ----------
        artifact_name
            Artifact name to resolve.

        Returns
        -------
        OutputTarget | None
            Target metadata if found.
        """
        return self.system.target_for_artifact(artifact_name)


@lru_cache(maxsize=1)
def get_target_metadata_service() -> TargetMetadataService:
    """Return the canonical target metadata service.

    Returns
    -------
    TargetMetadataService
        Singleton target metadata service.
    """
    system = load_target_system()
    derived = derive_target_outputs(system.runtime)
    inventory = OutputInventory(
        datasets_by_target=derived.datasets_by_target,
        artifacts_by_target=derived.artifacts_by_target,
    )
    tag_index = TagIndex.from_runtime(system.runtime)
    return TargetMetadataService(system=system, outputs=inventory, tag_index=tag_index)


__all__ = [
    "OutputInventory",
    "TargetMetadataService",
    "get_target_metadata_service",
]
