"""Output inventory types for build targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class OutputInventory:
    """Derived output inventory for build targets."""

    datasets_by_target: Mapping[str, tuple[str, ...]]
    artifacts_by_target: Mapping[str, tuple[str, ...]]
    artifact_templates_by_target: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

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

    def artifact_templates_for(self, target_name: str) -> Mapping[str, str]:
        """Return artifact path templates for a target.

        Parameters
        ----------
        target_name
            Target name to query.

        Returns
        -------
        Mapping[str, str]
            Mapping of artifact name to path template.
        """
        return self.artifact_templates_by_target.get(target_name, {})

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
