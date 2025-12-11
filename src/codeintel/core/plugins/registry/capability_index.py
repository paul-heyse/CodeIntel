"""Capability-based plugin registry index."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from codeintel.core.plugins.types.metadata import CorePluginMetadata


@dataclass(frozen=True)
class PluginRegistryIndex:
    """Index for looking up plugins by name, capability, or output table."""

    by_name: dict[str, CorePluginMetadata]
    by_capability: dict[str, CorePluginMetadata]
    by_output_table: dict[str, CorePluginMetadata]

    def get_by_name(self, name: str) -> CorePluginMetadata | None:
        """Look up plugin metadata by canonical name.

        Returns
        -------
        CorePluginMetadata | None
            Metadata when found, otherwise None.
        """
        return self.by_name.get(name)

    def get_provider(self, capability: str) -> CorePluginMetadata | None:
        """Look up the provider of a capability.

        Returns
        -------
        CorePluginMetadata | None
            Provider metadata when found, otherwise None.
        """
        return self.by_capability.get(capability)

    def get_producer(self, table: str) -> CorePluginMetadata | None:
        """Look up the producer of a table.

        Returns
        -------
        CorePluginMetadata | None
            Producer metadata when found, otherwise None.
        """
        return self.by_output_table.get(table)

    def provider_lookup(self) -> dict[str, str]:
        """Build a capability → provider name lookup.

        Returns
        -------
        dict[str, str]
            Mapping from capability name to provider plugin name.
        """
        return {capability: meta.name for capability, meta in self.by_capability.items()}

    def all_capabilities(self) -> tuple[str, ...]:
        """Return all registered capabilities.

        Returns
        -------
        tuple[str, ...]
            All capability names known to the registry.
        """
        return tuple(self.by_capability.keys())

    def all_tables(self) -> tuple[str, ...]:
        """Return all registered output tables.

        Returns
        -------
        tuple[str, ...]
            All table names produced by registered plugins.
        """
        return tuple(self.by_output_table.keys())


def build_registry_index(
    all_metadata: Iterable[CorePluginMetadata],
) -> PluginRegistryIndex:
    """Build a registry index from plugin metadata.

    Returns
    -------
    PluginRegistryIndex
        Populated registry index.
    """
    by_name: dict[str, CorePluginMetadata] = {}
    by_capability: dict[str, CorePluginMetadata] = {}
    by_output_table: dict[str, CorePluginMetadata] = {}

    for metadata in all_metadata:
        by_name[metadata.name] = metadata

        for capability in metadata.provides:
            by_capability[capability] = metadata

        for table in metadata.produces_tables:
            by_output_table[table] = metadata

    return PluginRegistryIndex(
        by_name=by_name,
        by_capability=by_capability,
        by_output_table=by_output_table,
    )


__all__ = [
    "PluginRegistryIndex",
    "build_registry_index",
]
