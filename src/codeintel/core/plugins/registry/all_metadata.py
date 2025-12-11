"""Central registry of all plugin metadata."""

from __future__ import annotations

from codeintel.analytics.plugins.functions.metrics import FUNCTION_METRICS_METADATA
from codeintel.analytics.plugins.types.coverage import TYPE_COVERAGE_METADATA
from codeintel.core.plugins.registry.capability_index import (
    PluginRegistryIndex,
    build_registry_index,
)
from codeintel.core.plugins.types.metadata import CorePluginMetadata
from codeintel.graphs.plugins.builders.callgraph import CALLGRAPH_METADATA
from codeintel.graphs.plugins.builders.import_graph import IMPORT_GRAPH_METADATA
from codeintel.ingestion.plugins.modules_plugin import MODULE_INGEST_METADATA
from codeintel.ingestion.plugins.scip_plugin import SCIP_INGEST_METADATA

ALL_PLUGIN_METADATA: tuple[CorePluginMetadata, ...] = (
    # Analytics
    FUNCTION_METRICS_METADATA,
    TYPE_COVERAGE_METADATA,
    # Graphs
    CALLGRAPH_METADATA,
    IMPORT_GRAPH_METADATA,
    # Ingestion
    MODULE_INGEST_METADATA,
    SCIP_INGEST_METADATA,
)

_GLOBAL_INDEX: PluginRegistryIndex | None = None


def get_global_registry_index() -> PluginRegistryIndex:
    """Return the global plugin registry index.

    Returns
    -------
    PluginRegistryIndex
        Registry index built from all plugin metadata.
    """
    global _GLOBAL_INDEX  # noqa: PLW0603
    if _GLOBAL_INDEX is None:
        _GLOBAL_INDEX = build_registry_index(ALL_PLUGIN_METADATA)
    return _GLOBAL_INDEX


def get_provider_lookup() -> dict[str, str]:
    """Return capability → provider name lookup.

    Returns
    -------
    dict[str, str]
        Mapping of capability name to provider plugin name.
    """
    return get_global_registry_index().provider_lookup()


__all__ = [
    "ALL_PLUGIN_METADATA",
    "get_global_registry_index",
    "get_provider_lookup",
]
