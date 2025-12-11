"""Shared helpers for analytics plugin metadata."""

from __future__ import annotations

from typing import cast

from codeintel.core.plugins.types.metadata import CorePluginMetadata
from codeintel.core.plugins.types.protocol import PluginKind, PluginMetadata, PluginStage

__all__ = [
    "to_plugin_metadata",
]


def to_plugin_metadata(core: CorePluginMetadata) -> PluginMetadata:
    """Convert CorePluginMetadata to the protocol-friendly PluginMetadata.

    Returns
    -------
    PluginMetadata
        Protocol-compatible metadata instance for registry consumers.
    """
    return PluginMetadata(
        name=core.name,
        version=core.version,
        description=core.description,
        kind=cast("PluginKind", core.kind),
        stage=cast("PluginStage", core.stage or "other"),
        provides=core.provides,
        requires=core.requires,
        produces_tables=core.produces_tables,
    )
