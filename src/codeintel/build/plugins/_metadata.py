"""Shared helpers for plugin metadata conversion.

This module provides the canonical implementation of metadata conversion
used by all plugins. Previously duplicated across 16+ plugin files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.core.plugins.types.protocol import PluginMetadata

if TYPE_CHECKING:
    from codeintel.core.plugins.types.metadata import CorePluginMetadata
    from codeintel.core.plugins.types.protocol import PluginKind, PluginStage

__all__ = [
    "to_plugin_metadata",
]


def to_plugin_metadata(core: CorePluginMetadata) -> PluginMetadata:
    """Convert CorePluginMetadata to the protocol-friendly PluginMetadata.

    This is the canonical conversion function used by all plugins to
    transform their internal CorePluginMetadata into the protocol-compatible
    PluginMetadata format.

    Parameters
    ----------
    core
        The core plugin metadata containing all plugin information.

    Returns
    -------
    PluginMetadata
        Protocol-compatible metadata instance for registry consumers.

    Examples
    --------
    >>> from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
    >>> core = CorePluginMetadata(
    ...     name="test",
    ...     version="1.0.0",
    ...     description="Test plugin",
    ...     domain=PluginDomain.ANALYTICS,
    ...     kind="compute",
    ... )
    >>> meta = to_plugin_metadata(core)
    >>> meta.name
    'test'
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
