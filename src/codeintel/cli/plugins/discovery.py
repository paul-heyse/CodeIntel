"""Plugin discovery and enumeration.

Provide utilities for discovering plugins in standard locations
using the manifest-based plugin format.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from codeintel.cli.plugins.manifest import PluginManifest

LOG = logging.getLogger(__name__)

DEFAULT_PLUGIN_PATHS = [
    Path.home() / ".codeintel" / "plugins",
    Path("/etc/codeintel/plugins"),
]


@dataclass
class DiscoveredPlugin:
    """Information about a discovered plugin.

    Parameters
    ----------
    path
        Path to plugin directory.
    manifest
        Plugin manifest.
    valid
        Whether plugin passed validation.
    errors
        Validation errors.
    """

    path: Path
    manifest: PluginManifest
    valid: bool
    errors: list[str]


def discover_plugins(
    search_paths: list[Path] | None = None,
) -> list[DiscoveredPlugin]:
    """Discover plugins in search paths.

    Parameters
    ----------
    search_paths
        Paths to search for plugins. Uses defaults if None.

    Returns
    -------
    list[DiscoveredPlugin]
        List of discovered plugins.
    """
    paths = search_paths or DEFAULT_PLUGIN_PATHS
    discovered: list[DiscoveredPlugin] = []

    for search_path in paths:
        if not search_path.exists():
            continue

        for item in search_path.iterdir():
            if not item.is_dir():
                continue

            manifest_path = item / "plugin.json"
            if not manifest_path.exists():
                LOG.debug("Skipping %s: no plugin.json manifest", item)
                continue

            try:
                manifest = PluginManifest.load(manifest_path)
                errors = manifest.validate()
                discovered.append(
                    DiscoveredPlugin(
                        path=item,
                        manifest=manifest,
                        valid=len(errors) == 0,
                        errors=errors,
                    ),
                )
            except (OSError, ValueError) as e:
                LOG.warning("Failed to load manifest %s: %s", manifest_path, e)
                discovered.append(
                    DiscoveredPlugin(
                        path=item,
                        manifest=PluginManifest(
                            name=item.name,
                            version="0.0.0",
                            api_version="0.0.0",
                        ),
                        valid=False,
                        errors=[str(e)],
                    ),
                )

    return discovered


def get_default_plugin_paths() -> list[Path]:
    """Get default plugin search paths.

    Returns
    -------
    list[Path]
        Default plugin search paths.
    """
    return list(DEFAULT_PLUGIN_PATHS)


__all__ = [
    "DEFAULT_PLUGIN_PATHS",
    "DiscoveredPlugin",
    "discover_plugins",
    "get_default_plugin_paths",
]
