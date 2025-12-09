"""Plugin discovery and enumeration.

Provide utilities for discovering plugins in standard locations,
supporting both manifest-based and legacy plugin formats.
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
    is_legacy
        Whether this is a legacy format plugin.
    """

    path: Path
    manifest: PluginManifest
    valid: bool
    errors: list[str]
    is_legacy: bool = False


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
                # Check for legacy plugin (bare .py file)
                if item.suffix == ".py" and not item.name.startswith("_"):
                    legacy = _check_legacy_plugin_file(item)
                    if legacy:
                        discovered.append(legacy)
                continue

            manifest_path = item / "plugin.json"
            if not manifest_path.exists():
                # Check for legacy plugin (directory without manifest)
                legacy = _check_legacy_plugin_dir(item)
                if legacy:
                    discovered.append(legacy)
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


def _check_legacy_plugin_file(path: Path) -> DiscoveredPlugin | None:
    """Check for legacy plugin format (single .py file with create_plugin).

    Parameters
    ----------
    path
        Path to Python file.

    Returns
    -------
    DiscoveredPlugin | None
        A DiscoveredPlugin with warnings about migration, or None.
    """
    try:
        content = path.read_text(encoding="utf-8")
        if "def create_plugin" in content:
            return DiscoveredPlugin(
                path=path,
                manifest=PluginManifest(
                    name=path.stem,
                    version="0.0.0",
                    api_version="0.0.0",
                    description="Legacy plugin (needs migration)",
                    entry_point=path.stem,
                ),
                valid=False,
                errors=[
                    "Legacy plugin format detected (single .py file). "
                    "Please migrate to manifest-based format with plugin.json. "
                    "See: https://docs.codeintel.dev/plugins/migration",
                ],
                is_legacy=True,
            )
    except OSError:
        pass
    return None


def _check_legacy_plugin_dir(path: Path) -> DiscoveredPlugin | None:
    """Check for legacy plugin format (directory with create_plugin).

    Parameters
    ----------
    path
        Path to plugin directory.

    Returns
    -------
    DiscoveredPlugin | None
        A DiscoveredPlugin with warnings about migration, or None.
    """
    # Look for .py files with create_plugin function
    for py_file in path.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            if "def create_plugin" in content:
                return DiscoveredPlugin(
                    path=path,
                    manifest=PluginManifest(
                        name=path.name,
                        version="0.0.0",
                        api_version="0.0.0",
                        description="Legacy plugin (needs migration)",
                        entry_point=py_file.stem,
                    ),
                    valid=False,
                    errors=[
                        "Legacy plugin format detected (missing plugin.json). "
                        "Please add plugin.json manifest and migrate to register() API. "
                        "See: https://docs.codeintel.dev/plugins/migration",
                    ],
                    is_legacy=True,
                )
        except OSError:
            continue
    return None


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
