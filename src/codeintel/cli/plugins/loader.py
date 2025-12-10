"""Unified plugin loading.

Provide a unified interface for loading manifest-based
plugins with optional sandboxing.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from codeintel.cli.plugins.discovery import DiscoveredPlugin, discover_plugins
from codeintel.cli.plugins.manifest import PluginCapability, PluginManifest
from codeintel.cli.plugins.sandbox import PluginSandbox, SandboxConfig

LOG = logging.getLogger(__name__)


@dataclass
class LoadedPlugin:
    """A loaded plugin ready for registration.

    Parameters
    ----------
    manifest
        Plugin manifest.
    module
        Loaded Python module.
    path
        Plugin path.
    """

    manifest: PluginManifest
    module: ModuleType
    path: Path

    def has_capability(self, cap: PluginCapability) -> bool:
        """Check if plugin has capability.

        Parameters
        ----------
        cap
            Capability to check.

        Returns
        -------
        bool
            True if plugin has capability.
        """
        return cap in self.manifest.capabilities


@dataclass
class PluginLoadResult:
    """Result of loading plugins.

    Parameters
    ----------
    loaded
        Successfully loaded plugins.
    failed
        Failed plugin paths with error messages.
    """

    loaded: list[LoadedPlugin] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)


class PluginLoader:
    """Load and initialize plugins.

    Parameters
    ----------
    sandbox_enabled
        Enable sandboxing for plugins.
    allowed_capabilities
        Additional capabilities to allow if not in manifest.
    """

    def __init__(
        self,
        *,
        sandbox_enabled: bool = True,
        allowed_capabilities: set[PluginCapability] | None = None,
    ) -> None:
        """Initialize plugin loader."""
        self._sandbox_enabled = sandbox_enabled
        self._allowed_caps = allowed_capabilities or set()

    def load_all(
        self,
        search_paths: list[Path] | None = None,
    ) -> PluginLoadResult:
        """Discover and load all plugins.

        Parameters
        ----------
        search_paths
            Paths to search for plugins.

        Returns
        -------
        PluginLoadResult
            Results of loading.
        """
        result = PluginLoadResult()

        discovered = discover_plugins(search_paths)

        for plugin in discovered:
            if not plugin.valid:
                result.failed.append((plugin.path, "; ".join(plugin.errors)))
                continue

            try:
                loaded = self._load_plugin(plugin)
                result.loaded.append(loaded)
            except (ImportError, OSError, RuntimeError) as e:
                LOG.exception("Failed to load plugin %s", plugin.path)
                result.failed.append((plugin.path, str(e)))

        return result

    def load_single(
        self,
        path: Path,
    ) -> LoadedPlugin:
        """Load a single plugin.

        Parameters
        ----------
        path
            Path to plugin directory.

        Returns
        -------
        LoadedPlugin
            Loaded plugin.

        Raises
        ------
        ValueError
            If plugin is invalid or cannot be loaded.
        """
        if not path.is_dir():
            msg = f"Plugin path must be a directory: {path}"
            raise ValueError(msg)

        manifest_path = path / "plugin.json"

        if not manifest_path.exists():
            msg = f"No plugin.json manifest found at {path}"
            raise ValueError(msg)

        manifest = PluginManifest.load(manifest_path)
        errors = manifest.validate()
        if errors:
            msg = f"Invalid manifest: {'; '.join(errors)}"
            raise ValueError(msg)

        discovered = DiscoveredPlugin(
            path=path,
            manifest=manifest,
            valid=True,
            errors=[],
        )
        return self._load_plugin(discovered)

    def _load_plugin(self, discovered: DiscoveredPlugin) -> LoadedPlugin:
        """Load a manifest-based plugin.

        Parameters
        ----------
        discovered
            Discovered plugin info.

        Returns
        -------
        LoadedPlugin
            Loaded plugin.
        """
        manifest = discovered.manifest

        if self._sandbox_enabled:
            config = SandboxConfig(
                allowed_capabilities=set(manifest.capabilities) | self._allowed_caps,
            )
            with PluginSandbox(manifest, config) as sandbox:
                module = sandbox.load_plugin()
        else:
            module = importlib.import_module(manifest.entry_point)

        return LoadedPlugin(
            manifest=manifest,
            module=module,
            path=discovered.path,
        )


def get_plugin_loader(
    *,
    sandbox_enabled: bool = True,
) -> PluginLoader:
    """Get a plugin loader instance.

    Parameters
    ----------
    sandbox_enabled
        Enable sandboxing for plugins.

    Returns
    -------
    PluginLoader
        Plugin loader instance.
    """
    return PluginLoader(sandbox_enabled=sandbox_enabled)


__all__ = [
    "LoadedPlugin",
    "PluginLoadResult",
    "PluginLoader",
    "get_plugin_loader",
]
