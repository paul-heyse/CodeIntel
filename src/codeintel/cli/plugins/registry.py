"""Plugin registration with operation registry.

Provide utilities for registering operations from loaded plugins
using the register() API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeintel.cli.execution.registry import OperationSpec
from codeintel.cli.introspection import OperationRegistry, get_registry
from codeintel.cli.plugins.discovery import DEFAULT_PLUGIN_PATHS
from codeintel.cli.plugins.loader import LoadedPlugin, PluginLoader, PluginLoadResult
from codeintel.cli.plugins.manifest import PluginCapability
from codeintel.core.singleton import SingletonHolder

LOG = logging.getLogger(__name__)


def register_plugin_operations(
    plugin: LoadedPlugin,
    registry: OperationRegistry | None = None,
) -> int:
    """Register operations from a plugin.

    Parameters
    ----------
    plugin
        Loaded plugin.
    registry
        Operation registry (uses global if None).

    Returns
    -------
    int
        Number of operations registered.
    """
    reg = registry or get_registry()

    if not plugin.has_capability(PluginCapability.REGISTER_OPERATIONS):
        LOG.warning(
            "Plugin %s does not have REGISTER_OPERATIONS capability",
            plugin.manifest.name,
        )
        return 0

    # Check for register() function
    if hasattr(plugin.module, "register"):
        return _register_new_api(plugin, reg)

    LOG.warning(
        "Plugin %s has no register() function",
        plugin.manifest.name,
    )
    return 0


def _register_new_api(
    plugin: LoadedPlugin,
    registry: OperationRegistry,
) -> int:
    """Register operations using new API.

    Parameters
    ----------
    plugin
        Loaded plugin.
    registry
        Operation registry.

    Returns
    -------
    int
        Number of operations registered.

    Raises
    ------
    RuntimeError
        If an error occurs during registration.
    """
    count = 0

    class RegistrationProxy:
        """Proxy for registry that counts registrations."""

        def register(self, spec: OperationSpec) -> OperationSpec:
            """Register an operation spec.

            Parameters
            ----------
            spec
                Operation spec to register.

            Returns
            -------
            OperationSpec
                The registered spec.
            """
            _ = self  # Required for method signature
            nonlocal count
            # Validate operation ID prefix
            if not spec.operation_id.startswith(f"{plugin.manifest.name}."):
                LOG.warning(
                    "Operation '%s' should be prefixed with '%s.'",
                    spec.operation_id,
                    plugin.manifest.name,
                )
            registry.register(spec)
            count += 1
            return spec

    try:
        plugin.module.register(RegistrationProxy())
        LOG.info(
            "Registered %d operations from plugin %s",
            count,
            plugin.manifest.name,
        )
    except (AttributeError, TypeError) as e:
        LOG.exception("Error registering operations from %s", plugin.manifest.name)
        msg = f"Error in plugin.register(): {e}"
        raise RuntimeError(msg) from e

    return count


def register_all_plugins(
    result: PluginLoadResult,
    registry: OperationRegistry | None = None,
) -> dict[str, int]:
    """Register operations from all loaded plugins.

    Parameters
    ----------
    result
        Plugin load result.
    registry
        Operation registry (uses global if None).

    Returns
    -------
    dict[str, int]
        Map of plugin name to number of operations registered.
    """
    reg = registry or get_registry()
    registrations: dict[str, int] = {}

    for plugin in result.loaded:
        try:
            count = register_plugin_operations(plugin, reg)
            registrations[plugin.manifest.name] = count
        except RuntimeError:
            LOG.exception("Failed to register plugin %s", plugin.manifest.name)
            registrations[plugin.manifest.name] = 0

    return registrations


def initialize_plugins(
    search_paths: list[Any] | None = None,
    *,
    sandbox_enabled: bool = True,
) -> dict[str, int]:
    """Initialize and load all plugins.

    Convenience function to discover, load, and register all plugins.

    Parameters
    ----------
    search_paths
        Paths to search for plugins.
    sandbox_enabled
        Enable sandboxing for plugins.

    Returns
    -------
    dict[str, int]
        Map of plugin name to number of operations registered.
    """
    loader = PluginLoader(sandbox_enabled=sandbox_enabled)
    result = loader.load_all(search_paths)

    # Log failures
    for path, error in result.failed:
        LOG.error("Failed to load plugin %s: %s", path, error)

    return register_all_plugins(result)


@dataclass
class PluginInfo:
    """Information about a loaded plugin.

    Parameters
    ----------
    name
        Plugin name.
    version
        Plugin version.
    description
        Plugin description.
    path
        Plugin file path.
    operations
        Number of operations provided.
    enabled
        Whether plugin is enabled.
    """

    name: str
    version: str
    description: str
    path: Path
    operations: int = 0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "path": str(self.path),
            "operations": self.operations,
            "enabled": self.enabled,
        }


@dataclass
class PluginManager:
    """Manage CLI plugins.

    Parameters
    ----------
    plugin_dirs
        Directories to search for plugins.
    loaded_plugins
        Currently loaded plugins.
    """

    plugin_dirs: list[Path] = field(default_factory=lambda: list(DEFAULT_PLUGIN_PATHS))
    loaded_plugins: dict[str, PluginInfo] = field(default_factory=dict)

    def discover(self) -> list[Path]:
        """Discover available plugins.

        Returns
        -------
        list[Path]
            Plugin file paths.
        """
        plugins: list[Path] = []
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
            plugin_files = [
                path for path in plugin_dir.glob("*.py") if not path.name.startswith("_")
            ]
            plugins.extend(plugin_files)
        return plugins

    def load_plugin(self, path: Path) -> PluginInfo | None:
        """Load a plugin from file.

        Parameters
        ----------
        path
            Plugin file path.

        Returns
        -------
        PluginInfo | None
            Plugin info or None on failure.
        """
        loader = PluginLoader(sandbox_enabled=False)
        try:
            loaded = loader.load_single(path)
            count = register_plugin_operations(loaded, get_registry())

            info = PluginInfo(
                name=loaded.manifest.name,
                version=loaded.manifest.version,
                description=loaded.manifest.description,
                path=path,
                operations=count,
            )
            self.loaded_plugins[info.name] = info
        except (OSError, ValueError, RuntimeError):
            LOG.exception("Failed to load plugin %s", path)
            return None
        else:
            LOG.info(
                "Loaded plugin: %s v%s (%d operations)",
                info.name,
                info.version,
                info.operations,
            )
            return info

    def load_all(self) -> list[PluginInfo]:
        """Load all discovered plugins.

        Returns
        -------
        list[PluginInfo]
            Loaded plugin info.
        """
        loaded: list[PluginInfo] = []
        for path in self.discover():
            info = self.load_plugin(path)
            if info:
                loaded.append(info)
        return loaded

    def list_plugins(self) -> list[PluginInfo]:
        """List loaded plugins.

        Returns
        -------
        list[PluginInfo]
            All loaded plugins.
        """
        return list(self.loaded_plugins.values())

    def get_plugin(self, name: str) -> PluginInfo | None:
        """Get plugin by name.

        Parameters
        ----------
        name
            Plugin name.

        Returns
        -------
        PluginInfo | None
            Plugin info or None.
        """
        return self.loaded_plugins.get(name)


class PluginManagerHolder(SingletonHolder[PluginManager]):
    """Thread-safe holder for the shared PluginManager."""


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager.

    Returns
    -------
    PluginManager
        Plugin manager instance.
    """
    return PluginManagerHolder.get(PluginManager)


__all__ = [
    "PluginInfo",
    "PluginManager",
    "get_plugin_manager",
    "initialize_plugins",
    "register_all_plugins",
    "register_plugin_operations",
]
