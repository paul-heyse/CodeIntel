"""Plugin architecture for CLI extensions.

Provide a framework for extending the CLI with custom operations
without modifying core code.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from codeintel.cli.executor import OperationSpec
from codeintel.cli.operation_registry import get_operation_registry

LOG = logging.getLogger(__name__)

DEFAULT_PLUGIN_DIRS = [
    Path.home() / ".codeintel" / "plugins",
    Path("/etc/codeintel/plugins"),
]


class PluginProtocol(Protocol):
    """Protocol for CLI plugins.

    Plugins must implement this interface to be loadable.
    """

    @property
    def name(self) -> str:
        """Plugin name."""
        ...

    @property
    def version(self) -> str:
        """Plugin version."""
        ...

    @property
    def description(self) -> str:
        """Plugin description."""
        ...

    def get_operations(self) -> list[OperationSpec[Any]]:
        """Get operations provided by this plugin.

        Returns
        -------
        list[OperationSpec[Any]]
            Operations to register.
        """
        ...

    def initialize(self) -> None:
        """Initialize the plugin.

        Called after loading but before operations are registered.
        """
        ...


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

    plugin_dirs: list[Path] = field(default_factory=lambda: list(DEFAULT_PLUGIN_DIRS))
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
        try:
            # Load module
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                LOG.warning("Could not load plugin: %s", path)
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)

            # Get plugin instance
            plugin_factory = getattr(module, "create_plugin", None)
            if plugin_factory is None:
                LOG.warning("Plugin missing create_plugin(): %s", path)
                return None

            plugin = plugin_factory()

            # Initialize
            plugin.initialize()

            # Register operations
            operations = plugin.get_operations()
            registry = get_operation_registry()
            for op_spec in operations:
                registry.register(op_spec)

            info = PluginInfo(
                name=plugin.name,
                version=plugin.version,
                description=plugin.description,
                path=path,
                operations=len(operations),
            )
            self.loaded_plugins[plugin.name] = info

            LOG.info(
                "Loaded plugin: %s v%s (%d operations)",
                info.name,
                info.version,
                info.operations,
            )

        except (OSError, ValueError, AttributeError, TypeError):
            LOG.exception("Failed to load plugin %s", path)
            return None
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


# Global plugin manager
_PLUGIN_MANAGER: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager.

    Returns
    -------
    PluginManager
        Plugin manager instance.
    """
    global _PLUGIN_MANAGER  # noqa: PLW0603
    if _PLUGIN_MANAGER is None:
        _PLUGIN_MANAGER = PluginManager()
    return _PLUGIN_MANAGER


def initialize_plugins() -> None:
    """Initialize and load all plugins.

    Called during CLI startup.
    """
    manager = get_plugin_manager()
    manager.load_all()


__all__ = [
    "DEFAULT_PLUGIN_DIRS",
    "PluginInfo",
    "PluginManager",
    "PluginProtocol",
    "get_plugin_manager",
    "initialize_plugins",
]
