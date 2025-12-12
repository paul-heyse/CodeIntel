"""Unified plugin architecture for CLI extensions.

This package provides a consolidated plugin infrastructure supporting:
- Manifest-based plugins with plugin.json
- Sandboxed execution with capability control
- Plugin discovery and loading
- Operation registration via register() API

Example
-------
Loading and registering plugins::

    from codeintel.cli.plugins import initialize_plugins


    registrations = initialize_plugins()

Creating a custom loader::

    from codeintel.cli.plugins import PluginLoader

    loader = PluginLoader(sandbox_enabled=True)
    result = loader.load_all()

    for plugin in result.loaded:
        print(f"Loaded: {plugin.manifest.name}")
"""

from __future__ import annotations

from codeintel.cli.plugins.discovery import (
    DEFAULT_PLUGIN_PATHS,
    DiscoveredPlugin,
    discover_plugins,
    get_default_plugin_paths,
)
from codeintel.cli.plugins.loader import (
    LoadedPlugin,
    PluginLoader,
    PluginLoadResult,
    get_plugin_loader,
)
from codeintel.cli.plugins.manifest import (
    CLI_API_VERSION,
    PluginCapability,
    PluginDependency,
    PluginManifest,
    SemanticVersion,
)
from codeintel.cli.plugins.registry import (
    PluginInfo,
    PluginManager,
    get_plugin_manager,
    initialize_plugins,
    register_all_plugins,
    register_plugin_operations,
)
from codeintel.cli.plugins.sandbox import (
    ALLOWED_MODULES,
    CAPABILITY_MODULES,
    PluginSandbox,
    SandboxConfig,
    SandboxedImporter,
)
from codeintel.cli.plugins.testing import (
    OperationSpecProtocol,
    PluginTestHarness,
    PluginTestResult,
    create_plugin_scaffold,
)

__all__ = [
    "ALLOWED_MODULES",
    "CAPABILITY_MODULES",
    "CLI_API_VERSION",
    "DEFAULT_PLUGIN_PATHS",
    "DiscoveredPlugin",
    "LoadedPlugin",
    "OperationSpecProtocol",
    "PluginCapability",
    "PluginDependency",
    "PluginInfo",
    "PluginLoadResult",
    "PluginLoader",
    "PluginManager",
    "PluginManifest",
    "PluginSandbox",
    "PluginTestHarness",
    "PluginTestResult",
    "SandboxConfig",
    "SandboxedImporter",
    "SemanticVersion",
    "create_plugin_scaffold",
    "discover_plugins",
    "get_default_plugin_paths",
    "get_plugin_loader",
    "get_plugin_manager",
    "initialize_plugins",
    "register_all_plugins",
    "register_plugin_operations",
]
