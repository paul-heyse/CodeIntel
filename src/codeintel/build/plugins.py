"""Plugin protocol for the build system.

This module defines the TargetPlugin protocol that all build system
plugins should implement. Plugins are pure executors - they receive
everything they need via TargetExecutionContext and don't declare
settings via ClassVars.

The build system is the single source of truth for:
- What tables/artifacts a target produces (OutputContract)
- What resources are needed (TargetResources)
- How to execute (TargetExecution)
- Tuning parameters (TargetParameters)

Plugins just implement execute() and use the context.

Example
-------
>>> from codeintel.build.plugins import TargetPlugin
>>> from codeintel.build.context import TargetExecutionContext, TargetResult
>>>
>>> class MyPlugin(TargetPlugin):
...     plugin_name = "my_plugin"
...     plugin_version = "1.0.0"
...
...     async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
...         # Get parameters from build config
...         max_items = ctx.parameters.get("max_items", int, default=100)
...
...         # Use resources
...         data = await ctx.resources.gateway.query(...)
...
...         # Write to tables (validated against contract)
...         ctx.write_table("analytics.my_table", rows)
...
...         return TargetResult.succeeded(row_counts={"analytics.my_table": len(rows)})
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext, TargetResult

__all__ = [
    "TargetPlugin",
]


@runtime_checkable
class TargetPlugin(Protocol):
    """Protocol for build system target plugins.

    Plugins implementing this protocol are pure executors. They receive
    all configuration and resources via TargetExecutionContext.

    Required class attributes:
    - plugin_name: Unique identifier for the plugin
    - plugin_version: Semantic version string

    Optional class attributes:
    - plugin_description: Human-readable description

    The execute() method is async to support I/O operations like
    subprocess execution, database queries, and file operations.

    Attributes
    ----------
    plugin_name
        Unique plugin identifier (e.g., "scip_ingest").
    plugin_version
        Semantic version string (e.g., "3.0.0").
    plugin_description
        Optional human-readable description.

    Examples
    --------
    >>> class HotspotsPlugin:
    ...     plugin_name = "hotspots"
    ...     plugin_version = "2.0.0"
    ...     plugin_description = "Compute file hotspots from git history"
    ...
    ...     async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
    ...         max_commits = ctx.parameters.get("max_commits", int, default=2000)
    ...         git = ctx.resources.git_history
    ...         entries = await git.log(ctx.repo_root, max_count=max_commits)
    ...         # ... compute hotspots ...
    ...         ctx.write_table("analytics.hotspots", hotspots)
    ...         return TargetResult.succeeded()
    """

    plugin_name: ClassVar[str]
    plugin_version: ClassVar[str]
    plugin_description: ClassVar[str]

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the plugin with the given context.

        Parameters
        ----------
        ctx
            Complete execution context with target info, resources,
            parameters, and write methods.

        Returns
        -------
        TargetResult
            Result indicating success or failure, with row counts
            and artifacts written.

        Raises
        ------
        Exception
            Any exception raised will be wrapped in PluginExecutionError
            by the BuildExecutor.
        """
        ...


# =============================================================================
# Plugin Registry (for discovery)
# =============================================================================


class PluginCatalog:
    """Container for decorator-registered plugins."""

    def __init__(self) -> None:
        self._registry: dict[str, type[TargetPlugin]] = {}

    def register(self, plugin_class: type[TargetPlugin]) -> type[TargetPlugin]:
        """Register a plugin class and return it."""
        name = plugin_class.plugin_name
        self._registry[name] = plugin_class
        return plugin_class

    def get(self, name: str) -> type[TargetPlugin] | None:
        """Return a plugin class by name."""
        return self._registry.get(name)

    def all(self) -> dict[str, type[TargetPlugin]]:
        """Return a copy of the registry."""
        return dict(self._registry)

    def clear(self) -> None:
        """Clear registry entries (for tests)."""
        self._registry.clear()


_DEFAULT_PLUGIN_CATALOG = PluginCatalog()


def register_plugin(
    plugin_class: type[TargetPlugin] | None = None, *, catalog: PluginCatalog | None = None
) -> type[TargetPlugin] | Callable[[type[TargetPlugin]], type[TargetPlugin]]:
    """Register a plugin class for discovery.

    Use as a decorator:

        @register_plugin
        class MyPlugin:
            plugin_name = "my_plugin"
            ...

    Parameters
    ----------
    plugin_class
        Plugin class to register.
    catalog
        Optional catalog to register into (defaults to module singleton).

    Returns
    -------
    type[TargetPlugin]
        The same class (for decorator use).
    """

    def _decorator(cls: type[TargetPlugin]) -> type[TargetPlugin]:
        target_catalog = catalog or _DEFAULT_PLUGIN_CATALOG
        return target_catalog.register(cls)

    if plugin_class is None:
        return _decorator
    return _decorator(plugin_class)


def get_plugin(name: str, *, catalog: PluginCatalog | None = None) -> type[TargetPlugin] | None:
    """Get a registered plugin by name.

    Parameters
    ----------
    name
        Plugin name to look up.

    Returns
    -------
    type[TargetPlugin] | None
        Plugin class if found.
    """
    target_catalog = catalog or _DEFAULT_PLUGIN_CATALOG
    return target_catalog.get(name)


def all_plugins(*, catalog: PluginCatalog | None = None) -> dict[str, type[TargetPlugin]]:
    """Get all registered plugins.

    Returns
    -------
    dict[str, type[TargetPlugin]]
        Mapping of plugin name to class.
    """
    target_catalog = catalog or _DEFAULT_PLUGIN_CATALOG
    return target_catalog.all()


__all__ = [
    "PluginCatalog",
    "TargetPlugin",
    "all_plugins",
    "get_plugin",
    "register_plugin",
]
