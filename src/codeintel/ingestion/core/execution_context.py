"""Execution context for ingestion plugins.

This module provides the execution context that plugins receive during
execution, enabling typed access to resources, configuration, and
shared scratch space.

IngestExecutionContext extends the core PluginExecutionContext with
ingestion-specific fields:
- code_profile: Code scanning profile
- config_profile: Config scanning profile
- tools: External tools configuration
- Plugin timing utilities for performance tracking
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.config.models import ToolsConfig
from codeintel.core.config_registry import ConfigNotFoundError, ConfigRegistry
from codeintel.core.plugins.context import PluginExecutionContext, PluginScratch
from codeintel.core.resources import ResourceNotFoundError, ResourceRegistry
from codeintel.ingestion.infrastructure.db_queries import safe_count

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.config.primitives import BuildPaths
    from codeintel.ingestion.infrastructure.scanning import ScanProfile
    from codeintel.ingestion.resources.protocol import ResourceProvider


def _empty_registry() -> ResourceRegistry:
    """Create an empty resource registry.

    This avoids importing ResourceRegistry at module level to prevent
    circular imports.

    Returns
    -------
    ResourceRegistry
        Empty registry instance.
    """
    return ResourceRegistry()


def _default_tools_config() -> ToolsConfig:
    """Construct a default tools configuration.

    Returns
    -------
    ToolsConfig
        Default tools configuration instance.
    """
    return ToolsConfig.default()


@dataclass
class IngestExecutionContext(PluginExecutionContext):
    """Execution context for ingestion plugins.

    Extend the core PluginExecutionContext with ingestion-specific
    functionality including scan profiles, tools configuration, and
    plugin timing utilities.

    Attributes
    ----------
    gateway
        StorageGateway providing DuckDB access.
    snapshot
        Repository snapshot reference.
    paths
        Build paths configuration (required for ingestion).
    code_profile
        Code scanning profile.
    config_profile
        Config scanning profile.
    tools
        Tools configuration.
    resources
        Resource registry for lazy resource access.
    scratch
        Shared scratch space for inter-plugin data.
    configs
        Configuration registry (uses ConfigRegistry for runtime validation).
    plugin_name
        Name of the executing plugin.
    run_id
        Unique identifier for this execution run.
    run_context
        Optional unified run context for cross-engine correlation.
    """

    # Override from base - required in ingestion
    paths: BuildPaths = field(default=None)  # type: ignore[assignment]

    # Ingestion-specific fields
    code_profile: ScanProfile = field(default=None)  # type: ignore[assignment]
    config_profile: ScanProfile = field(default=None)  # type: ignore[assignment]
    tools: ToolsConfig = field(default_factory=_default_tools_config)

    # Override from base - use ConfigRegistry instead of ConfigProvider
    configs: ConfigRegistry = field(default_factory=ConfigRegistry)  # type: ignore[assignment]

    # Plugin timing - internal tracking
    _plugin_start_times: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _plugin_durations: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    @property
    def build_dir(self) -> Path:
        """Build directory derived from execution config.

        Returns
        -------
        Path
            Path to the build directory.
        """
        return self.paths.build_dir

    def require[T: ResourceProvider[object]](self, resource_type: type[T]) -> T:
        """Get the resource provider, raising if unavailable.

        Override to provide more specific type bounds for ingestion.

        Parameters
        ----------
        resource_type
            The type of resource provider to retrieve.

        Returns
        -------
        T
            The resource provider instance.
        """
        return cast("T", self.resources.get(resource_type))

    def require_by_name(self, name: str) -> object:
        """Get a resource by name for duck-typing scenarios.

        Parameters
        ----------
        name
            Name of the resource provider.

        Returns
        -------
        object
            The resource provider instance. Caller should cast to expected type.
        """
        return self.resources.require_by_name(name)

    def has_resource[T: ResourceProvider[object]](self, resource_type: type[T]) -> bool:
        """Check if a resource is available.

        Parameters
        ----------
        resource_type
            The type of resource provider to check.

        Returns
        -------
        bool
            True if the resource is registered.
        """
        return self.resources.has(resource_type)

    def has_resource_by_name(self, name: str) -> bool:
        """Check if a resource is available by name.

        Parameters
        ----------
        name
            Name of the resource provider.

        Returns
        -------
        bool
            True if the resource is registered.
        """
        return self.resources.has_by_name(name)

    def register_config[T](self, config_type: type[T], config: T) -> None:
        """Register a configuration instance.

        Parameters
        ----------
        config_type
            The type to register the config under.
        config
            The configuration instance.
        """
        self.configs.register(config_type, config)

    def has_config(self, config_type: type[object]) -> bool:
        """Check if a config type is registered.

        Parameters
        ----------
        config_type
            The configuration type to check.

        Returns
        -------
        bool
            True if config is registered.
        """
        return self.configs.has(config_type)

    def get_config[T](self, config_type: type[T]) -> T:
        """Get a required configuration.

        Parameters
        ----------
        config_type
            The configuration type to retrieve.

        Returns
        -------
        T
            The configuration instance.

        Raises
        ------
        KeyError
            If the config type is not registered.
        """
        try:
            return self.configs.get(config_type)
        except ConfigNotFoundError as exc:
            raise KeyError(str(exc)) from exc

    def get_optional_config[T](self, config_type: type[T]) -> T | None:
        """Get an optional configuration.

        Parameters
        ----------
        config_type
            The configuration type to retrieve.

        Returns
        -------
        T | None
            The configuration instance or None.
        """
        return self.configs.get_optional(config_type)

    def count_produced_tables(
        self,
        tables: tuple[str, ...],
    ) -> Mapping[str, int]:
        """Count rows in the specified tables.

        Parameters
        ----------
        tables
            Table names to count.

        Returns
        -------
        Mapping[str, int]
            Mapping of table names to row counts.
        """
        counts: dict[str, int] = {}
        for table in tables:
            count = safe_count(self.gateway, table)
            counts[table] = count if count is not None else 0
        return counts

    def start_plugin_timer(self, plugin_name: str) -> None:
        """Record the start time for a plugin execution.

        Parameters
        ----------
        plugin_name
            Name of the plugin to time.
        """
        if plugin_name not in self._plugin_start_times:
            self._plugin_start_times[plugin_name] = time.perf_counter()
            self._plugin_durations.pop(plugin_name, None)

    def finish_plugin_timer(self, plugin_name: str) -> float:
        """Return elapsed time for a plugin execution.

        Parameters
        ----------
        plugin_name
            Name of the plugin.

        Returns
        -------
        float
            Duration in seconds for the specified plugin execution.
        """
        if plugin_name in self._plugin_durations:
            return self._plugin_durations[plugin_name]

        start_time = self._plugin_start_times.get(plugin_name)
        if start_time is None:
            return 0.0

        duration = time.perf_counter() - start_time
        self._plugin_durations[plugin_name] = duration
        self._plugin_start_times.pop(plugin_name, None)
        return duration


__all__ = [
    "IngestExecutionContext",
    "PluginScratch",
    "ResourceNotFoundError",
]
