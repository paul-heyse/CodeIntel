"""Execution context for ingestion plugins.

This module provides the execution context that plugins receive during
execution, enabling typed access to resources, configuration, and
shared scratch space.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.config.models import ToolsConfig
from codeintel.core.config_registry import ConfigNotFoundError, ConfigRegistry
from codeintel.ingestion.infrastructure.db_queries import safe_count
from codeintel.ingestion.plugins.protocol import IngestRuntimeScratch
from codeintel.ingestion.resources.registry import ResourceNotFoundError, ResourceRegistry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.ingestion.infrastructure.scanning import ScanProfile
    from codeintel.ingestion.resources.protocol import ResourceProvider
    from codeintel.runtime import RunContext
    from codeintel.storage.gateway import StorageGateway


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
class IngestExecutionContext:
    """Execution context for ingestion plugins.

    Provide access to storage, configuration, change tracking, and shared
    scratch space for inter-plugin communication.

    Attributes
    ----------
    gateway
        StorageGateway providing DuckDB access.
    snapshot
        Repository snapshot reference.
    paths
        Build paths configuration.
    tools
        Tools configuration.
    code_profile
        Code scanning profile.
    config_profile
        Config scanning profile.
    resources
        Resource registry for lazy resource access.
    scratch
        Shared scratch space for inter-plugin data.
    configs
        Mapping of config types to config instances.
    plugin_name
        Name of the executing plugin.
    run_id
        Unique identifier for this execution run.
    run_context
        Optional unified run context for cross-engine correlation.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    code_profile: ScanProfile
    config_profile: ScanProfile
    tools: ToolsConfig = field(default_factory=_default_tools_config)
    resources: ResourceRegistry = field(default_factory=_empty_registry)
    scratch: IngestRuntimeScratch = field(default_factory=IngestRuntimeScratch)
    configs: ConfigRegistry = field(default_factory=ConfigRegistry)
    plugin_name: str | None = None
    run_id: str | None = None
    run_context: RunContext | None = None
    _plugin_start_times: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _plugin_durations: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    @property
    def repo_root(self) -> Path:
        """Repository root for the current snapshot.

        Returns
        -------
        Path
            Absolute path to the repository root.
        """
        return self.snapshot.repo_root

    @property
    def repo(self) -> str:
        """Repository slug for the current snapshot.

        Returns
        -------
        str
            Repository identifier.
        """
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier for the current snapshot.

        Returns
        -------
        str
            Commit hash or identifier.
        """
        return self.snapshot.commit

    @property
    def build_dir(self) -> Path:
        """Build directory derived from execution config.

        Returns
        -------
        Path
            Path to the build directory.
        """
        return self.paths.build_dir

    @property
    def effective_run_id(self) -> str | None:
        """Get run ID preferring unified RunContext if present.

        Returns
        -------
        str | None
            Run ID from run_context if set, otherwise falls back to run_id.
        """
        if self.run_context is not None:
            return self.run_context.run_id
        return self.run_id

    def require[T: ResourceProvider[object]](self, provider_type: type[T]) -> T:
        """Get the resource provider, raising if unavailable.

        Parameters
        ----------
        provider_type
            The type of resource provider to retrieve.

        Returns
        -------
        T
            The resource provider instance.
        """
        return cast("T", self.resources.get(provider_type))

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

    def has_resource[T: ResourceProvider[object]](self, provider_type: type[T]) -> bool:
        """Check if a resource is available.

        Parameters
        ----------
        provider_type
            The type of resource provider to check.

        Returns
        -------
        bool
            True if the resource is registered.
        """
        return self.resources.has(provider_type)

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
        """Record the start time for a plugin execution."""
        if plugin_name not in self._plugin_start_times:
            self._plugin_start_times[plugin_name] = time.perf_counter()
            self._plugin_durations.pop(plugin_name, None)

    def finish_plugin_timer(self, plugin_name: str) -> float:
        """Return elapsed time for a plugin execution.

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
    "ResourceNotFoundError",
]
