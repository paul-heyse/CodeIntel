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

The module also provides IngestExecutionContextBuilder for fluent
construction of contexts with validation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

from codeintel.config.models import ToolsConfig
from codeintel.core.config_protocol import ConfigAccessor
from codeintel.core.config_registry import ConfigNotFoundError, ConfigRegistry
from codeintel.core.plugins.context import PluginExecutionContext, PluginScratch
from codeintel.core.resources import ResourceNotFoundError, ResourceRegistry
from codeintel.ingestion.infrastructure.db_queries import safe_count

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.ingestion.infrastructure.scanning import ScanProfile
    from codeintel.ingestion.resources.protocol import ResourceProvider
    from codeintel.runtime.context import RunContext
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
class IngestExecutionContext(PluginExecutionContext):
    """Execution context for ingestion plugins.

    Extend the core PluginExecutionContext with ingestion-specific
    functionality including scan profiles, tools configuration, and
    plugin timing utilities.

    For convenience, use IngestExecutionContextBuilder for fluent
    construction with validation.

    Attributes
    ----------
    gateway
        StorageGateway providing DuckDB access.
    snapshot
        Repository snapshot reference.
    paths
        Build paths configuration (optional, use validated_paths property).
    code_profile
        Code scanning profile (optional, use validated_code_profile property).
    config_profile
        Config scanning profile (optional, use validated_config_profile property).
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

    # Ingestion-specific fields with honest optional types
    # Use validated_* properties for safe access with runtime checks
    code_profile: ScanProfile | None = None
    config_profile: ScanProfile | None = None
    tools: ToolsConfig = field(default_factory=_default_tools_config)

    # Override from base - use ConfigRegistry for runtime validation
    configs: ConfigAccessor = field(default_factory=ConfigRegistry)

    # Plugin timing - internal tracking
    _plugin_start_times: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _plugin_durations: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    # Error messages for validation
    _ERR_PATHS_NOT_SET = "paths not initialized on IngestExecutionContext"
    _ERR_CODE_PROFILE_NOT_SET = "code_profile not initialized on IngestExecutionContext"
    _ERR_CONFIG_PROFILE_NOT_SET = "config_profile not initialized on IngestExecutionContext"

    @property
    def validated_paths(self) -> BuildPaths:
        """Access paths with runtime validation.

        Returns
        -------
        BuildPaths
            The build paths configuration.

        Raises
        ------
        RuntimeError
            If paths is not set.
        """
        if self.paths is None:
            raise RuntimeError(self._ERR_PATHS_NOT_SET)
        return self.paths

    @property
    def validated_code_profile(self) -> ScanProfile:
        """Access code_profile with runtime validation.

        Returns
        -------
        ScanProfile
            The code scanning profile.

        Raises
        ------
        RuntimeError
            If code_profile is not set.
        """
        if self.code_profile is None:
            raise RuntimeError(self._ERR_CODE_PROFILE_NOT_SET)
        return self.code_profile

    @property
    def validated_config_profile(self) -> ScanProfile:
        """Access config_profile with runtime validation.

        Returns
        -------
        ScanProfile
            The config scanning profile.

        Raises
        ------
        RuntimeError
            If config_profile is not set.
        """
        if self.config_profile is None:
            raise RuntimeError(self._ERR_CONFIG_PROFILE_NOT_SET)
        return self.config_profile

    @property
    def build_dir(self) -> Path:
        """Build directory derived from execution config.

        Returns
        -------
        Path
            Path to the build directory.
        """
        return self.validated_paths.build_dir

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


@dataclass
class IngestExecutionContextBuilder:
    """Builder for constructing IngestExecutionContext with validation.

    Provide a fluent API for configuring ingestion execution contexts,
    with validation that all required fields are set before building.

    Example
    -------
    >>> builder = IngestExecutionContextBuilder(gateway, snapshot)
    >>> builder = builder.with_paths(paths).with_code_profile(profile)
    >>> ctx = builder.build()  # raises if required fields missing
    """

    _gateway: StorageGateway
    _snapshot: SnapshotRef
    _run_id: str = ""
    _paths: BuildPaths | None = None
    _code_profile: ScanProfile | None = None
    _config_profile: ScanProfile | None = None
    _tools: ToolsConfig = field(default_factory=_default_tools_config)
    _resources: ResourceRegistry = field(default_factory=_empty_registry)
    _scratch: PluginScratch = field(default_factory=PluginScratch)
    _configs: ConfigRegistry = field(default_factory=ConfigRegistry)
    _plugin_name: str | None = None
    _run_context: RunContext | None = None

    def with_run_id(self, run_id: str) -> Self:
        """Set the run identifier.

        Parameters
        ----------
        run_id
            Unique identifier for this execution run.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._run_id = run_id
        return self

    def with_paths(self, paths: BuildPaths) -> Self:
        """Set the build paths configuration.

        Parameters
        ----------
        paths
            Build paths configuration.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._paths = paths
        return self

    def with_code_profile(self, profile: ScanProfile) -> Self:
        """Set the code scanning profile.

        Parameters
        ----------
        profile
            Code scanning profile.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._code_profile = profile
        return self

    def with_config_profile(self, profile: ScanProfile) -> Self:
        """Set the config scanning profile.

        Parameters
        ----------
        profile
            Config scanning profile.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._config_profile = profile
        return self

    def with_tools(self, tools: ToolsConfig) -> Self:
        """Set the tools configuration.

        Parameters
        ----------
        tools
            Tools configuration.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._tools = tools
        return self

    def with_resources(self, resources: ResourceRegistry) -> Self:
        """Set the resource registry.

        Parameters
        ----------
        resources
            Resource registry.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._resources = resources
        return self

    def with_scratch(self, scratch: PluginScratch) -> Self:
        """Set the shared scratch space.

        Parameters
        ----------
        scratch
            Scratch space for inter-plugin communication.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._scratch = scratch
        return self

    def with_configs(self, configs: ConfigRegistry) -> Self:
        """Set the configuration registry.

        Parameters
        ----------
        configs
            Configuration registry.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._configs = configs
        return self

    def with_plugin_name(self, name: str) -> Self:
        """Set the executing plugin name.

        Parameters
        ----------
        name
            Plugin name.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._plugin_name = name
        return self

    def with_run_context(self, run_context: RunContext) -> Self:
        """Set the unified run context.

        Parameters
        ----------
        run_context
            Run context for cross-engine correlation.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._run_context = run_context
        return self

    def build(self) -> IngestExecutionContext:
        """Build the IngestExecutionContext.

        Build the context. All fields are optional in the context itself,
        but callers should ensure required fields are set for their use case.

        Returns
        -------
        IngestExecutionContext
            The configured execution context.
        """
        return IngestExecutionContext(
            gateway=self._gateway,
            snapshot=self._snapshot,
            run_id=self._run_id,
            paths=self._paths,
            code_profile=self._code_profile,
            config_profile=self._config_profile,
            tools=self._tools,
            resources=self._resources,
            scratch=self._scratch,
            configs=self._configs,
            plugin_name=self._plugin_name,
            run_context=self._run_context,
        )

    def build_validated(self) -> IngestExecutionContext:
        """Build the IngestExecutionContext with full validation.

        Ensure all required fields are set before building.

        Returns
        -------
        IngestExecutionContext
            The configured execution context.

        Raises
        ------
        ValueError
            If any required field is not set.
        """
        errors: list[str] = []
        if self._paths is None:
            errors.append("paths is required")
        if self._code_profile is None:
            errors.append("code_profile is required")
        if self._config_profile is None:
            errors.append("config_profile is required")

        if errors:
            message = f"IngestExecutionContextBuilder validation failed: {', '.join(errors)}"
            raise ValueError(message)

        return self.build()


__all__ = [
    "IngestExecutionContext",
    "IngestExecutionContextBuilder",
    "PluginScratch",
    "ResourceNotFoundError",
]
