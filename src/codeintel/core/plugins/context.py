"""Unified execution context for plugins.

This module provides a minimal, protocol-driven execution context that
is used by both graph and analytics plugins. Plugins request what they
need through typed accessors and resource providers.

The base `PluginExecutionContext` can be extended by domain-specific
contexts (e.g., `GraphPluginExecutionContext`) that add specialized
methods and fields.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

from codeintel.core.resources.registry import ResourceNotFoundError, ResourceRegistry

if TYPE_CHECKING:
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.runtime import RunContext
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PluginScratch:
    """Ephemeral scratch/cache store shared across plugin executions in a run.

    Provides a simple key-value store for plugins to share intermediate
    results and register cleanup callbacks.
    """

    _store: dict[str, object] = field(default_factory=dict, repr=False)
    _cleanup: list[Callable[[], None]] = field(default_factory=list, repr=False)

    def declare(self, key: str, value: object) -> None:
        """Store a value for later consumption by other plugins.

        Parameters
        ----------
        key
            Unique key for the value.
        value
            Value to store.
        """
        self._store[key] = value

    def consume(self, key: str, default: T | None = None) -> T | None:
        """Retrieve a value populated by another plugin.

        Parameters
        ----------
        key
            Key to look up.
        default
            Default value if key is not found.

        Returns
        -------
        T | None
            Stored value or default.
        """
        return cast("T | None", self._store.get(key, default))

    def has(self, key: str) -> bool:
        """Check if a key exists in the scratch store.

        Parameters
        ----------
        key
            Key to check.

        Returns
        -------
        bool
            True if the key exists.
        """
        return key in self._store

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback executed after the run completes.

        Parameters
        ----------
        callback
            Cleanup function to call.
        """
        self._cleanup.append(callback)

    def cleanup(self) -> None:
        """Execute cleanup callbacks and clear stored values."""
        for callback in reversed(self._cleanup):
            try:
                callback()
            except (RuntimeError, OSError, ValueError):
                log.exception("scratch.cleanup_failed")
        self._store.clear()
        self._cleanup.clear()

    def keys(self) -> tuple[str, ...]:
        """Return all declared keys.

        Returns
        -------
        tuple[str, ...]
            Keys in the scratch store.
        """
        return tuple(self._store.keys())

    def __len__(self) -> int:
        """Return number of stored values.

        Returns
        -------
        int
            Count of stored key-value pairs.
        """
        return len(self._store)


class ConfigProvider:
    """Typed configuration accessor for plugins.

    Provides a clean interface for plugins to request their configuration
    without needing to know about all possible config types.
    """

    def __init__(self, configs: Mapping[type[Any], object] | None = None) -> None:
        """Initialize with a mapping of config types to instances.

        Parameters
        ----------
        configs
            Mapping of config type to config instance.
        """
        self._configs: dict[type[Any], object] = dict(configs) if configs else {}

    def get(self, config_type: type[T]) -> T:
        """Return configuration of the requested type.

        Parameters
        ----------
        config_type
            Type of configuration to retrieve.

        Returns
        -------
        T
            Configuration instance.

        Raises
        ------
        ValueError
            If the requested config type is not available.
        """
        if config_type not in self._configs:
            message = f"Configuration {config_type.__name__} not available in context"
            raise ValueError(message)
        return cast("T", self._configs[config_type])

    def get_optional(self, config_type: type[T]) -> T | None:
        """Return configuration if available, None otherwise.

        Parameters
        ----------
        config_type
            Type of configuration to retrieve.

        Returns
        -------
        T | None
            Configuration instance or None.
        """
        return cast("T | None", self._configs.get(config_type))

    def has(self, config_type: type[T]) -> bool:
        """Check if a config type is available.

        Parameters
        ----------
        config_type
            Type to check.

        Returns
        -------
        bool
            True if the config is available.
        """
        return config_type in self._configs

    def register(self, config_type: type[T], config: T) -> None:
        """Register a configuration instance.

        Parameters
        ----------
        config_type
            Type of configuration.
        config
            Configuration instance.
        """
        self._configs[config_type] = config


@dataclass
class PluginExecutionContext:
    """Unified execution context for all plugins.

    This context is used by both graph plugins (builders, metrics, validation)
    and analytics plugins, providing a consistent interface for resource
    access, configuration, and inter-plugin communication.

    Resource Access
    ---------------
    Use `ctx.require(ProviderType)` to access resources:

    - For graphs: `ctx.require(GraphResource)` - Graph data access
    - For analytics: `ctx.require(GraphProvider)` - Graph runtime access

    Configuration
    -------------
    Use `ctx.get_config(ConfigType)` to access typed configuration.

    Inter-Plugin Communication
    --------------------------
    Use `ctx.scratch` for sharing data between plugins in a run.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    run_id: str | None = None

    # Resource registry for typed resource access
    resources: ResourceRegistry = field(default_factory=ResourceRegistry)

    # Typed config accessor - uses ConfigAccessor protocol for flexibility
    configs: ConfigProvider = field(default_factory=ConfigProvider)

    # Scratch for inter-plugin communication
    scratch: PluginScratch = field(default_factory=PluginScratch)

    # Build paths configuration (for graph plugins)
    paths: BuildPaths | None = None

    # Plugin-specific options (validated by plugin)
    options: object | None = None

    # Current plugin name (set by executor)
    plugin_name: str | None = None

    # Additional metadata
    extra: MutableMapping[str, Any] = field(default_factory=dict)

    # Unified run context for cross-engine correlation
    run_context: RunContext | None = None

    @property
    def repo(self) -> str:
        """Repository identifier.

        Returns
        -------
        str
            Repository slug.
        """
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier.

        Returns
        -------
        str
            Commit hash.
        """
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path.

        Returns
        -------
        Path
            Absolute path to the repository root.
        """
        return self.snapshot.repo_root

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

    def get_config(self, config_type: type[T]) -> T:
        """Return configuration of the requested type.

        Parameters
        ----------
        config_type
            Type of configuration to retrieve.

        Returns
        -------
        T
            Configuration instance.
        """
        return self.configs.get(config_type)

    def get_optional_config(self, config_type: type[T]) -> T | None:
        """Return configuration if available, None otherwise.

        Parameters
        ----------
        config_type
            Type of configuration to retrieve.

        Returns
        -------
        T | None
            Configuration instance or None.
        """
        return self.configs.get_optional(config_type)

    def has_config(self, config_type: type[T]) -> bool:
        """Check if a config type is available.

        Parameters
        ----------
        config_type
            Type to check.

        Returns
        -------
        bool
            True if the config is available.
        """
        return self.configs.has(config_type)

    def require(self, resource_type: type[T]) -> T:
        """Get a resource from the registry.

        Parameters
        ----------
        resource_type
            Type of resource to retrieve.

        Returns
        -------
        T
            The loaded resource.
        """
        return self.resources.require(resource_type)

    def require_or_none(self, resource_type: type[T]) -> T | None:
        """Get a resource or None if unavailable.

        Parameters
        ----------
        resource_type
            Type of resource to retrieve.

        Returns
        -------
        T | None
            The resource, or None if unavailable.
        """
        return self.resources.require_or_none(resource_type)

    def has_resource(self, resource_type: type) -> bool:
        """Check if a resource type is registered.

        Parameters
        ----------
        resource_type
            Type to check.

        Returns
        -------
        bool
            True if the resource is available.
        """
        return self.resources.has(resource_type)

    def require_by_name(self, name: str) -> object:
        """Get a resource by string name.

        Parameters
        ----------
        name
            String name of the provider (typically the class name).

        Returns
        -------
        object
            The loaded resource.
        """
        return self.resources.require_by_name(name)

    def has_resource_by_name(self, name: str) -> bool:
        """Check if a resource is registered by string name.

        Parameters
        ----------
        name
            String name to check.

        Returns
        -------
        bool
            True if a resource with that name is available.
        """
        return self.resources.has_by_name(name)

    def register_resource(
        self,
        resource_type: type[T],
        provider: object,
    ) -> None:
        """Register a resource provider.

        Parameters
        ----------
        resource_type
            Type key for the provider.
        provider
            Resource provider instance.
        """
        self.resources.register(resource_type, provider)


@dataclass
class PluginExecutionContextBuilder:
    """Builder for constructing PluginExecutionContext instances.

    Provides a fluent API for configuring execution contexts.

    Example
    -------
    >>> builder = PluginExecutionContextBuilder(gateway, snapshot, run_id)
    >>> builder = builder.with_resource(GraphResource, graph_resource)
    >>> ctx = builder.build()
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    run_id: str
    _configs: dict[type[Any], object] = field(default_factory=dict)
    _resources: ResourceRegistry = field(default_factory=ResourceRegistry)
    _paths: BuildPaths | None = None
    _extra: dict[str, Any] = field(default_factory=dict)
    _options: object | None = None
    _plugin_name: str | None = None
    _run_context: RunContext | None = None

    def with_config(self, config_type: type[T], config: T) -> Self:
        """Add a configuration to the context.

        Parameters
        ----------
        config_type
            Type of configuration.
        config
            Configuration instance.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._configs[config_type] = config
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

    def with_options(self, options: object) -> Self:
        """Set plugin-specific options.

        Parameters
        ----------
        options
            Options object.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._options = options
        return self

    def with_plugin_name(self, name: str) -> Self:
        """Set the current plugin name.

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

    def with_extra(self, key: str, value: object) -> Self:
        """Add extra metadata.

        Parameters
        ----------
        key
            Metadata key.
        value
            Metadata value.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._extra[key] = value
        return self

    def with_run_context(self, run_context: RunContext) -> Self:
        """Set the unified run context.

        Parameters
        ----------
        run_context
            Unified run context for cross-engine correlation.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._run_context = run_context
        return self

    def with_resource(
        self,
        resource_type: type[T],
        provider: object,
    ) -> Self:
        """Register a resource provider.

        Parameters
        ----------
        resource_type
            Type key for the provider.
        provider
            Resource provider instance.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._resources.register(resource_type, provider)
        return self

    def with_resources(
        self,
        resources: ResourceRegistry,
    ) -> Self:
        """Set the resource registry.

        Replaces any previously configured resources.

        Parameters
        ----------
        resources
            Resource registry to use.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._resources = resources
        return self

    def build(self, *, scratch: PluginScratch | None = None) -> PluginExecutionContext:
        """Build the execution context.

        Parameters
        ----------
        scratch
            Optional shared scratch store.

        Returns
        -------
        PluginExecutionContext
            Configured execution context.
        """
        return PluginExecutionContext(
            gateway=self.gateway,
            snapshot=self.snapshot,
            run_id=self.run_id,
            resources=self._resources,
            configs=ConfigProvider(self._configs),
            scratch=scratch or PluginScratch(),
            paths=self._paths,
            options=self._options,
            plugin_name=self._plugin_name,
            extra=dict(self._extra),
            run_context=self._run_context,
        )


__all__ = [
    "ConfigProvider",
    "PluginExecutionContext",
    "PluginExecutionContextBuilder",
    "PluginScratch",
    "ResourceNotFoundError",
    "ResourceRegistry",
]
