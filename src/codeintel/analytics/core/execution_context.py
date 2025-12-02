"""Slim execution context for analytics plugins.

This module provides a minimal, protocol-driven execution context that
replaces the bloated AnalyticsExecutionContext with 19+ nullable config fields.
Plugins request what they need through typed accessors.

Architecture
------------
The context supports two resource access patterns:
1. **Legacy**: Direct lazy properties (graph_runtime, catalog, analytics_context)
2. **New**: ResourceRegistry for typed resource access (resources.require(T))

Both patterns are supported for backward compatibility.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, cast

from codeintel.analytics.resources.registry import ResourceNotFoundError, ResourceRegistry
from codeintel.analytics.runtime_manifest import AnalyticsScope
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from codeintel.analytics.context import AnalyticsContext
    from codeintel.analytics.graph_runtime import GraphRuntime
    from codeintel.analytics.resources.protocol import ResourceProvider
    from codeintel.graphs.catalog import FunctionCatalogProvider

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

    def __init__(self, configs: Mapping[type[Any], object]) -> None:
        """Initialize with a mapping of config types to instances.

        Parameters
        ----------
        configs
            Mapping of config type to config instance.
        """
        self._configs = dict(configs)

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
    """Minimal execution context for analytics plugins.

    This replaces the bloated AnalyticsExecutionContext by providing:
    - Core required fields (gateway, snapshot, run_id, scope)
    - Typed config accessor (get_config)
    - ResourceRegistry for typed resource access (require)
    - Lazy resolution of expensive resources (graph_runtime, catalog)
    - Scratch store for inter-plugin communication

    Resource Access Patterns
    ------------------------
    1. **Typed registry** (preferred): `ctx.require(GraphProvider)`
    2. **Legacy properties**: `ctx.graph_runtime`, `ctx.catalog`
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    run_id: str
    scope: AnalyticsScope

    # Typed config accessor
    configs: ConfigProvider = field(default_factory=lambda: ConfigProvider({}))

    # Resource registry for typed resource access
    resources: ResourceRegistry = field(default_factory=ResourceRegistry)

    # Scratch for inter-plugin communication
    scratch: PluginScratch = field(default_factory=PluginScratch)

    # Plugin-specific options (validated by plugin)
    options: object | None = None

    # Current plugin name (set by executor)
    plugin_name: str | None = None

    # Additional metadata
    extra: MutableMapping[str, Any] = field(default_factory=dict)

    # Lazy-initialized resources (legacy support)
    _graph_runtime: GraphRuntime | None = field(default=None, repr=False)
    _graph_runtime_factory: Callable[[], GraphRuntime] | None = field(default=None, repr=False)
    _catalog_provider: FunctionCatalogProvider | None = field(default=None, repr=False)
    _catalog_factory: Callable[[], FunctionCatalogProvider] | None = field(default=None, repr=False)
    _analytics_context: AnalyticsContext | None = field(default=None, repr=False)
    _analytics_context_factory: Callable[[], AnalyticsContext] | None = field(
        default=None, repr=False
    )

    @property
    def repo(self) -> str:
        """Repository identifier."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def graph_runtime(self) -> GraphRuntime:
        """Lazily resolved graph runtime.

        Returns
        -------
        GraphRuntime
            Graph runtime for this execution.

        Raises
        ------
        ValueError
            If no graph runtime is available.
        """
        if self._graph_runtime is not None:
            return self._graph_runtime
        if self._graph_runtime_factory is not None:
            self._graph_runtime = self._graph_runtime_factory()
            return self._graph_runtime
        message = "Graph runtime not available in this context"
        raise ValueError(message)

    @property
    def catalog(self) -> FunctionCatalogProvider:
        """Lazily resolved function catalog provider.

        Returns
        -------
        FunctionCatalogProvider
            Catalog provider for this execution.

        Raises
        ------
        ValueError
            If no catalog is available.
        """
        if self._catalog_provider is not None:
            return self._catalog_provider
        if self._catalog_factory is not None:
            self._catalog_provider = self._catalog_factory()
            return self._catalog_provider
        message = "Function catalog not available in this context"
        raise ValueError(message)

    @property
    def analytics_context(self) -> AnalyticsContext:
        """Lazily resolved analytics context.

        Returns
        -------
        AnalyticsContext
            Analytics context for this execution.

        Raises
        ------
        ValueError
            If no analytics context is available.
        """
        if self._analytics_context is not None:
            return self._analytics_context
        if self._analytics_context_factory is not None:
            self._analytics_context = self._analytics_context_factory()
            return self._analytics_context
        message = "Analytics context not available"
        raise ValueError(message)

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

    def has_graph_runtime(self) -> bool:
        """Check if graph runtime is available.

        Returns
        -------
        bool
            True if graph runtime is available.
        """
        return self._graph_runtime is not None or self._graph_runtime_factory is not None

    def has_catalog(self) -> bool:
        """Check if function catalog is available.

        Returns
        -------
        bool
            True if catalog is available.
        """
        return self._catalog_provider is not None or self._catalog_factory is not None

    def has_analytics_context(self) -> bool:
        """Check if analytics context is available.

        Returns
        -------
        bool
            True if analytics context is available.
        """
        return self._analytics_context is not None or self._analytics_context_factory is not None

    def require(self, resource_type: type[T]) -> T:
        """Get a resource from the registry.

        Typed access to resources registered with the ResourceRegistry.
        Preferred over legacy properties for new code.

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

    def register_resource(
        self,
        resource_type: type[T],
        provider: ResourceProvider[T],
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

    Provides a fluent API for configuring execution contexts with
    support for both legacy lazy properties and the ResourceRegistry.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    run_id: str
    scope: AnalyticsScope = field(default_factory=AnalyticsScope)
    _configs: dict[type[Any], object] = field(default_factory=dict)
    _resources: ResourceRegistry = field(default_factory=ResourceRegistry)
    _extra: dict[str, Any] = field(default_factory=dict)
    _graph_runtime: GraphRuntime | None = None
    _graph_runtime_factory: Callable[[], GraphRuntime] | None = None
    _catalog_provider: FunctionCatalogProvider | None = None
    _catalog_factory: Callable[[], FunctionCatalogProvider] | None = None
    _analytics_context: AnalyticsContext | None = None
    _analytics_context_factory: Callable[[], AnalyticsContext] | None = None
    _options: object | None = None
    _plugin_name: str | None = None

    def with_config(self, config_type: type[T], config: T) -> PluginExecutionContextBuilder:
        """Add a configuration to the context.

        Parameters
        ----------
        config_type
            Type of configuration.
        config
            Configuration instance.

        Returns
        -------
        PluginExecutionContextBuilder
            Self for chaining.
        """
        self._configs[config_type] = config
        return self

    def with_graph_runtime(
        self,
        runtime: GraphRuntime | None = None,
        *,
        factory: Callable[[], GraphRuntime] | None = None,
    ) -> PluginExecutionContextBuilder:
        """Set the graph runtime or factory.

        Parameters
        ----------
        runtime
            Graph runtime instance.
        factory
            Factory function to create runtime lazily.

        Returns
        -------
        PluginExecutionContextBuilder
            Self for chaining.
        """
        self._graph_runtime = runtime
        self._graph_runtime_factory = factory
        return self

    def with_catalog(
        self,
        catalog: FunctionCatalogProvider | None = None,
        *,
        factory: Callable[[], FunctionCatalogProvider] | None = None,
    ) -> PluginExecutionContextBuilder:
        """Set the function catalog or factory.

        Parameters
        ----------
        catalog
            Catalog provider instance.
        factory
            Factory function to create catalog lazily.

        Returns
        -------
        PluginExecutionContextBuilder
            Self for chaining.
        """
        self._catalog_provider = catalog
        self._catalog_factory = factory
        return self

    def with_analytics_context(
        self,
        context: AnalyticsContext | None = None,
        *,
        factory: Callable[[], AnalyticsContext] | None = None,
    ) -> PluginExecutionContextBuilder:
        """Set the analytics context or factory.

        Parameters
        ----------
        context
            Analytics context instance.
        factory
            Factory function to create context lazily.

        Returns
        -------
        PluginExecutionContextBuilder
            Self for chaining.
        """
        self._analytics_context = context
        self._analytics_context_factory = factory
        return self

    def with_options(self, options: object) -> PluginExecutionContextBuilder:
        """Set plugin-specific options.

        Parameters
        ----------
        options
            Options object.

        Returns
        -------
        PluginExecutionContextBuilder
            Self for chaining.
        """
        self._options = options
        return self

    def with_plugin_name(self, name: str) -> PluginExecutionContextBuilder:
        """Set the current plugin name.

        Parameters
        ----------
        name
            Plugin name.

        Returns
        -------
        PluginExecutionContextBuilder
            Self for chaining.
        """
        self._plugin_name = name
        return self

    def with_extra(self, key: str, value: object) -> PluginExecutionContextBuilder:
        """Add extra metadata.

        Parameters
        ----------
        key
            Metadata key.
        value
            Metadata value.

        Returns
        -------
        PluginExecutionContextBuilder
            Self for chaining.
        """
        self._extra[key] = value
        return self

    def with_resource(
        self,
        resource_type: type[T],
        provider: ResourceProvider[T],
    ) -> PluginExecutionContextBuilder:
        """Register a resource provider.

        Parameters
        ----------
        resource_type
            Type key for the provider.
        provider
            Resource provider instance.

        Returns
        -------
        PluginExecutionContextBuilder
            Self for chaining.
        """
        self._resources.register(resource_type, provider)
        return self

    def with_resources(
        self,
        resources: ResourceRegistry,
    ) -> PluginExecutionContextBuilder:
        """Set the resource registry.

        Replaces any previously configured resources.

        Parameters
        ----------
        resources
            Resource registry to use.

        Returns
        -------
        PluginExecutionContextBuilder
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
            scope=self.scope,
            configs=ConfigProvider(self._configs),
            resources=self._resources,
            scratch=scratch or PluginScratch(),
            options=self._options,
            plugin_name=self._plugin_name,
            extra=dict(self._extra),
            _graph_runtime=self._graph_runtime,
            _graph_runtime_factory=self._graph_runtime_factory,
            _catalog_provider=self._catalog_provider,
            _catalog_factory=self._catalog_factory,
            _analytics_context=self._analytics_context,
            _analytics_context_factory=self._analytics_context_factory,
        )


__all__ = [
    "ConfigProvider",
    "PluginExecutionContext",
    "PluginExecutionContextBuilder",
    "PluginScratch",
    "ResourceNotFoundError",
]
