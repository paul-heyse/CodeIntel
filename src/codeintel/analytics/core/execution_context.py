"""Slim execution context for analytics plugins.

This module provides a minimal, protocol-driven execution context that
replaces the bloated AnalyticsExecutionContext with 19+ nullable config fields.
Plugins request what they need through typed accessors.

Architecture
------------
The context uses ResourceRegistry for typed resource access:
- Access resources via `ctx.require(ProviderType)` or `ctx.require_or_none(ProviderType)`
- Common providers: GraphProvider, CatalogProvider, AstProvider, FeaturesProvider

All plugins have been migrated to use the resource provider pattern.
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
    from codeintel.analytics.resources.protocol import ResourceProvider

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
    - Scratch store for inter-plugin communication

    Resource Access
    ---------------
    Use `ctx.require(ProviderType)` to access resources:

    - `ctx.require(GraphProvider)` - Graph runtime access
    - `ctx.require(CatalogProvider)` - Function catalog
    - `ctx.require(AstProvider)` - Function AST data
    - `ctx.require(FeaturesProvider)` - Function AST features
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

    @property
    def repo(self) -> str:
        """Repository identifier."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

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

    def require_by_name(self, name: str) -> object:
        """Get a resource by string name.

        Use this for TYPE_CHECKING imports to avoid circular dependencies.

        Parameters
        ----------
        name
            String name of the provider (typically the class name).

        Returns
        -------
        object
            The loaded resource. Caller should cast to the expected type.
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
        provider: ResourceProvider[Any],
    ) -> None:
        """Register a resource provider.

        Parameters
        ----------
        resource_type
            Type key for the provider (typically the provider class).
        provider
            Resource provider instance.
        """
        self.resources.register(resource_type, provider)


@dataclass
class PluginExecutionContextBuilder:
    """Builder for constructing PluginExecutionContext instances.

    Provides a fluent API for configuring execution contexts using
    the ResourceRegistry pattern.

    Example
    -------
    >>> builder = PluginExecutionContextBuilder(gateway, snapshot, run_id)
    >>> builder = builder.with_resource_provider(GraphProvider, graph_provider)
    >>> ctx = builder.build()
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    run_id: str
    scope: AnalyticsScope = field(default_factory=AnalyticsScope)
    _configs: dict[type[Any], object] = field(default_factory=dict)
    _resources: ResourceRegistry = field(default_factory=ResourceRegistry)
    _extra: dict[str, Any] = field(default_factory=dict)
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
        provider: ResourceProvider[Any],
    ) -> PluginExecutionContextBuilder:
        """Register a resource provider.

        The resource_type is used as a lookup key and does not need to match
        the provider's generic type parameter.

        Parameters
        ----------
        resource_type
            Type key for the provider (typically the provider class).
        provider
            Resource provider instance.

        Returns
        -------
        PluginExecutionContextBuilder
            Self for chaining.
        """
        self._resources.register(resource_type, provider)
        return self

    def with_resource_provider(
        self,
        resource_type: type[T],
        provider: ResourceProvider[Any],
    ) -> PluginExecutionContextBuilder:
        """Register a resource provider (alias for with_resource).

        Parameters
        ----------
        resource_type
            Type key for the provider (typically the provider class).
        provider
            Resource provider instance.

        Returns
        -------
        PluginExecutionContextBuilder
            Self for chaining.
        """
        return self.with_resource(resource_type, provider)

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
        )


__all__ = [
    "ConfigProvider",
    "PluginExecutionContext",
    "PluginExecutionContextBuilder",
    "PluginScratch",
    "ResourceNotFoundError",
]
