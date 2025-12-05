"""Analytics plugin execution context.

This module provides the execution context for analytics plugins, extending
the unified `PluginExecutionContext` from `codeintel.core.plugins.context`
with analytics-specific functionality like `AnalyticsScope`.

Architecture
------------
**Inheritance from Core**
- `PluginExecutionContext` extends `CorePluginExecutionContext` from core
- `PluginExecutionContextBuilder` extends `CoreContextBuilder` from core
- All core fields (gateway, snapshot, run_id, resources, configs, scratch,
  paths, options, plugin_name, extra, run_context) are inherited
- Analytics adds: `scope` (AnalyticsScope)

**Resource Registry Pattern**
The context uses ResourceRegistry for typed resource access:
- Access resources via `ctx.require(ProviderType)` or `ctx.require_or_none(ProviderType)`
- Common providers: GraphProvider, CatalogProvider, AstProvider, FeaturesProvider

All plugins have been migrated to use the resource provider pattern.

See Also
--------
codeintel.core.plugins.context.PluginExecutionContext
    Core execution context that this class extends.
codeintel.core.plugins.context.PluginExecutionContextBuilder
    Core builder that PluginExecutionContextBuilder extends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self, TypeVar

from codeintel.analytics.runtime.manifest import AnalyticsScope
from codeintel.core.plugins.execution.context import (
    ConfigProvider,
    PluginScratch,
)
from codeintel.core.plugins.execution.context import (
    PluginExecutionContext as CorePluginExecutionContext,
)
from codeintel.core.plugins.execution.context import (
    PluginExecutionContextBuilder as CoreContextBuilder,
)
from codeintel.core.resources import ResourceNotFoundError, ResourceRegistry

T = TypeVar("T")


@dataclass
class PluginExecutionContext(CorePluginExecutionContext):
    """Execution context for analytics plugins.

    Extends `CorePluginExecutionContext` with analytics-specific functionality:
    - `scope` field for analytics execution scope

    This replaces the bloated AnalyticsExecutionContext by providing:
    - Core required fields (gateway, snapshot, run_id)
    - Analytics-specific scope field
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

    scope: AnalyticsScope = field(default_factory=AnalyticsScope)


@dataclass
class PluginExecutionContextBuilder(CoreContextBuilder):
    """Builder for constructing analytics PluginExecutionContext instances.

    Extends the core builder with analytics-specific configuration options.

    Provides a fluent API for configuring execution contexts using
    the ResourceRegistry pattern.

    Example
    -------
    >>> builder = PluginExecutionContextBuilder(gateway, snapshot, run_id)
    >>> builder = builder.with_scope(scope).with_resource(GraphProvider, provider)
    >>> ctx = builder.build()
    """

    _scope: AnalyticsScope = field(default_factory=AnalyticsScope)

    def with_scope(self, scope: AnalyticsScope) -> Self:
        """Set the analytics execution scope.

        Parameters
        ----------
        scope
            Analytics scope for execution.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._scope = scope
        return self

    def with_resource(
        self,
        resource_type: type[T],
        provider: object,
    ) -> Self:
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
        Self
            Self for chaining.
        """
        self._resources.register(resource_type, provider)
        return self

    def with_resource_provider(
        self,
        resource_type: type[T],
        provider: object,
    ) -> Self:
        """Register a resource provider (alias for with_resource).

        Parameters
        ----------
        resource_type
            Type key for the provider (typically the provider class).
        provider
            Resource provider instance.

        Returns
        -------
        Self
            Self for chaining.
        """
        return self.with_resource(resource_type, provider)

    def build(self, *, scratch: PluginScratch | None = None) -> PluginExecutionContext:
        """Build the analytics execution context.

        Parameters
        ----------
        scratch
            Optional shared scratch store.

        Returns
        -------
        PluginExecutionContext
            Configured analytics execution context.
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
            scope=self._scope,
        )


# Re-export core types for backward compatibility
__all__ = [
    "ConfigProvider",
    "PluginExecutionContext",
    "PluginExecutionContextBuilder",
    "PluginScratch",
    "ResourceNotFoundError",
    "ResourceRegistry",
]
