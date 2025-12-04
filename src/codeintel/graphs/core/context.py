"""Graph plugin execution context.

This module defines the execution context provided to graph plugins,
extending the unified `PluginExecutionContext` with graph-specific
functionality like `require_graphs()` and `GraphRunScope` support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from codeintel.core.plugins.context import (
    ConfigProvider,
    PluginExecutionContext,
    PluginExecutionContextBuilder,
    PluginScratch,
)
from codeintel.core.resources import ResourceProvider
from codeintel.graphs.resources.container import ResourceContainer
from codeintel.graphs.resources.graphs import GraphResource

if TYPE_CHECKING:
    from codeintel.config.steps_graphs import GraphRunScope
    from codeintel.graphs.catalog import FunctionCatalogProvider


@dataclass
class GraphPluginExecutionContext(PluginExecutionContext):
    """Execution context for graph plugins.

    Extends `PluginExecutionContext` with graph-specific functionality:
    - `require_graphs()` method for accessing graph data
    - `scope` field for incremental execution
    - `catalog_provider` property for function catalog access
    - Dual-mode resource access via both `ResourceRegistry` and `ResourceContainer`

    All I/O access should go through the resource container via `require()`.
    Use `require_graphs()` to access graph data via `GraphResource`.

    Attributes
    ----------
    scope
        Optional scoping for incremental graph execution.
    graph_resources
        Graph-specific resource container for legacy compatibility.
    """

    scope: GraphRunScope | None = None
    graph_resources: ResourceContainer = field(default_factory=ResourceContainer)
    _catalog_provider: FunctionCatalogProvider | None = field(default=None, repr=False)

    @property
    def catalog_provider(self) -> FunctionCatalogProvider | None:
        """Get the function catalog provider.

        Returns
        -------
        FunctionCatalogProvider | None
            Catalog provider if available.
        """
        return self._catalog_provider

    def require[T](self, resource_type: type[T]) -> T:
        """Get a resource, checking graph_resources first.

        Overrides the base method to check the graph-specific
        ResourceContainer first, then fall back to the unified
        ResourceRegistry.

        Parameters
        ----------
        resource_type
            Type of resource to retrieve.

        Returns
        -------
        T
            The resource instance.
        """
        # Get resource name from type
        resource_name = getattr(resource_type, "RESOURCE_NAME", resource_type.__name__)

        # Check graph_resources container first
        if self.graph_resources.has(resource_name):
            return cast("T", self.graph_resources.require_by_name(resource_name))

        # Fall back to unified resources registry
        return super().require(resource_type)

    def require_graphs(self) -> GraphResource:
        """Get the graph resource, raising if unavailable.

        This method provides access to graph data through resource injection.
        Use this instead of accessing ctx.engine directly.

        Checks both the graph_resources container (preferred) and the
        unified resources registry for compatibility.

        Returns
        -------
        GraphResource
            Graph resource for accessing call/import graphs.

        Raises
        ------
        RuntimeError
            If no GraphResource is registered in the context.
        """
        # Try graph-specific container first
        if self.graph_resources.has(GraphResource.RESOURCE_NAME):
            return cast(
                "GraphResource", self.graph_resources.require_by_name(GraphResource.RESOURCE_NAME)
            )

        # Fall back to unified resources registry
        if self.has_resource(GraphResource):
            return self.require(GraphResource)

        message = "No GraphResource registered in context"
        raise RuntimeError(message)

    def has_graph_resource(self, name: str) -> bool:
        """Check if a resource is available in graph resources.

        Parameters
        ----------
        name
            Resource name to check.

        Returns
        -------
        bool
            True if the resource is registered.
        """
        return self.graph_resources.has(name)

    def require_graph_resource_by_name(self, name: str) -> object:
        """Get a resource by name from the graph resource container.

        Parameters
        ----------
        name
            Resource name to look up.

        Returns
        -------
        object
            The resource value.
        """
        return self.graph_resources.require_by_name(name)


@dataclass
class GraphPluginExecutionContextBuilder(PluginExecutionContextBuilder):
    """Builder for constructing GraphPluginExecutionContext instances.

    Extends the base builder with graph-specific configuration options.

    Example
    -------
    >>> builder = GraphPluginExecutionContextBuilder(gateway, snapshot, run_id)
    >>> builder = builder.with_scope(scope).with_catalog_provider(catalog)
    >>> ctx = builder.build_graph_context()
    """

    _scope: GraphRunScope | None = None
    _graph_resources: ResourceContainer = field(default_factory=ResourceContainer)
    _catalog_provider: FunctionCatalogProvider | None = None

    def with_scope(self, scope: GraphRunScope) -> GraphPluginExecutionContextBuilder:
        """Set the graph run scope.

        Parameters
        ----------
        scope
            Graph run scope for incremental execution.

        Returns
        -------
        GraphPluginExecutionContextBuilder
            Self for chaining.
        """
        self._scope = scope
        return self

    def with_catalog_provider(
        self,
        catalog_provider: FunctionCatalogProvider,
    ) -> GraphPluginExecutionContextBuilder:
        """Set the function catalog provider.

        Parameters
        ----------
        catalog_provider
            Function catalog provider instance.

        Returns
        -------
        GraphPluginExecutionContextBuilder
            Self for chaining.
        """
        self._catalog_provider = catalog_provider
        return self

    def with_graph_resources(
        self,
        graph_resources: ResourceContainer,
    ) -> GraphPluginExecutionContextBuilder:
        """Set the graph-specific resource container.

        Parameters
        ----------
        graph_resources
            Graph resource container.

        Returns
        -------
        GraphPluginExecutionContextBuilder
            Self for chaining.
        """
        self._graph_resources = graph_resources
        return self

    def register_graph_resource(
        self,
        provider: object,
    ) -> GraphPluginExecutionContextBuilder:
        """Register a resource provider in the graph container.

        Parameters
        ----------
        provider
            Resource provider with a `resource_name` attribute.

        Returns
        -------
        GraphPluginExecutionContextBuilder
            Self for chaining.
        """
        if isinstance(provider, ResourceProvider):
            self._graph_resources.register(provider)
        return self

    def build_graph_context(
        self,
        *,
        scratch: PluginScratch | None = None,
    ) -> GraphPluginExecutionContext:
        """Build the graph execution context.

        Parameters
        ----------
        scratch
            Optional shared scratch store.

        Returns
        -------
        GraphPluginExecutionContext
            Configured graph execution context.
        """
        return GraphPluginExecutionContext(
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
            graph_resources=self._graph_resources,
            _catalog_provider=self._catalog_provider,
        )


__all__ = [
    "GraphPluginExecutionContext",
    "GraphPluginExecutionContextBuilder",
]
