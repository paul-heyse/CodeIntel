"""Graph plugin execution context.

This module defines the execution context provided to graph plugins,
extending the unified `PluginExecutionContext` with graph-specific
functionality like `require_graphs()` and `GraphRunScope` support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.core.plugins.context import (
    ConfigProvider,
    PluginExecutionContext,
    PluginExecutionContextBuilder,
    PluginScratch,
)
from codeintel.core.resources import ResourceNotFoundError
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

    All resource access should go through the unified `ResourceRegistry`
    via inherited `require()` and `require_by_name()` methods.
    Use `require_graphs()` for convenient access to `GraphResource`.

    Attributes
    ----------
    scope
        Optional scoping for incremental graph execution.
    """

    scope: GraphRunScope | None = None
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

    def require_graphs(self) -> GraphResource:
        """Get the graph resource, raising if unavailable.

        This method provides access to graph data through resource injection.
        Use this instead of accessing ctx.engine directly.

        Returns
        -------
        GraphResource
            Graph resource for accessing call/import graphs.

        Raises
        ------
        RuntimeError
            If no GraphResource is registered in the context.
        """
        # Try by type first
        if self.has_resource(GraphResource):
            return self.require(GraphResource)

        # Try by name as fallback
        if self.has_resource_by_name(GraphResource.RESOURCE_NAME):
            resource = self.require_by_name(GraphResource.RESOURCE_NAME)
            if isinstance(resource, GraphResource):
                return resource

        message = "No GraphResource registered in context"
        raise RuntimeError(message)

    def has_graph_resource(self, name: str) -> bool:
        """Check if a resource is available by name.

        Parameters
        ----------
        name
            Resource name to check.

        Returns
        -------
        bool
            True if the resource is registered.
        """
        return self.has_resource_by_name(name)

    def require_graph_resource_by_name(self, name: str) -> object:
        """Get a resource by name.

        Parameters
        ----------
        name
            Resource name to look up.

        Returns
        -------
        object
            The resource value.

        Raises
        ------
        ResourceNotFoundError
            If the resource is not registered.
        """
        try:
            return self.require_by_name(name)
        except KeyError as exc:
            raise ResourceNotFoundError(name) from exc


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

    def register_graph_resource(
        self,
        provider: object,
    ) -> GraphPluginExecutionContextBuilder:
        """Register a resource provider in the unified registry.

        Parameters
        ----------
        provider
            Resource provider with a RESOURCE_NAME attribute.

        Returns
        -------
        GraphPluginExecutionContextBuilder
            Self for chaining.
        """
        self._resources.register_provider(provider)
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
            _catalog_provider=self._catalog_provider,
        )


__all__ = [
    "GraphPluginExecutionContext",
    "GraphPluginExecutionContextBuilder",
]
