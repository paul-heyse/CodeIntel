"""Unified graph resource types and protocols.

This module provides canonical graph container types and protocols that
unify the graph resource handling across analytics and graphs packages.

Types
-----
GraphBundle
    Unified container for all graph types.
GraphProviderProtocol
    Protocol for graph resource providers.

Example
-------
```python
from codeintel.core.resources.graphs import GraphBundle, GraphProviderProtocol


class MyGraphProvider:
    RESOURCE_NAME = "graphs"

    def get(self) -> GraphBundle:
        return GraphBundle(call_graph=my_call_graph)

    def invalidate(self) -> None:
        self._cached = None

    @property
    def call_graph(self) -> GraphInput | None:
        return self.get().call_graph
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from codeintel.build.graphs.rx.algos import GraphInput

T_co = TypeVar("T_co", covariant=True)


@dataclass
class GraphBundle:
    """Unified container for all graph types.

    This dataclass provides a standard structure for passing graph
    collections between components. Individual graphs may be None
    if not loaded or not applicable.

    Attributes
    ----------
    call_graph
        Function call graph (directed).
    import_graph
        Module import graph (directed).
    symbol_module_graph
        Symbol to module bipartite graph.
    symbol_function_graph
        Symbol to function bipartite graph.
    config_module_bipartite
        Config key to module bipartite graph.
    cfg_graph
        Control flow graph (directed), if available.

    Examples
    --------
    >>> bundle = GraphBundle(call_graph=my_call_graph)
    >>> if bundle.call_graph is not None:
    ...     print(f"Call graph has {bundle.call_graph.number_of_nodes()} nodes")
    """

    call_graph: GraphInput | None = None
    import_graph: GraphInput | None = None
    symbol_module_graph: GraphInput | None = None
    symbol_function_graph: GraphInput | None = None
    config_module_bipartite: GraphInput | None = None
    cfg_graph: GraphInput | None = None

    @classmethod
    def empty(cls) -> GraphBundle:
        """Create an empty graph bundle.

        Returns
        -------
        GraphBundle
            Bundle with all graphs set to None.
        """
        return cls()

    @property
    def has_call_graph(self) -> bool:
        """Check if call graph is available.

        Returns
        -------
        bool
            True if call_graph is not None.
        """
        return self.call_graph is not None

    @property
    def has_import_graph(self) -> bool:
        """Check if import graph is available.

        Returns
        -------
        bool
            True if import_graph is not None.
        """
        return self.import_graph is not None

    @property
    def available_graphs(self) -> tuple[str, ...]:
        """List names of available (non-None) graphs.

        Returns
        -------
        tuple[str, ...]
            Names of graphs that are not None.
        """
        names: list[str] = []
        if self.call_graph is not None:
            names.append("call_graph")
        if self.import_graph is not None:
            names.append("import_graph")
        if self.symbol_module_graph is not None:
            names.append("symbol_module_graph")
        if self.symbol_function_graph is not None:
            names.append("symbol_function_graph")
        if self.config_module_bipartite is not None:
            names.append("config_module_bipartite")
        if self.cfg_graph is not None:
            names.append("cfg_graph")
        return tuple(names)


@runtime_checkable
class GraphProviderProtocol(Protocol[T_co]):
    """Protocol for graph resource providers.

    This protocol defines the interface for providers that supply graph
    resources to plugins and analytics components. Providers support
    lazy loading and cache invalidation.

    Type Parameters
    ---------------
    T_co
        The type returned by get(), typically GraphBundle or a subclass.

    Attributes
    ----------
    RESOURCE_NAME
        Class variable identifying this provider type.

    Examples
    --------
    >>> class MyProvider:
    ...     RESOURCE_NAME = "graphs"
    ...
    ...     def get(self) -> GraphBundle: ...
    ...     def invalidate(self) -> None: ...
    ...     @property
    ...     def call_graph(self) -> GraphInput | None: ...
    """

    RESOURCE_NAME: ClassVar[str]

    def get(self) -> T_co:
        """Load and return the graph resources.

        Returns
        -------
        T_co
            The loaded graph bundle or resources.
        """
        ...

    def invalidate(self) -> None:
        """Invalidate cached graph resources.

        Force a reload on the next get() call.
        """
        ...

    @property
    def call_graph(self) -> GraphInput | None:
        """Access call graph directly.

        Returns
        -------
        GraphInput | None
            The call graph, or None if not available.
        """
        ...

    @property
    def import_graph(self) -> GraphInput | None:
        """Access import graph directly.

        Returns
        -------
        GraphInput | None
            The import graph, or None if not available.
        """
        ...


@runtime_checkable
class ExtendedGraphProviderProtocol(GraphProviderProtocol[T_co], Protocol[T_co]):
    """Extended protocol with all graph accessors.

    This protocol extends GraphProviderProtocol with accessors for
    all graph types, providing a complete interface for graph access.
    """

    @property
    def symbol_module_graph(self) -> GraphInput | None:
        """Access symbol-module bipartite graph.

        Returns
        -------
        GraphInput | None
            The symbol-module graph, or None if not available.
        """
        ...

    @property
    def symbol_function_graph(self) -> GraphInput | None:
        """Access symbol-function bipartite graph.

        Returns
        -------
        GraphInput | None
            The symbol-function graph, or None if not available.
        """
        ...

    @property
    def config_module_bipartite(self) -> GraphInput | None:
        """Access config-module bipartite graph.

        Returns
        -------
        GraphInput | None
            The config-module graph, or None if not available.
        """
        ...

    @property
    def cfg_graph(self) -> GraphInput | None:
        """Access control flow graph.

        Returns
        -------
        GraphInput | None
            The CFG, or None if not available.
        """
        ...


__all__ = [
    "ExtendedGraphProviderProtocol",
    "GraphBundle",
    "GraphProviderProtocol",
]
