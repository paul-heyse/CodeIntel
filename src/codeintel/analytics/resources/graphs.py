"""Graph resource provider for lazy graph loading.

This module provides `GraphProvider` which wraps `GraphRuntime` to provide
lazy loading of call, import, and symbol graphs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx as nx

from codeintel.analytics.graph_runtime import (
    GraphRuntimeOptions,
    build_graph_runtime,
)
from codeintel.analytics.resources.protocol import LazyResource

if TYPE_CHECKING:
    from codeintel.analytics.graph_runtime import GraphRuntime
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass
class GraphResources:
    """Container for loaded graph resources.

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
    """

    call_graph: nx.DiGraph | None = None
    import_graph: nx.DiGraph | None = None
    symbol_module_graph: nx.Graph | None = None
    symbol_function_graph: nx.Graph | None = None


class GraphProvider(LazyResource[GraphResources]):
    """Provider for graph resources with lazy loading.

    This provider wraps a `GraphRuntime` and exposes individual graphs
    with lazy loading. Graphs are loaded on first access and cached.

    Example
    -------
    >>> provider = GraphProvider.from_gateway(gateway, snapshot)
    >>> resources = provider.get()
    >>> call_graph = resources.call_graph
    """

    def __init__(
        self,
        *,
        gateway: StorageGateway | None = None,
        snapshot: SnapshotRef | None = None,
        runtime: GraphRuntime | None = None,
        options: GraphRuntimeOptions | None = None,
    ) -> None:
        """Initialize the graph provider.

        Provide either:
        - A pre-built `runtime`
        - A `gateway` and `snapshot` to build a runtime
        - Options to customize runtime building

        Parameters
        ----------
        gateway
            Storage gateway for building runtime.
        snapshot
            Snapshot reference for building runtime.
        runtime
            Pre-built GraphRuntime instance.
        options
            Options for building a new runtime.
        """
        super().__init__("GraphResources")
        self._gateway = gateway
        self._snapshot = snapshot
        self._runtime = runtime
        self._options = options

    @classmethod
    def from_gateway(
        cls,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        *,
        options: GraphRuntimeOptions | None = None,
    ) -> GraphProvider:
        """Create a provider from a gateway and snapshot.

        Parameters
        ----------
        gateway
            Storage gateway for graph data.
        snapshot
            Repository snapshot reference.
        options
            Optional runtime options. If not provided, default options
            with the given snapshot will be used.

        Returns
        -------
        GraphProvider
            Configured provider.

        Example
        -------
        >>> from codeintel.analytics.graph_runtime import GraphRuntimeOptions
        >>> from codeintel.graphs.engine import GraphKind
        >>>
        >>> # With default options
        >>> provider = GraphProvider.from_gateway(gateway, snapshot)
        >>>
        >>> # With custom options
        >>> opts = GraphRuntimeOptions(snapshot=snapshot, graphs=GraphKind.CALL)
        >>> provider = GraphProvider.from_gateway(gateway, snapshot, options=opts)
        """
        resolved_options = (
            options if options is not None else GraphRuntimeOptions(snapshot=snapshot)
        )
        return cls(gateway=gateway, snapshot=snapshot, options=resolved_options)

    @classmethod
    def from_runtime(cls, runtime: GraphRuntime) -> GraphProvider:
        """Create a provider from an existing runtime.

        Parameters
        ----------
        runtime
            Pre-built GraphRuntime instance.

        Returns
        -------
        GraphProvider
            Provider wrapping the runtime.
        """
        return cls(runtime=runtime)

    def _load(self) -> GraphResources:
        """Load graph resources.

        Returns
        -------
        GraphResources
            Loaded graph resources.

        Notes
        -----
        May raise ValueError (via `_get_or_build_runtime`) if neither
        runtime nor gateway/snapshot are provided.
        """
        runtime = self._get_or_build_runtime()

        # Load graphs with type narrowing for DiGraph fields
        call_graph = self._ensure_graph(runtime, "call_graph")
        import_graph = self._ensure_graph(runtime, "import_graph")
        symbol_module_graph = self._ensure_graph(runtime, "symbol_module_graph")
        symbol_function_graph = self._ensure_graph(runtime, "symbol_function_graph")

        # Type narrow: call_graph and import_graph must be DiGraph or None
        if call_graph is not None and not isinstance(call_graph, nx.DiGraph):
            log.warning("call_graph is not a DiGraph, setting to None")
            call_graph = None
        if import_graph is not None and not isinstance(import_graph, nx.DiGraph):
            log.warning("import_graph is not a DiGraph, setting to None")
            import_graph = None

        return GraphResources(
            call_graph=call_graph,
            import_graph=import_graph,
            symbol_module_graph=symbol_module_graph,
            symbol_function_graph=symbol_function_graph,
        )

    def _get_or_build_runtime(self) -> GraphRuntime:
        """Get existing runtime or build a new one.

        Returns
        -------
        GraphRuntime
            The runtime to use.

        Raises
        ------
        ValueError
            If insufficient configuration provided.
        """
        if self._runtime is not None:
            return self._runtime

        if self._gateway is None or self._snapshot is None:
            message = "GraphProvider requires either runtime or gateway+snapshot"
            raise ValueError(message)

        options = self._options or GraphRuntimeOptions(snapshot=self._snapshot)
        self._runtime = build_graph_runtime(self._gateway, options)
        return self._runtime

    def _ensure_graph(  # noqa: PLR6301
        self,
        runtime: GraphRuntime,
        graph_attr: str,
    ) -> nx.DiGraph | nx.Graph | None:
        """Ensure a specific graph is loaded.

        Parameters
        ----------
        runtime
            The graph runtime.
        graph_attr
            Attribute name of the graph to load.

        Returns
        -------
        nx.DiGraph | nx.Graph | None
            The loaded graph, or None if not available.
        """
        ensure_method = f"ensure_{graph_attr}"
        if hasattr(runtime, ensure_method):
            try:
                return getattr(runtime, ensure_method)()
            except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                log.warning("Failed to load %s: %s", graph_attr, e, exc_info=True)
                return None
        return getattr(runtime, graph_attr, None)

    @property
    def runtime(self) -> GraphRuntime | None:
        """Return the underlying runtime if available.

        Returns
        -------
        GraphRuntime | None
            The runtime, or None if not yet built.
        """
        return self._runtime

    @property
    def call_graph(self) -> nx.DiGraph | None:
        """Access call graph directly, loading resources if needed.

        Convenience property for direct access without calling `get()`.

        Returns
        -------
        nx.DiGraph | None
            The function call graph, or None if unavailable.
        """
        return self.get().call_graph

    @property
    def import_graph(self) -> nx.DiGraph | None:
        """Access import graph directly, loading resources if needed.

        Convenience property for direct access without calling `get()`.

        Returns
        -------
        nx.DiGraph | None
            The module import graph, or None if unavailable.
        """
        return self.get().import_graph

    @property
    def symbol_module_graph(self) -> nx.Graph | None:
        """Access symbol-module graph directly, loading resources if needed.

        Convenience property for direct access without calling `get()`.

        Returns
        -------
        nx.Graph | None
            The symbol-module bipartite graph, or None if unavailable.
        """
        return self.get().symbol_module_graph

    @property
    def symbol_function_graph(self) -> nx.Graph | None:
        """Access symbol-function graph directly, loading resources if needed.

        Convenience property for direct access without calling `get()`.

        Returns
        -------
        nx.Graph | None
            The symbol-function bipartite graph, or None if unavailable.
        """
        return self.get().symbol_function_graph


@dataclass
class SingleGraphProvider(LazyResource[nx.DiGraph]):
    """Provider for a single graph type.

    Use this for fine-grained lazy loading of individual graph types.
    """

    _parent: GraphProvider = field(repr=False)
    _graph_name: str = "call_graph"

    def __init__(self, parent: GraphProvider, graph_name: str) -> None:
        """Initialize single graph provider.

        Parameters
        ----------
        parent
            Parent graph provider.
        graph_name
            Name of the graph attribute to access.
        """
        super().__init__(f"Graph:{graph_name}")
        self._parent = parent
        self._graph_name = graph_name

    def _load(self) -> nx.DiGraph:
        """Load the specific graph.

        Returns
        -------
        nx.DiGraph
            The loaded graph.

        Raises
        ------
        ValueError
            If the graph is not available.
        """
        resources = self._parent.get()
        graph = getattr(resources, self._graph_name, None)
        if graph is None:
            message = f"Graph {self._graph_name} not available"
            raise ValueError(message)
        return graph


__all__ = [
    "GraphProvider",
    "GraphResources",
    "SingleGraphProvider",
]
