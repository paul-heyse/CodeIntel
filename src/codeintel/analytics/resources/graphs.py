"""Graph resource provider for lazy graph loading.

This module provides `GraphProvider` which wraps `GraphRuntime` to provide
lazy loading of call, import, and symbol graphs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx as nx

from codeintel.analytics.resources.protocol import LazyResource

if TYPE_CHECKING:
    from codeintel.analytics.graph_runtime import GraphRuntime, GraphRuntimeOptions
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
        **kwargs: object,
    ) -> GraphProvider:
        """Create a provider from a gateway and snapshot.

        Parameters
        ----------
        gateway
            Storage gateway for graph data.
        snapshot
            Repository snapshot reference.
        **kwargs
            Additional options passed to GraphRuntimeOptions.

        Returns
        -------
        GraphProvider
            Configured provider.
        """
        from codeintel.analytics.graph_runtime import GraphRuntimeOptions

        options = GraphRuntimeOptions(snapshot=snapshot, **kwargs)  # type: ignore[arg-type]
        return cls(gateway=gateway, snapshot=snapshot, options=options)

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

        Raises
        ------
        ValueError
            If neither runtime nor gateway/snapshot are provided.
        """
        runtime = self._get_or_build_runtime()

        return GraphResources(
            call_graph=self._ensure_graph(runtime, "call_graph"),
            import_graph=self._ensure_graph(runtime, "import_graph"),
            symbol_module_graph=self._ensure_graph(runtime, "symbol_module_graph"),
            symbol_function_graph=self._ensure_graph(runtime, "symbol_function_graph"),
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

        from codeintel.analytics.graph_runtime import (
            GraphRuntimeOptions,
            build_graph_runtime,
        )

        options = self._options or GraphRuntimeOptions(snapshot=self._snapshot)
        self._runtime = build_graph_runtime(self._gateway, options)
        return self._runtime

    def _ensure_graph(
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
            except Exception:
                log.warning("Failed to load %s", graph_attr, exc_info=True)
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
