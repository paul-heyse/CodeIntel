"""GraphBundle provider for lazy graph loading.

This module provides `GraphProvider`, a graphs-domain resource that wraps
`GraphRuntime` to provide lazy loading of call, import, symbol, and bipartite graphs.

The `GraphBundle` type from `codeintel.core.resources.graphs` is used for bundling
graph resources together.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import networkx as nx

from codeintel.build.graphs.runtime import GraphRuntime, GraphRuntimeOptions, build_graph_runtime
from codeintel.core.resources import LazyResource
from codeintel.core.resources.graphs import GraphBundle

if TYPE_CHECKING:
    from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
    from codeintel.storage.gateway import StorageGateway


@runtime_checkable
class GraphRuntimeLike(Protocol):
    """Protocol for types providing graph attributes.

    This is a minimal interface for what GraphProvider.from_runtime actually
    needs to extract graphs. It allows test mocks to satisfy the type
    checker without implementing the full GraphRuntime class.

    Attributes
    ----------
    call_graph
        Optional call graph (directed).
    import_graph
        Optional import graph (directed).
    symbol_module_graph
        Optional symbol-to-module bipartite graph.
    symbol_function_graph
        Optional symbol-to-function bipartite graph.
    config_module_bipartite
        Optional config-to-module bipartite graph.
    test_function_bipartite
        Optional test-to-function bipartite graph.
    cfg_graph
        Optional control flow graph (directed).
    backend
        Optional backend configuration.
    use_gpu
        Whether GPU execution is enabled.
    """

    @property
    def call_graph(self) -> nx.DiGraph | None:
        """Call graph or None."""
        ...

    @property
    def import_graph(self) -> nx.DiGraph | None:
        """Import graph or None."""
        ...

    @property
    def symbol_module_graph(self) -> nx.Graph | None:
        """Symbol-module graph or None."""
        ...

    @property
    def symbol_function_graph(self) -> nx.Graph | None:
        """Symbol-function graph or None."""
        ...

    @property
    def config_module_bipartite(self) -> nx.Graph | None:
        """Config-module bipartite graph or None."""
        ...

    @property
    def test_function_bipartite(self) -> nx.Graph | None:
        """Test-function bipartite graph or None."""
        ...

    @property
    def cfg_graph(self) -> nx.DiGraph | None:
        """Control flow graph or None."""
        ...

    @property
    def backend(self) -> GraphBackendConfig | None:
        """Backend configuration or None."""
        ...

    @property
    def use_gpu(self) -> bool:
        """Whether GPU execution is enabled."""
        ...


log = logging.getLogger(__name__)


class GraphProvider(LazyResource[GraphBundle]):
    """Provider for graph resources with lazy loading.

    This provider wraps a `GraphRuntime` and exposes individual graphs
    with lazy loading. Graphs are loaded on first access and cached.

    Example
    -------
    >>> provider = GraphProvider.from_gateway(gateway, snapshot)
    >>> resources = provider.get()
    >>> call_graph = resources.call_graph
    """

    RESOURCE_NAME: ClassVar[str] = "GraphBundle"

    def __init__(
        self,
        *,
        gateway: StorageGateway | None = None,
        snapshot: SnapshotRef | None = None,
        runtime: GraphRuntime | GraphRuntimeLike | None = None,
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
            Pre-built GraphRuntime instance or GraphRuntimeLike mock.
        options
            Options for building a new runtime.
        """
        super().__init__("GraphBundle")
        self._gateway = gateway
        self._snapshot = snapshot

        self._runtime_internal: GraphRuntime | GraphRuntimeLike | None = runtime
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
        """
        resolved_options = (
            options if options is not None else GraphRuntimeOptions(snapshot=snapshot)
        )
        return cls(gateway=gateway, snapshot=snapshot, options=resolved_options)

    @classmethod
    def from_runtime(cls, runtime: GraphRuntime | GraphRuntimeLike) -> GraphProvider:
        """Create a provider from an existing runtime.

        Parameters
        ----------
        runtime
            Pre-built GraphRuntime instance or any GraphRuntimeLike protocol
            implementation (e.g., test mocks).

        Returns
        -------
        GraphProvider
            Provider wrapping the runtime.
        """
        return cls(runtime=runtime)

    def _load(self) -> GraphBundle:
        """Load graph resources.

        Returns
        -------
        GraphBundle
            Loaded graph resources.

        Notes
        -----
        May raise ValueError (via `_get_or_build_runtime`) if neither
        runtime nor gateway/snapshot are provided.
        """
        runtime = self._get_or_build_runtime()

        call_graph = _ensure_graph(runtime, "call_graph")
        import_graph = _ensure_graph(runtime, "import_graph")
        symbol_module_graph = _ensure_graph(runtime, "symbol_module_graph")
        symbol_function_graph = _ensure_graph(runtime, "symbol_function_graph")
        config_module_bipartite = _ensure_graph(runtime, "config_module_bipartite")
        test_function_bipartite = _ensure_graph(runtime, "test_function_bipartite")
        cfg_graph = _ensure_graph(runtime, "cfg_graph")

        if call_graph is not None and not isinstance(call_graph, nx.DiGraph):
            log.warning("call_graph is not a DiGraph, setting to None")
            call_graph = None
        if import_graph is not None and not isinstance(import_graph, nx.DiGraph):
            log.warning("import_graph is not a DiGraph, setting to None")
            import_graph = None
        if cfg_graph is not None and not isinstance(cfg_graph, nx.DiGraph):
            log.warning("cfg_graph is not a DiGraph, setting to None")
            cfg_graph = None

        return GraphBundle(
            call_graph=call_graph,
            import_graph=import_graph,
            symbol_module_graph=symbol_module_graph,
            symbol_function_graph=symbol_function_graph,
            config_module_bipartite=config_module_bipartite,
            test_function_bipartite=test_function_bipartite,
            cfg_graph=cfg_graph,
        )

    def _get_or_build_runtime(self) -> GraphRuntime | GraphRuntimeLike:
        """Get existing runtime or build a new one.

        Returns
        -------
        GraphRuntime | GraphRuntimeLike
            The runtime to use.

        Raises
        ------
        ValueError
            If insufficient configuration provided.
        """
        if self._runtime_internal is not None:
            return self._runtime_internal

        if self._gateway is None or self._snapshot is None:
            message = "GraphProvider requires either runtime or gateway+snapshot"
            raise ValueError(message)

        options = self._options or GraphRuntimeOptions(snapshot=self._snapshot)
        self._runtime_internal = build_graph_runtime(self._gateway, options)
        return self._runtime_internal

    @property
    def runtime(self) -> GraphRuntime | None:
        """Return the underlying runtime if available.

        Returns
        -------
        GraphRuntime | None
            The runtime, or None if not yet built. Returns None for
            GraphRuntimeLike test mocks (they don't implement full API).
        """
        if isinstance(self._runtime_internal, GraphRuntime):
            return self._runtime_internal
        return None

    @property
    def runtime_like(self) -> GraphRuntime | GraphRuntimeLike | None:
        """Return the underlying runtime or runtime-like mock if available.

        This property returns the internal runtime regardless of whether
        it's a full GraphRuntime or a test mock implementing GraphRuntimeLike.
        Use this for testing scenarios where you need to verify the stored
        runtime reference.

        Returns
        -------
        GraphRuntime | GraphRuntimeLike | None
            The runtime or runtime-like mock, or None if not yet built.
        """
        return self._runtime_internal

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

    @property
    def config_module_bipartite(self) -> nx.Graph | None:
        """Access config-module bipartite graph directly.

        Convenience property for direct access without calling `get()`.

        Returns
        -------
        nx.Graph | None
            The config-module bipartite graph, or None if unavailable.
        """
        return self.get().config_module_bipartite

    @property
    def test_function_bipartite(self) -> nx.Graph | None:
        """Access test-function bipartite graph directly.

        Convenience property for direct access without calling `get()`.

        Returns
        -------
        nx.Graph | None
            The test-function bipartite graph, or None if unavailable.
        """
        return self.get().test_function_bipartite

    @property
    def cfg_graph(self) -> nx.DiGraph | None:
        """Access control flow graph directly.

        Convenience property for direct access without calling `get()`.

        Returns
        -------
        nx.DiGraph | None
            The control flow graph, or None if unavailable.
        """
        return self.get().cfg_graph

    @property
    def backend(self) -> GraphBackendConfig | None:
        """Return the backend configuration for graph operations.

        Returns
        -------
        GraphBackendConfig | None
            The backend configuration, or None if runtime not built.
        """
        if self._runtime_internal is not None:
            return self._runtime_internal.backend
        return None

    @property
    def use_gpu(self) -> bool:
        """Return whether GPU execution is enabled.

        Returns
        -------
        bool
            True if GPU execution is enabled, False otherwise.
        """
        if self._runtime_internal is not None:
            return self._runtime_internal.use_gpu
        return False


def _ensure_graph(
    runtime: GraphRuntime | GraphRuntimeLike,
    graph_attr: str,
) -> nx.DiGraph | nx.Graph | None:
    """Ensure a specific graph is loaded.

    Returns
    -------
    nx.DiGraph | nx.Graph | None
        The loaded graph instance or None if unavailable.
    """
    ensure_method = f"ensure_{graph_attr}"
    if hasattr(runtime, ensure_method):
        try:
            return getattr(runtime, ensure_method)()
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            log.warning("Failed to load %s: %s", graph_attr, exc, exc_info=True)
            return None
    return getattr(runtime, graph_attr, None)


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
    "GraphBundle",
    "GraphProvider",
    "GraphRuntimeLike",
    "SingleGraphProvider",
]
