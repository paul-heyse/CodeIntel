"""Unified graph runtime/engine test double.

Loads graphs from seeded DuckDB tables when a gateway/snapshot is provided,
otherwise serves configured NetworkX graphs with defensive copies. All public
methods record call names for assertions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Final, TypedDict, TypeVar, Unpack, cast

import networkx as nx
from duckdb import Error as DuckDBError

from codeintel.analytics.resources.graphs import GraphRuntimeLike
from codeintel.analytics.runtime import GraphRuntimeOptions
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.engine.protocol import GraphEngine, GraphKind
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from codeintel.config.primitives import GraphBackendConfig
    from tests._helpers.graphs import GraphFixtures

CALL_GRAPH_TABLE: Final[str] = "graph.call_graph_edges"
IMPORT_GRAPH_TABLE: Final[str] = "graph.import_graph_edges"
SYMBOL_EDGE_TABLE: Final[str] = "graph.symbol_use_edges"


class GraphRuntimeInitKwargs(TypedDict, total=False):
    """Initialization kwargs for GraphRuntimeDouble."""

    gateway: StorageGateway | None
    snapshot: SnapshotRef | None
    call_graph: nx.DiGraph | None
    import_graph: nx.DiGraph | None
    symbol_module_graph: nx.Graph | None
    symbol_function_graph: nx.Graph | None
    config_graph: nx.Graph | None
    test_function_graph: nx.Graph | None
    cfg_graph: nx.DiGraph | None
    backend: GraphBackendConfig | None
    use_gpu: bool
    call_graph_obj: nx.DiGraph | None
    import_graph_obj: nx.DiGraph | None
    symbol_module_graph_obj: nx.Graph | None
    symbol_function_graph_obj: nx.Graph | None
    config_bipartite_obj: nx.Graph | None
    test_function_bipartite_obj: nx.Graph | None
    copy_graphs: bool
    calls: list[GraphCallRecord]


class GraphEngineSeedKwargs(TypedDict, total=False):
    """Graph seeds for building GraphEngine-compatible doubles."""

    call_graph: nx.DiGraph | None
    import_graph: nx.DiGraph | None
    symbol_module_graph: nx.Graph | None
    symbol_function_graph: nx.Graph | None
    config_graph: nx.Graph | None
    test_function_graph: nx.Graph | None
    cfg_graph: nx.DiGraph | None
    backend: GraphBackendConfig | None
    use_gpu: bool
    copy_graphs: bool


@dataclass
class GraphCallRecord:
    """Record of a graph retrieval call."""

    method: str


class GraphCallRecorder:
    """Recorder for graph calls with optional per-method counting."""

    def __init__(self, calls: list[GraphCallRecord] | None = None) -> None:
        self.calls: list[GraphCallRecord] = list(calls) if calls else []

    def record(self, method: str) -> GraphCallRecord:
        """Append a call record for the given method name.

        Returns
        -------
        GraphCallRecord
            The created call record.
        """
        record = GraphCallRecord(method=method)
        self.calls.append(record)
        return record

    def increment(self, method: str) -> int:
        """Record a call and return the count for that method.

        Returns
        -------
        int
            The number of calls recorded for the given method.
        """
        self.record(method)
        return self.count(method)

    def count(self, method: str | None = None) -> int:
        """Return total call count or per-method count when provided.

        Returns
        -------
        int
            Total call count, or count of a specific method when supplied.
        """
        if method is None:
            return len(self.calls)
        return sum(1 for call in self.calls if call.method == method)


@dataclass(init=False)
class GraphRuntimeDouble(GraphRuntimeLike):
    """Graph runtime/engine stand-in with DB-backed loading and call recording."""

    gateway: StorageGateway | None = None
    snapshot: SnapshotRef | None = None
    call_graph_obj: nx.DiGraph | None = None
    import_graph_obj: nx.DiGraph | None = None
    symbol_module_graph_obj: nx.Graph | None = None
    symbol_function_graph_obj: nx.Graph | None = None
    config_bipartite_obj: nx.Graph | None = None
    test_function_bipartite_obj: nx.Graph | None = None
    _cfg_graph_internal: nx.DiGraph | None = None
    _backend_internal: GraphBackendConfig | None = None
    _use_gpu_internal: bool = False
    _recorder: GraphCallRecorder
    copy_graphs: bool = True
    calls: list[GraphCallRecord] = field(default_factory=list)

    def __init__(self, **kwargs: Unpack[GraphRuntimeInitKwargs]) -> None:
        """Initialize the runtime with optional pre-seeded graphs."""
        self.gateway = kwargs.get("gateway")
        self.snapshot = kwargs.get("snapshot")
        self.call_graph_obj = kwargs.get("call_graph_obj") or kwargs.get("call_graph")
        self.import_graph_obj = kwargs.get("import_graph_obj") or kwargs.get("import_graph")
        self.symbol_module_graph_obj = kwargs.get("symbol_module_graph_obj") or kwargs.get(
            "symbol_module_graph"
        )
        self.symbol_function_graph_obj = kwargs.get("symbol_function_graph_obj") or kwargs.get(
            "symbol_function_graph"
        )
        self.config_bipartite_obj = kwargs.get("config_bipartite_obj") or kwargs.get("config_graph")
        self.test_function_bipartite_obj = kwargs.get("test_function_bipartite_obj") or kwargs.get(
            "test_function_graph"
        )
        self._cfg_graph_internal = kwargs.get("cfg_graph")
        self._backend_internal = kwargs.get("backend")
        self._use_gpu_internal = kwargs.get("use_gpu", False)
        self.copy_graphs = kwargs.get("copy_graphs", True)
        self._recorder = GraphCallRecorder(kwargs.get("calls"))
        self.calls = self._recorder.calls

    @classmethod
    def from_fixtures(
        cls,
        fixtures: GraphFixtures | None = None,
        *,
        gateway: StorageGateway | None = None,
        snapshot: SnapshotRef | None = None,
        backend: GraphBackendConfig | None = None,
        copy_graphs: bool = True,
    ) -> GraphRuntimeDouble:
        """Construct from standard graph fixtures.

        Returns
        -------
        GraphRuntimeDouble
            Runtime seeded with provided fixtures.
        """
        if fixtures is None:
            graphs_module = import_module("tests._helpers.graphs")
            graphs = graphs_module.standard_graph_fixtures()
        else:
            graphs = fixtures
        return cls(
            gateway=gateway,
            snapshot=snapshot,
            call_graph_obj=graphs.call_graph,
            import_graph_obj=graphs.import_graph,
            symbol_module_graph_obj=graphs.symbol_module_graph,
            symbol_function_graph_obj=graphs.symbol_function_graph,
            config_bipartite_obj=graphs.config_graph,
            test_function_bipartite_obj=nx.Graph(),
            cfg_graph=graphs.cfg_graph,
            backend=backend,
            copy_graphs=copy_graphs,
        )

    # ------------------------------------------------------------------ GraphRuntimeLike
    @property
    def call_graph(self) -> nx.DiGraph | None:
        self._recorder.record("call_graph")
        graph = self.call_graph_obj
        if graph is None and self.gateway and self.snapshot:
            graph = self._load_call_graph_from_db()
            self.call_graph_obj = graph
        return cast("nx.DiGraph | None", self._clone_graph(graph))

    @property
    def import_graph(self) -> nx.DiGraph | None:
        self._recorder.record("import_graph")
        graph = self.import_graph_obj
        if graph is None and self.gateway and self.snapshot:
            graph = self._load_import_graph_from_db()
            self.import_graph_obj = graph
        return cast("nx.DiGraph | None", self._clone_graph(graph))

    @property
    def symbol_module_graph(self) -> nx.Graph | None:
        self._recorder.record("symbol_module_graph")
        graph = self.symbol_module_graph_obj
        if graph is None and self.gateway and self.snapshot:
            graph = self._load_symbol_graph_from_db()
            self.symbol_module_graph_obj = graph
        return self._clone_graph(graph)

    @property
    def symbol_function_graph(self) -> nx.Graph | None:
        self._recorder.record("symbol_function_graph")
        return self._clone_graph(self.symbol_function_graph_obj)

    @property
    def config_module_bipartite(self) -> nx.Graph | None:
        self._recorder.record("config_module_bipartite")
        return self._clone_graph(self.config_bipartite_obj)

    @property
    def test_function_bipartite(self) -> nx.Graph | None:
        self._recorder.record("test_function_bipartite")
        return self._clone_graph(self.test_function_bipartite_obj)

    @property
    def cfg_graph(self) -> nx.DiGraph | None:
        self._recorder.record("cfg_graph")
        return cast("nx.DiGraph | None", self._clone_graph(self._cfg_graph_internal))

    @cfg_graph.setter
    def cfg_graph(self, graph: nx.DiGraph | None) -> None:
        self._cfg_graph_internal = graph

    @property
    def backend(self) -> GraphBackendConfig | None:
        return self._backend_internal

    @backend.setter
    def backend(self, value: GraphBackendConfig | None) -> None:
        self._backend_internal = value

    @property
    def use_gpu(self) -> bool:
        return self._use_gpu_internal

    @use_gpu.setter
    def use_gpu(self, value: bool) -> None:
        self._use_gpu_internal = value

    def ensure_call_graph(self) -> nx.DiGraph | None:
        self._recorder.record("ensure_call_graph")
        return self._graph_or_db(
            "call_graph_obj",
            loader=self._load_call_graph_from_db,
            default_type=nx.DiGraph,
            return_default_on_missing=False,
        )

    def ensure_import_graph(self) -> nx.DiGraph | None:
        self._recorder.record("ensure_import_graph")
        return self._graph_or_db(
            "import_graph_obj",
            loader=self._load_import_graph_from_db,
            default_type=nx.DiGraph,
            return_default_on_missing=False,
        )

    def ensure_symbol_module_graph(self) -> nx.Graph | None:
        self._recorder.record("ensure_symbol_module_graph")
        return self._graph_or_db(
            "symbol_module_graph_obj",
            loader=self._load_symbol_graph_from_db,
            default_type=nx.Graph,
            return_default_on_missing=False,
        )

    def ensure_symbol_function_graph(self) -> nx.Graph | None:
        self._recorder.record("ensure_symbol_function_graph")
        return self._graph_or_db(
            "symbol_function_graph_obj",
            loader=lambda: None,
            default_type=nx.Graph,
            return_default_on_missing=False,
        )

    def ensure_config_module_bipartite(self) -> nx.Graph | None:
        self._recorder.record("ensure_config_module_bipartite")
        return self._graph_or_db(
            "config_bipartite_obj",
            loader=lambda: None,
            default_type=nx.Graph,
            return_default_on_missing=False,
        )

    def ensure_test_function_bipartite(self) -> nx.Graph | None:
        self._recorder.record("ensure_test_function_bipartite")
        return self._graph_or_db(
            "test_function_bipartite_obj",
            loader=lambda: None,
            default_type=nx.Graph,
            return_default_on_missing=False,
        )

    def clear_graphs(self) -> None:
        """Clear any cached graph objects."""
        for attr in (
            "call_graph_obj",
            "import_graph_obj",
            "symbol_module_graph_obj",
            "symbol_function_graph_obj",
            "config_bipartite_obj",
            "test_function_bipartite_obj",
            "_cfg_graph_internal",
        ):
            if hasattr(self, attr):
                setattr(self, attr, None)

    def ensure_cfg_graph(self) -> nx.DiGraph | None:
        self._recorder.record("ensure_cfg_graph")
        return self._graph_or_db(
            "_cfg_graph_internal",
            loader=lambda: None,
            default_type=nx.DiGraph,
            return_default_on_missing=False,
        )

    # ------------------------------------------------------------------ Internal helpers
    def _graph_or_db(
        self,
        attr: str,
        *,
        loader: Callable[[], _GraphT | None],
        default_type: type[_GraphT],
        return_default_on_missing: bool = True,
    ) -> _GraphT | None:
        graph_candidate = getattr(self, attr, None)
        graph: _GraphT | None = (
            graph_candidate if isinstance(graph_candidate, default_type) else None
        )
        if graph is None and self.gateway and self.snapshot:
            graph = loader()
        if graph is None and return_default_on_missing:
            graph = default_type()
        return cast("_GraphT | None", self._clone_graph(graph))

    def _clone_graph(self, graph: nx.Graph | None) -> nx.Graph | None:
        if graph is None:
            return None
        if self.copy_graphs and hasattr(graph, "copy"):
            copied = graph.copy()
            return cast("nx.Graph", copied)
        return graph

    def _load_call_graph_from_db(self) -> nx.DiGraph | None:
        if self.gateway is None or self.snapshot is None:
            return None
        graph = nx.DiGraph()
        try:
            rows = self.gateway.con.execute(
                "SELECT caller_goid_h128, callee_goid_h128, confidence "
                "FROM graph.call_graph_edges WHERE repo=? AND commit=?",
                [self.snapshot.repo, self.snapshot.commit],
            ).fetchall()
        except DuckDBError:
            return None
        for caller, callee, weight in rows:
            if callee is None:
                continue
            graph.add_edge(caller, callee, weight=float(weight))
        return graph

    def _load_import_graph_from_db(self) -> nx.DiGraph | None:
        if self.gateway is None or self.snapshot is None:
            return None
        graph = nx.DiGraph()
        try:
            rows = self.gateway.con.execute(
                "SELECT src_module, dst_module FROM graph.import_graph_edges "
                "WHERE repo=? AND commit=?",
                [self.snapshot.repo, self.snapshot.commit],
            ).fetchall()
        except DuckDBError:
            return None
        for src, dst in rows:
            graph.add_edge(src, dst, weight=1.0)
        return graph

    def _load_symbol_graph_from_db(self) -> nx.Graph | None:
        if self.gateway is None or self.snapshot is None:
            return None
        graph = nx.Graph()
        try:
            rows = self.gateway.con.execute(
                "SELECT def_path, use_path FROM graph.symbol_use_edges"
            ).fetchall()
        except DuckDBError:
            return None
        for defin, use in rows:
            graph.add_edge(defin, use, weight=1.0)
        return graph


class GraphEngineAdapter(GraphEngine):
    """Adapter that exposes GraphRuntimeDouble as a GraphEngine."""

    def __init__(
        self,
        runtime: GraphRuntimeDouble,
        *,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        self._runtime = runtime
        self.gateway = gateway
        self._snapshot = snapshot
        if self._runtime.gateway is None:
            self._runtime.gateway = gateway
        if self._runtime.snapshot is None:
            self._runtime.snapshot = snapshot

    @property
    def snapshot(self) -> SnapshotRef:
        return self._snapshot

    @property
    def use_gpu(self) -> bool:
        return self._runtime.use_gpu

    @property
    def repo(self) -> str:
        return self._snapshot.repo

    @property
    def commit(self) -> str:
        return self._snapshot.commit

    def call_graph(self) -> nx.DiGraph:
        return self._ensure_graph(self._runtime.ensure_call_graph, nx.DiGraph)

    def load_call_graph(self) -> nx.DiGraph:
        return self._ensure_graph(self._runtime.ensure_call_graph, nx.DiGraph)

    def import_graph(self) -> nx.DiGraph:
        return self._ensure_graph(self._runtime.ensure_import_graph, nx.DiGraph)

    def load_import_graph(self) -> nx.DiGraph:
        return self._ensure_graph(self._runtime.ensure_import_graph, nx.DiGraph)

    def symbol_module_graph(self) -> nx.Graph:
        return self._ensure_graph(self._runtime.ensure_symbol_module_graph, nx.Graph)

    def load_symbol_module_graph(self) -> nx.Graph:
        return self._ensure_graph(self._runtime.ensure_symbol_module_graph, nx.Graph)

    def symbol_function_graph(self) -> nx.Graph:
        return self._ensure_graph(self._runtime.ensure_symbol_function_graph, nx.Graph)

    def load_symbol_function_graph(self) -> nx.Graph:
        return self._ensure_graph(self._runtime.ensure_symbol_function_graph, nx.Graph)

    def config_module_bipartite(self) -> nx.Graph:
        return self._ensure_graph(self._runtime.ensure_config_module_bipartite, nx.Graph)

    def load_config_module_bipartite(self) -> nx.Graph:
        return self._ensure_graph(self._runtime.ensure_config_module_bipartite, nx.Graph)

    def test_function_bipartite(self) -> nx.Graph:
        return self._ensure_graph(self._runtime.ensure_test_function_bipartite, nx.Graph)

    def load_test_function_bipartite(self) -> nx.Graph:
        return self._ensure_graph(self._runtime.ensure_test_function_bipartite, nx.Graph)

    def clear_cache(self) -> None:
        """Clear any cached graphs on the runtime."""
        self._runtime.clear_graphs()

    def close(self) -> None:
        if self._runtime.gateway is not None:
            self._runtime.gateway.close()

    @staticmethod
    def _ensure_graph(
        loader: Callable[[], _GraphT | None],
        graph_type: type[_GraphT],
    ) -> _GraphT:
        return loader() or graph_type()


class CountingGraphEngineAdapter(GraphEngineAdapter):
    """GraphEngineAdapter that records per-method invocation counts."""

    def __init__(
        self,
        runtime: GraphRuntimeDouble,
        *,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        super().__init__(runtime, gateway=gateway, snapshot=snapshot)
        self.method_counts: dict[str, int] = {}

    def _increment(self, name: str) -> None:
        self.method_counts[name] = self.method_counts.get(name, 0) + 1

    def load_call_graph(self) -> nx.DiGraph:
        self._increment("load_call_graph")
        return super().load_call_graph()

    def load_import_graph(self) -> nx.DiGraph:
        self._increment("load_import_graph")
        return super().load_import_graph()

    def load_symbol_module_graph(self) -> nx.Graph:
        self._increment("load_symbol_module_graph")
        return super().load_symbol_module_graph()

    def load_symbol_function_graph(self) -> nx.Graph:
        self._increment("load_symbol_function_graph")
        return super().load_symbol_function_graph()

    def load_config_module_bipartite(self) -> nx.Graph:
        self._increment("load_config_module_bipartite")
        return super().load_config_module_bipartite()

    def load_test_function_bipartite(self) -> nx.Graph:
        self._increment("load_test_function_bipartite")
        return super().load_test_function_bipartite()


def build_graph_engine_double(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    **graphs: Unpack[GraphEngineSeedKwargs],
) -> GraphEngineAdapter:
    """Build GraphEngine-compatible double with optional graph seeds.

    Returns
    -------
    GraphEngineAdapter
        Adapter exposing GraphEngine protocol backed by GraphRuntimeDouble.
    """
    runtime = GraphRuntimeDouble(
        gateway=gateway,
        snapshot=snapshot,
        **graphs,
    )
    return GraphEngineAdapter(runtime, gateway=gateway, snapshot=snapshot)


def graph_engine_with_cache(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    seed: Mapping[GraphKind, nx.Graph],
    *,
    copy_graphs: bool = True,
) -> GraphEngineAdapter:
    """Build an engine double seeded from a GraphKind -> graph mapping.

    Returns
    -------
    GraphEngineAdapter
        Adapter seeded with provided graphs and ready for cache assertions.
    """
    kind_to_kwarg: dict[GraphKind, str] = {
        GraphKind.CALL_GRAPH: "call_graph",
        GraphKind.IMPORT_GRAPH: "import_graph",
        GraphKind.CFG_GRAPH: "cfg_graph",
        GraphKind.SYMBOL_MODULE_GRAPH: "symbol_module_graph",
        GraphKind.SYMBOL_FUNCTION_GRAPH: "symbol_function_graph",
        GraphKind.CONFIG_MODULE_BIPARTITE: "config_graph",
        GraphKind.TEST_FUNCTION_BIPARTITE: "test_function_graph",
    }
    seed_kwargs: dict[str, object] = {"copy_graphs": copy_graphs}

    for kind, graph in seed.items():
        for flag, kwarg in kind_to_kwarg.items():
            if kind & flag:
                seed_kwargs[kwarg] = graph

    seeded_graphs = cast("GraphEngineSeedKwargs", seed_kwargs)
    return build_graph_engine_double(gateway, snapshot, **seeded_graphs)


def runtime_with_graphs(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    **graphs: Unpack[GraphEngineSeedKwargs],
) -> tuple[GraphRuntimeOptions, GraphEngineAdapter]:
    """Build GraphRuntimeOptions and engine double from provided graph seeds.

    Returns
    -------
    tuple[GraphRuntimeOptions, GraphEngineAdapter]
        Runtime options bound to the seeded engine, and the engine itself.
    """
    engine = build_graph_engine_double(gateway, snapshot, **graphs)
    options = GraphRuntimeOptions(snapshot=snapshot, engine=engine)
    return options, engine


# ---------------------------------------------------------------------------
# Convenience factory functions (migrated from deprecated graph_runtimes.py)
# ---------------------------------------------------------------------------


def create_mock_runtime_with_call_graph(
    edges: list[tuple[str, str]] | None = None,
) -> GraphRuntimeDouble:
    """Create a GraphRuntimeDouble with a populated call graph.

    Parameters
    ----------
    edges
        Optional list of (source, target) edges. Defaults to a simple chain.

    Returns
    -------
    GraphRuntimeDouble
        Runtime seeded with a call graph.
    """
    if edges is None:
        edges = [("func_a", "func_b"), ("func_b", "func_c")]
    call_g = nx.DiGraph()
    call_g.add_edges_from(edges)
    return GraphRuntimeDouble(call_graph=call_g)


def create_mock_runtime_with_import_graph(
    edges: list[tuple[str, str]] | None = None,
) -> GraphRuntimeDouble:
    """Create a GraphRuntimeDouble with a populated import graph.

    Parameters
    ----------
    edges
        Optional list of (source, target) edges. Defaults to a simple chain.

    Returns
    -------
    GraphRuntimeDouble
        Runtime seeded with an import graph.
    """
    if edges is None:
        edges = [("mod_a", "mod_b"), ("mod_b", "mod_c")]
    import_g = nx.DiGraph()
    import_g.add_edges_from(edges)
    return GraphRuntimeDouble(import_graph=import_g)


def create_mock_runtime_all_graphs() -> GraphRuntimeDouble:
    """Create a GraphRuntimeDouble with all graph types populated.

    Returns
    -------
    GraphRuntimeDouble
        Runtime seeded with all graph types.
    """
    call_g = nx.DiGraph([("f1", "f2"), ("f2", "f3")])
    import_g = nx.DiGraph([("m1", "m2"), ("m2", "m3")])
    symbol_mod_g = nx.Graph([("sym1", "mod1"), ("sym2", "mod2")])
    symbol_func_g = nx.Graph([("sym1", "func1"), ("sym2", "func2")])
    config_mod_g = nx.Graph([("config1", "mod1")])
    test_func_g = nx.Graph([("test1", "func1")])
    cfg_g = nx.DiGraph([("entry", "block1"), ("block1", "exit")])
    return GraphRuntimeDouble(
        call_graph=call_g,
        import_graph=import_g,
        symbol_module_graph=symbol_mod_g,
        symbol_function_graph=symbol_func_g,
        config_graph=config_mod_g,
        test_function_graph=test_func_g,
        cfg_graph=cfg_g,
    )


def create_mock_runtime_with_standard_graphs(
    fixtures: GraphFixtures | None = None,
) -> GraphRuntimeDouble:
    """Create a GraphRuntimeDouble seeded with standard graph shapes.

    Parameters
    ----------
    fixtures
        Optional pre-built graph fixtures. Defaults to standard_graph_fixtures().

    Returns
    -------
    GraphRuntimeDouble
        Runtime seeded with standard fixtures.
    """
    if fixtures is None:
        # Use lazy import to avoid circular dependency with graphs.py
        graphs_module = import_module("tests._helpers.graphs")
        graphs = graphs_module.standard_graph_fixtures()
    else:
        graphs = fixtures
    return GraphRuntimeDouble.from_fixtures(graphs)


# Alias for backward compatibility during migration
MockGraphRuntime = GraphRuntimeDouble


__all__ = [
    "CountingGraphEngineAdapter",
    "GraphCallRecord",
    "GraphEngineAdapter",
    "GraphRuntimeDouble",
    "MockGraphRuntime",  # Alias for backward compatibility
    "build_graph_engine_double",
    "create_mock_runtime_all_graphs",
    "create_mock_runtime_with_call_graph",
    "create_mock_runtime_with_import_graph",
    "create_mock_runtime_with_standard_graphs",
    "graph_engine_with_cache",
    "runtime_with_graphs",
]
_GraphT = TypeVar("_GraphT", bound=nx.Graph)
