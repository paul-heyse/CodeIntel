"""Unified graph runtime/engine test double.

Loads graphs from seeded DuckDB tables when a gateway/snapshot is provided,
otherwise serves configured NetworkX graphs with defensive copies. All public
methods record call names for assertions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, TypeVar

import networkx as nx
from duckdb import Error as DuckDBError

from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from tests._helpers.graphs import GraphFixtures, standard_graph_fixtures

if TYPE_CHECKING:
    from codeintel.config.primitives import GraphBackendConfig

CALL_GRAPH_TABLE: Final[str] = "graph.call_graph_edges"
IMPORT_GRAPH_TABLE: Final[str] = "graph.import_graph_edges"
SYMBOL_EDGE_TABLE: Final[str] = "graph.symbol_use_edges"


@dataclass
class GraphCallRecord:
    """Record of a graph retrieval call."""

    method: str


@dataclass
class GraphRuntimeDouble:
    """Graph runtime/engine stand-in with DB-backed loading and call recording."""

    gateway: StorageGateway | None = None
    snapshot: SnapshotRef | None = None
    call_graph_obj: nx.DiGraph | None = None
    import_graph_obj: nx.DiGraph | None = None
    symbol_module_graph_obj: nx.Graph | None = None
    symbol_function_graph_obj: nx.Graph | None = None
    config_bipartite_obj: nx.Graph | None = None
    test_function_bipartite_obj: nx.Graph | None = None
    cfg_graph: nx.DiGraph | None = None
    backend: GraphBackendConfig | None = None
    use_gpu: bool = False
    copy_graphs: bool = True
    calls: list[GraphCallRecord] = field(default_factory=list)

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
        graphs = fixtures or standard_graph_fixtures()
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
    def ensure_call_graph(self) -> nx.DiGraph | None:
        self.calls.append(GraphCallRecord(method="ensure_call_graph"))
        return self._graph_or_db(
            "call_graph_obj",
            loader=self._load_call_graph_from_db,
            default_type=nx.DiGraph,
        )

    def ensure_import_graph(self) -> nx.DiGraph | None:
        self.calls.append(GraphCallRecord(method="ensure_import_graph"))
        return self._graph_or_db(
            "import_graph_obj",
            loader=self._load_import_graph_from_db,
            default_type=nx.DiGraph,
        )

    def ensure_symbol_module_graph(self) -> nx.Graph | None:
        self.calls.append(GraphCallRecord(method="ensure_symbol_module_graph"))
        return self._graph_or_db(
            "symbol_module_graph_obj",
            loader=self._load_symbol_graph_from_db,
            default_type=nx.Graph,
        )

    def ensure_symbol_function_graph(self) -> nx.Graph | None:
        self.calls.append(GraphCallRecord(method="ensure_symbol_function_graph"))
        return self._clone(self.symbol_function_graph_obj or nx.Graph())

    def ensure_config_module_bipartite(self) -> nx.Graph | None:
        self.calls.append(GraphCallRecord(method="ensure_config_module_bipartite"))
        return self._clone(self.config_bipartite_obj or nx.Graph())

    def ensure_test_function_bipartite(self) -> nx.Graph | None:
        self.calls.append(GraphCallRecord(method="ensure_test_function_bipartite"))
        return self._clone(self.test_function_bipartite_obj or nx.Graph())

    def ensure_cfg_graph(self) -> nx.DiGraph | None:
        self.calls.append(GraphCallRecord(method="ensure_cfg_graph"))
        return self._clone(self.cfg_graph or nx.DiGraph())

    # ------------------------------------------------------------------ GraphEngine compatibility
    def call_graph(self) -> nx.DiGraph:
        return self._clone(self.ensure_call_graph() or nx.DiGraph())

    def load_call_graph(self) -> nx.DiGraph:
        return self.call_graph()

    def import_graph(self) -> nx.DiGraph:
        return self._clone(self.ensure_import_graph() or nx.DiGraph())

    def load_import_graph(self) -> nx.DiGraph:
        return self.import_graph()

    def symbol_module_graph(self) -> nx.Graph:
        return self._clone(self.ensure_symbol_module_graph() or nx.Graph())

    def load_symbol_module_graph(self) -> nx.Graph:
        return self.symbol_module_graph()

    def symbol_function_graph(self) -> nx.Graph:
        return self._clone(self.ensure_symbol_function_graph() or nx.Graph())

    def load_symbol_function_graph(self) -> nx.Graph:
        return self.symbol_function_graph()

    def config_module_bipartite(self) -> nx.Graph:
        return self._clone(self.ensure_config_module_bipartite() or nx.Graph())

    def load_config_module_bipartite(self) -> nx.Graph:
        return self.config_module_bipartite()

    def test_function_bipartite(self) -> nx.Graph:
        return self._clone(self.ensure_test_function_bipartite() or nx.Graph())

    def load_test_function_bipartite(self) -> nx.Graph:
        return self.test_function_bipartite()

    # ------------------------------------------------------------------ Internal helpers
    def _graph_or_db(
        self,
        attr: str,
        *,
        loader: Callable[[], _GraphT | None],
        default_type: type[_GraphT],
    ) -> _GraphT:
        graph_candidate = getattr(self, attr, None)
        graph: _GraphT | None = (
            graph_candidate if isinstance(graph_candidate, default_type) else None
        )
        if graph is None and self.gateway and self.snapshot:
            graph = loader()
        if graph is None:
            graph = default_type()
        return self._clone(graph)

    def _clone(self, graph: _GraphT) -> _GraphT:
        if self.copy_graphs:
            return graph.copy()
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


__all__ = ["GraphCallRecord", "GraphRuntimeDouble"]
_GraphT = TypeVar("_GraphT", bound=nx.Graph)
