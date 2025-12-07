"""Test-friendly GraphEngine stub.

Provides a minimal GraphEngine implementation for analytics runtime tests,
returning deterministic NetworkX graphs supplied at construction time, while
recording method calls for assertions.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.engine.protocol import GraphEngine
from codeintel.storage.gateway import StorageGateway
from tests._helpers.records import CallRecorder


@dataclass(frozen=True)
class GraphCall:
    """Record of a graph retrieval call."""

    method: str


class StubGraphEngine(GraphEngine):
    """GraphEngine stub that serves predefined graphs."""

    def __init__(
        self,
        *,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        call_graph: nx.DiGraph | None = None,
        import_graph: nx.DiGraph | None = None,
        use_gpu: bool = False,
    ) -> None:
        self.gateway = gateway
        self._snapshot = snapshot
        self._call_graph = call_graph or nx.DiGraph()
        self._import_graph = import_graph or nx.DiGraph()
        self._use_gpu = use_gpu
        self.calls: CallRecorder[GraphCall] = CallRecorder()

    @property
    def snapshot(self) -> SnapshotRef:
        return self._snapshot

    @property
    def use_gpu(self) -> bool:
        return self._use_gpu

    def call_graph(self) -> nx.DiGraph:
        self.calls.record(GraphCall(method="call_graph"))
        return self._call_graph

    def load_call_graph(self) -> nx.DiGraph:
        self.calls.record(GraphCall(method="load_call_graph"))
        return self._call_graph

    def import_graph(self) -> nx.DiGraph:
        self.calls.record(GraphCall(method="import_graph"))
        return self._import_graph

    def load_import_graph(self) -> nx.DiGraph:
        self.calls.record(GraphCall(method="load_import_graph"))
        return self._import_graph

    def symbol_module_graph(self) -> nx.Graph:
        self.calls.record(GraphCall(method="symbol_module_graph"))
        return nx.Graph()

    def load_symbol_module_graph(self) -> nx.Graph:
        self.calls.record(GraphCall(method="load_symbol_module_graph"))
        return nx.Graph()

    def symbol_function_graph(self) -> nx.Graph:
        self.calls.record(GraphCall(method="symbol_function_graph"))
        return nx.Graph()

    def load_symbol_function_graph(self) -> nx.Graph:
        self.calls.record(GraphCall(method="load_symbol_function_graph"))
        return nx.Graph()

    def config_module_bipartite(self) -> nx.Graph:
        self.calls.record(GraphCall(method="config_module_bipartite"))
        return nx.Graph()

    def load_config_module_bipartite(self) -> nx.Graph:
        self.calls.record(GraphCall(method="load_config_module_bipartite"))
        return nx.Graph()

    def test_function_bipartite(self) -> nx.Graph:
        self.calls.record(GraphCall(method="test_function_bipartite"))
        return nx.Graph()

    def load_test_function_bipartite(self) -> nx.Graph:
        self.calls.record(GraphCall(method="load_test_function_bipartite"))
        return nx.Graph()


__all__ = ["GraphCall", "StubGraphEngine"]
