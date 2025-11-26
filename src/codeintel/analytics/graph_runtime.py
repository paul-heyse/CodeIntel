"""Shared graph runtime options for analytics modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.analytics.graph_service import GraphContext
from codeintel.config.models import GraphBackendConfig
from codeintel.graphs.engine import GraphEngine, GraphKind, NxGraphEngine
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from codeintel.analytics.context import AnalyticsContext


@dataclass(frozen=True)
class GraphRuntimeOptions:
    """
    Runtime options for graph-based analytics functions.

    Attributes
    ----------
    context :
        Optional shared AnalyticsContext containing cached graphs.
    graph_ctx :
        Optional GraphContext carrying weights and sampling limits.
    graph_backend :
        Optional backend selection used to derive GPU preferences.
    engine :
        Optional pre-built GraphEngine to reuse across analytics modules.
    """

    context: AnalyticsContext | None = None
    graph_ctx: GraphContext | None = None
    graph_backend: GraphBackendConfig | None = None
    engine: GraphEngine | None = None

    @property
    def use_gpu(self) -> bool:
        """
        Compute whether GPU execution is preferred.

        Returns
        -------
        bool
            True when graph_backend or graph_ctx requests GPU usage.
        """
        if self.graph_backend is not None:
            return bool(self.graph_backend.use_gpu)
        if self.graph_ctx is not None:
            return bool(self.graph_ctx.use_gpu)
        return False

    def build_engine(
        self,
        gateway: StorageGateway,
        repo: str,
        commit: str,
    ) -> GraphEngine:
        """
        Construct or reuse a GraphEngine for the target snapshot.

        Parameters
        ----------
        gateway :
            Storage gateway for accessing DuckDB-backed graph views.
        repo : str
            Repository identifier anchoring the graph snapshot.
        commit : str
            Commit hash anchoring the graph snapshot.

        Returns
        -------
        GraphEngine
            Engine capable of materializing the configured graphs.
        """
        if self.engine is not None:
            return self.engine

        use_gpu = self.use_gpu
        engine = NxGraphEngine(
            gateway=gateway,
            repo=repo,
            commit=commit,
            use_gpu=use_gpu,
        )
        if self.context is not None and self.context.repo == repo and self.context.commit == commit:
            engine.seed(GraphKind.CALL_GRAPH, self.context.call_graph)
            engine.seed(GraphKind.IMPORT_GRAPH, self.context.import_graph)
            engine.seed(GraphKind.SYMBOL_MODULE_GRAPH, self.context.symbol_module_graph)
            engine.seed(GraphKind.SYMBOL_FUNCTION_GRAPH, self.context.symbol_function_graph)
        return engine
