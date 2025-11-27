"""Factory helpers for constructing graph engines across surfaces."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING

from codeintel.config.primitives import GraphBackendConfig
from codeintel.graphs.engine import GraphKind, NxGraphEngine
from codeintel.graphs.nx_backend import maybe_enable_nx_gpu
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from codeintel.analytics.context import AnalyticsContext


def build_graph_engine(
    gateway: StorageGateway,
    snapshot: tuple[str, str],
    *,
    graph_backend: GraphBackendConfig | None = None,
    context: AnalyticsContext | None = None,
    env: MutableMapping[str, str] | None = None,
) -> NxGraphEngine:
    """
    Construct an NxGraphEngine with optional cache seeding and backend hints.

    Parameters
    ----------
    gateway :
        Storage gateway providing graph tables and views.
    snapshot : tuple[str, str]
        Repository and commit anchoring the graph snapshot.
    graph_backend : GraphBackendConfig | None, optional
        Backend selection controlling GPU usage.
    context : AnalyticsContext | None, optional
        Optional analytics context used to seed engine caches when repo/commit match.
    env : MutableMapping[str, str] | None, optional
        Environment mapping mutated for backend configuration; defaults to os.environ.

    Returns
    -------
    NxGraphEngine
        Configured engine, seeded when possible.

    Raises
    ------
    ValueError
        If an unsupported graph backend is requested.
    """
    allowed_backends = {"auto", "cpu", "nx-cugraph"}
    if graph_backend is not None:
        if graph_backend.backend not in allowed_backends:
            message = f"Unsupported graph backend: {graph_backend.backend}"
            raise ValueError(message)
        maybe_enable_nx_gpu(graph_backend, env=env)
    use_gpu = bool(graph_backend.use_gpu) if graph_backend is not None else False
    repo, commit = snapshot
    engine = NxGraphEngine(
        gateway=gateway,
        repo=repo,
        commit=commit,
        use_gpu=use_gpu,
    )
    if context is not None and context.repo == repo and context.commit == commit:
        engine.seed(GraphKind.CALL_GRAPH, context.call_graph)
        engine.seed(GraphKind.IMPORT_GRAPH, context.import_graph)
        engine.seed(GraphKind.SYMBOL_MODULE_GRAPH, context.symbol_module_graph)
        engine.seed(GraphKind.SYMBOL_FUNCTION_GRAPH, context.symbol_function_graph)
        engine.seed(
            GraphKind.CONFIG_MODULE_BIPARTITE,
            getattr(context, "config_module_bipartite", None)
            or getattr(context, "config_module_graph", None),
        )
        engine.seed(
            GraphKind.TEST_FUNCTION_BIPARTITE,
            getattr(context, "test_function_bipartite", None)
            or getattr(context, "test_function_graph", None),
        )
    return engine
