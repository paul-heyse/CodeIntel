"""Factory helpers for constructing graph engines across surfaces."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.graphs.engine import GraphKind, NxGraphEngine
from codeintel.graphs.nx_backend import BackendEnablement, maybe_enable_nx_gpu
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from codeintel.analytics.context import AnalyticsContext


def build_graph_engine(  # noqa: PLR0913
    gateway: StorageGateway,
    snapshot: SnapshotRef | tuple[str, str],
    *,
    graph_backend: GraphBackendConfig | None = None,
    context: AnalyticsContext | None = None,
    env: MutableMapping[str, str] | None = None,
    enabler: Callable[[], None] | None = None,
) -> NxGraphEngine:
    """
    Construct an NxGraphEngine with optional cache seeding and backend hints.

    Parameters
    ----------
    gateway :
        Storage gateway providing graph tables and views.
    snapshot :
        Repository snapshot anchoring the graph build or a (repo, commit) tuple.
    graph_backend : GraphBackendConfig | None, optional
        Backend selection controlling GPU usage.
    context : AnalyticsContext | None, optional
        Optional analytics context used to seed engine caches when repo/commit match.
    env : MutableMapping[str, str] | None, optional
        Environment mapping mutated for backend configuration; defaults to os.environ.
    enabler : Callable[[], None] | None, optional
        Optional hook to enable the GPU backend (used for testing).

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
    enablement: BackendEnablement | None = None
    use_gpu_preference = bool(graph_backend.use_gpu) if graph_backend is not None else False
    if graph_backend is not None:
        if graph_backend.backend not in allowed_backends:
            message = f"Unsupported graph backend: {graph_backend.backend}"
            raise ValueError(message)
        enablement = maybe_enable_nx_gpu(graph_backend, env=env, enabler=enabler)
    effective_use_gpu = (
        bool(enablement.gpu_enabled) if enablement is not None else use_gpu_preference
    )
    normalized_snapshot = (
        snapshot
        if isinstance(snapshot, SnapshotRef)
        else SnapshotRef(repo=snapshot[0], commit=snapshot[1], repo_root=Path())
    )
    engine = NxGraphEngine(
        gateway=gateway,
        snapshot=normalized_snapshot,
        use_gpu=use_gpu_preference,
        effective_use_gpu=effective_use_gpu,
        backend_info=enablement,
    )
    if (
        context is not None
        and context.repo == normalized_snapshot.repo
        and context.commit == normalized_snapshot.commit
    ):
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
