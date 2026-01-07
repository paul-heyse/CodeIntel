"""Factory helpers for constructing graph engines across surfaces.

Engines are configured from the graph_backend build config and consume
Parquet-backed datasets (no view-registry fallbacks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.graphs.engine.backend import maybe_enable_nx_gpu
from codeintel.build.graphs.engine.nx_engine import NxGraphEngine
from codeintel.build.graphs.engine.rx_engine import RxGraphEngine
from codeintel.config.primitives import SnapshotRef

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping

    from codeintel.build.graphs.engine.backend import BackendEnablement
    from codeintel.config.primitives import GraphBackendConfig


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineBuildOptions:
    """Optional configuration for building graph engines.

    The graph_backend field mirrors the build config selection routed through
    GraphRuntimeOptions and build_graph_runtime.
    """

    graph_backend: GraphBackendConfig | None = None
    env: MutableMapping[str, str] | None = None
    enabler: Callable[[], None] | None = None


def build_graph_engine(
    *,
    snapshot: SnapshotRef | tuple[str, str],
    dataset_root_dir: Path | None,
    options: EngineBuildOptions | None = None,
) -> NxGraphEngine | RxGraphEngine:
    """
    Construct a graph engine with optional cache seeding and backend hints.

    Parameters
    ----------
    snapshot :
        Repository snapshot anchoring the graph build or a (repo, commit) tuple.
    dataset_root_dir :
        Dataset root directory for Parquet snapshots.
    options : EngineBuildOptions | None
        Optional bundle controlling backend selection, cache seeding, and env/enabler hooks.

    Returns
    -------
    NxGraphEngine | RxGraphEngine
        Configured engine, seeded when possible.

    Raises
    ------
    ValueError
        If an unsupported graph backend is requested.
    """
    opts = options or EngineBuildOptions()
    allowed_backends = {"auto", "cpu", "nx-cugraph"}
    allowed_engines = {"networkx", "rustworkx"}
    enablement: BackendEnablement | None = None
    engine_name = (
        opts.graph_backend.engine if opts.graph_backend is not None else "networkx"
    )
    if engine_name not in allowed_engines:
        message = f"Unsupported graph engine: {engine_name}"
        raise ValueError(message)
    use_gpu_preference = (
        bool(opts.graph_backend.use_gpu) if opts.graph_backend is not None else False
    )
    if opts.graph_backend is not None:
        if opts.graph_backend.backend not in allowed_backends:
            message = f"Unsupported graph backend: {opts.graph_backend.backend}"
            raise ValueError(message)
        if engine_name == "networkx":
            enablement = maybe_enable_nx_gpu(
                opts.graph_backend,
                env=opts.env,
                enabler=opts.enabler,
            )
        elif use_gpu_preference:
            log.info("rustworkx engine selected; ignoring GPU backend preference.")
    effective_use_gpu = (
        bool(enablement.gpu_enabled)
        if enablement is not None
        else use_gpu_preference
        if engine_name == "networkx"
        else False
    )
    if engine_name == "rustworkx":
        use_gpu_preference = False
    normalized_snapshot = (
        snapshot
        if isinstance(snapshot, SnapshotRef)
        else SnapshotRef(repo=snapshot[0], commit=snapshot[1], repo_root=Path())
    )
    engine = NxGraphEngine(
        dataset_root_dir=dataset_root_dir,
        snapshot=normalized_snapshot,
        use_gpu=use_gpu_preference,
        effective_use_gpu=effective_use_gpu,
        backend_info=enablement,
    )
    if engine_name == "rustworkx":
        log.info("Using NetworkX compatibility shim for rustworkx engine selection.")
        return RxGraphEngine(delegate=engine)
    return engine
