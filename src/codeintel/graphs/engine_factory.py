"""Factory helpers for constructing graph engines across surfaces."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from pathlib import Path

from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.graphs.engine import NxGraphEngine
from codeintel.graphs.nx_backend import BackendEnablement, maybe_enable_nx_gpu
from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class EngineBuildOptions:
    """Optional configuration for building graph engines."""

    graph_backend: GraphBackendConfig | None = None
    env: MutableMapping[str, str] | None = None
    enabler: Callable[[], None] | None = None


def build_graph_engine(
    gateway: StorageGateway,
    snapshot: SnapshotRef | tuple[str, str],
    options: EngineBuildOptions | None = None,
) -> NxGraphEngine:
    """
    Construct an NxGraphEngine with optional cache seeding and backend hints.

    Parameters
    ----------
    gateway :
        Storage gateway providing graph tables and views.
    snapshot :
        Repository snapshot anchoring the graph build or a (repo, commit) tuple.
    options : EngineBuildOptions | None
        Optional bundle controlling backend selection, cache seeding, and env/enabler hooks.

    Returns
    -------
    NxGraphEngine
        Configured engine, seeded when possible.

    Raises
    ------
    ValueError
        If an unsupported graph backend is requested.
    """
    opts = options or EngineBuildOptions()
    allowed_backends = {"auto", "cpu", "nx-cugraph"}
    enablement: BackendEnablement | None = None
    use_gpu_preference = (
        bool(opts.graph_backend.use_gpu) if opts.graph_backend is not None else False
    )
    if opts.graph_backend is not None:
        if opts.graph_backend.backend not in allowed_backends:
            message = f"Unsupported graph backend: {opts.graph_backend.backend}"
            raise ValueError(message)
        enablement = maybe_enable_nx_gpu(opts.graph_backend, env=opts.env, enabler=opts.enabler)
    effective_use_gpu = (
        bool(enablement.gpu_enabled) if enablement is not None else use_gpu_preference
    )
    normalized_snapshot = (
        snapshot
        if isinstance(snapshot, SnapshotRef)
        else SnapshotRef(repo=snapshot[0], commit=snapshot[1], repo_root=Path())
    )
    return NxGraphEngine(
        gateway=gateway,
        snapshot=normalized_snapshot,
        use_gpu=use_gpu_preference,
        effective_use_gpu=effective_use_gpu,
        backend_info=enablement,
    )
