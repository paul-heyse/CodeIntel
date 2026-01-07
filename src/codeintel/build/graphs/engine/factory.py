"""Factory helpers for constructing graph engines across surfaces.

Engines are configured from the graph_backend build config and consume
Parquet-backed datasets (no view-registry fallbacks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.graphs.engine.rx_engine import RxGraphEngine
from codeintel.config.primitives import SnapshotRef

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping

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
) -> RxGraphEngine:
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
    RxGraphEngine
        Configured engine, seeded when possible.

    """
    opts = options or EngineBuildOptions()
    if opts.graph_backend is not None and opts.graph_backend.use_gpu:
        log.info("GPU preference ignored; rustworkx is CPU-only.")
    normalized_snapshot = (
        snapshot
        if isinstance(snapshot, SnapshotRef)
        else SnapshotRef(repo=snapshot[0], commit=snapshot[1], repo_root=Path())
    )
    return RxGraphEngine(
        dataset_root_dir=dataset_root_dir,
        snapshot=normalized_snapshot,
        use_gpu=False,
        effective_use_gpu=False,
        backend_info=None,
    )
