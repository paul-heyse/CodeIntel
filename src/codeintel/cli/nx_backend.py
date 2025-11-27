"""CLI-facing shim around graphs.nx_backend."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping

from codeintel.config.primitives import GraphBackendConfig
from codeintel.graphs.nx_backend import maybe_enable_nx_gpu as _maybe_enable_nx_gpu

__all__ = ["maybe_enable_nx_gpu"]


def maybe_enable_nx_gpu(
    cfg: GraphBackendConfig,
    *,
    env: MutableMapping[str, str] | None = None,
    enabler: Callable[[], None] | None = None,
) -> None:
    """Delegate NetworkX backend configuration to the graphs helper."""
    _maybe_enable_nx_gpu(cfg, env=env, enabler=enabler)
