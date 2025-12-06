"""Helpers for configuring NetworkX backends (CPU vs GPU)."""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass

from codeintel.config.primitives import GraphBackendConfig

LOG = logging.getLogger(__name__)
_GPU_AUTOCONFIG_ENV = "NX_CUGRAPH_AUTOCONFIG"


@dataclass(frozen=True)
class BackendEnablement:
    """Result of backend enablement attempt."""

    requested_backend: str
    requested_gpu: bool
    effective_backend: str
    gpu_enabled: bool
    fallback_reason: str | None = None


def _enable_nx_cugraph_backend() -> None:
    """
    Enable the nx-cugraph backend when available.

    Support both old API (set_default_backend) and NetworkX 3.x config API.

    Raises
    ------
    RuntimeError
        If nx_cugraph is missing or cannot be enabled.
    """
    try:
        nx_cugraph = importlib.import_module("nx_cugraph")
    except ImportError as exc:  # pragma: no cover - environment dependent
        message = "Requested GPU backend, but nx_cugraph is not installed."
        raise RuntimeError(message) from exc

    # Try old API first (nx_cugraph < 25.x)
    set_backend = getattr(nx_cugraph, "set_default_backend", None)
    if set_backend is not None:
        set_backend()
        LOG.info("NetworkX GPU backend enabled via nx_cugraph.set_default_backend.")
        return

    # NetworkX 3.x uses nx.config.backend_priority
    try:
        nx = importlib.import_module("networkx")
        config = getattr(nx, "config", None)
        if config is not None and hasattr(config, "backend_priority"):
            # Set cugraph as the priority backend
            config.backend_priority = ["cugraph"]
            # Suppress verbose cache warnings - we don't mutate graphs after creation
            if hasattr(config, "warnings_to_ignore"):
                config.warnings_to_ignore.add("cache")
            LOG.info("NetworkX GPU backend enabled via nx.config.backend_priority=['cugraph'].")
            return
    except (ImportError, AttributeError) as exc:
        LOG.debug("NetworkX config API not available: %s", exc)

    # Fallback: set environment variable for automatic backend dispatch
    # NetworkX 3.x will pick up cugraph automatically if available
    os.environ.setdefault("NETWORKX_BACKEND_PRIORITY", "cugraph")
    LOG.info("NetworkX GPU backend enabled via NETWORKX_BACKEND_PRIORITY env var.")


def maybe_enable_nx_gpu(
    cfg: GraphBackendConfig,
    *,
    env: MutableMapping[str, str] | None = None,
    enabler: Callable[[], None] | None = None,
) -> BackendEnablement:
    """
    Configure NetworkX backend based on GraphBackendConfig.

    Parameters
    ----------
    cfg : GraphBackendConfig
        Backend selection options.
    env : MutableMapping[str, str] | None, optional
        Environment mapping to mutate; defaults to os.environ.
    enabler : Callable[[], None] | None, optional
        Callback that enables the GPU backend; defaults to nx-cugraph enabler.

    Returns
    -------
    BackendEnablement
        Outcome describing effective backend, GPU status, and fallback reason.

    Raises
    ------
    RuntimeError
        If strict mode is enabled and the GPU backend cannot be configured.
    """
    env_vars = env if env is not None else os.environ
    enable_backend = enabler or _enable_nx_cugraph_backend
    requested = cfg.backend
    base = BackendEnablement(
        requested_backend=requested,
        requested_gpu=cfg.use_gpu,
        effective_backend="cpu",
        gpu_enabled=False,
        fallback_reason=None,
    )

    if not cfg.use_gpu:
        LOG.debug("Graph backend: CPU (use_gpu=False).")
        return base

    backend = cfg.backend
    LOG.info("Graph backend requested: %s", backend)
    if backend == "cpu":
        LOG.info("Graph backend pinned to CPU.")
        return base

    if backend in {"auto", "nx-cugraph"}:
        env_vars.setdefault(_GPU_AUTOCONFIG_ENV, "True")
        try:
            enable_backend()
            return BackendEnablement(
                requested_backend=requested,
                requested_gpu=True,
                effective_backend="nx-cugraph",
                gpu_enabled=True,
                fallback_reason=None,
            )
        except RuntimeError as exc:
            if cfg.strict:
                LOG.exception("Failed to enable GPU backend (strict=True).")
                raise
            LOG.exception("Failed to enable GPU backend; continuing with CPU backend.")
            return BackendEnablement(
                requested_backend=requested,
                requested_gpu=True,
                effective_backend="cpu",
                gpu_enabled=False,
                fallback_reason=str(exc),
            )

    LOG.warning("Unknown graph backend '%s'; using CPU backend.", backend)
    return BackendEnablement(
        requested_backend=requested,
        requested_gpu=cfg.use_gpu,
        effective_backend="cpu",
        gpu_enabled=False,
        fallback_reason=f"unknown backend {backend}",
    )


__all__ = ["BackendEnablement", "maybe_enable_nx_gpu"]
