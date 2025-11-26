"""Helpers for configuring NetworkX backends (CPU vs GPU)."""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable, MutableMapping

from codeintel.config.primitives import GraphBackendConfig

LOG = logging.getLogger(__name__)
_GPU_AUTOCONFIG_ENV = "NX_CUGRAPH_AUTOCONFIG"


def _enable_nx_cugraph_backend() -> None:
    """
    Enable the nx-cugraph backend when available.

    Raises
    ------
    RuntimeError
        If nx_cugraph is missing or exposes no backend setter.
    """
    try:
        nx_cugraph = importlib.import_module("nx_cugraph")
    except ImportError as exc:  # pragma: no cover - environment dependent
        message = "Requested GPU backend, but nx_cugraph is not installed."
        raise RuntimeError(message) from exc

    try:
        nx_cugraph.set_default_backend()  # type: ignore[attr-defined]
        LOG.info("NetworkX GPU backend enabled via nx_cugraph.")
    except AttributeError as exc:  # pragma: no cover - version dependent
        message = "nx_cugraph.set_default_backend is not available for this version."
        raise RuntimeError(message) from exc


def maybe_enable_nx_gpu(
    cfg: GraphBackendConfig,
    *,
    env: MutableMapping[str, str] | None = None,
    enabler: Callable[[], None] | None = None,
) -> None:
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

    Raises
    ------
    RuntimeError
        If strict mode is enabled and the GPU backend cannot be configured.
    """
    env_vars = env if env is not None else os.environ
    enable_backend = enabler or _enable_nx_cugraph_backend

    if not cfg.use_gpu:
        LOG.debug("Graph backend: CPU (use_gpu=False).")
        return

    backend = cfg.backend
    LOG.info("Graph backend requested: %s", backend)
    if backend == "cpu":
        LOG.info("Graph backend pinned to CPU.")
        return

    if backend in {"auto", "nx-cugraph"}:
        env_vars.setdefault(_GPU_AUTOCONFIG_ENV, "True")
        try:
            enable_backend()
        except RuntimeError:
            if cfg.strict:
                LOG.exception("Failed to enable GPU backend (strict=True).")
                raise
            LOG.exception("Failed to enable GPU backend; continuing with CPU backend.")
        return

    LOG.warning("Unknown graph backend '%s'; using CPU backend.", backend)
