"""Helpers for configuring rustworkx execution (CPU only)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping

    from codeintel.config.primitives import GraphBackendConfig

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendEnablement:
    """Result of backend enablement attempt."""

    requested_backend: str
    requested_gpu: bool
    effective_backend: str
    gpu_enabled: bool
    fallback_reason: str | None = None


def _enable_nx_cugraph_backend() -> None:
    """Raise because rustworkx has no GPU backend."""
    message = "Rustworkx execution is CPU-only; GPU backend is unavailable."
    raise RuntimeError(message)


def maybe_enable_nx_gpu(
    cfg: GraphBackendConfig,
    *,
    env: MutableMapping[str, str] | None = None,
    enabler: Callable[[], None] | None = None,
) -> BackendEnablement:
    """
    Validate GPU intent and return rustworkx-only enablement status.

    Parameters
    ----------
    cfg : GraphBackendConfig
        Backend selection options.
    env : MutableMapping[str, str] | None, optional
        Environment mapping (unused; kept for compatibility).
    enabler : Callable[[], None] | None, optional
        Callback invoked when GPU was requested (unused by default).

    Returns
    -------
    BackendEnablement
        Outcome describing effective backend, GPU status, and fallback reason.

    Raises
    ------
    RuntimeError
        If strict mode is enabled and GPU usage was requested.
    """
    if env is not None:
        LOG.debug("Ignoring backend env overrides for rustworkx-only execution.")
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
        LOG.debug("Graph backend: CPU (rustworkx-only).")
        return base

    LOG.info("GPU backend requested but rustworkx is CPU-only.")
    try:
        enable_backend()
    except RuntimeError as exc:
        if cfg.strict:
            LOG.exception("Failed to enable GPU backend (strict=True).")
            raise
        return BackendEnablement(
            requested_backend=requested,
            requested_gpu=True,
            effective_backend="cpu",
            gpu_enabled=False,
            fallback_reason=str(exc),
        )
    return BackendEnablement(
        requested_backend=requested,
        requested_gpu=True,
        effective_backend="cpu",
        gpu_enabled=False,
        fallback_reason="rustworkx cpu-only",
    )


__all__ = ["BackendEnablement", "maybe_enable_nx_gpu"]
