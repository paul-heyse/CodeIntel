"""Tests for GPU backend selection and fallback handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.graph_runtime import GraphRuntimeOptions, build_graph_runtime
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.graphs.nx_backend import maybe_enable_nx_gpu
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.schemas import apply_all_schemas


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    message = detail
    raise AssertionError(message)


def test_maybe_enable_nx_gpu_success() -> None:
    """GPU enablement should return enabled status when enabler succeeds."""
    cfg = GraphBackendConfig(use_gpu=True, backend="nx-cugraph", strict=True)
    status = maybe_enable_nx_gpu(cfg, enabler=lambda: None)
    _expect(condition=status.gpu_enabled, detail="GPU backend should be enabled")
    _expect(
        condition=status.effective_backend == "nx-cugraph",
        detail="Unexpected backend effective value",
    )
    _expect(
        condition=status.fallback_reason is None,
        detail="Fallback reason should be None on success",
    )


def test_maybe_enable_nx_gpu_fallback() -> None:
    """Non-strict mode should fall back to CPU and capture reason."""
    cfg = GraphBackendConfig(use_gpu=True, backend="auto", strict=False)

    def _fail() -> None:
        err = RuntimeError("no gpu")
        raise err

    status = maybe_enable_nx_gpu(cfg, enabler=_fail)
    _expect(condition=not status.gpu_enabled, detail="GPU should be disabled after fallback")
    _expect(condition=status.effective_backend == "cpu", detail="Expected CPU fallback")
    _expect(
        condition=status.fallback_reason is not None, detail="Fallback reason should be populated"
    )


def test_maybe_enable_nx_gpu_strict_raises() -> None:
    """Strict mode should raise when GPU cannot be enabled."""
    cfg = GraphBackendConfig(use_gpu=True, backend="auto", strict=True)

    def _fail() -> None:
        err = RuntimeError("no gpu")
        raise err

    with pytest.raises(RuntimeError):
        maybe_enable_nx_gpu(cfg, enabler=_fail)


def test_build_graph_runtime_captures_backend_info(
    fresh_gateway: StorageGateway,
) -> None:
    """Runtime should expose backend metadata recorded during engine construction."""
    apply_all_schemas(fresh_gateway.con)
    snapshot = SnapshotRef(repo="r", commit="c", repo_root=Path())
    cfg = GraphBackendConfig(use_gpu=True, backend="nx-cugraph", strict=False)

    runtime = build_graph_runtime(
        fresh_gateway,
        GraphRuntimeOptions(snapshot=snapshot, backend=cfg),
        enabler=lambda: None,
    )
    info = runtime.backend_info
    _expect(condition=info is not None, detail="backend_info should be set on runtime")
    _expect(
        condition=info is not None and info.gpu_enabled,
        detail="GPU should be marked enabled in backend_info",
    )
    _expect(
        condition=info is not None and info.effective_backend == "nx-cugraph",
        detail="Unexpected backend recorded on runtime",
    )
