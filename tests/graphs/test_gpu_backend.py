"""Tests for optional GPU backend enablement helper."""

from __future__ import annotations

import pytest

from codeintel.config.primitives import GraphBackendConfig
from codeintel.graphs.engine.backend import maybe_enable_nx_gpu


def test_enable_nx_cugraph_backend_missing_module() -> None:
    """Helper should raise RuntimeError when nx_cugraph is absent."""

    def _missing_enabler() -> None:
        message = "nx_cugraph is not installed"
        raise RuntimeError(message)

    cfg = GraphBackendConfig(use_gpu=True, backend="nx-cugraph", strict=True)
    with pytest.raises(RuntimeError, match="nx_cugraph is not installed"):
        maybe_enable_nx_gpu(cfg, enabler=_missing_enabler)


def test_enable_nx_cugraph_backend_invokes_setter() -> None:
    """Helper should call set_default_backend when present."""
    called = {"set": False}

    def _enabler() -> None:
        called["set"] = True

    cfg = GraphBackendConfig(use_gpu=True, backend="nx-cugraph", strict=True)
    result = maybe_enable_nx_gpu(cfg, enabler=_enabler)
    if not called["set"]:
        pytest.fail("Expected set_default_backend to be invoked")
    if not result.gpu_enabled or result.effective_backend != "nx-cugraph":
        pytest.fail("Expected GPU backend to be enabled")


def test_maybe_enable_nx_gpu_falls_back_when_missing() -> None:
    """maybe_enable_nx_gpu should return CPU fallback when enablement fails in non-strict mode."""

    def _failing_enabler() -> None:
        message = "missing"
        raise RuntimeError(message)

    cfg = GraphBackendConfig(use_gpu=True, backend="nx-cugraph", strict=False)
    result = maybe_enable_nx_gpu(cfg, enabler=_failing_enabler)
    if result.effective_backend != "cpu" or result.gpu_enabled:
        pytest.fail("Expected CPU fallback when enabler fails in non-strict mode")
    if result.fallback_reason is None:
        pytest.fail("Expected fallback reason to be populated")


def test_maybe_enable_nx_gpu_raises_when_strict() -> None:
    """maybe_enable_nx_gpu should raise when strict=True and enablement fails."""

    def _failing_enabler() -> None:
        message = "missing"
        raise RuntimeError(message)

    cfg = GraphBackendConfig(use_gpu=True, backend="nx-cugraph", strict=True)
    with pytest.raises(RuntimeError):
        maybe_enable_nx_gpu(cfg, enabler=_failing_enabler)
