"""Tests for optional GPU backend enablement helper."""

from __future__ import annotations

import pytest

from codeintel.build.graphs.engine.backend import maybe_enable_nx_gpu
from codeintel.config.primitives import GraphBackendConfig
from tests._helpers.assertions import expect_true


def test_enable_gpu_strict_raises() -> None:
    """Helper should raise RuntimeError when GPU is requested with strict mode."""
    cfg = GraphBackendConfig(use_gpu=True, backend="auto", strict=True)
    with pytest.raises(RuntimeError, match="CPU-only"):
        maybe_enable_nx_gpu(cfg)


def test_enable_gpu_invokes_enabler() -> None:
    """Helper should call enabler when GPU is requested."""
    called = {"set": False}

    def _enabler() -> None:
        called["set"] = True

    cfg = GraphBackendConfig(use_gpu=True, backend="auto", strict=False)
    result = maybe_enable_nx_gpu(cfg, enabler=_enabler)
    expect_true(called["set"], message="Expected enabler to be invoked")
    expect_true(result.effective_backend == "cpu" and not result.gpu_enabled)


def test_maybe_enable_nx_gpu_falls_back_when_missing() -> None:
    """maybe_enable_nx_gpu should return CPU fallback when enablement fails in non-strict mode."""

    def _failing_enabler() -> None:
        message = "missing"
        raise RuntimeError(message)

    cfg = GraphBackendConfig(use_gpu=True, backend="auto", strict=False)
    result = maybe_enable_nx_gpu(cfg, enabler=_failing_enabler)
    expect_true(
        result.effective_backend == "cpu" and not result.gpu_enabled,
        message="Expected CPU fallback when enabler fails in non-strict mode",
    )
    expect_true(
        result.fallback_reason is not None, message="Expected fallback reason to be populated"
    )


def test_maybe_enable_nx_gpu_raises_when_strict() -> None:
    """maybe_enable_nx_gpu should raise when strict=True and enablement fails."""

    def _failing_enabler() -> None:
        message = "missing"
        raise RuntimeError(message)

    cfg = GraphBackendConfig(use_gpu=True, backend="auto", strict=True)
    with pytest.raises(RuntimeError, match="missing"):
        maybe_enable_nx_gpu(cfg, enabler=_failing_enabler)
