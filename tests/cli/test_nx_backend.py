"""Unit coverage for NetworkX GPU backend wiring."""

from __future__ import annotations

import os

import pytest

from codeintel.cli import nx_backend
from codeintel.config.models import GraphBackendConfig
from tests._helpers.assertions import expect_equal, expect_true


def test_maybe_enable_nx_gpu_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip backend wiring when use_gpu is False."""
    monkeypatch.delenv("NX_CUGRAPH_AUTOCONFIG", raising=False)
    nx_backend.maybe_enable_nx_gpu(GraphBackendConfig(use_gpu=False))
    expect_true("NX_CUGRAPH_AUTOCONFIG" not in os.environ)


def test_maybe_enable_nx_gpu_soft_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Log and continue when GPU backend setup fails in non-strict mode."""
    monkeypatch.delenv("NX_CUGRAPH_AUTOCONFIG", raising=False)
    message = "backend missing"

    def _fail() -> None:
        raise RuntimeError(message)

    monkeypatch.setattr(nx_backend, "_enable_nx_cugraph_backend", _fail)
    nx_backend.maybe_enable_nx_gpu(GraphBackendConfig(use_gpu=True, strict=False))
    expect_equal(os.environ.get("NX_CUGRAPH_AUTOCONFIG"), "True")


def test_maybe_enable_nx_gpu_raises_when_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bubble errors when strict mode is enabled."""
    monkeypatch.delenv("NX_CUGRAPH_AUTOCONFIG", raising=False)
    message = "backend missing"

    def _fail() -> None:
        raise RuntimeError(message)

    monkeypatch.setattr(nx_backend, "_enable_nx_cugraph_backend", _fail)
    with pytest.raises(RuntimeError):
        nx_backend.maybe_enable_nx_gpu(GraphBackendConfig(use_gpu=True, strict=True))
    expect_equal(os.environ.get("NX_CUGRAPH_AUTOCONFIG"), "True")
