"""Unit coverage for NetworkX GPU backend wiring."""

from __future__ import annotations

from collections import UserDict

import pytest

from codeintel.cli import nx_backend
from codeintel.config.models import GraphBackendConfig
from tests._helpers.assertions import expect_equal, expect_true


def test_maybe_enable_nx_gpu_noop_when_disabled() -> None:
    """Skip backend wiring when use_gpu is False."""
    env: UserDict[str, str] = UserDict()
    nx_backend.maybe_enable_nx_gpu(GraphBackendConfig(use_gpu=False), env=env)
    expect_true("NX_CUGRAPH_AUTOCONFIG" not in env)


def test_maybe_enable_nx_gpu_soft_fallback() -> None:
    """Log and continue when GPU backend setup fails in non-strict mode."""
    env: UserDict[str, str] = UserDict()

    def _fail() -> None:
        message = "backend missing"
        raise RuntimeError(message)

    nx_backend.maybe_enable_nx_gpu(
        GraphBackendConfig(use_gpu=True, strict=False),
        env=env,
        enabler=_fail,
    )
    expect_equal(env.get("NX_CUGRAPH_AUTOCONFIG"), "True")


def test_maybe_enable_nx_gpu_raises_when_strict() -> None:
    """Bubble errors when strict mode is enabled."""
    env: UserDict[str, str] = UserDict()

    def _fail() -> None:
        message = "backend missing"
        raise RuntimeError(message)

    with pytest.raises(RuntimeError):
        nx_backend.maybe_enable_nx_gpu(
            GraphBackendConfig(use_gpu=True, strict=True),
            env=env,
            enabler=_fail,
        )
    expect_equal(env.get("NX_CUGRAPH_AUTOCONFIG"), "True")
