"""Tests for GPU backend selection and fallback handling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from codeintel.build.graphs.engine import GraphKind
from codeintel.build.graphs.engine.backend import BackendEnablement, maybe_enable_nx_gpu
from codeintel.build.graphs.runtime import build_graph_runtime
from codeintel.config.primitives import GraphBackendConfig
from tests._helpers.assertions import expect_true
from tests._helpers.factories import make_graph_runtime_options
from tests._helpers.fakes.graph_runtime import graph_engine_with_cache
from tests._helpers.fixtures.graphs import call_graph_fixture

if TYPE_CHECKING:
    from tests._helpers.fakes.graph_contexts import GraphTestEnv


def test_maybe_enable_nx_gpu_success() -> None:
    """GPU enablement should return enabled status when enabler succeeds."""
    cfg = GraphBackendConfig(use_gpu=True, backend="nx-cugraph", strict=True)
    status = maybe_enable_nx_gpu(cfg, enabler=lambda: None)
    expect_true(status.gpu_enabled, message="GPU backend should be enabled")
    expect_true(
        status.effective_backend == "nx-cugraph", message="Unexpected backend effective value"
    )
    expect_true(status.fallback_reason is None, message="Fallback reason should be None on success")


def test_maybe_enable_nx_gpu_fallback() -> None:
    """Non-strict mode should fall back to CPU and capture reason."""
    cfg = GraphBackendConfig(use_gpu=True, backend="auto", strict=False)

    def _fail() -> None:
        err = RuntimeError("no gpu")
        raise err

    status = maybe_enable_nx_gpu(cfg, enabler=_fail)
    expect_true(not status.gpu_enabled, message="GPU should be disabled after fallback")
    expect_true(status.effective_backend == "cpu", message="Expected CPU fallback")
    expect_true(status.fallback_reason is not None, message="Fallback reason should be populated")


def test_maybe_enable_nx_gpu_strict_raises() -> None:
    """Strict mode should raise when GPU cannot be enabled."""
    cfg = GraphBackendConfig(use_gpu=True, backend="auto", strict=True)

    def _fail() -> None:
        err = RuntimeError("no gpu")
        raise err

    with pytest.raises(RuntimeError):
        maybe_enable_nx_gpu(cfg, enabler=_fail)


def test_build_graph_runtime_captures_backend_info(graph_executor_env: GraphTestEnv) -> None:
    """Runtime should expose backend metadata recorded during engine construction."""
    cfg = GraphBackendConfig(use_gpu=True, backend="nx-cugraph", strict=False)
    engine_with_metadata: Any = graph_engine_with_cache(
        graph_executor_env.gateway,
        graph_executor_env.snapshot,
        {GraphKind.CALL_GRAPH: call_graph_fixture()},
    )
    engine_with_metadata.backend_info = BackendEnablement(
        requested_backend=cfg.backend,
        requested_gpu=True,
        effective_backend="nx-cugraph",
        gpu_enabled=True,
        fallback_reason=None,
    )

    runtime = build_graph_runtime(
        graph_executor_env.gateway,
        make_graph_runtime_options(
            snapshot=graph_executor_env.snapshot,
            backend=cfg,
            engine=engine_with_metadata,
        ),
        enabler=lambda: None,
    )
    info = runtime.backend_info
    expect_true(info is not None, message="backend_info should be set on runtime")
    expect_true(
        info is not None and info.gpu_enabled,
        message="GPU should be marked enabled in backend_info",
    )
    expect_true(
        info is not None and info.effective_backend == "nx-cugraph",
        message="Unexpected backend recorded on runtime",
    )
