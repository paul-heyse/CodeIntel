"""Tests for Hamilton parallel execution adapters."""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, cast

import pytest
from hamilton.lifecycle import base as lifecycle_base

from codeintel.build.hamilton.adapters.parallel import (
    ExecutionBackend,
    ParallelConfig,
    ThreadPoolAdapter,
    create_parallel_adapter,
    get_available_backends,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_none,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextlib.contextmanager
def _temporary_env(values: dict[str, str | None]) -> Iterator[None]:
    saved: dict[str, str | None] = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield None
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TestExecutionBackend:
    """Test suite for ExecutionBackend enum."""

    @staticmethod
    def test_values() -> None:
        """Test enum values."""
        expect_equal(ExecutionBackend.SEQUENTIAL.value, "sequential")
        expect_equal(ExecutionBackend.THREADPOOL.value, "threadpool")
        expect_equal(ExecutionBackend.AUTO.value, "auto")

    @staticmethod
    def test_from_string() -> None:
        """Test creating from string."""
        expect_equal(ExecutionBackend("sequential"), ExecutionBackend.SEQUENTIAL)
        expect_equal(ExecutionBackend("threadpool"), ExecutionBackend.THREADPOOL)


class TestParallelConfig:
    """Test suite for ParallelConfig dataclass."""

    @staticmethod
    def test_default_config() -> None:
        """Test default configuration."""
        config = ParallelConfig()
        expect_equal(config.backend, ExecutionBackend.SEQUENTIAL)
        expect_is_none(config.max_workers)
        expect_equal(config.thread_name_prefix, "hamilton-build")

    @staticmethod
    def test_custom_config() -> None:
        """Test custom configuration."""
        config = ParallelConfig(
            backend=ExecutionBackend.THREADPOOL,
            max_workers=8,
            thread_name_prefix="custom",
        )
        expect_equal(config.backend, ExecutionBackend.THREADPOOL)
        expect_equal(config.max_workers, 8)
        expect_equal(config.thread_name_prefix, "custom")

    @staticmethod
    def test_from_env() -> None:
        """Test creating config from environment."""
        with _temporary_env({"HAMILTON_BACKEND": "threadpool", "HAMILTON_MAX_WORKERS": "4"}):
            config = ParallelConfig.from_env()
            expect_equal(config.backend, ExecutionBackend.THREADPOOL)
            expect_equal(config.max_workers, 4)

    @staticmethod
    def test_from_env_defaults() -> None:
        """Test defaults when env vars not set."""
        with _temporary_env({"HAMILTON_BACKEND": None, "HAMILTON_MAX_WORKERS": None}):
            config = ParallelConfig.from_env()
            expect_equal(config.backend, ExecutionBackend.SEQUENTIAL)
            expect_is_none(config.max_workers)

    @staticmethod
    def test_from_env_invalid_backend() -> None:
        """Test handling invalid backend in env."""
        with _temporary_env({"HAMILTON_BACKEND": "invalid_backend"}):
            config = ParallelConfig.from_env()
            expect_equal(config.backend, ExecutionBackend.SEQUENTIAL)  # Falls back

    @staticmethod
    def test_from_cli_args() -> None:
        """Test creating from CLI arguments."""
        config = ParallelConfig.from_cli_args(
            backend="threadpool",
            max_workers=16,
        )
        expect_equal(config.backend, ExecutionBackend.THREADPOOL)
        expect_equal(config.max_workers, 16)

    @staticmethod
    def test_from_cli_args_none() -> None:
        """Test None args fall back to env."""
        with _temporary_env({"HAMILTON_BACKEND": "threadpool"}):
            config = ParallelConfig.from_cli_args(backend=None)
            expect_equal(config.backend, ExecutionBackend.THREADPOOL)


class TestGetAvailableBackends:
    """Test suite for get_available_backends."""

    @staticmethod
    def test_always_includes_basic() -> None:
        """Test that basic backends are always available."""
        backends = get_available_backends()
        expect_in("sequential", backends)
        expect_in("threadpool", backends)


class TestThreadPoolAdapter:
    """Test suite for ThreadPoolAdapter."""

    @staticmethod
    def test_creation() -> None:
        """Test creating adapter."""
        adapter = ThreadPoolAdapter(max_workers=4)
        expect_equal(adapter.max_workers, 4)
        expect_equal(adapter.thread_name_prefix, "hamilton-build")

    @staticmethod
    def test_implements_remote_execute_and_build_result() -> None:
        """ThreadPoolAdapter should implement Hamilton lifecycle adapter interfaces."""
        adapter = ThreadPoolAdapter()
        expect_is_instance(adapter, lifecycle_base.BaseDoRemoteExecute)
        expect_is_instance(adapter, lifecycle_base.BaseDoBuildResult)


class TestCreateParallelAdapter:
    """Test suite for create_parallel_adapter factory."""

    @staticmethod
    def test_sequential_returns_none() -> None:
        """Test sequential backend returns None."""
        adapter = create_parallel_adapter("sequential")
        expect_is_none(adapter)

    @staticmethod
    def test_threadpool_returns_adapter() -> None:
        """Test threadpool returns ThreadPoolAdapter."""
        adapter = create_parallel_adapter("threadpool")
        expect_is_instance(adapter, ThreadPoolAdapter)

    @staticmethod
    def test_auto_selects_threadpool() -> None:
        """Test auto mode selects threadpool."""
        adapter = create_parallel_adapter("auto")
        if adapter is None:
            return
        expect_is_instance(adapter, ThreadPoolAdapter)

    @staticmethod
    def test_with_max_workers() -> None:
        """Test passing max_workers."""
        adapter = create_parallel_adapter("threadpool", max_workers=8)
        expect_is_instance(adapter, ThreadPoolAdapter)
        expect_equal(cast("ThreadPoolAdapter", adapter).max_workers, 8)

    @staticmethod
    def test_invalid_backend_falls_back() -> None:
        """Test invalid backend falls back to sequential."""
        adapter = create_parallel_adapter("invalid_backend")
        expect_is_none(adapter)  # Sequential returns None

    @staticmethod
    def test_with_enum() -> None:
        """Test passing ExecutionBackend enum."""
        adapter = create_parallel_adapter(ExecutionBackend.THREADPOOL)
        expect_is_instance(adapter, ThreadPoolAdapter)


@pytest.mark.parametrize(
    ("backend_str", "expected_type"),
    [
        pytest.param("sequential", type(None), id="sequential"),
        pytest.param("threadpool", ThreadPoolAdapter, id="threadpool"),
        pytest.param("THREADPOOL", ThreadPoolAdapter, id="uppercase"),
        pytest.param("ThreadPool", ThreadPoolAdapter, id="mixedcase"),
    ],
)
def test_backend_string_normalization(
    backend_str: str,
    expected_type: type,
) -> None:
    """Parametrized test for backend string normalization."""
    adapter = create_parallel_adapter(backend_str)
    expect_is_instance(adapter, expected_type)
