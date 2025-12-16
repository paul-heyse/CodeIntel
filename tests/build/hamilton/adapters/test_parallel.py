"""Tests for Hamilton parallel execution adapters."""

from __future__ import annotations

import pytest

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
    expect_true,
)


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
    def test_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
        """Test creating config from environment."""
        monkeypatch.setenv("HAMILTON_BACKEND", "threadpool")
        monkeypatch.setenv("HAMILTON_MAX_WORKERS", "4")

        config = ParallelConfig.from_env()
        expect_equal(config.backend, ExecutionBackend.THREADPOOL)
        expect_equal(config.max_workers, 4)

    @staticmethod
    def test_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
        """Test defaults when env vars not set."""
        monkeypatch.delenv("HAMILTON_BACKEND", raising=False)
        monkeypatch.delenv("HAMILTON_MAX_WORKERS", raising=False)

        config = ParallelConfig.from_env()
        expect_equal(config.backend, ExecutionBackend.SEQUENTIAL)
        expect_is_none(config.max_workers)

    @staticmethod
    def test_from_env_invalid_backend(monkeypatch: pytest.MonkeyPatch) -> None:
        """Test handling invalid backend in env."""
        monkeypatch.setenv("HAMILTON_BACKEND", "invalid_backend")

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
    def test_from_cli_args_none(monkeypatch: pytest.MonkeyPatch) -> None:
        """Test None args fall back to env."""
        monkeypatch.setenv("HAMILTON_BACKEND", "threadpool")

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
    def test_lazy_delegate() -> None:
        """Test delegate is created lazily."""
        adapter = ThreadPoolAdapter()
        expect_is_none(adapter._delegate)

        # Access delegate
        _ = adapter._ensure_delegate()
        expect_true(adapter._delegate is not None)


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
        expect_is_instance(adapter, ThreadPoolAdapter)

    @staticmethod
    def test_with_max_workers() -> None:
        """Test passing max_workers."""
        adapter = create_parallel_adapter("threadpool", max_workers=8)
        expect_true(adapter is not None)
        expect_equal(adapter.max_workers, 8)

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
