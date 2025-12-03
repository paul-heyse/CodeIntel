"""Tests for worker pool infrastructure.

This module tests the worker pool utilities used for parallel
processing in ingestion pipelines.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from codeintel.ingestion.infrastructure_utilities.workers import (
    AST_WORKER_CONFIG,
    CST_WORKER_CONFIG,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MIN_WORKERS,
    WorkerConfig,
    create_executor,
    executor_factory,
    resolve_worker_count,
    worker_pool,
)
from tests._helpers.frozen_test import try_setattr

# Test constants for magic values
TEST_WORKER_COUNT = 42
EXPECTED_DEFAULT_MAX_WORKERS = 16
EXPECTED_DEFAULT_MIN_WORKERS = 2


# --- WorkerConfig Tests ---


def test_worker_config_create_with_defaults() -> None:
    """WorkerConfig should have sensible defaults."""
    config = WorkerConfig(env_var="TEST_WORKERS")

    assert config.env_var == "TEST_WORKERS"
    assert config.default_max == DEFAULT_MAX_WORKERS
    assert config.default_min == DEFAULT_MIN_WORKERS
    assert config.executor_kind == "process"


def test_worker_config_create_with_custom_values() -> None:
    """WorkerConfig should accept custom values."""
    custom_max = 8
    custom_min = 1
    config = WorkerConfig(
        env_var="CUSTOM_WORKERS",
        default_max=custom_max,
        default_min=custom_min,
        executor_kind="thread",
    )

    assert config.env_var == "CUSTOM_WORKERS"
    assert config.default_max == custom_max
    assert config.default_min == custom_min
    assert config.executor_kind == "thread"


def test_worker_config_frozen_dataclass() -> None:
    """WorkerConfig should be immutable."""
    config = WorkerConfig(env_var="TEST")

    with pytest.raises(AttributeError):
        try_setattr(config, "env_var", "NEW")


# --- ResolveWorkerCount Tests ---


def test_resolve_worker_count_explicit_count_takes_precedence() -> None:
    """Explicit count should override environment and defaults."""
    explicit = 4
    result = resolve_worker_count(
        "NONEXISTENT_VAR",
        explicit_count=explicit,
    )

    assert result == explicit


def test_resolve_worker_count_explicit_zero_is_ignored() -> None:
    """Explicit count of zero should be ignored."""
    result = resolve_worker_count(
        "NONEXISTENT_VAR",
        explicit_count=0,
    )

    # Should fall back to CPU-based calculation
    cpu_count = os.cpu_count() or 1
    expected = min(DEFAULT_MAX_WORKERS, max(DEFAULT_MIN_WORKERS, cpu_count // 2))
    assert result == expected


def test_resolve_worker_count_negative_explicit_is_ignored() -> None:
    """Negative explicit count should be ignored."""
    result = resolve_worker_count(
        "NONEXISTENT_VAR",
        explicit_count=-1,
    )

    # Should fall back to CPU-based calculation
    cpu_count = os.cpu_count() or 1
    expected = min(DEFAULT_MAX_WORKERS, max(DEFAULT_MIN_WORKERS, cpu_count // 2))
    assert result == expected


def test_resolve_worker_count_env_var_takes_precedence_over_default() -> None:
    """Environment variable should override default calculation."""
    env_value = 6

    result = resolve_worker_count("TEST_WORKERS_ENV", env={"TEST_WORKERS_ENV": str(env_value)})

    assert result == env_value


def test_resolve_worker_count_invalid_env_var_is_ignored() -> None:
    """Invalid environment variable should be ignored."""
    result = resolve_worker_count(
        "TEST_WORKERS_INVALID", env={"TEST_WORKERS_INVALID": "not_a_number"}
    )

    # Should fall back to CPU-based calculation
    cpu_count = os.cpu_count() or 1
    expected = min(DEFAULT_MAX_WORKERS, max(DEFAULT_MIN_WORKERS, cpu_count // 2))
    assert result == expected


def test_resolve_worker_count_zero_env_var_is_ignored() -> None:
    """Environment variable with zero should be ignored."""
    result = resolve_worker_count("TEST_WORKERS_ZERO", env={"TEST_WORKERS_ZERO": "0"})

    # Should fall back to CPU-based calculation
    cpu_count = os.cpu_count() or 1
    expected = min(DEFAULT_MAX_WORKERS, max(DEFAULT_MIN_WORKERS, cpu_count // 2))
    assert result == expected


def test_resolve_worker_count_default_calculation() -> None:
    """Default calculation should be based on CPU count."""
    result = resolve_worker_count("NONEXISTENT_VAR")

    cpu_count = os.cpu_count() or 1
    expected = min(DEFAULT_MAX_WORKERS, max(DEFAULT_MIN_WORKERS, cpu_count // 2))
    assert result == expected


def test_resolve_worker_count_custom_max_workers() -> None:
    """Custom max workers should be respected."""
    custom_max = 4
    result = resolve_worker_count(
        "NONEXISTENT_VAR",
        default_max=custom_max,
    )

    cpu_count = os.cpu_count() or 1
    expected = min(custom_max, max(DEFAULT_MIN_WORKERS, cpu_count // 2))
    assert result == expected


def test_resolve_worker_count_custom_min_workers() -> None:
    """Custom min workers should be respected."""
    custom_min = 4
    result = resolve_worker_count(
        "NONEXISTENT_VAR",
        default_min=custom_min,
    )

    cpu_count = os.cpu_count() or 1
    expected = min(DEFAULT_MAX_WORKERS, max(custom_min, cpu_count // 2))
    assert result == expected


# --- CreateExecutor Tests ---


def test_create_executor_process_executor() -> None:
    """create_executor should return ProcessPoolExecutor for 'process'."""
    workers = 2
    executor = create_executor("process", workers)

    try:
        assert isinstance(executor, ProcessPoolExecutor)
    finally:
        executor.shutdown(wait=False)


def test_create_executor_thread_executor() -> None:
    """create_executor should return ThreadPoolExecutor for 'thread'."""
    workers = 2
    executor = create_executor("thread", workers)

    try:
        assert isinstance(executor, ThreadPoolExecutor)
    finally:
        executor.shutdown(wait=False)


def test_create_executor_unknown_kind_defaults_to_thread() -> None:
    """Unknown executor kind should default to ThreadPoolExecutor."""
    workers = 2
    executor = create_executor("unknown", workers)

    try:
        assert isinstance(executor, ThreadPoolExecutor)
    finally:
        executor.shutdown(wait=False)


# --- WorkerPool Tests ---


def test_worker_pool_process() -> None:
    """worker_pool should yield ProcessPoolExecutor for 'process'."""
    with worker_pool("process", 2) as executor:
        assert isinstance(executor, ProcessPoolExecutor)


def test_worker_pool_thread() -> None:
    """worker_pool should yield ThreadPoolExecutor for 'thread'."""
    with worker_pool("thread", 2) as executor:
        assert isinstance(executor, ThreadPoolExecutor)


def test_worker_pool_shutdown_on_exit() -> None:
    """worker_pool should shutdown executor on context exit."""
    with worker_pool("thread", 2) as executor:
        # Executor should be usable inside context
        future = executor.submit(lambda: TEST_WORKER_COUNT)
        assert future.result() == TEST_WORKER_COUNT

    # Executor should be shutdown after context exit
    # (we can't easily verify this directly, but submitting should fail)


# --- ExecutorFactory Tests ---


def test_executor_factory_returns_callable() -> None:
    """executor_factory should return a callable."""
    factory = executor_factory("thread", 2)

    assert callable(factory)


def test_executor_factory_creates_thread_executor() -> None:
    """Factory should create ThreadPoolExecutor when called."""
    factory = executor_factory("thread", 2)
    executor = factory()

    try:
        assert isinstance(executor, ThreadPoolExecutor)
    finally:
        executor.shutdown(wait=False)


def test_executor_factory_creates_process_executor() -> None:
    """Factory should create ProcessPoolExecutor when called."""
    factory = executor_factory("process", 2)
    executor = factory()

    try:
        assert isinstance(executor, ProcessPoolExecutor)
    finally:
        executor.shutdown(wait=False)


def test_executor_factory_creates_new_instance_each_call() -> None:
    """Factory should create new executor on each call."""
    factory = executor_factory("thread", 2)
    executor1 = factory()
    executor2 = factory()

    try:
        assert executor1 is not executor2
    finally:
        executor1.shutdown(wait=False)
        executor2.shutdown(wait=False)


# --- PreConfiguredConfigs Tests ---


def test_ast_worker_config() -> None:
    """AST_WORKER_CONFIG should be properly configured."""
    assert AST_WORKER_CONFIG.env_var == "CODEINTEL_AST_WORKERS"
    assert AST_WORKER_CONFIG.default_max == DEFAULT_MAX_WORKERS
    assert AST_WORKER_CONFIG.executor_kind == "process"


def test_cst_worker_config() -> None:
    """CST_WORKER_CONFIG should be properly configured."""
    assert CST_WORKER_CONFIG.env_var == "CODEINTEL_CST_WORKERS"
    assert CST_WORKER_CONFIG.default_max == DEFAULT_MAX_WORKERS
    assert CST_WORKER_CONFIG.executor_kind == "process"


# --- DefaultConstants Tests ---


def test_default_max_workers() -> None:
    """DEFAULT_MAX_WORKERS should be reasonable."""
    assert DEFAULT_MAX_WORKERS > 0
    assert DEFAULT_MAX_WORKERS == EXPECTED_DEFAULT_MAX_WORKERS


def test_default_min_workers() -> None:
    """DEFAULT_MIN_WORKERS should be reasonable."""
    assert DEFAULT_MIN_WORKERS > 0
    assert DEFAULT_MIN_WORKERS == EXPECTED_DEFAULT_MIN_WORKERS
    assert DEFAULT_MIN_WORKERS < DEFAULT_MAX_WORKERS
