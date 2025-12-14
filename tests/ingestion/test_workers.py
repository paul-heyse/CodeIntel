"""Tests for worker pool infrastructure.

This module tests the worker pool utilities used for parallel
processing in ingestion pipelines.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from codeintel.core.concurrency import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_MIN_WORKERS,
    WorkerConfig,
    create_executor,
    executor_factory,
    resolve_worker_count,
    worker_pool,
)
from tests._helpers.assertions import (
    assert_cannot_setattr,
    expect_equal,
    expect_is_instance,
    expect_true,
)

TEST_WORKER_COUNT = 42
EXPECTED_DEFAULT_MAX_WORKERS = 16
EXPECTED_DEFAULT_MIN_WORKERS = 2


def test_worker_config_create_with_defaults() -> None:
    """WorkerConfig should have sensible defaults."""
    config = WorkerConfig(env_var="TEST_WORKERS")

    expect_equal(config.env_var, "TEST_WORKERS")
    expect_equal(config.default_max, DEFAULT_MAX_WORKERS)
    expect_equal(config.default_min, DEFAULT_MIN_WORKERS)
    expect_equal(config.executor_type, "thread")


def test_worker_config_create_with_custom_values() -> None:
    """WorkerConfig should accept custom values."""
    custom_max = 8
    custom_min = 1
    config = WorkerConfig(
        env_var="CUSTOM_WORKERS",
        default_max=custom_max,
        default_min=custom_min,
        executor_type="process",
    )

    expect_equal(config.env_var, "CUSTOM_WORKERS")
    expect_equal(config.default_max, custom_max)
    expect_equal(config.default_min, custom_min)
    expect_equal(config.executor_type, "process")


def test_worker_config_frozen_dataclass() -> None:
    """WorkerConfig should be immutable."""
    config = WorkerConfig(env_var="TEST")

    assert_cannot_setattr(config, "env_var", "NEW")


def test_resolve_worker_count_explicit_count_takes_precedence() -> None:
    """Explicit count should override environment and defaults."""
    explicit = 4
    result = resolve_worker_count(explicit)

    expect_equal(result, explicit)


def test_resolve_worker_count_explicit_zero_is_ignored() -> None:
    """Explicit count of zero should be ignored."""
    result = resolve_worker_count(0)

    cpu_count = os.cpu_count() or 1
    expected = min(DEFAULT_MAX_WORKERS, max(DEFAULT_MIN_WORKERS, cpu_count // 2))
    expect_equal(result, expected)


def test_resolve_worker_count_negative_explicit_is_ignored() -> None:
    """Negative explicit count should be ignored."""
    result = resolve_worker_count(-1)

    cpu_count = os.cpu_count() or 1
    expected = min(DEFAULT_MAX_WORKERS, max(DEFAULT_MIN_WORKERS, cpu_count // 2))
    expect_equal(result, expected)


def test_resolve_worker_count_env_var_takes_precedence_over_default() -> None:
    """Environment variable should override default calculation."""
    env_value = 6

    result = resolve_worker_count(env_var="TEST_WORKERS_ENV", env={"TEST_WORKERS_ENV": str(env_value)})

    expect_equal(result, env_value)


def test_resolve_worker_count_invalid_env_var_is_ignored() -> None:
    """Invalid environment variable should be ignored."""
    result = resolve_worker_count(
        env_var="TEST_WORKERS_INVALID", env={"TEST_WORKERS_INVALID": "not_a_number"}
    )

    cpu_count = os.cpu_count() or 1
    expected = min(DEFAULT_MAX_WORKERS, max(DEFAULT_MIN_WORKERS, cpu_count // 2))
    expect_equal(result, expected)


def test_resolve_worker_count_zero_env_var_is_ignored() -> None:
    """Environment variable with zero should be ignored."""
    result = resolve_worker_count(env_var="TEST_WORKERS_ZERO", env={"TEST_WORKERS_ZERO": "0"})

    cpu_count = os.cpu_count() or 1
    expected = min(DEFAULT_MAX_WORKERS, max(DEFAULT_MIN_WORKERS, cpu_count // 2))
    expect_equal(result, expected)


def test_resolve_worker_count_default_calculation() -> None:
    """Default calculation should be based on CPU count."""
    result = resolve_worker_count(env_var="NONEXISTENT_VAR")

    cpu_count = os.cpu_count() or 1
    expected = min(DEFAULT_MAX_WORKERS, max(DEFAULT_MIN_WORKERS, cpu_count // 2))
    expect_equal(result, expected)


def test_resolve_worker_count_custom_max_workers() -> None:
    """Custom max workers should be respected."""
    custom_max = 4
    result = resolve_worker_count(
        env_var="NONEXISTENT_VAR",
        default_max=custom_max,
    )

    cpu_count = os.cpu_count() or 1
    expected = min(custom_max, max(DEFAULT_MIN_WORKERS, cpu_count // 2))
    expect_equal(result, expected)


def test_resolve_worker_count_custom_min_workers() -> None:
    """Custom min workers should be respected."""
    custom_min = 4
    result = resolve_worker_count(
        env_var="NONEXISTENT_VAR",
        default_min=custom_min,
    )

    cpu_count = os.cpu_count() or 1
    expected = min(DEFAULT_MAX_WORKERS, max(custom_min, cpu_count // 2))
    expect_equal(result, expected)


def test_create_executor_process_executor() -> None:
    """create_executor should return ProcessPoolExecutor for 'process'."""
    workers = 2
    config = WorkerConfig(max_workers=workers, executor_type="process")
    executor = create_executor(config)

    try:
        expect_is_instance(executor, ProcessPoolExecutor)
    finally:
        executor.shutdown(wait=False)


def test_create_executor_thread_executor() -> None:
    """create_executor should return ThreadPoolExecutor for 'thread'."""
    workers = 2
    config = WorkerConfig(max_workers=workers, executor_type="thread")
    executor = create_executor(config)

    try:
        expect_is_instance(executor, ThreadPoolExecutor)
    finally:
        executor.shutdown(wait=False)


def test_create_executor_default_is_thread() -> None:
    """Default executor type should be thread."""
    config = WorkerConfig(max_workers=2)
    executor = create_executor(config)

    try:
        expect_is_instance(executor, ThreadPoolExecutor)
    finally:
        executor.shutdown(wait=False)


def test_worker_pool_process() -> None:
    """worker_pool should yield ProcessPoolExecutor for 'process'."""
    with worker_pool("process", 2) as executor:
        expect_is_instance(executor, ProcessPoolExecutor)


def test_worker_pool_thread() -> None:
    """worker_pool should yield ThreadPoolExecutor for 'thread'."""
    with worker_pool("thread", 2) as executor:
        expect_is_instance(executor, ThreadPoolExecutor)


def test_worker_pool_shutdown_on_exit() -> None:
    """worker_pool should shutdown executor on context exit."""
    with worker_pool("thread", 2) as executor:
        future = executor.submit(lambda: TEST_WORKER_COUNT)
        expect_equal(future.result(), TEST_WORKER_COUNT)


def test_executor_factory_returns_callable() -> None:
    """executor_factory should return a callable."""
    factory = executor_factory("thread", 2)

    expect_true(callable(factory))


def test_executor_factory_creates_thread_executor() -> None:
    """Factory should create ThreadPoolExecutor when called."""
    factory = executor_factory("thread", 2)
    executor = factory()

    try:
        expect_is_instance(executor, ThreadPoolExecutor)
    finally:
        executor.shutdown(wait=False)


def test_executor_factory_creates_process_executor() -> None:
    """Factory should create ProcessPoolExecutor when called."""
    factory = executor_factory("process", 2)
    executor = factory()

    try:
        expect_is_instance(executor, ProcessPoolExecutor)
    finally:
        executor.shutdown(wait=False)


def test_executor_factory_creates_new_instance_each_call() -> None:
    """Factory should create new executor on each call."""
    factory = executor_factory("thread", 2)
    executor1 = factory()
    executor2 = factory()

    try:
        expect_true(executor1 is not executor2)
    finally:
        executor1.shutdown(wait=False)
        executor2.shutdown(wait=False)


def test_default_max_workers() -> None:
    """DEFAULT_MAX_WORKERS should be reasonable."""
    expect_true(DEFAULT_MAX_WORKERS > 0)
    expect_equal(DEFAULT_MAX_WORKERS, EXPECTED_DEFAULT_MAX_WORKERS)


def test_default_min_workers() -> None:
    """DEFAULT_MIN_WORKERS should be reasonable."""
    expect_true(DEFAULT_MIN_WORKERS > 0)
    expect_equal(DEFAULT_MIN_WORKERS, EXPECTED_DEFAULT_MIN_WORKERS)
    expect_true(DEFAULT_MIN_WORKERS < DEFAULT_MAX_WORKERS)
