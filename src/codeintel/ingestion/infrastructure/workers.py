"""Shared worker pool infrastructure for ingestion pipelines.

This module provides consolidated worker pool management used by AST, CST,
and other parser-based ingestion pipelines. It eliminates duplicate
implementations across ingest modules.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from concurrent.futures import Executor

log = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 16
DEFAULT_MIN_WORKERS = 2


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration for worker pool behavior.

    Attributes
    ----------
    env_var
        Environment variable name for worker count override.
    default_max
        Maximum worker count when not overridden.
    default_min
        Minimum worker count floor.
    executor_kind
        Default executor type ("thread" or "process").
    """

    env_var: str
    default_max: int = DEFAULT_MAX_WORKERS
    default_min: int = DEFAULT_MIN_WORKERS
    executor_kind: str = "process"


def resolve_worker_count(
    env_var: str,
    *,
    explicit_count: int | None = None,
    default_max: int = DEFAULT_MAX_WORKERS,
    default_min: int = DEFAULT_MIN_WORKERS,
    env: Mapping[str, str] | None = None,
) -> int:
    """
    Resolve worker pool size from explicit value, environment, or CPU count.

    Parameters
    ----------
    env_var
        Environment variable name to check for override.
    explicit_count
        Explicit worker count if provided (takes precedence).
    default_max
        Maximum worker count when derived from CPU.
    default_min
        Minimum worker count floor.
    env
        Optional environment mapping to read overrides from.

    Returns
    -------
    int
        Resolved worker count.
    """
    environment = env or os.environ

    if explicit_count is not None and explicit_count > 0:
        return explicit_count

    env_value = environment.get(env_var)
    if env_value:
        try:
            value = int(env_value)
            if value > 0:
                return value
        except ValueError:
            log.warning("Ignoring invalid %s=%s", env_var, env_value)

    cpu_count = os.cpu_count() or 1
    return min(default_max, max(default_min, cpu_count // 2))


def create_executor(
    kind: str,
    workers: int,
) -> ThreadPoolExecutor | ProcessPoolExecutor:
    """
    Create an executor of the specified type.

    Parameters
    ----------
    kind
        Executor type: "thread" or "process".
    workers
        Maximum worker count.

    Returns
    -------
    ThreadPoolExecutor | ProcessPoolExecutor
        Configured executor instance.
    """
    if kind == "process":
        return ProcessPoolExecutor(max_workers=workers)
    return ThreadPoolExecutor(max_workers=workers)


@contextmanager
def worker_pool(
    kind: str,
    workers: int,
) -> Iterator[Executor]:
    """
    Context manager for worker pool lifecycle.

    Parameters
    ----------
    kind
        Executor type: "thread" or "process".
    workers
        Maximum worker count.

    Yields
    ------
    Executor
        Configured executor for use within the context.
    """
    executor = create_executor(kind, workers)
    try:
        yield executor
    finally:
        executor.shutdown(wait=True)


def executor_factory(
    kind: str,
    workers: int,
) -> Callable[[], Executor]:
    """
    Create a factory function that produces executors.

    This is useful for deferred executor creation in pipeline execution.

    Parameters
    ----------
    kind
        Executor type: "thread" or "process".
    workers
        Maximum worker count.

    Returns
    -------
    Callable[[], Executor]
        Factory function that creates executors.
    """

    def _factory() -> Executor:
        return create_executor(kind, workers)

    return _factory


AST_WORKER_CONFIG = WorkerConfig(
    env_var="CODEINTEL_AST_WORKERS",
    default_max=DEFAULT_MAX_WORKERS,
    executor_kind="process",
)

CST_WORKER_CONFIG = WorkerConfig(
    env_var="CODEINTEL_CST_WORKERS",
    default_max=DEFAULT_MAX_WORKERS,
    executor_kind="process",
)


__all__ = [
    "AST_WORKER_CONFIG",
    "CST_WORKER_CONFIG",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_MIN_WORKERS",
    "WorkerConfig",
    "create_executor",
    "executor_factory",
    "resolve_worker_count",
    "worker_pool",
]
