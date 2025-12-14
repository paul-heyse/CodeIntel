"""Shared worker pool infrastructure for ingestion pipelines.

This module provides consolidated worker pool management used by AST, CST,
and other parser-based ingestion pipelines.

.. deprecated:: 1.0
    Import from ``codeintel.core.concurrency`` instead.
    This module will be removed in a future version.

Examples
--------
Instead of:

>>> from codeintel.ingestion.infrastructure.workers import WorkerConfig

Use:

>>> from codeintel.core.concurrency import WorkerConfig
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.concurrency import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_MIN_WORKERS,
)
from codeintel.core.concurrency import (
    executor_factory as _core_executor_factory,
)
from codeintel.core.concurrency import (
    resolve_worker_count as _core_resolve_worker_count,
)
from codeintel.core.concurrency import (
    worker_pool as _core_worker_pool,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from concurrent.futures import Executor

if not TYPE_CHECKING:
    warnings.warn(
        "Importing from codeintel.ingestion.infrastructure.workers is deprecated. "
        "Import from codeintel.core.concurrency instead.",
        DeprecationWarning,
        stacklevel=2,
    )


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration for worker pool behavior.

    .. deprecated:: 1.0
        Use ``codeintel.core.concurrency.WorkerConfig`` instead.

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
    """Resolve worker pool size from explicit value, environment, or CPU count.

    .. deprecated:: 1.0
        Use ``codeintel.core.concurrency.resolve_worker_count`` instead.

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
    return _core_resolve_worker_count(
        explicit_count,
        env_var=env_var,
        default_max=default_max,
        default_min=default_min,
        env=env,
    )


def create_executor(
    kind: str,
    workers: int,
) -> ThreadPoolExecutor | ProcessPoolExecutor:
    """Create an executor of the specified type.

    .. deprecated:: 1.0
        Use ``codeintel.core.concurrency.create_executor`` instead.

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
    """Context manager for worker pool lifecycle.

    .. deprecated:: 1.0
        Use ``codeintel.core.concurrency.worker_pool`` instead.

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
    # The core version expects Literal["thread", "process"]
    # but we accept any string here for backward compatibility
    with _core_worker_pool(kind, workers) as executor:  # type: ignore[arg-type]
        yield executor


def executor_factory(
    kind: str,
    workers: int,
) -> Callable[[], Executor]:
    """Create a factory function that produces executors.

    .. deprecated:: 1.0
        Use ``codeintel.core.concurrency.executor_factory`` instead.

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
    # The core version expects Literal["thread", "process"]
    return _core_executor_factory(kind, workers)  # type: ignore[arg-type]


# Domain-specific worker configurations for ingestion pipelines
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
