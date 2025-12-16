"""Parallel execution adapters for Hamilton builds.

This module provides adapters for parallel execution of Hamilton DAGs,
enabling better resource utilization for I/O-bound workloads.

Supported backends:
- threadpool: Multi-threaded execution using ThreadPoolExecutor
- sequential: Default single-threaded execution (no adapter needed)
- auto: Automatically select best available backend

Examples
--------
Create a threadpool adapter:

>>> from codeintel.build.hamilton.adapters.parallel import create_parallel_adapter
>>> adapter = create_parallel_adapter("threadpool", max_workers=4)
>>> dr = driver.Builder().with_adapters(adapter).build()

Check available backends:

>>> from codeintel.build.hamilton.adapters.parallel import get_available_backends
>>> backends = get_available_backends()
>>> print(backends)  # ['sequential', 'threadpool']
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, cast

from hamilton.lifecycle import base as lifecycle_base

from codeintel.build.hamilton import tags as ht
from codeintel.build.hamilton.contracts.enforced_gateway import ContractEnforcingStorageGateway
from codeintel.build.hamilton.env import BuildEnv
from codeintel.storage.gateway import open_gateway

if TYPE_CHECKING:
    from collections.abc import Callable

    from hamilton.graph import FunctionGraph
    from hamilton.lifecycle import ResultBuilder
    from hamilton.node import Node

    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.gateway.protocol import StorageGateway

__all__ = [
    "ExecutionBackend",
    "ParallelConfig",
    "ThreadPoolAdapter",
    "create_parallel_adapter",
    "get_available_backends",
]

log = logging.getLogger(__name__)


class ExecutionBackend(Enum):
    """Available execution backends.

    Attributes
    ----------
    SEQUENTIAL
        Default single-threaded execution.
    THREADPOOL
        Multi-threaded execution using ThreadPoolExecutor.
    AUTO
        Automatically select best available backend.
    """

    SEQUENTIAL = "sequential"
    THREADPOOL = "threadpool"
    AUTO = "auto"


@dataclass
class ParallelConfig:
    """Configuration for parallel execution.

    Attributes
    ----------
    backend
        The execution backend to use.
    max_workers
        Maximum number of parallel workers.
        For threadpool: number of threads.
    thread_name_prefix
        Prefix for thread names (threadpool only).

    Examples
    --------
    >>> config = ParallelConfig(
    ...     backend=ExecutionBackend.THREADPOOL,
    ...     max_workers=8,
    ... )
    """

    backend: ExecutionBackend = ExecutionBackend.SEQUENTIAL
    max_workers: int | None = None
    thread_name_prefix: str = "hamilton-build"

    @classmethod
    def from_env(cls) -> ParallelConfig:
        """Create config from environment variables.

        Reads:
        - HAMILTON_BACKEND: Backend name (sequential, threadpool, auto)
        - HAMILTON_MAX_WORKERS: Number of workers

        Returns
        -------
        ParallelConfig
            Configuration from environment.
        """
        backend_str = os.getenv("HAMILTON_BACKEND", "sequential").lower()
        try:
            backend = ExecutionBackend(backend_str)
        except ValueError:
            log.warning("Unknown backend %s, using sequential", backend_str)
            backend = ExecutionBackend.SEQUENTIAL

        max_workers_str = os.getenv("HAMILTON_MAX_WORKERS")
        max_workers = int(max_workers_str) if max_workers_str else None

        return cls(backend=backend, max_workers=max_workers)

    @classmethod
    def from_cli_args(
        cls,
        backend: str | None = None,
        max_workers: int | None = None,
    ) -> ParallelConfig:
        """Create config from CLI arguments.

        Parameters
        ----------
        backend
            Backend name from CLI.
        max_workers
            Number of workers from CLI.

        Returns
        -------
        ParallelConfig
            Configuration from CLI arguments.
        """
        if backend is None:
            return cls.from_env()

        try:
            backend_enum = ExecutionBackend(backend.lower())
        except ValueError:
            log.warning("Unknown backend %s, using sequential", backend)
            backend_enum = ExecutionBackend.SEQUENTIAL

        return cls(backend=backend_enum, max_workers=max_workers)


def get_available_backends() -> list[str]:
    """Get list of available execution backends.

    Returns
    -------
    list[str]
        Names of available backends.
    """
    available = ["sequential", "threadpool"]  # Always available

    # Check for optional backends (future support)
    if importlib.util.find_spec("ray") is not None:
        available.append("ray")

    if importlib.util.find_spec("dask") is not None:
        available.append("dask")

    return available


class ThreadPoolAdapter(
    lifecycle_base.BaseDoRemoteExecute,
    lifecycle_base.BaseDoBuildResult,
    lifecycle_base.BasePostGraphExecute,
):
    """ThreadPool execution adapter with a global write lock for materialize nodes.

    This adapter executes nodes in a ThreadPoolExecutor. Nodes tagged as
    `node_type=materialize` or `node_type=artifact` are executed under a global
    lock to prevent concurrent DuckDB writes.

    Parameters
    ----------
    max_workers
        Maximum number of parallel workers (threads). Defaults to
        ``min(32, os.cpu_count() + 4)`` in the standard library.
    thread_name_prefix
        Prefix for thread names.
    result_builder
        Optional Hamilton ResultBuilder for output aggregation. When omitted,
        the adapter returns a plain ``dict[str, object]`` of computed outputs.
    write_lock
        Optional lock to use for write nodes. When omitted, a new lock is
        created.
    """

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        thread_name_prefix: str = "hamilton-build",
        result_builder: ResultBuilder | None = None,
        write_lock: threading.Lock | None = None,
    ) -> None:
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self._result_builder = result_builder
        self._write_lock = write_lock or threading.Lock()
        self._primary_thread_id = threading.get_ident()
        self._gateways: dict[tuple[int, bool], StorageGateway] = {}
        self._gateway_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._closed = False

    @staticmethod
    def _is_write_node(node: Node) -> bool:
        tags = node.tags if isinstance(node.tags, dict) else {}
        node_type = tags.get(ht.TAG_NODE_TYPE)
        return node_type in {ht.NODE_TYPE_MATERIALIZE, ht.NODE_TYPE_ARTIFACT}

    def do_remote_execute(
        self,
        *,
        node: Node,
        kwargs: dict[str, object],
        execute_lifecycle_for_node: Callable[..., object],
    ) -> Future[object]:
        """Execute a node in the threadpool.

        Parameters
        ----------
        node
            Node to execute.
        kwargs
            Keyword arguments for the node (may contain Future values).
        execute_lifecycle_for_node
            Hamilton-provided callable that runs lifecycle hooks and executes the node.

        Returns
        -------
        Future[object]
            Future for the node result.
        """
        if self._is_write_node(node):
            return self._executor.submit(
                self._execute_with_lock, execute_lifecycle_for_node, kwargs
            )
        return self._executor.submit(self._execute_without_lock, execute_lifecycle_for_node, kwargs)

    def _execute_without_lock(self, fn: Callable[..., object], kwargs: dict[str, object]) -> object:
        resolved = _resolve_futures(kwargs)
        resolved = self._maybe_inject_thread_gateway(resolved, read_only=True)
        return fn(**resolved)

    def _execute_with_lock(self, fn: Callable[..., object], kwargs: dict[str, object]) -> object:
        resolved = _resolve_futures(kwargs)
        resolved = self._maybe_inject_thread_gateway(resolved, read_only=False)
        with self._write_lock:
            return fn(**resolved)

    def _maybe_inject_thread_gateway(
        self,
        resolved: dict[str, object],
        *,
        read_only: bool,
    ) -> dict[str, object]:
        env = resolved.get("env")
        if not isinstance(env, BuildEnv):
            return resolved

        # Main-thread execution can reuse the primary gateway safely.
        if threading.get_ident() == self._primary_thread_id:
            return resolved

        thread_gateway = self._get_thread_gateway(env, read_only=read_only)
        resolved["env"] = replace(env, gateway=thread_gateway)
        return resolved

    def _get_thread_gateway(self, env: BuildEnv, *, read_only: bool) -> StorageGateway:
        key = (threading.get_ident(), read_only)
        with self._gateway_lock:
            existing = self._gateways.get(key)
            if existing is not None:
                return existing

            cfg = _thread_storage_config(env.gateway.config, read_only=read_only)
            gw = open_gateway(cfg)
            if env.strict_contracts:
                gw = cast("StorageGateway", ContractEnforcingStorageGateway(gw))

            self._gateways[key] = gw
            return gw

    def do_build_result(self, *, outputs: object) -> object:
        """Build the final execution result, resolving any futures.

        Parameters
        ----------
        outputs
            Outputs dictionary from Hamilton execution.

        Returns
        -------
        object
            Resolved result object. Defaults to a ``dict[str, object]`` when no
            result_builder is configured.

        Raises
        ------
        TypeError
            If ``outputs`` is not a ``dict[str, object]`` or contains non-string keys.
        """
        if not isinstance(outputs, dict):
            msg = f"Expected outputs to be a dict, got {type(outputs)}"
            raise TypeError(msg)

        outputs_by_name: dict[str, object] = {}
        for key, value in outputs.items():
            if not isinstance(key, str):
                msg = f"Expected output key to be a str, got {type(key)}"
                raise TypeError(msg)
            outputs_by_name[key] = value

        resolved = _resolve_futures(outputs_by_name)
        if self._result_builder is not None:
            return self._result_builder.build_result(**resolved)
        return resolved

    def post_graph_execute(
        self,
        *,
        run_id: str,
        graph: FunctionGraph,
        success: bool,
        error: Exception | None,
        results: object | None,
    ) -> None:
        """Shutdown threadpool resources after execution.

        Parameters
        ----------
        run_id
            Hamilton run identifier (unused).
        graph
            Hamilton function graph (unused).
        success
            Whether the graph execution was successful.
        error
            Exception raised during execution, if any (unused).
        results
            Execution results, if available (unused).
        """
        _ = run_id
        _ = graph
        _ = error
        _ = results

        if self._closed:
            return

        self._closed = True
        self._executor.shutdown(wait=success, cancel_futures=not success)
        self._close_thread_gateways()

    def _close_thread_gateways(self) -> None:
        with self._gateway_lock:
            gateways = list(self._gateways.values())
            self._gateways.clear()
        for gw in gateways:
            gw.close()


def create_parallel_adapter(
    backend: str | ExecutionBackend = ExecutionBackend.SEQUENTIAL,
    *,
    max_workers: int | None = None,
    thread_name_prefix: str = "hamilton-build",
    result_builder: ResultBuilder | None = None,
) -> ThreadPoolAdapter | None:
    """Create a parallel execution adapter.

    Factory function for creating execution adapters based on the
    specified backend.

    Parameters
    ----------
    backend
        Execution backend to use. Can be string or ExecutionBackend enum.
    max_workers
        Maximum number of parallel workers.
    thread_name_prefix
        Prefix for thread names (threadpool only).
    result_builder
        Optional ResultBuilder for output aggregation.

    Returns
    -------
    ThreadPoolAdapter | None
        Adapter instance, or None for sequential execution.

    Examples
    --------
    >>> adapter = create_parallel_adapter("threadpool", max_workers=4)
    >>> if adapter:
    ...     dr = driver.Builder().with_adapters(adapter).build()

    >>> # Auto-select backend:
    >>> adapter = create_parallel_adapter("auto")
    """
    # Convert string to enum
    if isinstance(backend, str):
        try:
            backend = ExecutionBackend(backend.lower())
        except ValueError:
            log.warning("Unknown backend %s, using sequential", backend)
            backend = ExecutionBackend.SEQUENTIAL

    # Handle auto selection
    if backend == ExecutionBackend.AUTO:
        # For now, default to threadpool for auto
        # Future: Could check workload characteristics
        backend = ExecutionBackend.THREADPOOL
        log.info("Auto-selected backend: %s", backend.value)

    # Create adapter based on backend
    if backend == ExecutionBackend.SEQUENTIAL:
        # No adapter needed for sequential execution
        return None

    if backend == ExecutionBackend.THREADPOOL:
        return ThreadPoolAdapter(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
            result_builder=result_builder,
        )

    # Future backends would be handled here
    log.warning("Backend %s not yet implemented, using sequential", backend.value)
    return None


def create_adapter_from_config(config: ParallelConfig) -> ThreadPoolAdapter | None:
    """Create adapter from ParallelConfig.

    Parameters
    ----------
    config
        Parallel execution configuration.

    Returns
    -------
    ThreadPoolAdapter | None
        Adapter instance, or None for sequential execution.
    """
    return create_parallel_adapter(
        backend=config.backend,
        max_workers=config.max_workers,
        thread_name_prefix=config.thread_name_prefix,
    )


def _resolve_futures(kwargs: dict[str, object]) -> dict[str, object]:
    resolved: dict[str, object] = {}
    for key, value in kwargs.items():
        current = value
        while isinstance(current, Future):
            current = current.result()
        resolved[key] = current
    return resolved


def _thread_storage_config(config: StorageConfig, *, read_only: bool) -> StorageConfig:
    return replace(
        config,
        read_only=read_only,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
    )
