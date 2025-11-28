"""Tests for GraphRuntimePool caching and eviction."""

from __future__ import annotations

from pathlib import Path

from codeintel.analytics.graph_runtime import GraphRuntimeOptions, GraphRuntimePool
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.schemas import apply_all_schemas


def test_pool_reuses_runtime_within_ttl(fresh_gateway: StorageGateway) -> None:
    """
    Pool should return the same runtime when within TTL.

    Raises
    ------
    AssertionError
        When the pooled runtime is not reused.
    """
    gateway = fresh_gateway
    apply_all_schemas(gateway.con)
    snapshot = SnapshotRef(repo="r", commit="c", repo_root=Path())
    options = GraphRuntimeOptions(snapshot=snapshot)
    current = [0.0]

    def _time() -> float:
        return current[0]

    pool = GraphRuntimePool(ttl_seconds=10.0, time_func=_time)
    first = pool.get(gateway, options)
    current[0] += 1.0
    second = pool.get(gateway, options)
    if first is not second:
        message = "Expected runtime reuse within TTL"
        raise AssertionError(message)


def test_pool_expires_runtime_after_ttl(fresh_gateway: StorageGateway) -> None:
    """
    Pool should rebuild runtime after TTL expires.

    Raises
    ------
    AssertionError
        When the runtime is not refreshed after expiration.
    """
    gateway = fresh_gateway
    apply_all_schemas(gateway.con)
    snapshot = SnapshotRef(repo="r", commit="c", repo_root=Path())
    options = GraphRuntimeOptions(snapshot=snapshot)
    current = [0.0]

    def _time() -> float:
        return current[0]

    pool = GraphRuntimePool(ttl_seconds=0.5, time_func=_time)
    first = pool.get(gateway, options)
    current[0] += 1.0
    second = pool.get(gateway, options)
    if first is second:
        message = "Expected runtime to expire after TTL"
        raise AssertionError(message)


def test_pool_lru_eviction(fresh_gateway: StorageGateway) -> None:
    """
    LRU eviction should drop least-recently-used runtime when capacity exceeded.

    Raises
    ------
    AssertionError
        When the oldest runtime is not evicted.
    """
    gateway = fresh_gateway
    apply_all_schemas(gateway.con)
    snapshot1 = SnapshotRef(repo="r", commit="c1", repo_root=Path())
    snapshot2 = SnapshotRef(repo="r", commit="c2", repo_root=Path())
    opts1 = GraphRuntimeOptions(snapshot=snapshot1)
    opts2 = GraphRuntimeOptions(snapshot=snapshot2)
    current = [0.0]

    def _time() -> float:
        return current[0]

    pool = GraphRuntimePool(max_size=1, time_func=_time)
    first = pool.get(gateway, opts1)
    current[0] += 1.0
    pool.get(gateway, opts2)
    current[0] += 1.0
    again = pool.get(gateway, opts1)
    if again is first:
        message = "Expected LRU eviction of the oldest runtime"
        raise AssertionError(message)
