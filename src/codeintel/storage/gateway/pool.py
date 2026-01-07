"""Read-only Warehouse pool for serving workloads.

DuckDB connections are not designed to be shared concurrently across threads. For
serving, we instead maintain a pool of *read-only* connections and hand out one
exclusive handle per request. Each handle is wrapped with:

- ``MinimalStorageGateway`` (policy + DuckDB context access over a raw connection)
- ``Warehouse`` (single I/O boundary façade used by higher layers)

This keeps serving code free from ad-hoc gateway creation and centralizes
connection management inside the storage layer.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Empty, LifoQueue
from typing import TYPE_CHECKING, cast

from codeintel.core.storage import StorageContext
from codeintel.storage.backend import DuckDBSession
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.gateway.minimal import MinimalStorageGateway
from codeintel.storage.warehouse import Warehouse

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.storage.backend.duckdb_session import DuckDBConnectConfig
    from codeintel.storage.gateway.protocol import StorageGateway

__all__ = ["PoolConfig", "ReadPoolWarehouse"]


@dataclass(frozen=True)
class PoolConfig:
    """Pool configuration parameters.

    Parameters
    ----------
    size
        Number of read-only handles in the pool.
    threads
        DuckDB threads per connection (None = default).
    memory_limit
        DuckDB memory limit per connection (e.g., "2GB").
    temp_directory
        Temporary directory for spilling.
    """

    size: int = 4
    threads: int | None = None
    memory_limit: str | None = None
    temp_directory: str | None = None


class ReadPoolWarehouse:
    """Thread-safe pool of read-only ``Warehouse`` handles.

    Parameters
    ----------
    db_path
        Path to DuckDB database file.
    cfg
        Pool configuration.
    storage_config
        Optional StorageConfig to reuse for read-only connections.
    """

    def __init__(
        self,
        db_path: Path,
        cfg: PoolConfig,
        *,
        storage_config: StorageConfig | None = None,
    ) -> None:
        self._db_path = db_path
        if storage_config is None:
            storage_config = StorageConfig.for_readonly(db_path)
        elif storage_config.db_path != db_path:
            msg = "StorageConfig db_path does not match pool db_path"
            raise ValueError(msg)
        self._storage_config = storage_config
        self._cfg = cfg
        self._available: LifoQueue[Warehouse] = LifoQueue()
        self._lock = threading.Lock()
        self._closing = False
        self._init_handles()

    def _connect_config(self) -> DuckDBConnectConfig:
        config: DuckDBConnectConfig = {}
        if self._cfg.threads is not None:
            config["threads"] = self._cfg.threads
        if self._cfg.memory_limit is not None:
            config["memory_limit"] = self._cfg.memory_limit
        if self._cfg.temp_directory is not None:
            config["temp_directory"] = self._cfg.temp_directory
        return config

    def _open(self) -> Warehouse:
        session = DuckDBSession(
            self._storage_config,
            duckdb_config=self._connect_config(),
        )
        con = session.open_reader()
        gateway = MinimalStorageGateway(con, config=self._storage_config)
        context = StorageContext(gateway=cast("StorageGateway", gateway))
        return Warehouse(context=context)

    def _init_handles(self) -> None:
        for _ in range(max(1, self._cfg.size)):
            self._available.put(self._open())

    @contextmanager
    def acquire(self) -> Iterator[Warehouse]:
        """Acquire a ``Warehouse`` handle from the pool.

        Yields
        ------
        Warehouse
            A request-scoped warehouse handle backed by an exclusive DuckDB
            connection.

        Raises
        ------
        RuntimeError
            If the pool is closing.
        """
        with self._lock:
            if self._closing:
                msg = "Pool is closing"
                raise RuntimeError(msg)

        handle = self._available.get()
        try:
            yield handle
        finally:
            self._release(handle)

    def _release(self, handle: Warehouse) -> None:
        with self._lock:
            closing = self._closing
        if closing:
            handle.gateway.close()
            return
        self._available.put(handle)

    def close_gracefully(self) -> None:
        """Mark the pool as closing and drain available handles."""
        with self._lock:
            self._closing = True

        while True:
            try:
                handle = self._available.get_nowait()
            except Empty:
                break
            handle.gateway.close()
