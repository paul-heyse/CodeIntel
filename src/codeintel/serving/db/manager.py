"""Serving database manager with hot-swap support.

Watches the pointer file and swaps connection pools when the snapshot changes,
enabling zero-downtime deployments.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.db.pool import DuckDBPoolConfig, DuckDBReadPool

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.storage.gateway.protocol import DuckDBConnection


@dataclass
class ServingDBManager:
    """Manages serving database connections with hot-swap support.

    Parameters
    ----------
    pointer_path
        Path to current.json pointer file.
    pool_cfg
        Connection pool configuration.
    poll_interval_s
        Seconds between pointer file checks.
    """

    pointer_path: Path
    pool_cfg: DuckDBPoolConfig = field(default_factory=DuckDBPoolConfig)
    poll_interval_s: float = 1.0

    _pointer: ServingSnapshotPointer | None = field(default=None, init=False)
    _pool: DuckDBReadPool | None = field(default=None, init=False)
    _watch_task: asyncio.Task[None] | None = field(default=None, init=False)
    _last_mtime_ns: int | None = field(default=None, init=False)

    async def start(self) -> None:
        """Initialize manager and start watch loop."""
        await self._reload_if_needed(force=True)
        self._watch_task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop watch loop and close pool."""
        if self._watch_task is not None:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task
        if self._pool is not None:
            self._pool.close_gracefully()

    def current_pointer(self) -> ServingSnapshotPointer:
        """Return current snapshot pointer.

        Returns
        -------
        ServingSnapshotPointer
            Active snapshot pointer.

        Raises
        ------
        RuntimeError
            If manager not started or pointer not yet available.
        """
        if self._pointer is None:
            msg = "ServingDBManager has no active snapshot pointer"
            raise RuntimeError(msg)
        return self._pointer

    @contextmanager
    def connect(self) -> Iterator[tuple[DuckDBConnection, ServingSnapshotPointer]]:
        """Yield a database connection plus the current pointer.

        Yields
        ------
        tuple[DuckDBConnection, ServingSnapshotPointer]
            Connection and current pointer.

        Raises
        ------
        RuntimeError
            If manager not started.
        """
        pool = self._pool
        pointer = self._pointer
        if pool is None or pointer is None:
            msg = "ServingDBManager not started"
            raise RuntimeError(msg)

        con = pool.acquire()
        try:
            yield con, pointer
        finally:
            pool.release(con)

    async def _watch_loop(self) -> None:
        """Background task watching for pointer changes."""
        while True:
            await self._reload_if_needed(force=False)
            await asyncio.sleep(self.poll_interval_s)

    async def _reload_if_needed(self, *, force: bool) -> None:
        """Reload snapshot if pointer file changed."""
        if not self.pointer_path.exists():
            return

        st = self.pointer_path.stat()
        if not force and self._last_mtime_ns == st.st_mtime_ns:
            return
        self._last_mtime_ns = st.st_mtime_ns

        new_ptr = ServingSnapshotPointer.load(self.pointer_path)

        # Skip if same DB path (metadata-only update)
        if self._pointer is not None and new_ptr.db_path == self._pointer.db_path:
            self._pointer = new_ptr
            return

        new_pool = DuckDBReadPool(new_ptr.db_path, self.pool_cfg)
        old_pool = self._pool
        self._pool = new_pool
        self._pointer = new_ptr

        if old_pool is not None:
            old_pool.close_gracefully()


__all__ = ["ServingDBManager"]
