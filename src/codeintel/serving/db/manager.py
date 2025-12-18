"""Serving database manager with hot-swap support.

Watches the pointer file and swaps connection pools when the snapshot changes,
enabling zero-downtime deployments.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.build.spec import buildspec_from_json
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.semantic.inventory import SchemaInventory
from codeintel.serving.semantic.registry import SemanticRegistry
from codeintel.storage.gateway.pool import PoolConfig, ReadPoolWarehouse

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from codeintel.build.spec import BuildSpec
    from codeintel.storage.warehouse import Warehouse


_POOL_CLOSING_ERROR = "Pool is closing"
_POOL_ACQUIRE_MAX_ATTEMPTS = 3


def _default_export_pool_cfg() -> PoolConfig:
    return PoolConfig(size=1)


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
    hot_swap
        When True, watch the pointer file and hot-swap pools.
    """

    pointer_path: Path
    pool_cfg: PoolConfig = field(default_factory=PoolConfig)
    export_pool_cfg: PoolConfig = field(default_factory=_default_export_pool_cfg)
    poll_interval_s: float = 1.0
    hot_swap: bool = True

    _pointer: ServingSnapshotPointer | None = field(default=None, init=False)
    _pool: ReadPoolWarehouse | None = field(default=None, init=False)
    _export_pool: ReadPoolWarehouse | None = field(default=None, init=False)
    _snapshot_cache: dict[Path, ServingSnapshotContext] = field(default_factory=dict, init=False)
    _watch_task: asyncio.Task[None] | None = field(default=None, init=False)
    _last_mtime_ns: int | None = field(default=None, init=False)

    async def start(self) -> None:
        """Initialize manager and start watch loop."""
        await self._reload_if_needed(force=True)
        if self.hot_swap:
            self._watch_task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop watch loop and close pool."""
        if self._watch_task is not None:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task
        if self._pool is not None:
            self._pool.close_gracefully()
        if self._export_pool is not None:
            self._export_pool.close_gracefully()

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
    def connect(self) -> Iterator[tuple[Warehouse, ServingSnapshotPointer]]:
        """Yield a warehouse handle plus the current pointer.

        Yields
        ------
        tuple[Warehouse, ServingSnapshotPointer]
            Warehouse handle and current pointer.

        Raises
        ------
        RuntimeError
            If manager not started.
        """
        attempts = 0
        while True:
            pool = self._pool
            pointer = self._pointer
            if pool is None or pointer is None:
                msg = "ServingDBManager not started"
                raise RuntimeError(msg)
            try:
                with pool.acquire() as warehouse:
                    yield warehouse, pointer
            except RuntimeError as exc:
                if str(exc) != _POOL_CLOSING_ERROR:
                    raise
                attempts += 1
                if attempts >= _POOL_ACQUIRE_MAX_ATTEMPTS:
                    msg = "ServingDBManager could not acquire a warehouse handle"
                    raise RuntimeError(msg) from exc
                continue
            else:
                return

    @contextmanager
    def connect_export(self) -> Iterator[tuple[Warehouse, ServingSnapshotPointer]]:
        """Yield a warehouse handle from the export pool plus the current pointer.

        This isolates long-lived export streams from interactive query capacity.

        Yields
        ------
        tuple[Warehouse, ServingSnapshotPointer]
            Warehouse handle and current pointer.

        Raises
        ------
        RuntimeError
            If manager not started.
        """
        attempts = 0
        while True:
            pool = self._export_pool
            pointer = self._pointer
            if pool is None or pointer is None:
                msg = "ServingDBManager not started"
                raise RuntimeError(msg)
            try:
                with pool.acquire() as warehouse:
                    yield warehouse, pointer
            except RuntimeError as exc:
                if str(exc) != _POOL_CLOSING_ERROR:
                    raise
                attempts += 1
                if attempts >= _POOL_ACQUIRE_MAX_ATTEMPTS:
                    msg = "ServingDBManager could not acquire an export warehouse handle"
                    raise RuntimeError(msg) from exc
                continue
            else:
                return

    def snapshot_context(self, pointer: ServingSnapshotPointer) -> ServingSnapshotContext:
        """Return cached snapshot context for the given pointer.

        Parameters
        ----------
        pointer
            Snapshot pointer whose artifacts should be loaded.

        Returns
        -------
        ServingSnapshotContext
            In-memory context containing registry/inventory/buildspec for the snapshot.
        """
        cached = self._snapshot_cache.get(pointer.db_path)
        if cached is not None and cached.pointer == pointer:
            return cached

        context = _load_snapshot_context(pointer)
        self._snapshot_cache[pointer.db_path] = context
        return context

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
            self._snapshot_cache[new_ptr.db_path] = _load_snapshot_context(new_ptr)
            return

        new_pool = ReadPoolWarehouse(new_ptr.db_path, self.pool_cfg)
        new_export_pool = ReadPoolWarehouse(new_ptr.db_path, self.export_pool_cfg)
        old_pool = self._pool
        old_export_pool = self._export_pool
        self._pool = new_pool
        self._export_pool = new_export_pool
        self._pointer = new_ptr
        self._snapshot_cache[new_ptr.db_path] = _load_snapshot_context(new_ptr)

        if old_pool is not None:
            old_pool.close_gracefully()
        if old_export_pool is not None:
            old_export_pool.close_gracefully()


__all__ = ["ServingDBManager"]


@dataclass(frozen=True)
class ServingSnapshotContext:
    """Cached snapshot-scoped context used by serving operations.

    Parameters
    ----------
    pointer
        Snapshot pointer describing the active artifact paths.
    registry
        Semantic registry loaded from the snapshot artifact.
    inventory
        Schema inventory loaded from the snapshot artifact.
    buildspec
        BuildSpec contract loaded from the snapshot artifact.
    environment
        Optional environment metadata artifact (tool versions, settings).
    """

    pointer: ServingSnapshotPointer
    registry: SemanticRegistry
    inventory: SchemaInventory
    buildspec: BuildSpec
    environment: dict[str, object] | None = None

    def to_summary(self) -> Mapping[str, object]:
        """Return a compact summary for observability endpoints.

        Returns
        -------
        Mapping[str, object]
            Stable snapshot metadata for health/observability surfaces.
        """
        summary: dict[str, object] = {
            "repo": self.pointer.repo,
            "commit": self.pointer.commit,
            "run_id": self.pointer.run_id,
            "semantic_layer_version": self.pointer.semantic_layer_version,
            "semantic_registry_version": self.registry.version,
            "schema_inventory": self.inventory.summary(),
            "buildspec_version": self.buildspec.spec_version,
        }
        tools = None
        if isinstance(self.environment, dict):
            tools_obj = self.environment.get("tools")
            if isinstance(tools_obj, dict):
                tools = {str(k): str(v) for k, v in tools_obj.items()}
        if tools is not None:
            summary["tools"] = tools
        return summary


def _load_snapshot_context(pointer: ServingSnapshotPointer) -> ServingSnapshotContext:
    registry = SemanticRegistry.load(pointer.semantic_registry_path)
    inventory = SchemaInventory.load(pointer.schema_manifest_path)
    buildspec_payload = pointer.buildspec_path.read_text(encoding="utf-8")
    buildspec = buildspec_from_json(buildspec_payload)
    env_path = pointer.schema_manifest_path.parent / "environment.json"
    environment: dict[str, object] | None = None
    if env_path.is_file():
        try:
            raw = json.loads(env_path.read_text(encoding="utf-8"))
        except ValueError:
            raw = None
        if isinstance(raw, dict):
            environment = raw
    return ServingSnapshotContext(
        pointer=pointer,
        registry=registry,
        inventory=inventory,
        buildspec=buildspec,
        environment=environment,
    )
