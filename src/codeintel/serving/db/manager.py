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
from codeintel.config.primitives import (
    BuildPaths,
    GraphBackendConfig,
    GraphFeatureFlags,
    SnapshotRef,
)
from codeintel.core.execution import ExecutionContext, new_run_context
from codeintel.core.registry import RegistryService
from codeintel.core.runtime.loader import (
    RuntimeInputs,
    build_runtime_primitives,
    load_execution_context,
)
from codeintel.core.tools import ToolBinaries
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.semantic.datasets import DatasetManifestIndex, load_dataset_manifests
from codeintel.serving.semantic.inventory import SchemaInventory
from codeintel.serving.semantic.view_registry import ViewRegistry, view_spec_modules
from codeintel.storage.gateway.pool import PoolConfig, ReadPoolWarehouse

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from codeintel.build.spec import BuildSpec
    from codeintel.serving.semantic.registry import SemanticRegistry
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
    _ready_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _summary: dict[str, object] | None = field(default=None, init=False)

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

    def current_summary(self) -> dict[str, object]:
        """Return cached snapshot summary for health/ready routes.

        Returns
        -------
        dict[str, object]
            Cached summary for the active snapshot.

        Raises
        ------
        RuntimeError
            If manager not started or summary not yet available.
        """
        if self._summary is None:
            msg = "ServingDBManager has no cached snapshot summary"
            raise RuntimeError(msg)
        return dict(self._summary)

    async def wait_ready(self, *, timeout_s: float | None = None) -> bool:
        """Wait for a snapshot pointer to be available.

        Parameters
        ----------
        timeout_s
            Optional timeout (seconds). When provided, returns False if not ready.

        Returns
        -------
        bool
            True when ready, False on timeout.
        """
        if self._pointer is not None:
            return True
        if timeout_s is None:
            await self._ready_event.wait()
            return self._pointer is not None
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout_s)
        except TimeoutError:
            return False
        return self._pointer is not None

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

        context = _load_snapshot_context(pointer, pointer_path=self.pointer_path)
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
            context = _load_snapshot_context(new_ptr, pointer_path=self.pointer_path)
            self._snapshot_cache[new_ptr.db_path] = context
            self._summary = dict(context.summary)
            self._ready_event.set()
            return

        new_pool = ReadPoolWarehouse(new_ptr.db_path, self.pool_cfg)
        new_export_pool = ReadPoolWarehouse(new_ptr.db_path, self.export_pool_cfg)
        old_pool = self._pool
        old_export_pool = self._export_pool
        context = _load_snapshot_context(new_ptr, pointer_path=self.pointer_path)
        self._pool = new_pool
        self._export_pool = new_export_pool
        self._pointer = new_ptr
        self._snapshot_cache[new_ptr.db_path] = context
        self._summary = dict(context.summary)
        self._ready_event.set()

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
    execution_context
        Unified execution context derived from the serving snapshot metadata.
    registry_service
        Canonical registry service used for semantic discovery.
    dataset_manifests
        Dataset manifest metadata for Arrow-backed tables.
    view_registry
        Polars view registry for semantic views.
    """

    pointer: ServingSnapshotPointer
    registry: SemanticRegistry
    inventory: SchemaInventory
    buildspec: BuildSpec
    environment: dict[str, object] | None = None
    execution_context: ExecutionContext | None = None
    registry_service: RegistryService | None = None
    dataset_manifests: DatasetManifestIndex = field(
        default_factory=lambda: DatasetManifestIndex({})
    )
    view_registry: ViewRegistry = field(
        default_factory=lambda: ViewRegistry.load(modules=view_spec_modules())
    )
    summary: dict[str, object] = field(default_factory=dict)

    def to_summary(self) -> Mapping[str, object]:
        """Return a compact summary for observability endpoints.

        Returns
        -------
        Mapping[str, object]
            Stable snapshot metadata for health/observability surfaces.
        """
        return self.summary


def _build_execution_context(pointer: ServingSnapshotPointer) -> ExecutionContext:
    repo_root = pointer.semantic_registry_path.parent
    snapshot = SnapshotRef.from_args(
        repo=pointer.repo,
        commit=pointer.commit,
        repo_root=repo_root,
    )
    primitives = build_runtime_primitives(
        RuntimeInputs(
            snapshot=snapshot,
            paths=BuildPaths.from_repo_root(repo_root),
            tools=ToolBinaries(),
            graph_backend=GraphBackendConfig(),
            graph_features=GraphFeatureFlags(),
            profiles=None,
        )
    )
    run_context = new_run_context(snapshot=snapshot, kind="op_prereqs", trigger="api")
    return load_execution_context(primitives=primitives, run=run_context)


def _resolve_environment_path(
    pointer: ServingSnapshotPointer,
    *,
    pointer_path: Path,
) -> Path | None:
    candidates = (
        pointer_path.parent / "environment.json",
        pointer.schema_manifest_path.parent / "environment.json",
        pointer_path.parent / "artifacts" / "environment.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_snapshot_context(
    pointer: ServingSnapshotPointer,
    *,
    pointer_path: Path,
) -> ServingSnapshotContext:
    registry_service = RegistryService.from_semantic_registry_path(pointer.semantic_registry_path)
    registry = registry_service.semantic_registry
    if registry is None:
        msg = "Semantic registry was not loaded for the serving snapshot"
        raise ValueError(msg)
    inventory = SchemaInventory.load(pointer.schema_manifest_path)
    buildspec_payload = pointer.buildspec_path.read_text(encoding="utf-8")
    buildspec = buildspec_from_json(buildspec_payload)
    dataset_manifests = load_dataset_manifests(pointer.dataset_manifest_paths)
    env_path = _resolve_environment_path(pointer, pointer_path=pointer_path)
    environment: dict[str, object] | None = None
    if env_path is not None:
        try:
            raw = json.loads(env_path.read_text(encoding="utf-8"))
        except ValueError:
            raw = None
        if isinstance(raw, dict):
            environment = raw
    summary: dict[str, object] = {
        "repo": pointer.repo,
        "commit": pointer.commit,
        "run_id": pointer.run_id,
        "semantic_layer_version": pointer.semantic_layer_version,
        "semantic_registry_version": registry.version,
        "schema_inventory": inventory.summary(),
        "buildspec_version": buildspec.spec_version,
    }
    tools = None
    if isinstance(environment, dict):
        tools_obj = environment.get("tools")
        if isinstance(tools_obj, dict):
            tools = {str(k): str(v) for k, v in tools_obj.items()}
    if tools is not None:
        summary["tools"] = tools
    return ServingSnapshotContext(
        pointer=pointer,
        registry=registry,
        inventory=inventory,
        buildspec=buildspec,
        environment=environment,
        execution_context=_build_execution_context(pointer),
        registry_service=registry_service,
        dataset_manifests=dataset_manifests,
        view_registry=ViewRegistry.load(modules=view_spec_modules()),
        summary=summary,
    )
