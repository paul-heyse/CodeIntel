"""Cache adapter integration for Hamilton builds."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import Field, dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeGuard, cast

import pyarrow as pa
from hamilton.caching.adapter import (
    CachingBehavior,
    CachingEventType,
    HamiltonCacheAdapter,
    NodeRoleInTaskExecution,
)
from hamilton.caching.stores.base import ResultStore
from hamilton.caching.stores.file import FileResultStore

from codeintel.build.hamilton.cache_index import CacheIndex, CacheProbeResult
from codeintel.build.manifest.records import CacheManifestEntry
from codeintel.build.manifest.writer import CacheManifestWriter
from codeintel.build.tabular.conversion import reader_to_table, table_to_reader
from codeintel.core.hamilton import tags as ht
from codeintel.core.telemetry.hooks.cache_events import CacheEventMetrics

if TYPE_CHECKING:
    from hamilton.caching.stores.base import DataLoader, DataSaver, MetadataStore
    from hamilton.node import Node

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CacheAdapterOptions:
    """Configuration for Hamilton cache adapter defaults."""

    cache_store: CacheStore | None = None
    metadata_store: MetadataStore | None = None
    result_store: ResultStore | None = None
    default_behavior: Literal["default", "recompute", "disable", "ignore"] = "default"
    default_loader_behavior: Literal["default", "recompute", "disable", "ignore"] = "default"
    default_saver_behavior: Literal["default", "recompute", "disable", "ignore"] = "default"
    log_to_file: bool = False


@dataclass(frozen=True, slots=True)
class ArrowCachedResult:
    """Cached Arrow data wrapper that preserves the original return kind."""

    kind: Literal["table", "reader"]
    table: pa.Table


class ArrowFileResultStore(ResultStore):
    """ResultStore that materializes Arrow streams for file-backed caching."""

    def __init__(self, path: str | Path, *, create_dir: bool = True) -> None:
        self._store: FileResultStore = FileResultStore(
            path=str(path),
            create_dir=create_dir,
        )
        self.path: Path = self._store.path

    def set(
        self,
        data_version: str,
        result: object,
        **kwargs: object,
    ) -> None:
        """Store a cached result, materializing Arrow readers."""
        normalized = _normalize_arrow_cached_result(result)
        saver_cls = cast("DataSaver | None", kwargs.get("saver_cls"))
        loader_cls = cast("DataLoader | None", kwargs.get("loader_cls"))
        FileResultStore.set(
            self._store,
            data_version=data_version,
            result=normalized,
            saver_cls=saver_cls,
            loader_cls=loader_cls,
        )

    def get(self, data_version: str, **_kwargs: object) -> object | None:
        """Load a cached result, restoring Arrow readers when needed.

        Returns
        -------
        object | None
            Cached result when present.
        """
        stored = FileResultStore.get(self._store, data_version=data_version)
        return _resolve_arrow_cached_result(stored)

    def delete(self, data_version: str) -> None:
        """Delete a cached result by data version."""
        FileResultStore.delete(self._store, data_version)

    def delete_all(self) -> None:
        """Delete all cached results in the store."""
        FileResultStore.delete_all(self._store)

    def exists(self, data_version: str) -> bool:
        """Return True when a cached result exists for data_version.

        Returns
        -------
        bool
            True when the cache entry is present.
        """
        return FileResultStore.exists(self._store, data_version)


@dataclass(frozen=True, slots=True)
class _CacheEvent:
    run_id: str
    node_name: str
    actor: Literal["adapter", "metadata_store", "result_store"]
    event_type: CachingEventType
    msg: str | None
    value: object | None
    task_id: str | None


@dataclass(frozen=True, slots=True)
class CacheStore(CacheIndex):
    """Shared cache store that supports read probes for planning."""

    metadata_store: MetadataStore
    result_store: ResultStore | None = None

    def has(self, *, node: str, version: str) -> bool:
        """Return True if the cache entry exists and data is available.

        Returns
        -------
        bool
            True when metadata (and result data, if configured) is present.
        """
        _ = node
        if not version:
            return False
        if not self.metadata_store.exists(version):
            return False
        data_version = self.metadata_store.get(version)
        if data_version is None:
            return False
        if self.result_store is None:
            return True
        return self.result_store.exists(data_version)

    def batch_has(self, pairs: Iterable[tuple[str, str]]) -> tuple[CacheProbeResult, ...]:
        """Batch-check cache hits for node/version pairs.

        Returns
        -------
        tuple[CacheProbeResult, ...]
            Probe results for each input pair.
        """
        results: list[CacheProbeResult] = []
        for node, version in pairs:
            results.append(
                CacheProbeResult(
                    node=node,
                    version=version,
                    hit=self.has(node=node, version=version),
                )
            )
        return tuple(results)

    def get_data_version(self, cache_key: str) -> str | None:
        """Return the data version for a cache key, if present.

        Returns
        -------
        str | None
            Data version string if available.
        """
        if not cache_key:
            return None
        return self.metadata_store.get(cache_key)


_ARROW_CACHE_BEHAVIORS = {
    CachingBehavior.DEFAULT,
    CachingBehavior.RECOMPUTE,
    CachingBehavior.IGNORE,
}


class ManifestBackedCacheAdapter(HamiltonCacheAdapter):
    """Hamilton cache adapter that emits cache events to the manifest writer.

    Manifest output is audit-only and must not drive control flow.
    """

    def __init__(
        self,
        *,
        path: str | Path,
        manifest_writer: CacheManifestWriter | None = None,
        manifest_run_id: str | None = None,
        strict_manifest: bool = False,
        options: CacheAdapterOptions | None = None,
    ) -> None:
        resolved = options or CacheAdapterOptions()
        cache_store = resolved.cache_store
        metadata_store = (
            cache_store.metadata_store if cache_store is not None else resolved.metadata_store
        )
        result_store = (
            cache_store.result_store if cache_store is not None else resolved.result_store
        )
        super().__init__(
            path=str(path),
            metadata_store=metadata_store,
            result_store=result_store,
            default_behavior=resolved.default_behavior,
            default_loader_behavior=resolved.default_loader_behavior,
            default_saver_behavior=resolved.default_saver_behavior,
            log_to_file=resolved.log_to_file,
        )
        self._manifest_writer = manifest_writer
        self._manifest_run_id = manifest_run_id
        self._strict_manifest = strict_manifest
        self._metrics = CacheEventMetrics()
        self.cache_store = cache_store or CacheStore(
            metadata_store=self.metadata_store,
            result_store=self.result_store,
        )
        self._arrow_cache_tables: dict[tuple[str, str, str | None], pa.Table] = {}

    def do_node_execute(
        self,
        *,
        run_id: str,
        node_: Node,
        kwargs: dict[str, object],
        task_id: str | None = None,
        **future_kwargs: object,
    ) -> object:
        """Execute a node and materialize Arrow readers for caching.

        Returns
        -------
        object
            Node output with Arrow readers normalized for caching.
        """
        result = super().do_node_execute(
            run_id=run_id,
            node_=node_,
            kwargs=kwargs,
            task_id=task_id,
            **future_kwargs,
        )
        if not isinstance(result, pa.RecordBatchReader):
            return result
        node_name = getattr(node_, "name", None)
        if not isinstance(node_name, str):
            return result
        if not self._should_materialize_arrow_result(run_id=run_id, node_name=node_name):
            return result
        table = reader_to_table(result)
        self._arrow_cache_tables[_arrow_cache_key(run_id, node_name, task_id)] = table
        return table_to_reader(table)

    def post_node_execute(self, **kwargs: object) -> None:
        """Persist cache results after node execution."""
        run_id = cast("str", kwargs["run_id"])
        node_ = cast("Node", kwargs["node_"])
        result = kwargs.get("result")
        success = cast("bool", kwargs.get("success", True))
        error = cast("Exception | None", kwargs.get("error"))
        task_id = cast("str | None", kwargs.get("task_id"))
        future_kwargs = _future_kwargs(kwargs)
        node_name = getattr(node_, "name", None)
        table: pa.Table | None = None
        if isinstance(node_name, str):
            cache_key = _arrow_cache_key(run_id, node_name, task_id)
            table = self._arrow_cache_tables.pop(cache_key, None)
        if table is not None and isinstance(result, pa.RecordBatchReader):
            result = ArrowCachedResult(kind="reader", table=table)
        super().post_node_execute(
            run_id=run_id,
            node_=node_,
            result=cast("str | None", result),
            success=success,
            error=error,
            task_id=task_id,
            **future_kwargs,
        )

    def resolve_behaviors(self, run_id: str) -> dict[str, CachingBehavior]:
        """Resolve caching behaviors for a run, overriding materialization nodes.

        Returns
        -------
        dict[str, CachingBehavior]
            Resolved caching behavior per node name.
        """
        behaviors = super().resolve_behaviors(run_id)
        graph = self._fn_graphs.get(run_id)
        if graph is None:
            return behaviors
        for node in graph.get_nodes():
            tags = node.tags if isinstance(node.tags, dict) else {}
            if tags.get(ht.TAG_NODE_TYPE) == ht.NODE_TYPE_MATERIALIZE:
                behaviors[node.name] = CachingBehavior.RECOMPUTE
        return behaviors

    def _should_materialize_arrow_result(self, *, run_id: str, node_name: str) -> bool:
        behavior = self.behaviors.get(run_id, {}).get(node_name)
        return behavior in _ARROW_CACHE_BEHAVIORS

    def _log_event(self, *args: object, **kwargs: object) -> None:
        event = _parse_cache_event(args, kwargs)
        if event is None:
            return
        super()._log_event(
            run_id=event.run_id,
            node_name=event.node_name,
            actor=event.actor,
            event_type=event.event_type,
            msg=event.msg,
            value=event.value,
            task_id=event.task_id,
        )
        if self._manifest_writer is None:
            return
        if event.event_type == CachingEventType.GET_RESULT and event.msg == "hit":
            self._emit_cache_event(
                run_id=event.run_id,
                node_name=event.node_name,
                status="hit",
                cache_version=_coerce_str(event.value),
                task_id=event.task_id,
            )
            return
        if event.event_type == CachingEventType.EXECUTE_NODE:
            self._emit_cache_event(
                run_id=event.run_id,
                node_name=event.node_name,
                status="miss",
                cache_version=None,
                task_id=event.task_id,
            )
            return
        if event.event_type == CachingEventType.SET_RESULT:
            self._emit_cache_event(
                run_id=event.run_id,
                node_name=event.node_name,
                status="store",
                cache_version=_coerce_str(event.value),
                task_id=event.task_id,
            )

    def _emit_cache_event(
        self,
        *,
        run_id: str,
        node_name: str,
        status: Literal["hit", "miss", "store"],
        cache_version: str | None,
        task_id: str | None,
    ) -> None:
        behavior = self.behaviors.get(run_id, {}).get(node_name)
        if behavior in {CachingBehavior.DISABLE, CachingBehavior.IGNORE}:
            return
        cache_key = self._peek_cache_key(run_id, node_name, task_id)
        cache_path, size_bytes = self._cache_artifact(cache_version)
        target = _target_for_node(self._fn_graphs.get(run_id), node_name)
        manifest_run_id = self._manifest_run_id or run_id
        entry = CacheManifestEntry(
            run_id=manifest_run_id,
            node_name=node_name,
            status=status,
            recorded_at=datetime.now(tz=UTC),
            cache_key=cache_key,
            cache_version=cache_version,
            cache_path=cache_path,
            size_bytes=size_bytes,
            target=target,
        )
        manifest_writer = self._manifest_writer
        if manifest_writer is None:
            return
        try:
            if status == "hit":
                manifest_writer.record_hit(entry)
                self._metrics.record_hit()
            elif status == "miss":
                manifest_writer.record_miss(entry)
                self._metrics.record_miss()
            else:
                manifest_writer.record_store(entry)
                self._metrics.record_store()
        except Exception as exc:
            if self._strict_manifest:
                raise
            log.warning(
                "build.cache.manifest_event_failed run_id=%s node=%s status=%s error=%s",
                run_id,
                node_name,
                status,
                exc,
            )

    def _peek_cache_key(
        self,
        run_id: str,
        node_name: str,
        task_id: str | None,
    ) -> str | None:
        node_role = self._get_node_role(run_id=run_id, node_name=node_name, task_id=task_id)
        cache_keys = self.cache_keys.get(run_id, {})
        if node_role == NodeRoleInTaskExecution.INSIDE:
            nested = cache_keys.get(node_name, {})
            if isinstance(nested, dict):
                if task_id is None:
                    return None
                return nested.get(task_id)
            return None
        value = cache_keys.get(node_name)
        return value if isinstance(value, str) else None

    def _cache_artifact(self, cache_version: str | None) -> tuple[str | None, int | None]:
        if cache_version is None:
            return None, None
        result_store = self.result_store
        if result_store is None:
            return None, None
        path_root = getattr(result_store, "path", None)
        if path_root is None:
            return None, None
        root = path_root if isinstance(path_root, Path) else Path(str(path_root))
        path = root / cache_version
        if not path.exists():
            return str(path), None
        try:
            return str(path), path.stat().st_size
        except OSError:
            return str(path), None


def _coerce_str(value: object | None) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _parse_cache_event(
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> _CacheEvent | None:
    param_names = (
        "run_id",
        "node_name",
        "actor",
        "event_type",
        "msg",
        "value",
        "task_id",
    )
    values: dict[str, object | None] = dict.fromkeys(param_names, None)
    values.update(dict(zip(param_names, args, strict=False)))
    values.update({name: value for name, value in kwargs.items() if name in values})
    run_id = values["run_id"]
    node_name = values["node_name"]
    actor = values["actor"]
    event_type = values["event_type"]
    if not isinstance(run_id, str) or not isinstance(node_name, str):
        return None
    if not isinstance(actor, str):
        return None
    if not isinstance(event_type, CachingEventType):
        return None
    if actor == "adapter":
        actor_literal: Literal["adapter", "metadata_store", "result_store"] = "adapter"
    elif actor == "metadata_store":
        actor_literal = "metadata_store"
    elif actor == "result_store":
        actor_literal = "result_store"
    else:
        return None
    msg = values["msg"] if isinstance(values["msg"], str) else None
    task_id = values["task_id"] if isinstance(values["task_id"], str) else None
    return _CacheEvent(
        run_id=run_id,
        node_name=node_name,
        actor=actor_literal,
        event_type=event_type,
        msg=msg,
        value=values["value"],
        task_id=task_id,
    )


def _target_for_node(fn_graph: object | None, node_name: str) -> str | None:
    if fn_graph is None:
        return None
    nodes = getattr(fn_graph, "nodes", None)
    if not isinstance(nodes, dict):
        return None
    node = nodes.get(node_name)
    if node is None:
        return None
    tags = node.tags if isinstance(node.tags, dict) else None
    target = tags.get(ht.TAG_TARGET) if tags else None
    return target if isinstance(target, str) else None


def _future_kwargs(values: Mapping[str, object]) -> dict[str, object]:
    ignored = {"run_id", "node_", "result", "success", "error", "task_id"}
    return {key: value for key, value in values.items() if key not in ignored}


def _arrow_cache_key(
    run_id: str,
    node_name: str,
    task_id: str | None,
) -> tuple[str, str, str | None]:
    return run_id, node_name, task_id


class _DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


def _is_dataclass_instance(value: object) -> TypeGuard[_DataclassInstance]:
    return is_dataclass(value) and not isinstance(value, type)


def _normalize_arrow_cached_result(result: object) -> object:
    normalized = result
    if not isinstance(result, ArrowCachedResult):
        if isinstance(result, pa.RecordBatchReader):
            normalized = ArrowCachedResult(kind="reader", table=reader_to_table(result))
        elif isinstance(result, pa.Table):
            normalized = ArrowCachedResult(kind="table", table=result)
        elif _is_dataclass_instance(result):
            replaced = _normalize_dataclass(result)
            if replaced is not None:
                normalized = replaced
        else:
            collection = _normalize_collection(result)
            if collection is not None:
                normalized = collection
    return normalized


def _normalize_dataclass[T: _DataclassInstance](result: T) -> T | None:
    updates: dict[str, object] = {}
    for field_info in fields(result):
        value = getattr(result, field_info.name)
        normalized = _normalize_arrow_cached_result(value)
        if normalized is not value:
            updates[field_info.name] = normalized
    if not updates:
        return None
    return replace(result, **updates)


def _resolve_dataclass[T: _DataclassInstance](result: T) -> T | None:
    updates: dict[str, object] = {}
    for field_info in fields(result):
        value = getattr(result, field_info.name)
        resolved = _resolve_arrow_cached_result(value)
        if resolved is not value:
            updates[field_info.name] = resolved
    if not updates:
        return None
    return replace(result, **updates)


def _resolve_arrow_cached_result(result: object | None) -> object | None:
    if isinstance(result, ArrowCachedResult):
        if result.kind == "reader":
            return table_to_reader(result.table)
        if result.kind == "table":
            return result.table
        msg = f"Unsupported cached Arrow result kind: {result.kind}"
        raise ValueError(msg)

    resolved = result
    if _is_dataclass_instance(result):
        replaced = _resolve_dataclass(result)
        if replaced is not None:
            resolved = replaced
    else:
        collection = _resolve_collection(result)
        if collection is not None:
            resolved = collection
    return resolved


def _normalize_collection(result: object) -> object | None:
    if isinstance(result, dict):
        return {key: _normalize_arrow_cached_result(value) for key, value in result.items()}
    if isinstance(result, list):
        return [_normalize_arrow_cached_result(value) for value in result]
    if isinstance(result, tuple):
        return tuple(_normalize_arrow_cached_result(value) for value in result)
    return None


def _resolve_collection(result: object | None) -> object | None:
    if isinstance(result, dict):
        return {key: _resolve_arrow_cached_result(value) for key, value in result.items()}
    if isinstance(result, list):
        return [_resolve_arrow_cached_result(value) for value in result]
    if isinstance(result, tuple):
        return tuple(_resolve_arrow_cached_result(value) for value in result)
    return None


__all__ = [
    "ArrowCachedResult",
    "ArrowFileResultStore",
    "CacheAdapterOptions",
    "CacheStore",
    "ManifestBackedCacheAdapter",
]
