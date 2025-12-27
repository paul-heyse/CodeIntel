"""Cache adapter integration for Hamilton builds."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from hamilton.caching.adapter import (
    CachingBehavior,
    CachingEventType,
    HamiltonCacheAdapter,
    NodeRoleInTaskExecution,
)
from hamilton.caching.stores.file import FileResultStore

from codeintel.build.manifest.records import CacheManifestEntry
from codeintel.build.manifest.writer import CacheManifestWriter
from codeintel.core.hamilton import tags as ht
from codeintel.core.telemetry.hooks.cache_events import CacheEventMetrics

if TYPE_CHECKING:
    from hamilton.caching.stores.base import MetadataStore, ResultStore

log = logging.getLogger(__name__)


class ManifestBackedCacheAdapter(HamiltonCacheAdapter):
    """Hamilton cache adapter that emits cache events to the manifest writer."""

    def __init__(
        self,
        *,
        path: str | Path,
        manifest_writer: CacheManifestWriter | None = None,
        manifest_run_id: str | None = None,
        strict_manifest: bool = False,
        metadata_store: MetadataStore | None = None,
        result_store: ResultStore | None = None,
        default_behavior: Literal["default", "recompute", "disable", "ignore"] = "default",
        default_loader_behavior: Literal["default", "recompute", "disable", "ignore"] = "default",
        default_saver_behavior: Literal["default", "recompute", "disable", "ignore"] = "default",
        log_to_file: bool = False,
    ) -> None:
        super().__init__(
            path=str(path),
            metadata_store=metadata_store,
            result_store=result_store,
            default_behavior=default_behavior,
            default_loader_behavior=default_loader_behavior,
            default_saver_behavior=default_saver_behavior,
            log_to_file=log_to_file,
        )
        self._manifest_writer = manifest_writer
        self._manifest_run_id = manifest_run_id
        self._strict_manifest = strict_manifest
        self._metrics = CacheEventMetrics()

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

    def _log_event(
        self,
        run_id: str,
        node_name: str,
        actor: Literal["adapter", "metadata_store", "result_store"],
        event_type: CachingEventType,
        msg: str | None = None,
        value: object | None = None,
        task_id: str | None = None,
    ) -> None:
        super()._log_event(
            run_id=run_id,
            node_name=node_name,
            actor=actor,
            event_type=event_type,
            msg=msg,
            value=value,
            task_id=task_id,
        )
        if self._manifest_writer is None:
            return
        if event_type == CachingEventType.GET_RESULT and msg == "hit":
            self._emit_cache_event(
                run_id=run_id,
                node_name=node_name,
                status="hit",
                cache_version=_coerce_str(value),
                task_id=task_id,
            )
            return
        if event_type == CachingEventType.EXECUTE_NODE:
            self._emit_cache_event(
                run_id=run_id,
                node_name=node_name,
                status="miss",
                cache_version=None,
                task_id=task_id,
            )
            return
        if event_type == CachingEventType.SET_RESULT:
            self._emit_cache_event(
                run_id=run_id,
                node_name=node_name,
                status="store",
                cache_version=_coerce_str(value),
                task_id=task_id,
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
        try:
            if status == "hit":
                self._manifest_writer.record_hit(entry)
                self._metrics.record_hit()
            elif status == "miss":
                self._manifest_writer.record_miss(entry)
                self._metrics.record_miss()
            else:
                self._manifest_writer.record_store(entry)
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
                return nested.get(task_id)
            return None
        value = cache_keys.get(node_name)
        return value if isinstance(value, str) else None

    def _cache_artifact(self, cache_version: str | None) -> tuple[str | None, int | None]:
        if cache_version is None:
            return None, None
        if not isinstance(self.result_store, FileResultStore):
            return None, None
        path = self.result_store._path_from_data_version(cache_version)
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


__all__ = ["ManifestBackedCacheAdapter"]
