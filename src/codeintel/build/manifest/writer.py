"""Cache manifest writer backed by pipeline step tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from codeintel.build.manifest.records import CacheManifestEntry
from codeintel.storage.tracking import ModuleKind, PipelineStepRecord

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CacheManifestWriter:
    """Persist cache events to the pipeline_steps table."""

    gateway: StorageGateway
    module: ModuleKind = "build"
    stage: str = "cache"
    strict: bool = False

    def record_hit(self, entry: CacheManifestEntry) -> None:
        """Persist a cache hit event."""
        self._record(entry)

    def record_miss(self, entry: CacheManifestEntry) -> None:
        """Persist a cache miss event."""
        self._record(entry)

    def record_store(self, entry: CacheManifestEntry) -> None:
        """Persist a cache store event."""
        self._record(entry)

    def _record(self, entry: CacheManifestEntry) -> None:
        extra = _build_extra(entry)
        if extra is None:
            extra = {}
        extra["cache_status"] = entry.status
        started_at = entry.recorded_at
        if entry.duration_ms is not None:
            started_at = entry.recorded_at - timedelta(milliseconds=entry.duration_ms)
        record = PipelineStepRecord(
            run_id=entry.run_id,
            module=self.module,
            stage=self.stage,
            name=entry.node_name,
            status="succeeded",
            started_at=started_at,
            completed_at=entry.recorded_at,
            row_counts=None,
            extra=extra,
        )
        try:
            self.gateway.runs.record_step(record)
        except Exception as exc:
            if self.strict:
                raise
            log.warning(
                "build.cache.manifest_write_failed run_id=%s node=%s error=%s",
                entry.run_id,
                entry.node_name,
                exc,
            )


def _build_extra(entry: CacheManifestEntry) -> dict[str, object] | None:
    extra: dict[str, object] = {}
    if entry.cache_key is not None:
        extra["cache_key"] = entry.cache_key
    if entry.cache_version is not None:
        extra["cache_version"] = entry.cache_version
    if entry.cache_path is not None:
        extra["cache_path"] = entry.cache_path
    if entry.duration_ms is not None:
        extra["duration_ms"] = entry.duration_ms
    if entry.size_bytes is not None:
        extra["size_bytes"] = entry.size_bytes
    if entry.target is not None:
        extra["target"] = entry.target
    if not extra:
        return None
    return extra


__all__ = ["CacheManifestWriter"]
