"""Cache manifest reader for pipeline step tracking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING

from codeintel.build.manifest.records import CacheEventStatus, CacheManifestEntry
from codeintel.storage.tracking import StepStatus

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True, slots=True)
class CacheManifestReader:
    """Read cache events from pipeline step tracking."""

    gateway: StorageGateway
    module: str = "build"
    stage: str = "cache"

    def fetch(self, run_id: str) -> tuple[CacheManifestEntry, ...]:
        """Fetch cache events for a run.

        Parameters
        ----------
        run_id
            Run identifier to load.

        Returns
        -------
        tuple[CacheManifestEntry, ...]
            Cache manifest entries for the run.
        """
        steps = self.gateway.runs.fetch_steps(run_id)
        entries: list[CacheManifestEntry] = []
        for step in steps:
            if step.module != self.module or step.stage != self.stage:
                continue
            extra = dict(step.extra) if step.extra else {}
            recorded_at = step.completed_at or step.started_at
            if recorded_at.tzinfo is None:
                recorded_at = recorded_at.replace(tzinfo=UTC)
            cache_status = _coerce_cache_status(extra.get("cache_status"))
            if cache_status is None:
                cache_status = _status_from_step(step.status)
            entries.append(
                CacheManifestEntry(
                    run_id=step.run_id,
                    node_name=step.name,
                    status=cache_status,
                    recorded_at=recorded_at,
                    cache_key=_coerce_str(extra.get("cache_key")),
                    cache_version=_coerce_str(extra.get("cache_version")),
                    cache_path=_coerce_str(extra.get("cache_path")),
                    duration_ms=_coerce_float(extra.get("duration_ms")),
                    size_bytes=_coerce_int(extra.get("size_bytes")),
                    target=_coerce_str(extra.get("target")),
                )
            )
        return tuple(entries)


def _coerce_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _coerce_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _coerce_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _coerce_cache_status(value: object) -> CacheEventStatus | None:
    if value == "hit":
        return "hit"
    if value == "miss":
        return "miss"
    if value == "store":
        return "store"
    return None


def _status_from_step(status: StepStatus) -> CacheEventStatus:
    if status == "skipped":
        return "hit"
    if status == "succeeded":
        return "store"
    return "miss"


__all__ = ["CacheManifestReader"]
