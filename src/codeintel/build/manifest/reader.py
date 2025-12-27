"""Cache manifest reader for pipeline step tracking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING

from codeintel.build.manifest.records import CacheManifestEntry

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
            entries.append(
                CacheManifestEntry(
                    run_id=step.run_id,
                    node_name=step.name,
                    status=step.status,
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


__all__ = ["CacheManifestReader"]
