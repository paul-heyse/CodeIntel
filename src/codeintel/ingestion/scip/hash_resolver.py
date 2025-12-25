"""Hash resolver utilities for incremental SCIP indexing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.ingestion.adapters.hash_change_detection import HashChangeDetectionAdapter

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.ingestion.ports.change_detection import FileDigest
    from codeintel.ingestion.ports.discovery import ModuleRecord


@dataclass(frozen=True)
class HashSourceSummary:
    """Summary of hash source usage during planning."""

    hash_source: str | None
    hash_reused: int
    hash_computed: int
    breakdown: str | None


class FileDigestResolver:
    """Resolve file digests from multiple sources with provenance tracking."""

    def __init__(
        self,
        *,
        file_state_by_path: Mapping[str, FileDigest] | None = None,
        module_state_by_path: Mapping[str, FileDigest] | None = None,
    ) -> None:
        self._file_state_by_path = file_state_by_path or {}
        self._module_state_by_path = module_state_by_path or {}
        self._counts = {
            "file_state": 0,
            "module_state": 0,
            "computed": 0,
        }

    def resolve(self, module: ModuleRecord) -> FileDigest | None:
        """Resolve a digest for a module from configured sources.

        Returns
        -------
        FileDigest | None
            Resolved digest or None when unavailable.
        """
        digest = self._file_state_by_path.get(module.rel_path)
        if digest is not None:
            self._counts["file_state"] += 1
            return digest
        digest = self._module_state_by_path.get(module.rel_path)
        if digest is not None:
            self._counts["module_state"] += 1
            return digest
        digest = HashChangeDetectionAdapter.compute_file_digest(module.file_path)
        if digest is None:
            return None
        self._counts["computed"] += 1
        return digest

    def summary(self) -> HashSourceSummary:
        """Summarize hash source usage.

        Returns
        -------
        HashSourceSummary
            Summary of hash provenance counts.
        """
        used_sources = [name for name, count in self._counts.items() if count > 0]
        if not used_sources:
            source = None
        elif len(used_sources) == 1:
            source = used_sources[0]
        else:
            source = "mixed"
        reused = self._counts["file_state"] + self._counts["module_state"]
        computed = self._counts["computed"]
        breakdown = _format_breakdown(self._counts)
        return HashSourceSummary(
            hash_source=source,
            hash_reused=reused,
            hash_computed=computed,
            breakdown=breakdown,
        )


def _format_breakdown(counts: Mapping[str, int]) -> str | None:
    parts = [
        f"{name}={counts[name]}"
        for name in ("file_state", "module_state", "computed")
        if counts[name] > 0
    ]
    return ",".join(parts) if parts else None


__all__ = [
    "FileDigestResolver",
    "HashSourceSummary",
]
