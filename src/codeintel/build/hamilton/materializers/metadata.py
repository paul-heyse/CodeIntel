"""Canonical materialization metadata schema for Hamilton DataSavers.

Hamilton's DataSaver API returns untyped metadata dicts. This module provides the single typed
schema for those dicts, so producers (savers) and consumers (record builders) cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

MaterializationStatus = Literal["succeeded", "skipped", "failed"]

_MATERIALIZATION_STATUS: dict[str, MaterializationStatus] = {
    "failed": "failed",
    "skipped": "skipped",
    "succeeded": "succeeded",
}

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class DuckDBMaterializationMetadata:
    """Canonical metadata for a single DuckDB table materialization."""

    status: MaterializationStatus
    table_key: str
    row_count: int | None
    duration_ms: float
    input_hash: str
    error: str | None

    def to_dict(self) -> dict[str, object]:
        """Convert this metadata to the untyped dict shape expected by Hamilton.

        Returns
        -------
        dict[str, object]
            Dictionary representation of this metadata.
        """
        return {
            "status": self.status,
            "table_key": self.table_key,
            "row_count": self.row_count,
            "duration_ms": self.duration_ms,
            "input_hash": self.input_hash,
            "error": self.error,
        }

    @classmethod
    def from_mapping(
        cls,
        materialization: Mapping[str, object],
        *,
        default_table_key: str,
    ) -> DuckDBMaterializationMetadata:
        """Parse metadata from a mapping produced by a DataSaver.

        Parameters
        ----------
        materialization
            Mapping of materialization metadata fields.
        default_table_key
            Table key to use when the mapping does not provide one.

        Returns
        -------
        DuckDBMaterializationMetadata
            Parsed and normalized materialization metadata.
        """
        status_raw = materialization.get("status")
        if isinstance(status_raw, str) and status_raw in _MATERIALIZATION_STATUS:
            status = _MATERIALIZATION_STATUS[status_raw]
        else:
            status = "failed"

        table_key_raw = materialization.get("table_key")
        if isinstance(table_key_raw, str) and table_key_raw:
            table_key = table_key_raw
        else:
            table_key = default_table_key

        row_count_raw = materialization.get("row_count")
        row_count = row_count_raw if isinstance(row_count_raw, int) else None

        duration_raw = materialization.get("duration_ms")
        duration_ms = float(duration_raw) if isinstance(duration_raw, (int, float)) else 0.0

        input_hash_raw = materialization.get("input_hash")
        input_hash = input_hash_raw if isinstance(input_hash_raw, str) else ""

        error_raw = materialization.get("error")
        error = error_raw if isinstance(error_raw, str) else None

        return cls(
            status=status,
            table_key=table_key,
            row_count=row_count,
            duration_ms=duration_ms,
            input_hash=input_hash,
            error=error,
        )


@dataclass(frozen=True, slots=True)
class FileArtifactMaterializationMetadata:
    """Canonical metadata for a single file artifact materialization."""

    status: MaterializationStatus
    artifact_name: str
    path: str | None
    size_bytes: int | None
    duration_ms: float
    input_hash: str
    error: str | None

    def to_dict(self) -> dict[str, object]:
        """Convert this metadata to the untyped dict shape expected by Hamilton.

        Returns
        -------
        dict[str, object]
            Dictionary representation of this metadata.
        """
        return {
            "status": self.status,
            "artifact_name": self.artifact_name,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "duration_ms": self.duration_ms,
            "input_hash": self.input_hash,
            "error": self.error,
        }

    @classmethod
    def from_mapping(
        cls,
        materialization: Mapping[str, object],
        *,
        default_artifact_name: str,
    ) -> FileArtifactMaterializationMetadata:
        """Parse metadata from a mapping produced by a DataSaver.

        Parameters
        ----------
        materialization
            Mapping of materialization metadata fields.
        default_artifact_name
            Artifact name to use when the mapping does not provide one.

        Returns
        -------
        FileArtifactMaterializationMetadata
            Parsed and normalized materialization metadata.
        """
        status_raw = materialization.get("status")
        if isinstance(status_raw, str) and status_raw in _MATERIALIZATION_STATUS:
            status = _MATERIALIZATION_STATUS[status_raw]
        else:
            status = "failed"

        artifact_name_raw = materialization.get("artifact_name")
        artifact_name = (
            artifact_name_raw
            if isinstance(artifact_name_raw, str) and artifact_name_raw
            else default_artifact_name
        )

        path_raw = materialization.get("path")
        path = path_raw if isinstance(path_raw, str) else None

        size_bytes_raw = materialization.get("size_bytes")
        size_bytes = size_bytes_raw if isinstance(size_bytes_raw, int) else None

        duration_raw = materialization.get("duration_ms")
        duration_ms = float(duration_raw) if isinstance(duration_raw, (int, float)) else 0.0

        input_hash_raw = materialization.get("input_hash")
        input_hash = input_hash_raw if isinstance(input_hash_raw, str) else ""

        error_raw = materialization.get("error")
        error = error_raw if isinstance(error_raw, str) else None

        return cls(
            status=status,
            artifact_name=artifact_name,
            path=path,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            input_hash=input_hash,
            error=error,
        )


__all__ = [
    "DuckDBMaterializationMetadata",
    "FileArtifactMaterializationMetadata",
    "MaterializationStatus",
]
