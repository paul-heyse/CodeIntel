"""Unified materialization results for CodeIntel pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

MaterializationStatus = Literal["succeeded", "skipped", "failed"]

_STATUS_MAP: dict[str, MaterializationStatus] = {
    "failed": "failed",
    "skipped": "skipped",
    "succeeded": "succeeded",
}


def _coerce_status(value: object) -> MaterializationStatus:
    if isinstance(value, str) and value in _STATUS_MAP:
        return _STATUS_MAP[value]
    return "failed"


def _coerce_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _coerce_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _coerce_duration(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    """Unified metadata for table and artifact materializations.

    Attributes
    ----------
    status
        Materialization status for the output.
    table_key
        Fully-qualified table key for table writes.
    row_count
        Rows written for table outputs when available.
    artifact_name
        Artifact name for file outputs.
    path
        Resolved artifact path for file outputs.
    iceberg_snapshot_id
        Iceberg snapshot identifier for table outputs when available.
    validation_id
        Validation identifier persisted for table outputs when available.
    validation_status
        Validation status for table outputs when available.
    size_bytes
        Size of the artifact payload when available.
    duration_ms
        Duration of the materialization step in milliseconds.
    input_hash
        Input hash used for manifest-based incremental decisions.
    error
        Error message when the materialization fails.
    """

    status: MaterializationStatus
    table_key: str | None = None
    row_count: int | None = None
    artifact_name: str | None = None
    path: str | None = None
    iceberg_snapshot_id: int | None = None
    validation_id: str | None = None
    validation_status: str | None = None
    size_bytes: int | None = None
    duration_ms: float = 0.0
    input_hash: str = ""
    error: str | None = None

    def to_mapping(self) -> dict[str, object]:
        """Convert the result to a metadata mapping.

        Returns
        -------
        dict[str, object]
            Dictionary representation of the materialization result.
        """
        return {
            "status": self.status,
            "table_key": self.table_key,
            "row_count": self.row_count,
            "artifact_name": self.artifact_name,
            "path": self.path,
            "iceberg_snapshot_id": self.iceberg_snapshot_id,
            "validation_id": self.validation_id,
            "validation_status": self.validation_status,
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
        default_table_key: str | None = None,
        default_artifact_name: str | None = None,
    ) -> MaterializationResult:
        """Parse a materialization result from a metadata mapping.

        Parameters
        ----------
        materialization
            Mapping of materialization results fields.
        default_table_key
            Table key to use when the mapping does not provide one.
        default_artifact_name
            Artifact name to use when the mapping does not provide one.

        Returns
        -------
        MaterializationResult
            Parsed materialization result.
        """
        status = _coerce_status(materialization.get("status"))
        table_key = _coerce_str(materialization.get("table_key")) or default_table_key
        artifact_name = _coerce_str(materialization.get("artifact_name")) or default_artifact_name
        input_hash = _coerce_str(materialization.get("input_hash")) or ""

        return cls(
            status=status,
            table_key=table_key,
            row_count=_coerce_int(materialization.get("row_count")),
            artifact_name=artifact_name,
            path=_coerce_str(materialization.get("path")),
            iceberg_snapshot_id=_coerce_int(materialization.get("iceberg_snapshot_id")),
            validation_id=_coerce_str(materialization.get("validation_id")),
            validation_status=_coerce_str(materialization.get("validation_status")),
            size_bytes=_coerce_int(materialization.get("size_bytes")),
            duration_ms=_coerce_duration(materialization.get("duration_ms")),
            input_hash=input_hash,
            error=_coerce_str(materialization.get("error")),
        )


@dataclass(frozen=True, slots=True)
class TableMaterializationMetadata:
    """Optional metadata for table materialization results."""

    iceberg_snapshot_id: int | None = None
    validation_id: str | None = None
    validation_status: str | None = None


def failed_artifact_result(
    *, artifact_name: str, duration_ms: float, input_hash: str, error: str
) -> MaterializationResult:
    """Build a failed artifact result.

    Returns
    -------
    MaterializationResult
        Failed artifact materialization.
    """
    return MaterializationResult(
        status="failed",
        artifact_name=artifact_name,
        path=None,
        size_bytes=None,
        duration_ms=duration_ms,
        input_hash=input_hash,
        error=error,
    )


def skipped_artifact_result(
    *, artifact_name: str, duration_ms: float, input_hash: str, path: str | None
) -> MaterializationResult:
    """Build a skipped artifact result.

    Returns
    -------
    MaterializationResult
        Skipped artifact materialization.
    """
    return MaterializationResult(
        status="skipped",
        artifact_name=artifact_name,
        path=path,
        size_bytes=None,
        duration_ms=duration_ms,
        input_hash=input_hash,
        error=None,
    )


def succeeded_artifact_result(
    *,
    artifact_name: str,
    duration_ms: float,
    input_hash: str,
    path: str,
    size_bytes: int,
) -> MaterializationResult:
    """Build a succeeded artifact result.

    Returns
    -------
    MaterializationResult
        Succeeded artifact materialization.
    """
    return MaterializationResult(
        status="succeeded",
        artifact_name=artifact_name,
        path=path,
        size_bytes=size_bytes,
        duration_ms=duration_ms,
        input_hash=input_hash,
        error=None,
    )


def failed_table_result(
    *,
    table_key: str,
    duration_ms: float,
    input_hash: str,
    error: str,
    metadata: TableMaterializationMetadata | None = None,
) -> MaterializationResult:
    """Build a failed table result.

    Returns
    -------
    MaterializationResult
        Failed table materialization.
    """
    return MaterializationResult(
        status="failed",
        table_key=table_key,
        row_count=None,
        duration_ms=duration_ms,
        input_hash=input_hash,
        error=error,
        iceberg_snapshot_id=metadata.iceberg_snapshot_id if metadata else None,
        validation_id=metadata.validation_id if metadata else None,
        validation_status=metadata.validation_status if metadata else None,
    )


def skipped_table_result(
    *,
    table_key: str,
    duration_ms: float,
    input_hash: str,
    row_count: int | None,
    metadata: TableMaterializationMetadata | None = None,
) -> MaterializationResult:
    """Build a skipped table result.

    Returns
    -------
    MaterializationResult
        Skipped table materialization.
    """
    return MaterializationResult(
        status="skipped",
        table_key=table_key,
        row_count=row_count,
        duration_ms=duration_ms,
        input_hash=input_hash,
        error=None,
        iceberg_snapshot_id=metadata.iceberg_snapshot_id if metadata else None,
        validation_id=metadata.validation_id if metadata else None,
        validation_status=metadata.validation_status if metadata else None,
    )


def succeeded_table_result(
    *,
    table_key: str,
    duration_ms: float,
    input_hash: str,
    row_count: int,
    metadata: TableMaterializationMetadata | None = None,
) -> MaterializationResult:
    """Build a succeeded table result.

    Returns
    -------
    MaterializationResult
        Succeeded table materialization.
    """
    return MaterializationResult(
        status="succeeded",
        table_key=table_key,
        row_count=row_count,
        duration_ms=duration_ms,
        input_hash=input_hash,
        error=None,
        iceberg_snapshot_id=metadata.iceberg_snapshot_id if metadata else None,
        validation_id=metadata.validation_id if metadata else None,
        validation_status=metadata.validation_status if metadata else None,
    )


__all__ = [
    "MaterializationResult",
    "MaterializationStatus",
    "TableMaterializationMetadata",
    "failed_artifact_result",
    "failed_table_result",
    "skipped_artifact_result",
    "skipped_table_result",
    "succeeded_artifact_result",
    "succeeded_table_result",
]
