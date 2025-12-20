"""Shared metadata helpers for serving exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.serving.export.formats import ExportFormat
from codeintel.serving.export.models import ExportArtifactSpec
from codeintel.serving.snapshot.models import ServingExportSnapshot

if TYPE_CHECKING:
    from codeintel.serving.operations.protocols import ServingSnapshotPointerProtocol


def build_export_snapshot_dict(
    pointer: ServingSnapshotPointerProtocol, *, buildspec_hash: str | None
) -> dict[str, str]:
    """Build a canonical export snapshot dictionary.

    Parameters
    ----------
    pointer
        Serving snapshot pointer.
    buildspec_hash
        Buildspec hash string when available.

    Returns
    -------
    dict[str, str]
        Snapshot metadata with stringified values.
    """
    snapshot = ServingExportSnapshot.from_pointer(pointer).model_dump(mode="json")
    if buildspec_hash is not None:
        snapshot["buildspec_hash"] = buildspec_hash
    return {str(key): str(value) for key, value in snapshot.items() if value is not None}


@dataclass(frozen=True, slots=True)
class ExportArtifactInputs:
    """Inputs required to build an export artifact spec."""

    view_id: str
    columns: tuple[str, ...]
    column_types: dict[str, str]
    compiled_sql: str
    snapshot: dict[str, str]
    export_format: ExportFormat
    query_hash: str | None
    schema_hash: str | None


def build_export_artifact_spec(inputs: ExportArtifactInputs) -> ExportArtifactSpec:
    """Build a normalized export artifact specification.

    Parameters
    ----------
    inputs
        Export artifact inputs bundle.

    Returns
    -------
    ExportArtifactSpec
        Export artifact specification.
    """
    return ExportArtifactSpec(
        view_id=inputs.view_id,
        columns=inputs.columns,
        column_types=inputs.column_types,
        compiled_sql=inputs.compiled_sql,
        snapshot=inputs.snapshot,
        format=inputs.export_format,
        query_hash=inputs.query_hash,
        schema_hash=inputs.schema_hash,
    )


__all__ = ["ExportArtifactInputs", "build_export_artifact_spec", "build_export_snapshot_dict"]
