"""Shared export planning and execution helpers for serving."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from codeintel.serving.export.formats import (
    ExportFormat,
    mime_type_for_export_format,
    suffix_for_export_format,
)

_BINARY_EXPORT_FORMATS: set[str] = {"arrow", "parquet"}

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.serving.operations.ops import ServingOperations
    from codeintel.serving.semantic.models import SemanticExportRequest


class ExportDelivery(Enum):
    """Export delivery strategy used by adapters."""

    ndjson_stream = "ndjson_stream"
    json_rows = "json_rows"
    binary_file = "binary_file"


@dataclass(frozen=True, slots=True)
class ExportPlan:
    """Resolved export plan for adapters.

    Parameters
    ----------
    format
        Export format.
    delivery
        Delivery strategy for the format.
    mime_type
        MIME type for the export.
    suffix
        File suffix for the export.
    """

    format: ExportFormat
    delivery: ExportDelivery
    mime_type: str
    suffix: str


def build_export_plan(request: SemanticExportRequest) -> ExportPlan:
    """Build an export plan for a semantic export request.

    Parameters
    ----------
    request
        Export request payload.

    Returns
    -------
    ExportPlan
        Planned export strategy.

    Raises
    ------
    ValueError
        If the export format is unsupported.
    """
    if request.format == "ndjson":
        delivery = ExportDelivery.ndjson_stream
    elif request.format == "json":
        delivery = ExportDelivery.json_rows
    elif request.format in _BINARY_EXPORT_FORMATS:
        delivery = ExportDelivery.binary_file
    else:
        msg = f"Unsupported export format: {request.format}"
        raise ValueError(msg)

    return ExportPlan(
        format=request.format,
        delivery=delivery,
        mime_type=mime_type_for_export_format(request.format),
        suffix=suffix_for_export_format(request.format),
    )


def write_export_file(
    ops: ServingOperations,
    request: SemanticExportRequest,
    *,
    output_path: Path,
) -> int:
    """Write an export payload to a file for binary formats.

    Parameters
    ----------
    ops
        Serving operations facade.
    request
        Export request payload.
    output_path
        Path to write the file to.

    Returns
    -------
    int
        Number of rows written.

    Raises
    ------
    ValueError
        If the export format is unsupported for file export.
    """
    if request.format == "parquet":
        return ops.export_to_parquet(request, output_path=output_path)
    if request.format == "arrow":
        return ops.export_to_arrow_ipc(request, output_path=output_path)
    msg = f"Unsupported file export format: {request.format}"
    raise ValueError(msg)


__all__ = ["ExportDelivery", "ExportPlan", "build_export_plan", "write_export_file"]
