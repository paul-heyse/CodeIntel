"""FastMCP export dispatch helpers.

This module centralizes mapping from export format -> ResourceStore writer so
FastMCP tool handlers remain thin and consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.serving.export.engine import ExportDelivery, build_export_plan, write_export_file
from codeintel.serving.export.models import ExportArtifactSpec
from codeintel.serving.export.ndjson import NdjsonBatchOptions, iter_ndjson_bytes_from_batches
from codeintel.serving.mcp.resource_store import ResourceStore
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.models import SemanticExportRequest

if TYPE_CHECKING:
    from codeintel.serving.mcp.resource_store import StoredArtifact, StoredMetadata
    from codeintel.serving.operations.cancellation import CancelCheck


@dataclass(frozen=True, slots=True)
class ExportStoreRequest:
    """Input payload for storing a semantic export."""

    ops: ServingOperations
    store: ResourceStore
    request: SemanticExportRequest
    spec: ExportArtifactSpec
    export_id: str
    cancel_check: CancelCheck | None = None


def write_export_to_store(
    payload: ExportStoreRequest,
) -> tuple[str, StoredArtifact, StoredMetadata]:
    """Write a semantic export to the ResourceStore based on the requested format.

    Returns
    -------
    tuple[str, StoredArtifact, StoredMetadata]
        Resource URI, stored artifact, and associated metadata.

    Raises
    ------
    ValueError
        If the export format is unsupported.
    """
    plan = build_export_plan(payload.request)

    if plan.delivery is ExportDelivery.ndjson_stream:
        return payload.store.put_with_metadata_stream(
            iter_ndjson_bytes_from_batches(
                payload.ops.export_record_batches(
                    payload.request,
                    cancel_check=payload.cancel_check,
                ),
                options=NdjsonBatchOptions(cancel_check=payload.cancel_check),
            ),
            spec=payload.spec,
            export_id=payload.export_id,
        )
    if plan.delivery is ExportDelivery.json_rows:
        rows = list(payload.ops.export_rows(payload.request, cancel_check=payload.cancel_check))
        return payload.store.put_with_metadata(
            rows,
            spec=payload.spec,
            export_id=payload.export_id,
        )
    if plan.delivery is ExportDelivery.binary_file:
        return payload.store.put_generated_file_with_metadata(
            spec=payload.spec,
            export_id=payload.export_id,
            write_fn=lambda path: write_export_file(
                payload.ops,
                payload.request,
                output_path=path,
                cancel_check=payload.cancel_check,
            ),
        )
    msg = f"Unsupported export format: {payload.request.format}"
    raise ValueError(msg)


__all__ = ["ExportStoreRequest", "write_export_to_store"]
