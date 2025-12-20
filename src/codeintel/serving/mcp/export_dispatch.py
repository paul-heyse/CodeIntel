"""FastMCP export dispatch helpers.

This module centralizes mapping from export format -> ResourceStore writer so
FastMCP tool handlers remain thin and consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.serving.export.engine import ExportDelivery, build_export_plan, write_export_file
from codeintel.serving.export.models import ExportArtifactSpec
from codeintel.serving.mcp.resource_store import ResourceStore
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.models import SemanticExportRequest

if TYPE_CHECKING:
    from codeintel.serving.mcp.resource_store import StoredArtifact, StoredMetadata


def write_export_to_store(
    *,
    ops: ServingOperations,
    store: ResourceStore,
    request: SemanticExportRequest,
    spec: ExportArtifactSpec,
    export_id: str,
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
    plan = build_export_plan(request)

    if plan.delivery is ExportDelivery.ndjson_stream:
        return store.put_with_metadata_stream(
            ops.export_rows(request),
            spec=spec,
            export_id=export_id,
        )
    if plan.delivery is ExportDelivery.json_rows:
        rows = list(ops.export_rows(request))
        return store.put_with_metadata(rows, spec=spec, export_id=export_id)
    if plan.delivery is ExportDelivery.binary_file:
        return store.put_generated_file_with_metadata(
            spec=spec,
            export_id=export_id,
            write_fn=lambda path: write_export_file(ops, request, output_path=path),
        )
    msg = f"Unsupported export format: {request.format}"
    raise ValueError(msg)


__all__ = ["write_export_to_store"]
