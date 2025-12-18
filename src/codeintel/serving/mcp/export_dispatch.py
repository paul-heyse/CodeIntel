"""FastMCP export dispatch helpers.

This module centralizes mapping from export format -> ResourceStore writer so
FastMCP tool handlers remain thin and consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.serving.export.formats import is_text_export_format
from codeintel.serving.mcp.resource_store import ExportArtifactSpec, ResourceStore
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
    """Write a semantic export to the ResourceStore based on the requested format."""
    if is_text_export_format(spec.format):
        if spec.format == "ndjson":
            return store.put_with_metadata_stream(
                ops.export_rows(request),
                spec=spec,
                export_id=export_id,
            )
        rows = list(ops.export_rows(request))
        return store.put_with_metadata(rows, spec=spec, export_id=export_id)

    if spec.format == "parquet":
        return store.put_generated_file_with_metadata(
            spec=spec,
            export_id=export_id,
            write_fn=lambda path: ops.export_to_parquet(request, output_path=path),
        )

    if spec.format == "arrow":
        return store.put_generated_file_with_metadata(
            spec=spec,
            export_id=export_id,
            write_fn=lambda path: ops.export_to_arrow_ipc(request, output_path=path),
        )

    msg = f"Unsupported export format: {spec.format}"
    raise ValueError(msg)


__all__ = ["write_export_to_store"]

