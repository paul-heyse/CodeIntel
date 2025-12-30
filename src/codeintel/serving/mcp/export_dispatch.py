"""FastMCP export dispatch helpers.

This module centralizes mapping from export format -> ResourceStore writer so
FastMCP tool handlers remain thin and consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.serving.export.dispatch import (
    ExportDispatchHandlers,
    ExportRowProvider,
    dispatch_export,
)
from codeintel.serving.export.models import ExportArtifactSpec
from codeintel.serving.mcp.resource_store import ResourceStore
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.models import SemanticExportRequest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from codeintel.serving.export.engine import ExportPlan
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
    RuntimeError
        If dispatch returns an unexpected async result.
    """

    def handle_ndjson(
        _plan: ExportPlan,
        provider: ExportRowProvider,
    ) -> tuple[str, StoredArtifact, StoredMetadata]:
        return payload.store.put_with_metadata_stream(
            provider.iter_rows(),
            spec=payload.spec,
            export_id=payload.export_id,
        )

    def handle_json_rows(
        _plan: ExportPlan,
        provider: ExportRowProvider,
    ) -> tuple[str, StoredArtifact, StoredMetadata]:
        return payload.store.put_with_metadata(
            provider.collect_rows(),
            spec=payload.spec,
            export_id=payload.export_id,
        )

    def handle_binary_file(
        _plan: ExportPlan,
        write_fn: Callable[[Path], int],
    ) -> tuple[str, StoredArtifact, StoredMetadata]:
        return payload.store.put_generated_file_with_metadata(
            spec=payload.spec,
            export_id=payload.export_id,
            write_fn=write_fn,
        )

    handlers = ExportDispatchHandlers(
        ndjson_stream=handle_ndjson,
        json_rows=handle_json_rows,
        binary_file=handle_binary_file,
    )
    result = dispatch_export(
        payload.ops,
        payload.request,
        cancel_check=payload.cancel_check,
        handlers=handlers,
    )
    if isinstance(result, tuple):
        return result
    msg = "Export dispatch returned unexpected async result"
    raise RuntimeError(msg)


__all__ = ["ExportStoreRequest", "write_export_to_store"]
