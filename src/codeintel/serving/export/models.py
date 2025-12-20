"""Shared export models for serving."""

from __future__ import annotations

from dataclasses import dataclass, field

from codeintel.serving.export.formats import ExportFormat


@dataclass(frozen=True, slots=True)
class ExportArtifactSpec:
    """Input specification for an export artifact plus its metadata sidecar."""

    view_id: str
    columns: tuple[str, ...] = ()
    column_types: dict[str, str] = field(default_factory=dict)
    compiled_sql: str | None = None
    snapshot: dict[str, str] = field(default_factory=dict)
    format: ExportFormat = "ndjson"
    query_hash: str | None = None
    schema_hash: str | None = None


__all__ = ["ExportArtifactSpec"]
