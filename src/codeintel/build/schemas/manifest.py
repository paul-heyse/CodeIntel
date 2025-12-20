"""Schema manifest types for build-time schema products.

This module defines the SchemaManifest and related types for capturing
build-time schema state. The v2 format extends v1 with views and artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from codeintel.build.manifest_base import ManifestBase

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import TableSchema


ExportArtifactKind = Literal["parquet", "jsonl", "json", "csv"]


@dataclass(frozen=True)
class ExportArtifact(ManifestBase):
    """Specification for an export artifact (Parquet, JSONL, etc.).

    Export artifacts represent file outputs tied to table data, enabling
    tracking of export filenames alongside their source table schemas.

    Parameters
    ----------
    kind
        Type of export file: parquet, jsonl, json, or csv.
    filename
        Default filename for the export (e.g., "function_metrics.parquet").
    table_key
        Fully qualified table key (schema.table) this artifact exports from.
        None for artifacts not tied to a specific table.
    description
        Optional description of the artifact's purpose.

    Examples
    --------
    >>> artifact = ExportArtifact(
    ...     kind="parquet",
    ...     filename="function_metrics.parquet",
    ...     table_key="analytics.function_metrics",
    ... )
    >>> artifact.kind
    'parquet'
    """

    kind: ExportArtifactKind
    filename: str
    table_key: str | None = None
    description: str | None = None

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable representation.

        Returns
        -------
        dict[str, object]
            JSON-serializable artifact payload.
        """
        result: dict[str, object] = {
            "kind": self.kind,
            "filename": self.filename,
        }
        if self.table_key is not None:
            result["table_key"] = self.table_key
        if self.description is not None:
            result["description"] = self.description
        return result


@dataclass(frozen=True)
class SchemaManifest(ManifestBase):
    """Stable manifest of schemas compiled for a build selection.

    The manifest captures table schemas, view schemas (v2), and export
    artifact specifications (v2) at build time. This enables schema drift
    detection and version tracking.

    Parameters
    ----------
    version
        Manifest version identifier (v1 or v2).
    tables
        Table schemas included in this manifest.
    views
        View schemas included in this manifest (v2 only).
    artifacts
        Export artifact specifications (v2 only).

    Notes
    -----
    - v1 manifests only contain tables
    - v2 manifests may contain tables, views, and/or artifacts
    - Empty tuples are omitted from JSON output for cleaner diffs
    """

    version: str
    tables: tuple[TableSchema, ...] = ()
    views: tuple[TableSchema, ...] = field(default_factory=tuple)
    artifacts: tuple[ExportArtifact, ...] = field(default_factory=tuple)

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable manifest representation.

        Returns
        -------
        dict[str, object]
            JSON-serializable manifest payload.
        """
        result: dict[str, object] = {"version": self.version}

        if self.tables:
            result["tables"] = [table.to_json_obj() for table in self.tables]

        if self.views:
            result["views"] = [view.to_json_obj() for view in self.views]

        if self.artifacts:
            result["artifacts"] = [artifact.to_json_obj() for artifact in self.artifacts]

        return result

    @property
    def is_v2(self) -> bool:
        """Check if this manifest uses v2 format features.

        Returns
        -------
        bool
            True if manifest contains views or artifacts.
        """
        return bool(self.views or self.artifacts)


__all__ = ["ExportArtifact", "ExportArtifactKind", "SchemaManifest"]
