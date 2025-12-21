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
ManifestDerivationKind = Literal[
    "explicit_override",
    "inferred_ibis",
    "declared_source",
    "view_inferred",
]
InferenceStatus = Literal["inferred", "override", "disabled", "error", "pending"]


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
class TableProvenance(ManifestBase):
    """Describe schema provenance for a table or view.

    Parameters
    ----------
    schema_hash
        Stable hash of the schema definition.
    derivation_kind
        Label describing how the schema was derived.
    derivation_source
        Source identifier for the derivation.
    inference_status
        Optional inference status when the table is inferable.
    inference_error
        Optional inference error message when inference failed.
    """

    schema_hash: str
    derivation_kind: ManifestDerivationKind
    derivation_source: str
    inference_status: InferenceStatus | None = None
    inference_error: str | None = None

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable representation.

        Returns
        -------
        dict[str, object]
            JSON-serializable provenance payload.
        """
        result: dict[str, object] = {
            "schema_hash": self.schema_hash,
            "derivation_kind": self.derivation_kind,
            "derivation_source": self.derivation_source,
        }
        if self.inference_status is not None:
            result["inference_status"] = self.inference_status
        if self.inference_error is not None:
            result["inference_error"] = self.inference_error
        return result


@dataclass(frozen=True)
class ArtifactProvenance(ManifestBase):
    """Describe lineage metadata for an export artifact.

    Parameters
    ----------
    source_table_keys
        Ordered table keys that feed this artifact.
    source_schema_hashes
        Schema hashes aligned to the source table keys.
    """

    source_table_keys: tuple[str, ...]
    source_schema_hashes: tuple[str, ...]

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable representation.

        Returns
        -------
        dict[str, object]
            JSON-serializable artifact provenance payload.
        """
        return {
            "source_table_keys": list(self.source_table_keys),
            "source_schema_hashes": list(self.source_schema_hashes),
        }


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
    table_provenance
        Optional per-table provenance metadata (v2 additive).
    view_provenance
        Optional per-view provenance metadata (v2 additive).
    artifact_provenance
        Optional per-artifact provenance metadata (v2 additive).

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
    table_provenance: dict[str, TableProvenance] = field(default_factory=dict)
    view_provenance: dict[str, TableProvenance] = field(default_factory=dict)
    artifact_provenance: dict[str, ArtifactProvenance] = field(default_factory=dict)

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable manifest representation.

        Returns
        -------
        dict[str, object]
            JSON-serializable manifest payload.
        """
        result: dict[str, object] = {"version": self.version}

        if self.tables:
            tables: list[dict[str, object]] = []
            for table in self.tables:
                table_obj = table.to_json_obj()
                provenance = self.table_provenance.get(table.table_key)
                if provenance is not None:
                    table_obj.update(provenance.to_json_obj())
                tables.append(table_obj)
            result["tables"] = tables

        if self.views:
            views: list[dict[str, object]] = []
            for view in self.views:
                view_obj = view.to_json_obj()
                provenance = self.view_provenance.get(view.table_key)
                if provenance is not None:
                    view_obj.update(provenance.to_json_obj())
                views.append(view_obj)
            result["views"] = views

        if self.artifacts:
            artifacts: list[dict[str, object]] = []
            for artifact in self.artifacts:
                artifact_obj = artifact.to_json_obj()
                provenance = self.artifact_provenance.get(artifact.filename)
                if provenance is not None:
                    artifact_obj["provenance"] = provenance.to_json_obj()
                artifacts.append(artifact_obj)
            result["artifacts"] = artifacts

        return result

    @property
    def is_v2(self) -> bool:
        """Check if this manifest uses v2 format features.

        Returns
        -------
        bool
            True if manifest contains views or artifacts.
        """
        return self.version == "v2"


__all__ = [
    "ArtifactProvenance",
    "ExportArtifact",
    "ExportArtifactKind",
    "InferenceStatus",
    "ManifestDerivationKind",
    "SchemaManifest",
    "TableProvenance",
]
