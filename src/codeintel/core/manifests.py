"""Shared manifest models and JSON helpers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from codeintel.core.schemas.primitives import TableSchema


def write_manifest_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON manifest with deterministic formatting.

    Parameters
    ----------
    path
        Destination path for the manifest file.
    payload
        JSON-serializable manifest payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest_json(path: Path) -> dict[str, Any]:
    """Read a JSON manifest file.

    Parameters
    ----------
    path
        Path to the manifest file.

    Returns
    -------
    dict[str, Any]
        Parsed JSON payload.
    """
    return json.loads(path.read_text(encoding="utf-8"))


class ManifestBase(ABC):
    """Base class for deterministic manifest serialization."""

    @abstractmethod
    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable representation.

        Returns
        -------
        dict[str, object]
            JSON-serializable manifest payload.
        """
        ...

    def to_json(self) -> str:
        """Serialize the manifest to a deterministic JSON string.

        Returns
        -------
        str
            JSON string representation of this manifest.
        """
        return json.dumps(self.to_json_obj(), indent=2, sort_keys=True)

    def write_json(self, path: Path) -> Path:
        """Write the manifest to disk with deterministic formatting.

        Parameters
        ----------
        path
            Destination path for the manifest file.

        Returns
        -------
        Path
            Path to the written manifest file.
        """
        write_manifest_json(path, self.to_json_obj())
        return path


@dataclass(frozen=True)
class ExportManifestData(ManifestBase):
    """Structured manifest metadata for a single dataset export."""

    dataset: str
    artifact: str | None
    schema_id: str | None
    schema_version: str | None
    schema_digest: str | None
    validation_profile: str
    row_count: int
    data_hash: str
    started_at: str
    completed_at: str
    extras: Mapping[str, Any] | None = None

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable export manifest payload.

        Returns
        -------
        dict[str, object]
            JSON-serializable export manifest payload.
        """
        payload: dict[str, object] = {
            "dataset": self.dataset,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "schema_digest": self.schema_digest,
            "validation_profile": self.validation_profile,
            "row_count": self.row_count,
            "data_hash": self.data_hash,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        if self.artifact is not None:
            payload["artifact"] = self.artifact
        if self.extras:
            payload["extras"] = dict(self.extras)
        return payload


@dataclass(frozen=True)
class ArrowDatasetManifest(ManifestBase):
    """Manifest describing an Arrow dataset snapshot."""

    dataset_id: str
    snapshot_id: str
    table_key: str
    schema_hash: str | None
    partition_columns: tuple[str, ...]
    files: tuple[str, ...]
    row_count: int | None = None
    stats: Mapping[str, Any] | None = None
    created_at: str | None = None
    extras: Mapping[str, Any] | None = None

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable dataset manifest payload.

        Returns
        -------
        dict[str, object]
            JSON-serializable dataset manifest payload.
        """
        payload: dict[str, object] = {
            "dataset_id": self.dataset_id,
            "snapshot_id": self.snapshot_id,
            "table_key": self.table_key,
            "partition_columns": list(self.partition_columns),
            "files": list(self.files),
        }
        if self.schema_hash is not None:
            payload["schema_hash"] = self.schema_hash
        if self.row_count is not None:
            payload["row_count"] = self.row_count
        if self.stats:
            payload["stats"] = dict(self.stats)
        if self.created_at is not None:
            payload["created_at"] = self.created_at
        if self.extras:
            payload["extras"] = dict(self.extras)
        return payload


@dataclass(frozen=True)
class IncrementalMarker(ManifestBase):
    """Metadata persisted to decide if an export can be reused."""

    dataset: str
    row_count: int
    schema_version: str | None
    validation_profile: str
    schema_digest: str | None = None
    extras: Mapping[str, Any] | None = None
    exported_at: str | None = None

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable marker payload.

        Returns
        -------
        dict[str, object]
            JSON-serializable marker payload.

        Raises
        ------
        ValueError
            If ``exported_at`` is not set before serialization.
        """
        if self.exported_at is None:
            msg = "IncrementalMarker.exported_at must be set before serialization"
            raise ValueError(msg)
        payload: dict[str, object] = {
            "dataset": self.dataset,
            "row_count": self.row_count,
            "schema_version": self.schema_version,
            "validation_profile": self.validation_profile,
            "schema_digest": self.schema_digest,
            "exported_at": self.exported_at,
        }
        if self.extras:
            payload["extras"] = dict(self.extras)
        return payload


@dataclass(frozen=True)
class SkipCriteria:
    """Inputs used to decide whether an export can be reused."""

    row_count: int | None
    schema_version: str | None
    validation_profile: str
    schema_digest: str | None
    force_full_export: bool


@dataclass(frozen=True)
class ServingSnapshotManifest(ManifestBase):
    """Manifest describing a published serving snapshot.

    Parameters
    ----------
    run_id
        Unique build run identifier.
    repo
        Repository identifier.
    commit
        Commit SHA.
    published_at
        ISO timestamp when snapshot was published.
    db_path
        Path to DuckDB snapshot file.
    semantic_registry_path
        Path to semantic_registry.json.
    schema_manifest_path
        Path to schema_manifest.json.
    dataset_manifest_paths
        Paths to dataset manifest files (Arrow datasets).
    buildspec_path
        Path to buildspec.json.
    semantic_layer_version
        Version hash of semantic layer.
    """

    run_id: str
    repo: str
    commit: str
    published_at: str
    db_path: str
    semantic_registry_path: str
    schema_manifest_path: str
    buildspec_path: str
    semantic_layer_version: str
    dataset_manifest_paths: tuple[str, ...] = ()

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable manifest payload.

        Returns
        -------
        dict[str, object]
            JSON-serializable manifest payload.
        """
        return {
            "run_id": self.run_id,
            "repo": self.repo,
            "commit": self.commit,
            "published_at": self.published_at,
            "db_path": self.db_path,
            "semantic_registry_path": self.semantic_registry_path,
            "schema_manifest_path": self.schema_manifest_path,
            **(
                {"dataset_manifest_paths": list(self.dataset_manifest_paths)}
                if self.dataset_manifest_paths
                else {}
            ),
            "buildspec_path": self.buildspec_path,
            "semantic_layer_version": self.semantic_layer_version,
        }

    @classmethod
    def from_path(cls, path: Path) -> ServingSnapshotManifest:
        """Load manifest from JSON file.

        Parameters
        ----------
        path
            Path to the JSON manifest file.

        Returns
        -------
        ServingSnapshotManifest
            Loaded manifest instance.
        """
        data = read_manifest_json(path)
        raw_paths = data.get("dataset_manifest_paths") or ()
        data["dataset_manifest_paths"] = tuple(str(path) for path in raw_paths)
        return cls(**data)


ExportArtifactKind = Literal["parquet", "jsonl", "json", "csv"]
ManifestDerivationKind = Literal[
    "explicit_override",
    "inferred_relation",
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

    The manifest captures table schemas, view schemas, and export
    artifact specifications at build time. This enables schema drift
    detection and version tracking.

    Parameters
    ----------
    version
        Manifest version identifier (v2).
    tables
        Table schemas included in this manifest.
    views
        View schemas included in this manifest.
    artifacts
        Export artifact specifications.
    table_provenance
        Optional per-table provenance metadata.
    view_provenance
        Optional per-view provenance metadata.
    artifact_provenance
        Optional per-artifact provenance metadata.
    """

    version: str
    tables: tuple[TableSchema, ...] = ()
    views: tuple[TableSchema, ...] = field(default_factory=tuple)
    artifacts: tuple[ExportArtifact, ...] = field(default_factory=tuple)
    table_provenance: dict[str, TableProvenance] = field(default_factory=dict)
    view_provenance: dict[str, TableProvenance] = field(default_factory=dict)
    artifact_provenance: dict[str, ArtifactProvenance] = field(default_factory=dict)

    @property
    def is_v2(self) -> bool:
        """Return True when the manifest version is v2.

        Returns
        -------
        bool
            True if this manifest uses the v2 format.
        """
        return self.version == "v2"

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
                artifact_obj["provenance"] = (
                    provenance.to_json_obj()
                    if provenance is not None
                    else {"source_table_keys": [], "source_schema_hashes": []}
                )
                artifacts.append(artifact_obj)
            result["artifacts"] = artifacts

        return result


__all__ = [
    "ArrowDatasetManifest",
    "ArtifactProvenance",
    "ExportArtifact",
    "ExportArtifactKind",
    "ExportManifestData",
    "IncrementalMarker",
    "InferenceStatus",
    "ManifestBase",
    "ManifestDerivationKind",
    "SchemaManifest",
    "ServingSnapshotManifest",
    "SkipCriteria",
    "TableProvenance",
    "read_manifest_json",
    "write_manifest_json",
]
