"""Shared manifest models and JSON helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import msgspec

from codeintel.core.serialization.msgspec_json import (
    decode_json_bytes,
    encode_json_bytes,
    encode_json_text,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from codeintel.core.schemas.primitives import TableSchema


def write_manifest_json(path: Path, payload: object) -> None:
    """Write a JSON manifest with deterministic formatting.

    Parameters
    ----------
    path
        Destination path for the manifest file.
    payload
        JSON-serializable manifest payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = payload.to_json_obj() if isinstance(payload, ManifestBase) else payload
    path.write_bytes(_encode_manifest_bytes(normalized))


def read_manifest_json(path: Path) -> dict[str, object]:
    """Read a JSON manifest file.

    Parameters
    ----------
    path
        Path to the manifest file.

    Returns
    -------
    dict[str, object]
        Parsed JSON payload.
    """
    return decode_json_bytes(path.read_bytes(), payload_type=dict[str, object])


def _encode_manifest_bytes(payload: object) -> bytes:
    """Encode a manifest payload to deterministic JSON bytes.

    Returns
    -------
    bytes
        JSON-encoded manifest payload with stable formatting.
    """
    return encode_json_bytes(payload, indent=2, newline=True)


def _encode_manifest_text(payload: object) -> str:
    """Encode a manifest payload to deterministic JSON text.

    Returns
    -------
    str
        JSON-encoded manifest payload with stable formatting.
    """
    return encode_json_text(payload, indent=2, newline=False)


class ManifestBase:
    """Mixin for deterministic manifest serialization."""

    def to_json_obj(self) -> object:
        """Return a JSON-serializable representation.

        Returns
        -------
        object
            JSON-serializable payload.
        """
        return self

    def to_json(self) -> str:
        """Serialize the manifest to a deterministic JSON string.

        Returns
        -------
        str
            Deterministic JSON representation of the manifest.
        """
        return _encode_manifest_text(self.to_json_obj())

    def write_json(self, path: Path) -> Path:
        """Write the manifest to disk with deterministic formatting.

        Returns
        -------
        Path
            Path to the written manifest file.
        """
        write_manifest_json(path, self)
        return path


class ExportManifestData(msgspec.Struct, ManifestBase, frozen=True):
    """Structured manifest metadata for a single dataset export."""

    dataset: str
    schema_id: str | None
    schema_version: str | None
    schema_digest: str | None
    validation_profile: str
    row_count: int
    data_hash: str
    started_at: str
    completed_at: str
    artifact: str | None = None
    extras: Mapping[str, object] | None = None

    def to_json_obj(self) -> object:
        """Return a JSON-serializable export manifest payload.

        Returns
        -------
        object
            JSON-serializable export manifest payload.
        """
        return self


class InferencePlanLoaderOverride(msgspec.Struct, frozen=True):
    """Loader override mapping for inference execution."""

    node: str
    table_key: str

    def to_json_obj(self) -> object:
        """Return JSON-serializable loader override payload.

        Returns
        -------
        object
            JSON-serializable loader override payload.
        """
        return self


class InferencePlanDatasetRef(msgspec.Struct, frozen=True):
    """Dataset reference mapping for inference execution."""

    param: str
    table_key: str

    def to_json_obj(self) -> object:
        """Return JSON-serializable dataset reference payload.

        Returns
        -------
        object
            JSON-serializable dataset reference payload.
        """
        return self


class InferencePlanSeedDataset(msgspec.Struct, frozen=True):
    """Seed dataset settings captured for inference runs."""

    dataset_root_dir: str | None
    snapshot_id: str | None
    scan_mode: str
    sample_rows: int
    batch_size: int
    fragment_readahead: int | None

    def to_json_obj(self) -> object:
        """Return JSON-serializable seed dataset payload.

        Returns
        -------
        object
            JSON-serializable seed dataset payload.
        """
        return self


class InferencePlanSettings(msgspec.Struct, frozen=True):
    """Runtime settings snapshot used for inference."""

    engine_version: str
    polars_profile: bool
    polars_inspect: bool
    polars_query_opt_flags: tuple[str, ...]
    polars_streaming: bool
    polars_streaming_fallback: bool

    def to_json_obj(self) -> object:
        """Return JSON-serializable settings payload.

        Returns
        -------
        object
            JSON-serializable settings payload.
        """
        return self


class InferencePlanManifest(msgspec.Struct, ManifestBase, frozen=True):
    """Manifest describing a deterministic inference plan."""

    manifest_version: int
    run_id: str
    repo: str
    commit: str
    repo_root: str
    generated_at: str
    table_keys: tuple[str, ...]
    targets: tuple[str, ...]
    qparams: tuple[str, ...]
    loader_overrides: tuple[InferencePlanLoaderOverride, ...]
    dataset_refs: tuple[InferencePlanDatasetRef, ...]
    settings: InferencePlanSettings
    seed_dataset: InferencePlanSeedDataset | None = None

    def to_json_obj(self) -> object:
        """Return JSON-serializable inference plan manifest payload.

        Returns
        -------
        object
            JSON-serializable inference plan manifest payload.
        """
        return self


class ArrowDatasetManifest(msgspec.Struct, ManifestBase, frozen=True):
    """Manifest describing an Arrow dataset snapshot."""

    dataset_id: str
    snapshot_id: str
    table_key: str
    partition_columns: tuple[str, ...]
    files: tuple[str, ...]
    schema_hash: str | None = None
    row_count: int | None = None
    stats: Mapping[str, object] | None = None
    created_at: str | None = None
    extras: Mapping[str, object] | None = None

    def to_json_obj(self) -> object:
        """Return a JSON-serializable dataset manifest payload.

        Returns
        -------
        object
            JSON-serializable dataset manifest payload.
        """
        return self


class DatasetSuiteManifest(msgspec.Struct, ManifestBase, frozen=True):
    """Manifest describing a suite of dataset snapshots."""

    suite_manifest_version: int
    suite_kind: str
    repo: str
    commit: str
    created_at: str
    dataset_manifest_paths: Mapping[str, str]
    tool_versions: Mapping[str, str] | None = None

    def to_json_obj(self) -> object:
        """Return a JSON-serializable suite manifest payload.

        Returns
        -------
        object
            JSON-serializable suite manifest payload.
        """
        return self

    @classmethod
    def from_path(cls, path: Path) -> DatasetSuiteManifest:
        """Load a suite manifest from JSON.

        Parameters
        ----------
        path
            Path to the suite manifest JSON file.

        Returns
        -------
        DatasetSuiteManifest
            Loaded suite manifest.
        """
        data = read_manifest_json(path)
        dataset_manifest_paths = _parse_suite_manifest_paths(data.get("dataset_manifest_paths"))
        tool_versions = _coerce_str_mapping(data.get("tool_versions"))
        return cls(
            suite_manifest_version=_require_int(data, "suite_manifest_version"),
            suite_kind=_require_str(data, "suite_kind"),
            repo=_require_str(data, "repo"),
            commit=_require_str(data, "commit"),
            created_at=_require_str(data, "created_at"),
            dataset_manifest_paths=dataset_manifest_paths,
            tool_versions=tool_versions,
        )


class SnapshotDatasetEntry(msgspec.Struct, ManifestBase, frozen=True):
    """Summary pointer to a dataset manifest within a serving snapshot."""

    manifest_path: str
    partition_columns: tuple[str, ...]
    schema_hash: str | None = None
    row_count: int | None = None
    stats: Mapping[str, object] | None = None

    def to_json_obj(self) -> object:
        """Return a JSON-serializable dataset entry payload.

        Returns
        -------
        object
            JSON-serializable dataset entry payload.
        """
        return self


class IncrementalMarker(msgspec.Struct, ManifestBase, frozen=True):
    """Metadata persisted to decide if an export can be reused."""

    dataset: str
    row_count: int
    schema_version: str | None
    validation_profile: str
    schema_digest: str | None = None
    extras: Mapping[str, object] | None = None
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


class SkipCriteria(msgspec.Struct, frozen=True):
    """Inputs used to decide whether an export can be reused."""

    row_count: int | None
    schema_version: str | None
    validation_profile: str
    schema_digest: str | None
    force_full_export: bool


class ServingSnapshotManifest(msgspec.Struct, ManifestBase, frozen=True):
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
    datasets
        Mapping of table key to dataset manifest metadata.
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
    datasets: dict[str, SnapshotDatasetEntry] = msgspec.field(default_factory=dict)

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
                {
                    "datasets": {
                        table_key: entry.to_json_obj() for table_key, entry in self.datasets.items()
                    }
                }
                if self.datasets
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

        Raises
        ------
        TypeError
            If the datasets payload is not a mapping.
        """
        data = read_manifest_json(path)
        raw_datasets = data.get("datasets") or {}
        if not isinstance(raw_datasets, dict):
            msg = "Snapshot manifest datasets must be a mapping"
            raise TypeError(msg)
        datasets: dict[str, SnapshotDatasetEntry] = {}
        for table_key, raw_entry in raw_datasets.items():
            datasets[str(table_key)] = _parse_snapshot_dataset_entry(
                raw_entry,
                table_key=str(table_key),
            )
        return cls(
            run_id=_require_str(data, "run_id"),
            repo=_require_str(data, "repo"),
            commit=_require_str(data, "commit"),
            published_at=_require_str(data, "published_at"),
            db_path=_require_str(data, "db_path"),
            semantic_registry_path=_require_str(data, "semantic_registry_path"),
            schema_manifest_path=_require_str(data, "schema_manifest_path"),
            buildspec_path=_require_str(data, "buildspec_path"),
            semantic_layer_version=_require_str(data, "semantic_layer_version"),
            datasets=datasets,
        )


def _parse_snapshot_dataset_entry(
    raw_entry: object,
    *,
    table_key: str,
) -> SnapshotDatasetEntry:
    if not isinstance(raw_entry, dict):
        msg = f"Snapshot dataset entry must be an object: {table_key}"
        raise TypeError(msg)
    manifest_path = raw_entry.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path:
        msg = f"Snapshot dataset entry manifest_path is required: {table_key}"
        raise TypeError(msg)
    partition_columns = _parse_optional_str_list(
        raw_entry.get("partition_columns"),
        ctx=f"datasets[{table_key}].partition_columns",
    )
    schema_hash = _optional_str(raw_entry.get("schema_hash"))
    row_count = _optional_int(raw_entry.get("row_count"))
    stats = _coerce_mapping(raw_entry.get("stats"))
    return SnapshotDatasetEntry(
        manifest_path=manifest_path,
        schema_hash=schema_hash,
        partition_columns=partition_columns,
        row_count=row_count,
        stats=stats,
    )


def _parse_suite_manifest_paths(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        msg = "Suite manifest dataset_manifest_paths must be a mapping"
        raise TypeError(msg)
    result: dict[str, str] = {}
    for key, raw in value.items():
        if raw is None:
            continue
        table_key = str(key)
        if not isinstance(raw, str) or not raw:
            msg = f"Suite manifest path missing for {table_key}"
            raise TypeError(msg)
        result[table_key] = raw
    return result


def _parse_optional_str_list(value: object, *, ctx: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        msg = f"Expected list for {ctx}"
        raise TypeError(msg)
    return tuple(str(item) for item in value)


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        msg = "row_count must be an integer"
        raise TypeError(msg)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    msg = f"row_count must be an integer, got {type(value).__name__}"
    raise TypeError(msg)


def _coerce_mapping(value: object | None) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): val for key, val in value.items()}
    msg = "Snapshot dataset stats must be an object"
    raise TypeError(msg)


def _coerce_str_mapping(value: object | None) -> dict[str, str] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): str(val) for key, val in value.items()}
    msg = "Suite manifest tool_versions must be an object"
    raise TypeError(msg)


def _require_int(payload: dict[str, object], key: str) -> int:
    raw = payload.get(key)
    if isinstance(raw, bool):
        msg = f"Suite manifest {key} must be an integer"
        raise TypeError(msg)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    msg = f"Suite manifest {key} must be an integer"
    raise TypeError(msg)


def _require_str(payload: dict[str, object], key: str) -> str:
    raw = payload.get(key)
    if isinstance(raw, str) and raw:
        return raw
    msg = f"Suite manifest {key} must be a non-empty string"
    raise TypeError(msg)


def manifest_json_schema(manifest_type: type[object]) -> dict[str, object]:
    """Return a JSON Schema for a manifest type.

    Returns
    -------
    dict[str, object]
        JSON Schema describing the manifest payload.
    """
    return cast("dict[str, object]", msgspec.json.schema(manifest_type))


ExportArtifactKind = Literal["parquet", "jsonl", "json", "csv"]
ManifestDerivationKind = Literal[
    "explicit_override",
    "inferred_relation",
    "declared_source",
    "view_inferred",
]
InferenceStatus = Literal["inferred", "override", "disabled", "error", "pending"]


class ExportArtifact(msgspec.Struct, ManifestBase, frozen=True):
    """Specification for an export artifact (Parquet, JSONL, etc.).

    Export artifacts represent file outputs tied to table data, enabling
    tracking of export filenames alongside their source table schemas.

    Parameters
    ----------
    kind
        Type of export file: parquet, jsonl, json, or csv.
    filename
        Default filename for the export (e.g., "modules.parquet").
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


class TableProvenance(msgspec.Struct, ManifestBase, frozen=True):
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
    producer_target
        Optional Hamilton target name responsible for the schema output.
    producer_module
        Optional Hamilton target module (ingestion/graphs/analytics/export/views).
    producer_version
        Optional Hamilton target spec version string.
    """

    schema_hash: str
    derivation_kind: ManifestDerivationKind
    derivation_source: str
    inference_status: InferenceStatus | None = None
    inference_error: str | None = None
    producer_target: str | None = None
    producer_module: str | None = None
    producer_version: str | None = None

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
        if self.producer_target is not None:
            result["producer_target"] = self.producer_target
        if self.producer_module is not None:
            result["producer_module"] = self.producer_module
        if self.producer_version is not None:
            result["producer_version"] = self.producer_version
        return result


class ArtifactProvenance(msgspec.Struct, ManifestBase, frozen=True):
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


class SchemaManifest(msgspec.Struct, ManifestBase, frozen=True):
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
    views: tuple[TableSchema, ...] = msgspec.field(default_factory=tuple)
    artifacts: tuple[ExportArtifact, ...] = msgspec.field(default_factory=tuple)
    table_provenance: dict[str, TableProvenance] = msgspec.field(default_factory=dict)
    view_provenance: dict[str, TableProvenance] = msgspec.field(default_factory=dict)
    artifact_provenance: dict[str, ArtifactProvenance] = msgspec.field(default_factory=dict)

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
    "DatasetSuiteManifest",
    "ExportArtifact",
    "ExportArtifactKind",
    "ExportManifestData",
    "IncrementalMarker",
    "InferencePlanDatasetRef",
    "InferencePlanLoaderOverride",
    "InferencePlanManifest",
    "InferencePlanSeedDataset",
    "InferencePlanSettings",
    "InferenceStatus",
    "ManifestBase",
    "ManifestDerivationKind",
    "SchemaManifest",
    "ServingSnapshotManifest",
    "SkipCriteria",
    "SnapshotDatasetEntry",
    "TableProvenance",
    "manifest_json_schema",
    "read_manifest_json",
    "write_manifest_json",
]
