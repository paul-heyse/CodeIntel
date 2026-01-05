"""Ingest build metadata bundles into DuckDB metadata tables."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.schema_catalog_models import (
    DEFAULT_SCHEMA_MANIFEST_KIND,
    ColumnStatsPayload,
    DatasetStatsPayload,
    DerivedSettingsPayload,
    SchemaManifestRunRecord,
    SchemaObservationRecord,
    SchemaVersionRecord,
    TableSchemaRegistryRecord,
)
from codeintel.storage.constants import META_CATALOG_NAME
from codeintel.storage.contracts.provider import load_contract_catalog_from_connection
from codeintel.storage.gateway.minimal import MinimalStorageGateway
from codeintel.storage.metadata.bootstrap import (
    replace_dataset_dataflow_edges,
    replace_dataset_dataflow_nodes,
    replace_derived_lineage_columns,
    replace_derived_lineage_edges,
)
from codeintel.storage.metadata.catalogs import build_catalog_entry, upsert_canonical_catalog
from codeintel.storage.metadata.ddl import apply_metadata_ddl
from codeintel.storage.metadata.sync import bootstrap_metadata_datasets
from codeintel.storage.tracking.schema_catalog import SchemaCatalogTracking

if TYPE_CHECKING:
    from collections.abc import Iterable

    from duckdb import DuckDBPyConnection


@dataclass(frozen=True, slots=True)
class BundleManifestFile:
    """Manifest entry describing a file in the metadata bundle."""

    path: Path
    sha256: str
    size_bytes: int
    record_count: int | None
    schema_version: str | None


@dataclass(frozen=True, slots=True)
class BundleManifest:
    """Metadata bundle manifest contents."""

    bundle_schema_version: str
    generated_at: datetime | None
    repo: str | None
    commit: str | None
    run_id: str | None
    files: tuple[BundleManifestFile, ...]


@dataclass(frozen=True, slots=True)
class BundleValidation:
    """Validation results for a metadata bundle."""

    ok: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BundleIngestReport:
    """Summary counts for metadata bundle ingestion."""

    repo: str | None
    commit: str | None
    run_id: str | None
    contract_catalog_hash: str | None
    schema_manifest_hash: str | None
    schema_versions_rows: int
    table_schema_registry_rows: int
    schema_observations_rows: int
    dataflow_nodes: int
    dataflow_edges: int
    lineage_edges: int
    lineage_columns: int
    export_audit_rows: int

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serializable summary payload.

        Returns
        -------
        dict[str, object]
            Summary payload for API/CLI output.
        """
        return {
            "repo": self.repo,
            "commit": self.commit,
            "run_id": self.run_id,
            "contract_catalog_hash": self.contract_catalog_hash,
            "schema_manifest_hash": self.schema_manifest_hash,
            "schema_versions_rows": self.schema_versions_rows,
            "table_schema_registry_rows": self.table_schema_registry_rows,
            "schema_observations_rows": self.schema_observations_rows,
            "dataflow_nodes": self.dataflow_nodes,
            "dataflow_edges": self.dataflow_edges,
            "lineage_edges": self.lineage_edges,
            "lineage_columns": self.lineage_columns,
            "export_audit_rows": self.export_audit_rows,
        }


def bundle_manifest_from_path(bundle_root: Path) -> BundleManifest:
    """Load the bundle manifest from the bundle root.

    Returns
    -------
    BundleManifest
        Parsed bundle manifest.
    """
    path = bundle_root / "bundle_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    files_raw = payload.get("files", [])
    files: list[BundleManifestFile] = []
    for item in files_raw:
        if not isinstance(item, Mapping):
            continue
        file_path = item.get("path")
        if not isinstance(file_path, str):
            continue
        files.append(
            BundleManifestFile(
                path=bundle_root / file_path,
                sha256=str(item.get("sha256", "")),
                size_bytes=int(item.get("size_bytes", 0)),
                record_count=_optional_int(item.get("record_count")),
                schema_version=_optional_str(item.get("schema_version")),
            )
        )
    return BundleManifest(
        bundle_schema_version=str(payload.get("bundle_schema_version", "")),
        generated_at=_parse_datetime(payload.get("generated_at")),
        repo=_optional_str(payload.get("repo")),
        commit=_optional_str(payload.get("commit")),
        run_id=_optional_str(payload.get("run_id")),
        files=tuple(files),
    )


def validate_build_metadata_bundle(bundle_root: Path) -> BundleValidation:
    """Validate bundle files against the recorded manifest hashes.

    Returns
    -------
    BundleValidation
        Validation results with any errors encountered.
    """
    errors: list[str] = []
    try:
        manifest = bundle_manifest_from_path(bundle_root)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return BundleValidation(ok=False, errors=(f"Failed to read bundle_manifest.json: {exc}",))

    for entry in manifest.files:
        if not entry.path.is_file():
            errors.append(f"Missing bundle file: {entry.path}")
            continue
        if entry.size_bytes and entry.path.stat().st_size != entry.size_bytes:
            errors.append(f"Size mismatch for {entry.path}")
        if entry.sha256 and _sha256_path(entry.path) != entry.sha256:
            errors.append(f"Hash mismatch for {entry.path}")
    return BundleValidation(ok=not errors, errors=tuple(errors))


def load_build_metadata_bundle(
    bundle_root: Path,
    con: DuckDBPyConnection,
) -> BundleIngestReport:
    """Ingest a build metadata bundle into metadata tables.

    Returns
    -------
    BundleIngestReport
        Summary of ingested bundle counts.

    Raises
    ------
    ValueError
        If the bundle manifest or bundle files fail validation.
    """
    manifest = bundle_manifest_from_path(bundle_root)
    validation = validate_build_metadata_bundle(bundle_root)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    apply_metadata_ddl(con, catalog=META_CATALOG_NAME, include_views=True)
    gateway = MinimalStorageGateway(con)
    tracker = SchemaCatalogTracking(gateway)

    contract_hash = _ingest_contract_catalog(bundle_root, con)
    schema_manifest_hash = _ingest_schema_manifest(bundle_root, con, manifest)

    load_contract_catalog_from_connection(con)
    bootstrap_metadata_datasets(con)

    registry_rows = _ingest_schema_registry(bundle_root, tracker)
    version_rows = _ingest_schema_versions(bundle_root, tracker)
    observation_rows = _ingest_schema_observations(bundle_root, tracker)

    dataflow_nodes = _ingest_dataflow_nodes(bundle_root, con)
    dataflow_edges = _ingest_dataflow_edges(bundle_root, con)
    lineage_edges = _ingest_lineage_edges(bundle_root, con)
    lineage_columns = _ingest_lineage_columns(bundle_root, con)
    export_audit_rows = _ingest_export_audit(bundle_root, gateway)

    return BundleIngestReport(
        repo=manifest.repo,
        commit=manifest.commit,
        run_id=manifest.run_id,
        contract_catalog_hash=contract_hash,
        schema_manifest_hash=schema_manifest_hash,
        schema_versions_rows=version_rows,
        table_schema_registry_rows=registry_rows,
        schema_observations_rows=observation_rows,
        dataflow_nodes=dataflow_nodes,
        dataflow_edges=dataflow_edges,
        lineage_edges=lineage_edges,
        lineage_columns=lineage_columns,
        export_audit_rows=export_audit_rows,
    )


def _ingest_contract_catalog(bundle_root: Path, con: DuckDBPyConnection) -> str | None:
    path = bundle_root / "contracts" / "contract_catalog.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    catalog_hash = fingerprint(payload)
    entry = build_catalog_entry(
        catalog_kind="dataset_contracts",
        catalog_hash=catalog_hash,
        payload=payload,
        inputs={"source": "bundle_ingest"},
    )
    upsert_canonical_catalog(MinimalStorageGateway(con), entry)
    return catalog_hash


def _ingest_schema_manifest(
    bundle_root: Path,
    con: DuckDBPyConnection,
    manifest: BundleManifest,
) -> str | None:
    path = bundle_root / "schema" / "schema_manifest.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    catalog_hash = fingerprint(payload)
    entry = build_catalog_entry(
        catalog_kind=DEFAULT_SCHEMA_MANIFEST_KIND,
        catalog_hash=catalog_hash,
        payload=payload,
        inputs={"source": "bundle_ingest"},
    )
    upsert_canonical_catalog(MinimalStorageGateway(con), entry)
    if manifest.repo and manifest.commit and manifest.run_id:
        tracker = SchemaCatalogTracking(MinimalStorageGateway(con))
        tracker.record_schema_manifest_runs_batch(
            [
                SchemaManifestRunRecord(
                    run_id=manifest.run_id,
                    repo=manifest.repo,
                    commit=manifest.commit,
                    manifest_kind=DEFAULT_SCHEMA_MANIFEST_KIND,
                    catalog_hash=catalog_hash,
                    created_at=manifest.generated_at or datetime.now(tz=UTC),
                )
            ]
        )
    return catalog_hash


def _ingest_schema_registry(
    bundle_root: Path,
    tracker: SchemaCatalogTracking,
) -> int:
    path = bundle_root / "schema" / "schema_registry.json"
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    records: list[TableSchemaRegistryRecord] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        table_key = _optional_str(entry.get("table_key"))
        schema_digest = _optional_str(entry.get("schema_digest"))
        schema_hash = _optional_str(entry.get("schema_hash"))
        derivation_kind = _optional_str(entry.get("derivation_kind"))
        derivation_source = _optional_str(entry.get("derivation_source"))
        if not (
            table_key and schema_digest and schema_hash and derivation_kind and derivation_source
        ):
            continue
        records.append(
            TableSchemaRegistryRecord(
                table_key=table_key,
                schema_digest=schema_digest,
                schema_hash=schema_hash,
                derivation_kind=derivation_kind,
                derivation_source=derivation_source,
                inference_status=_optional_str(entry.get("inference_status")),
                inference_error=_optional_str(entry.get("inference_error")),
                catalog_hash=_optional_str(entry.get("catalog_hash")),
                updated_at=_parse_datetime(entry.get("updated_at")),
            )
        )
    return tracker.record_table_schema_registry_batch(records)


def _ingest_schema_versions(
    bundle_root: Path,
    tracker: SchemaCatalogTracking,
) -> int:
    path = bundle_root / "schema" / "schema_versions.jsonl"
    if not path.is_file():
        return 0
    records: list[SchemaVersionRecord] = []
    for item in _iter_jsonl(path):
        schema_digest = _optional_str(item.get("schema_digest"))
        schema_hash = _optional_str(item.get("schema_hash"))
        if not (schema_digest and schema_hash):
            continue
        records.append(
            SchemaVersionRecord(
                schema_digest=schema_digest,
                schema_hash=schema_hash,
                schema_json=_optional_mapping(item.get("schema_json")) or {},
                renderer_cache=_optional_mapping(item.get("renderer_cache")),
                created_at=_parse_datetime(item.get("created_at")),
            )
        )
    return tracker.record_schema_versions_batch(records)


def _ingest_schema_observations(
    bundle_root: Path,
    tracker: SchemaCatalogTracking,
) -> int:
    path = bundle_root / "schema" / "schema_observations.jsonl"
    if not path.is_file():
        return 0
    records: list[SchemaObservationRecord] = []
    for item in _iter_jsonl(path):
        table_key = _optional_str(item.get("table_key"))
        schema_digest = _optional_str(item.get("schema_digest"))
        schema_hash = _optional_str(item.get("schema_hash"))
        arrow_schema = _optional_str(item.get("arrow_schema_ipc_b64"))
        if not (table_key and schema_digest and schema_hash and arrow_schema):
            continue
        records.append(
            SchemaObservationRecord(
                observation_id=_optional_str(item.get("observation_id")),
                table_key=table_key,
                repo=_optional_str(item.get("repo")),
                commit=_optional_str(item.get("commit")),
                target_name=_optional_str(item.get("target_name")),
                schema_digest=schema_digest,
                schema_hash=schema_hash,
                arrow_schema_ipc_b64=arrow_schema,
                column_stats=_optional_column_stats(item.get("column_stats")),
                dataset_stats=_optional_dataset_stats(item.get("dataset_stats")),
                derived_settings=_optional_derived_settings(item.get("derived_settings")),
                drift_summary=_optional_mapping(item.get("drift_summary")),
                observed_at=_parse_datetime(item.get("observed_at")),
            )
        )
    return tracker.record_schema_observations_batch(records)


def _ingest_dataflow_nodes(bundle_root: Path, con: DuckDBPyConnection) -> int:
    path = bundle_root / "dataflow" / "dataset_nodes.jsonl"
    if not path.is_file():
        return 0
    rows = [
        (
            str(item.get("id", "")),
            str(item.get("kind", "")),
            _optional_str(item.get("family")),
            _optional_str(item.get("owner_package")),
            _optional_str(item.get("description")),
        )
        for item in _iter_jsonl(path)
    ]
    replace_dataset_dataflow_nodes(con, rows=rows)
    return len(rows)


def _ingest_dataflow_edges(bundle_root: Path, con: DuckDBPyConnection) -> int:
    path = bundle_root / "dataflow" / "dataset_edges.jsonl"
    if not path.is_file():
        return 0
    rows = [
        (
            str(item.get("src", "")),
            str(item.get("dst", "")),
            str(item.get("edge_type", "")),
        )
        for item in _iter_jsonl(path)
    ]
    replace_dataset_dataflow_edges(con, rows=rows)
    return len(rows)


def _ingest_lineage_edges(bundle_root: Path, con: DuckDBPyConnection) -> int:
    path = bundle_root / "lineage" / "derived_edges.jsonl"
    if not path.is_file():
        return 0
    rows = []
    repo: str | None = None
    commit: str | None = None
    edge_type = "derived_depends_on"
    for item in _iter_jsonl(path):
        repo = _optional_str(item.get("repo")) or repo
        commit = _optional_str(item.get("commit")) or commit
        edge_type = _optional_str(item.get("edge_type")) or edge_type
        if repo is None or commit is None:
            continue
        rows.append(
            (
                repo,
                commit,
                str(item.get("downstream", "")),
                str(item.get("upstream", "")),
                str(item.get("edge_type", edge_type)),
            )
        )
    if repo is None or commit is None:
        return 0
    replace_derived_lineage_edges(
        con,
        repo=repo,
        commit=commit,
        edge_type=edge_type,
        rows=rows,
    )
    return len(rows)


def _ingest_lineage_columns(bundle_root: Path, con: DuckDBPyConnection) -> int:
    path = bundle_root / "lineage" / "derived_columns.jsonl"
    if not path.is_file():
        return 0
    rows = []
    repo: str | None = None
    commit: str | None = None
    edge_type = "derived_column_depends_on"
    for item in _iter_jsonl(path):
        repo = _optional_str(item.get("repo")) or repo
        commit = _optional_str(item.get("commit")) or commit
        edge_type = _optional_str(item.get("edge_type")) or edge_type
        if repo is None or commit is None:
            continue
        rows.append(
            (
                repo,
                commit,
                str(item.get("downstream_table", "")),
                str(item.get("downstream_column", "")),
                str(item.get("upstream_table", "")),
                str(item.get("upstream_column", "")),
                str(item.get("edge_type", edge_type)),
            )
        )
    if repo is None or commit is None:
        return 0
    replace_derived_lineage_columns(
        con,
        repo=repo,
        commit=commit,
        edge_type=edge_type,
        rows=rows,
    )
    return len(rows)


def _ingest_export_audit(bundle_root: Path, gateway: MinimalStorageGateway) -> int:
    path = bundle_root / "exports" / "export_audit.jsonl"
    if not path.is_file():
        return 0
    rows = [
        (
            str(item.get("dataset", "")),
            str(item.get("macro", "")),
            _optional_int(item.get("rows")),
            _optional_float(item.get("duration_s")) or 0.0,
            str(item.get("output_path", "")),
            _optional_str(item.get("sql")),
            _optional_str(item.get("plan")),
            _parse_datetime(item.get("created_at")) or datetime.now(tz=UTC),
        )
        for item in _iter_jsonl(path)
    ]
    if not rows:
        return 0
    return gateway.policy.bulk_insert(
        "metadata.export_audit",
        rows,
        columns=[
            "dataset",
            "macro",
            "rows",
            "duration_s",
            "output_path",
            "sql",
            "plan",
            "created_at",
        ],
        catalog=META_CATALOG_NAME,
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                yield payload


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_str(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _optional_mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return None


def _optional_column_stats(value: object) -> ColumnStatsPayload | None:
    mapping = _optional_mapping(value)
    if mapping is None:
        return None
    return cast("ColumnStatsPayload", mapping)


def _optional_dataset_stats(value: object) -> DatasetStatsPayload | None:
    mapping = _optional_mapping(value)
    if mapping is None:
        return None
    return cast("DatasetStatsPayload", mapping)


def _optional_derived_settings(value: object) -> DerivedSettingsPayload | None:
    mapping = _optional_mapping(value)
    if mapping is None:
        return None
    return cast("DerivedSettingsPayload", mapping)


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                return float(stripped)
            except ValueError:
                return None
    return None


__all__ = [
    "BundleIngestReport",
    "BundleManifest",
    "BundleValidation",
    "bundle_manifest_from_path",
    "load_build_metadata_bundle",
    "validate_build_metadata_bundle",
]
