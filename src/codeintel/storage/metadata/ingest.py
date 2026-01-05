"""Ingest build metadata bundles into DuckDB metadata tables."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.provider import MappingSchemaProvider
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
from codeintel.core.schemas.table_registry import TABLE_SCHEMAS
from codeintel.core.serialization.payload import encode_payload
from codeintel.core.time import utc_now
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
from codeintel.storage.metadata.meta_catalog import meta_table_ref
from codeintel.storage.metadata.sync import bootstrap_metadata_datasets
from codeintel.storage.tracking.schema_catalog import SchemaCatalogTracking
from codeintel.storage.upsert import UpsertSpec

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
    run_index_rows: int
    run_metadata_rows: int
    run_tag_summary_rows: int
    output_catalog_rows: int
    asset_versions_rows: int
    asset_version_events_rows: int
    run_asset_versions_rows: int
    asset_lineage_rows: int

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
            "run_index_rows": self.run_index_rows,
            "run_metadata_rows": self.run_metadata_rows,
            "run_tag_summary_rows": self.run_tag_summary_rows,
            "output_catalog_rows": self.output_catalog_rows,
            "asset_versions_rows": self.asset_versions_rows,
            "asset_version_events_rows": self.asset_version_events_rows,
            "run_asset_versions_rows": self.run_asset_versions_rows,
            "asset_lineage_rows": self.asset_lineage_rows,
        }


@dataclass(frozen=True, slots=True)
class _BundleIngestResults:
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
    run_index_rows: int
    run_metadata_rows: int
    run_tag_summary_rows: int
    output_catalog_rows: int
    asset_versions_rows: int
    asset_version_events_rows: int
    run_asset_versions_rows: int
    asset_lineage_rows: int


@dataclass(frozen=True, slots=True)
class _BundleIngestCounts:
    schema_versions_rows: int
    table_schema_registry_rows: int
    schema_observations_rows: int
    dataflow_nodes: int
    dataflow_edges: int
    lineage_edges: int
    lineage_columns: int
    export_audit_rows: int
    run_index_rows: int
    run_metadata_rows: int
    run_tag_summary_rows: int
    output_catalog_rows: int
    asset_versions_rows: int
    asset_version_events_rows: int
    run_asset_versions_rows: int
    asset_lineage_rows: int


@dataclass(slots=True)
class _RunReportRows:
    run_metadata_rows: list[tuple[object, ...]] = field(default_factory=list)
    tag_summary_rows: list[tuple[object, ...]] = field(default_factory=list)
    output_rows: list[tuple[object, ...]] = field(default_factory=list)
    run_ids: set[str] = field(default_factory=set)


_SUPPORTED_BUNDLE_SCHEMA_VERSIONS: frozenset[str] = frozenset({"v1"})
_REQUIRED_BUNDLE_FILES: tuple[str, ...] = (
    "contracts/contract_catalog.json",
    "contracts/contract_catalog.hash",
    "schema/schema_manifest.json",
    "schema/schema_registry.json",
    "schema/schema_versions.jsonl",
    "schema/schema_observations.jsonl",
    "dataflow/dataset_nodes.jsonl",
    "dataflow/dataset_edges.jsonl",
    "lineage/derived_edges.jsonl",
    "lineage/derived_columns.jsonl",
    "assets/asset_versions.jsonl",
    "assets/asset_version_events.jsonl",
    "assets/run_asset_versions.jsonl",
    "assets/asset_lineage.jsonl",
    "runs/run_index.jsonl",
    "exports/export_audit.jsonl",
)


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


def _manifest_relative_path(path: Path, bundle_root: Path) -> str:
    try:
        return path.relative_to(bundle_root).as_posix()
    except ValueError:
        return path.as_posix()


def _collect_manifest_paths(
    bundle_root: Path,
    manifest: BundleManifest,
    errors: list[str],
) -> set[str]:
    manifest_paths: set[str] = set()
    for entry in manifest.files:
        relative_path = _manifest_relative_path(entry.path, bundle_root)
        manifest_paths.add(relative_path)
        if not entry.path.is_file():
            errors.append(f"Missing bundle file: {entry.path}")
            continue
        if entry.size_bytes and entry.path.stat().st_size != entry.size_bytes:
            errors.append(f"Size mismatch for {entry.path}")
        if entry.sha256 and _sha256_path(entry.path) != entry.sha256:
            errors.append(f"Hash mismatch for {entry.path}")
    return manifest_paths


def _validate_required_files(
    *,
    bundle_root: Path,
    manifest_paths: set[str],
    errors: list[str],
) -> None:
    for required_path in _REQUIRED_BUNDLE_FILES:
        if required_path not in manifest_paths:
            errors.append(f"Missing bundle file in manifest: {required_path}")
        full_path = bundle_root / required_path
        if not full_path.is_file():
            errors.append(f"Missing required bundle file: {full_path}")


def _validate_run_report_paths(manifest_paths: set[str], errors: list[str]) -> None:
    has_run_reports = any(
        path.startswith("runs/run_report_") and path.endswith(".jsonl") for path in manifest_paths
    )
    if not has_run_reports:
        errors.append("Missing run report file in bundle (runs/run_report_<run_id>.jsonl)")


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

    if manifest.bundle_schema_version not in _SUPPORTED_BUNDLE_SCHEMA_VERSIONS:
        errors.append(
            f"Unsupported bundle_schema_version: {manifest.bundle_schema_version or '<missing>'}"
        )

    manifest_paths = _collect_manifest_paths(bundle_root, manifest, errors)
    _validate_required_files(
        bundle_root=bundle_root,
        manifest_paths=manifest_paths,
        errors=errors,
    )
    _validate_run_report_paths(manifest_paths, errors)
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

    results = _ingest_bundle_records(bundle_root, con, manifest)

    return BundleIngestReport(
        repo=manifest.repo,
        commit=manifest.commit,
        run_id=manifest.run_id,
        contract_catalog_hash=results.contract_catalog_hash,
        schema_manifest_hash=results.schema_manifest_hash,
        schema_versions_rows=results.schema_versions_rows,
        table_schema_registry_rows=results.table_schema_registry_rows,
        schema_observations_rows=results.schema_observations_rows,
        dataflow_nodes=results.dataflow_nodes,
        dataflow_edges=results.dataflow_edges,
        lineage_edges=results.lineage_edges,
        lineage_columns=results.lineage_columns,
        export_audit_rows=results.export_audit_rows,
        run_index_rows=results.run_index_rows,
        run_metadata_rows=results.run_metadata_rows,
        run_tag_summary_rows=results.run_tag_summary_rows,
        output_catalog_rows=results.output_catalog_rows,
        asset_versions_rows=results.asset_versions_rows,
        asset_version_events_rows=results.asset_version_events_rows,
        run_asset_versions_rows=results.run_asset_versions_rows,
        asset_lineage_rows=results.asset_lineage_rows,
    )


def _ingest_bundle_records(
    bundle_root: Path,
    con: DuckDBPyConnection,
    manifest: BundleManifest,
) -> _BundleIngestResults:
    apply_metadata_ddl(con, catalog=META_CATALOG_NAME, include_views=True)
    schema_provider = MappingSchemaProvider(TABLE_SCHEMAS)
    gateway = MinimalStorageGateway(con, schema_provider=schema_provider)
    tracker = SchemaCatalogTracking(gateway)

    contract_hash = _ingest_contract_catalog(bundle_root, con)
    schema_manifest_hash = _ingest_schema_manifest(bundle_root, con, manifest)

    load_contract_catalog_from_connection(con)
    bootstrap_metadata_datasets(con)

    counts = _ingest_bundle_payloads(
        bundle_root=bundle_root,
        con=con,
        gateway=gateway,
        tracker=tracker,
    )

    return _BundleIngestResults(
        contract_catalog_hash=contract_hash,
        schema_manifest_hash=schema_manifest_hash,
        schema_versions_rows=counts.schema_versions_rows,
        table_schema_registry_rows=counts.table_schema_registry_rows,
        schema_observations_rows=counts.schema_observations_rows,
        dataflow_nodes=counts.dataflow_nodes,
        dataflow_edges=counts.dataflow_edges,
        lineage_edges=counts.lineage_edges,
        lineage_columns=counts.lineage_columns,
        export_audit_rows=counts.export_audit_rows,
        run_index_rows=counts.run_index_rows,
        run_metadata_rows=counts.run_metadata_rows,
        run_tag_summary_rows=counts.run_tag_summary_rows,
        output_catalog_rows=counts.output_catalog_rows,
        asset_versions_rows=counts.asset_versions_rows,
        asset_version_events_rows=counts.asset_version_events_rows,
        run_asset_versions_rows=counts.run_asset_versions_rows,
        asset_lineage_rows=counts.asset_lineage_rows,
    )


def _ingest_bundle_payloads(
    *,
    bundle_root: Path,
    con: DuckDBPyConnection,
    gateway: MinimalStorageGateway,
    tracker: SchemaCatalogTracking,
) -> _BundleIngestCounts:
    registry_rows = _ingest_schema_registry(bundle_root, tracker)
    version_rows = _ingest_schema_versions(bundle_root, tracker)
    observation_rows = _ingest_schema_observations(bundle_root, tracker)

    dataflow_nodes = _ingest_dataflow_nodes(bundle_root, con)
    dataflow_edges = _ingest_dataflow_edges(bundle_root, con)
    lineage_edges = _ingest_lineage_edges(bundle_root, con)
    lineage_columns = _ingest_lineage_columns(bundle_root, con)
    export_audit_rows = _ingest_export_audit(bundle_root, gateway)
    run_index_rows = _ingest_run_index(bundle_root, gateway)
    run_report_rows = _ingest_run_reports(bundle_root, gateway)
    asset_versions_rows = _ingest_asset_versions(bundle_root, gateway)
    asset_version_events_rows = _ingest_asset_version_events(bundle_root, gateway)
    run_asset_versions_rows = _ingest_run_asset_versions(bundle_root, gateway)
    asset_lineage_rows = _ingest_asset_lineage(bundle_root, gateway)

    return _BundleIngestCounts(
        schema_versions_rows=version_rows,
        table_schema_registry_rows=registry_rows,
        schema_observations_rows=observation_rows,
        dataflow_nodes=dataflow_nodes,
        dataflow_edges=dataflow_edges,
        lineage_edges=lineage_edges,
        lineage_columns=lineage_columns,
        export_audit_rows=export_audit_rows,
        run_index_rows=run_index_rows,
        run_metadata_rows=run_report_rows[0],
        run_tag_summary_rows=run_report_rows[1],
        output_catalog_rows=run_report_rows[2],
        asset_versions_rows=asset_versions_rows,
        asset_version_events_rows=asset_version_events_rows,
        run_asset_versions_rows=run_asset_versions_rows,
        asset_lineage_rows=asset_lineage_rows,
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


def _ingest_run_index(bundle_root: Path, gateway: MinimalStorageGateway) -> int:
    path = bundle_root / "runs" / "run_index.jsonl"
    if not path.is_file():
        return 0
    rows: list[tuple[object, ...]] = []
    run_ids: set[str] = set()
    for item in _iter_jsonl(path):
        run_id = _optional_str(item.get("run_id"))
        if not run_id:
            continue
        run_ids.add(run_id)
        rows.append(
            (
                run_id,
                _optional_str(item.get("repo")),
                _optional_str(item.get("commit")),
                _parse_datetime(item.get("started_at")),
                _optional_float(item.get("duration_ms")),
                _optional_bool(item.get("success")),
                _optional_str(item.get("report_path")),
                _optional_int(item.get("computed_targets_count")),
                _optional_int(item.get("skipped_targets_count")),
                _optional_int(item.get("failed_targets_count")),
            )
        )
    if not rows:
        return 0
    _delete_run_ids(gateway.con, "metadata.build_run_index", run_ids)
    return gateway.policy.bulk_insert(
        "metadata.build_run_index",
        rows,
        columns=[
            "run_id",
            "repo",
            "commit",
            "started_at",
            "duration_ms",
            "success",
            "report_path",
            "computed_targets_count",
            "skipped_targets_count",
            "failed_targets_count",
        ],
        catalog=META_CATALOG_NAME,
    )


def _ingest_run_reports(
    bundle_root: Path,
    gateway: MinimalStorageGateway,
) -> tuple[int, int, int]:
    report_paths = _run_report_paths(bundle_root)
    if not report_paths:
        return 0, 0, 0

    rows = _collect_run_report_rows(report_paths)
    if rows.run_ids:
        _delete_run_ids(gateway.con, "metadata.build_run_metadata", rows.run_ids)
        _delete_run_ids(gateway.con, "metadata.build_run_tag_summary", rows.run_ids)
        _delete_run_ids(gateway.con, "metadata.build_output_catalog", rows.run_ids)
    if rows.run_metadata_rows:
        gateway.policy.bulk_insert(
            "metadata.build_run_metadata",
            rows.run_metadata_rows,
            columns=[
                "run_id",
                "repo",
                "commit",
                "snapshot_id",
                "started_at",
                "duration_ms",
                "success",
                "computed_targets",
                "skipped_targets",
                "failed_targets",
                "error_summary",
            ],
            catalog=META_CATALOG_NAME,
        )
    if rows.tag_summary_rows:
        gateway.policy.bulk_insert(
            "metadata.build_run_tag_summary",
            rows.tag_summary_rows,
            columns=[
                "run_id",
                "repo",
                "commit",
                "snapshot_id",
                "summary",
            ],
            catalog=META_CATALOG_NAME,
        )
    if rows.output_rows:
        gateway.policy.bulk_insert(
            "metadata.build_output_catalog",
            rows.output_rows,
            columns=[
                "run_id",
                "output_kind",
                "output_key",
                "table_key",
                "artifact_name",
                "artifact_type",
                "artifact_path",
                "target",
                "status",
                "row_count",
                "manifest_row_count",
                "schema_hash",
                "dataset_manifest_path",
                "output_role",
                "saver_node",
                "sink",
                "tags",
                "repo",
                "commit",
                "snapshot_id",
            ],
            catalog=META_CATALOG_NAME,
        )
    return len(rows.run_metadata_rows), len(rows.tag_summary_rows), len(rows.output_rows)


def _ingest_asset_versions(bundle_root: Path, gateway: MinimalStorageGateway) -> int:
    path = bundle_root / "assets" / "asset_versions.jsonl"
    if not path.is_file():
        return 0
    rows: list[tuple[object, ...]] = []
    for item in _iter_jsonl(path):
        asset_kind = _optional_str(item.get("asset_kind"))
        asset_key = _optional_str(item.get("asset_key"))
        version_hash = _optional_str(item.get("version_hash"))
        if not (asset_kind and asset_key and version_hash):
            continue
        rows.append(
            (
                asset_kind,
                asset_key,
                version_hash,
                _optional_str(item.get("schema_hash")),
                _optional_int(item.get("row_count")),
                _optional_int(item.get("bytes")),
                _parse_datetime(item.get("created_at")) or utc_now(),
                encode_payload(_optional_mapping(item.get("meta")) or {}),
            )
        )
    if not rows:
        return 0
    gateway.policy.ensure_table("build.asset_versions", create_if_missing=True)
    return gateway.policy.upsert(
        "build.asset_versions",
        rows,
        columns=(
            "asset_kind",
            "asset_key",
            "version_hash",
            "schema_hash",
            "row_count",
            "bytes",
            "created_at",
            "meta",
        ),
        upsert=UpsertSpec(
            conflict_columns=("asset_kind", "asset_key", "version_hash"),
            update_columns=("schema_hash", "row_count", "bytes", "created_at", "meta"),
        ),
    )


def _ingest_asset_version_events(bundle_root: Path, gateway: MinimalStorageGateway) -> int:
    path = bundle_root / "assets" / "asset_version_events.jsonl"
    if not path.is_file():
        return 0
    rows: list[tuple[object, ...]] = []
    for item in _iter_jsonl(path):
        run_id = _optional_str(item.get("run_id"))
        repo = _optional_str(item.get("repo"))
        commit = _optional_str(item.get("commit"))
        asset_kind = _optional_str(item.get("asset_kind"))
        asset_key = _optional_str(item.get("asset_key"))
        version_hash = _optional_str(item.get("version_hash"))
        status = _optional_str(item.get("status"))
        if not (
            run_id and repo and commit and asset_kind and asset_key and version_hash and status
        ):
            continue
        rows.append(
            (
                run_id,
                repo,
                commit,
                asset_kind,
                asset_key,
                version_hash,
                _optional_str(item.get("target")),
                _optional_str(item.get("impl_kind")),
                status,
                _optional_str(item.get("location")),
                _optional_str(item.get("input_hash")),
                _optional_str(item.get("options_hash")),
                _parse_datetime(item.get("recorded_at")) or utc_now(),
                encode_payload(_optional_mapping(item.get("meta")) or {}),
            )
        )
    if not rows:
        return 0
    gateway.policy.ensure_table("build.asset_version_events", create_if_missing=True)
    return gateway.policy.upsert(
        "build.asset_version_events",
        rows,
        columns=(
            "run_id",
            "repo",
            "commit",
            "asset_kind",
            "asset_key",
            "version_hash",
            "target",
            "impl_kind",
            "status",
            "location",
            "input_hash",
            "options_hash",
            "recorded_at",
            "meta",
        ),
        upsert=UpsertSpec(
            conflict_columns=("run_id", "asset_kind", "asset_key"),
            update_columns=(
                "repo",
                "commit",
                "version_hash",
                "target",
                "impl_kind",
                "status",
                "location",
                "input_hash",
                "options_hash",
                "recorded_at",
                "meta",
            ),
        ),
    )


def _ingest_run_asset_versions(bundle_root: Path, gateway: MinimalStorageGateway) -> int:
    path = bundle_root / "assets" / "run_asset_versions.jsonl"
    if not path.is_file():
        return 0
    rows: list[tuple[object, ...]] = []
    for item in _iter_jsonl(path):
        run_id = _optional_str(item.get("run_id"))
        repo = _optional_str(item.get("repo"))
        commit = _optional_str(item.get("commit"))
        asset_kind = _optional_str(item.get("asset_kind"))
        asset_key = _optional_str(item.get("asset_key"))
        version_hash = _optional_str(item.get("version_hash"))
        resolution_kind = _optional_str(item.get("resolution_kind"))
        if not (
            run_id
            and repo
            and commit
            and asset_kind
            and asset_key
            and version_hash
            and resolution_kind
        ):
            continue
        rows.append(
            (
                run_id,
                repo,
                commit,
                asset_kind,
                asset_key,
                version_hash,
                _optional_str(item.get("target")),
                resolution_kind,
                _parse_datetime(item.get("recorded_at")) or utc_now(),
                encode_payload(_optional_mapping(item.get("meta")) or {}),
            )
        )
    if not rows:
        return 0
    gateway.policy.ensure_table("build.run_asset_versions", create_if_missing=True)
    return gateway.policy.upsert(
        "build.run_asset_versions",
        rows,
        columns=(
            "run_id",
            "repo",
            "commit",
            "asset_kind",
            "asset_key",
            "version_hash",
            "target",
            "resolution_kind",
            "recorded_at",
            "meta",
        ),
        upsert=UpsertSpec(
            conflict_columns=("run_id", "asset_kind", "asset_key"),
            update_columns=("version_hash", "target", "resolution_kind", "recorded_at", "meta"),
        ),
    )


def _ingest_asset_lineage(bundle_root: Path, gateway: MinimalStorageGateway) -> int:
    path = bundle_root / "assets" / "asset_lineage.jsonl"
    if not path.is_file():
        return 0
    rows: list[tuple[object, ...]] = []
    for item in _iter_jsonl(path):
        downstream_kind = _optional_str(item.get("downstream_kind"))
        downstream_key = _optional_str(item.get("downstream_key"))
        downstream_version = _optional_str(item.get("downstream_version"))
        upstream_kind = _optional_str(item.get("upstream_kind"))
        upstream_key = _optional_str(item.get("upstream_key"))
        upstream_version = _optional_str(item.get("upstream_version"))
        edge_kind = _optional_str(item.get("edge_kind"))
        if not (
            downstream_kind
            and downstream_key
            and downstream_version
            and upstream_kind
            and upstream_key
            and upstream_version
            and edge_kind
        ):
            continue
        rows.append(
            (
                downstream_kind,
                downstream_key,
                downstream_version,
                upstream_kind,
                upstream_key,
                upstream_version,
                edge_kind,
                _parse_datetime(item.get("created_at")) or utc_now(),
                encode_payload(_optional_mapping(item.get("meta")) or {}),
            )
        )
    if not rows:
        return 0
    gateway.policy.ensure_table("build.asset_lineage", create_if_missing=True)
    return gateway.policy.upsert(
        "build.asset_lineage",
        rows,
        columns=(
            "downstream_kind",
            "downstream_key",
            "downstream_version",
            "upstream_kind",
            "upstream_key",
            "upstream_version",
            "edge_kind",
            "created_at",
            "meta",
        ),
        upsert=UpsertSpec(
            conflict_columns=(
                "downstream_kind",
                "downstream_key",
                "downstream_version",
                "upstream_kind",
                "upstream_key",
                "upstream_version",
                "edge_kind",
            ),
            update_columns=("created_at", "meta"),
        ),
    )


def _run_report_paths(bundle_root: Path) -> list[Path]:
    runs_root = bundle_root / "runs"
    if not runs_root.is_dir():
        return []
    return sorted(runs_root.glob("run_report_*.jsonl"))


def _collect_run_report_rows(report_paths: list[Path]) -> _RunReportRows:
    rows = _RunReportRows()
    row_targets = {
        "run_metadata": (rows.run_metadata_rows, _run_metadata_row),
        "tag_schema_summary": (rows.tag_summary_rows, _run_tag_summary_row),
        "output_catalog": (rows.output_rows, _run_output_row),
    }
    for path in report_paths:
        for item in _iter_jsonl(path):
            record_type = _optional_str(item.get("record_type"))
            if not record_type:
                continue
            target = row_targets.get(record_type)
            if target is None:
                continue
            target_rows, parser = target
            row = parser(item)
            if row is None:
                continue
            rows.run_ids.add(str(row[0]))
            target_rows.append(row)
    return rows


def _delete_run_ids(
    con: DuckDBPyConnection,
    table_key: str,
    run_ids: set[str],
) -> None:
    if not run_ids:
        return
    placeholders = ", ".join("?" for _ in run_ids)
    table_ref = meta_table_ref(table_key)
    con.execute(
        f"DELETE FROM {table_ref} WHERE run_id IN ({placeholders})",
        sorted(run_ids),
    )


def _run_metadata_row(item: Mapping[str, object]) -> tuple[object, ...] | None:
    run_id = _optional_str(item.get("run_id"))
    if not run_id:
        return None
    return (
        run_id,
        _optional_str(item.get("repo")),
        _optional_str(item.get("commit")),
        _optional_str(item.get("snapshot_id")),
        _parse_datetime(item.get("started_at")),
        _optional_float(item.get("duration_ms")),
        _optional_bool(item.get("success")),
        _optional_json(item.get("computed_targets")),
        _optional_json(item.get("skipped_targets")),
        _optional_json(item.get("failed_targets")),
        _optional_str(item.get("error_summary")),
    )


def _run_tag_summary_row(item: Mapping[str, object]) -> tuple[object, ...] | None:
    run_id = _optional_str(item.get("run_id"))
    summary = _optional_json(item.get("summary"))
    if not run_id or summary is None:
        return None
    return (
        run_id,
        _optional_str(item.get("repo")),
        _optional_str(item.get("commit")),
        _optional_str(item.get("snapshot_id")),
        summary,
    )


def _run_output_row(item: Mapping[str, object]) -> tuple[object, ...] | None:
    run_id = _optional_str(item.get("run_id"))
    output_kind = _optional_str(item.get("output_kind"))
    target = _optional_str(item.get("target"))
    status = _optional_str(item.get("status"))
    if not run_id or not output_kind or not target or not status:
        return None
    table_key = _optional_str(item.get("table_key"))
    artifact_name = _optional_str(item.get("artifact_name"))
    output_key = table_key if output_kind == "table" else artifact_name
    if not output_key:
        return None
    return (
        run_id,
        output_kind,
        output_key,
        table_key,
        artifact_name,
        _optional_str(item.get("artifact_type")),
        _optional_str(item.get("artifact_path")),
        target,
        status,
        _optional_int(item.get("row_count")),
        _optional_int(item.get("manifest_row_count")),
        _optional_str(item.get("schema_hash")),
        _optional_str(item.get("dataset_manifest_path")),
        _optional_str(item.get("output_role")),
        _optional_str(item.get("saver_node")),
        _optional_str(item.get("sink")),
        _optional_json(item.get("tags")),
        _optional_str(item.get("repo")),
        _optional_str(item.get("commit")),
        _optional_str(item.get("snapshot_id")),
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


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized == "true"
    return None


def _optional_json(value: object) -> str | None:
    result: str | None = None
    if value is None:
        result = None
    elif isinstance(value, (str, bytes, bytearray, memoryview)):
        if isinstance(value, memoryview):
            text = value.tobytes().decode("utf-8")
        elif isinstance(value, (bytes, bytearray)):
            text = value.decode("utf-8")
        else:
            text = value
        result = text if text.strip() else None
    elif isinstance(value, Mapping):
        try:
            result = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError):
            result = None
    elif isinstance(value, (list, tuple)):
        try:
            result = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        except (TypeError, ValueError):
            result = None
    return result


__all__ = [
    "BundleIngestReport",
    "BundleManifest",
    "BundleValidation",
    "bundle_manifest_from_path",
    "load_build_metadata_bundle",
    "validate_build_metadata_bundle",
]
