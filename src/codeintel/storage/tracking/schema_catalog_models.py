"""Shared models for schema catalog persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from datetime import datetime

DEFAULT_SCHEMA_MANIFEST_KIND = "schema_manifest_v2"


class ColumnStatsEntry(TypedDict, total=False):
    """Typed payload for a single column statistics entry."""

    null_count: int
    non_null_count: int
    distinct_count_max: int
    min: object
    max: object
    avg_length: float


type ColumnStatsPayload = dict[str, ColumnStatsEntry]


type ParquetStatsPayload = Mapping[str, object]


class IcebergStatsPayload(TypedDict, total=False):
    """Typed payload for Iceberg metadata-derived statistics."""

    snapshot_id: int
    schema_id: int
    snapshot_count: int
    manifest_count: int
    data_file_count: int
    delete_file_count: int
    total_records: int
    total_bytes: int
    tombstone_rows: int
    tombstone_ratio: float
    deleted_rows: int


class DatasetStatsPayload(TypedDict, total=False):
    """Typed payload for dataset-level stats observations."""

    row_count: int
    batch_count: int
    total_bytes: int
    manifest_row_count: int
    parquet_stats: ParquetStatsPayload
    iceberg_stats: IcebergStatsPayload


class DerivedSettingsPayload(TypedDict, total=False):
    """Typed payload for derived dataset settings."""

    extras_policy: str
    dictionary_encode_columns: list[str]
    dictionary_max_cardinality: int
    unify_dictionaries: bool
    row_group_size: int
    data_page_size: int
    avg_row_bytes: float


@dataclass(frozen=True)
class SchemaVersionRecord:
    """Record of a content-addressed schema version."""

    schema_digest: str
    schema_hash: str
    schema_json: dict[str, object]
    renderer_cache: dict[str, object] | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class TableSchemaRegistryRecord:
    """Current schema pointer for a table key."""

    table_key: str
    schema_digest: str
    schema_hash: str
    derivation_kind: str
    derivation_source: str
    inference_status: str | None = None
    inference_error: str | None = None
    catalog_hash: str | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class SchemaManifestRunRecord:
    """Schema manifest catalog linkage for a build run."""

    run_id: str
    repo: str
    commit: str
    manifest_kind: str
    catalog_hash: str
    created_at: datetime | None = None


@dataclass(frozen=True)
class SchemaObservationRecord:
    """Observed schema payload persisted for inference tracking."""

    table_key: str
    schema_digest: str
    schema_hash: str
    arrow_schema_ipc_b64: str
    repo: str | None = None
    commit: str | None = None
    target_name: str | None = None
    column_stats: ColumnStatsPayload | None = None
    dataset_stats: DatasetStatsPayload | None = None
    derived_settings: DerivedSettingsPayload | None = None
    drift_summary: Mapping[str, object] | None = None
    observed_at: datetime | None = None
    observation_id: str | None = None


@dataclass(frozen=True)
class MaterializationValidationRecord:
    """Validation record persisted for materialized outputs."""

    validation_id: str
    table_key: str
    repo: str | None = None
    commit: str | None = None
    target_name: str | None = None
    output_role: str = "contract"
    validation_scope: str = "internal"
    validation_profile: str | None = None
    status: str = "skipped"
    issues: list[dict[str, object]] | None = None
    checks: Mapping[str, object] | None = None
    skipped_checks: Mapping[str, str] | None = None
    iceberg_snapshot_id: int | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class TableSchemaOverrideVersionRecord:
    """Record of a schema override version for an inferable table."""

    version_id: str
    table_key: str
    schema_digest: str
    schema_hash: str
    catalog_hash: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class TableSchemaOverrideRegistryRecord:
    """Current override pointer for an inferable table."""

    table_key: str
    schema_digest: str
    schema_hash: str
    version_id: str
    updated_at: datetime | None = None


@dataclass(frozen=True)
class OverrideRegistryRefreshResult:
    """Summary of an override registry refresh attempt."""

    status: str
    reason: str | None
    version_id: str | None
    tables: int
    schema_versions_rows: int
    override_versions_rows: int
    override_registry_rows: int


@dataclass(frozen=True)
class SchemaCatalogRequest:
    """Inputs for compiling or persisting a schema catalog."""

    run_id: str
    repo: str
    commit: str
    catalog_inputs: Mapping[str, object] | None = None
    include_views: bool = True
    strict_provenance: bool = True
    strict_hash_match: bool = True
    now: datetime | None = None
    catalog_kind: str = DEFAULT_SCHEMA_MANIFEST_KIND
    manifest_kind: str = DEFAULT_SCHEMA_MANIFEST_KIND


__all__ = [
    "DEFAULT_SCHEMA_MANIFEST_KIND",
    "ColumnStatsEntry",
    "ColumnStatsPayload",
    "DatasetStatsPayload",
    "DerivedSettingsPayload",
    "IcebergStatsPayload",
    "MaterializationValidationRecord",
    "OverrideRegistryRefreshResult",
    "ParquetStatsPayload",
    "SchemaCatalogRequest",
    "SchemaManifestRunRecord",
    "SchemaObservationRecord",
    "SchemaVersionRecord",
    "TableSchemaOverrideRegistryRecord",
    "TableSchemaOverrideVersionRecord",
    "TableSchemaRegistryRecord",
]
