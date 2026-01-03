"""Core gateway protocols for build-time access."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, Self

from codeintel.core.duckdb_types import DuckDBConnection, DuckDBRelation
from codeintel.core.ports.export import ExportRelation

if TYPE_CHECKING:
    from codeintel.core.build_manifest import BuildRunRecord, BuildStatus, OutputManifest
    from codeintel.core.config.settings import ExportAuditSettings
    from codeintel.core.hamilton.records import NodeExecutionRecord, TargetRunRecord
    from codeintel.core.manifests import SchemaManifest
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider
    from codeintel.core.schemas.schema_catalog_models import (
        OverrideRegistryRefreshResult,
        SchemaCatalogRequest,
        SchemaObservationRecord,
        SchemaVersionRecord,
        TableSchemaRegistryRecord,
    )


class GatewayConfig(Protocol):
    """Protocol for gateway configuration access."""

    @property
    def read_only(self) -> bool:
        """Return whether the gateway is read-only."""
        ...

    @property
    def db_path(self) -> str | Path:
        """Return the configured database path."""
        ...

    @property
    def dataset_root_dir(self) -> Path | None:
        """Return the dataset root directory when configured."""
        ...


class DatasetRegistryProtocol(Protocol):
    """Protocol for dataset registry access."""

    @property
    def by_name(self) -> Mapping[str, DatasetContract]:
        """Mapping of dataset name to contract."""
        ...

    @property
    def by_table_key(self) -> Mapping[str, DatasetContract]:
        """Mapping of table key to contract."""
        ...

    @property
    def jsonl_datasets(self) -> Mapping[str, str]:
        """Mapping of dataset name to JSONL filename."""
        ...

    @property
    def parquet_datasets(self) -> Mapping[str, str]:
        """Mapping of dataset name to Parquet filename."""
        ...

    @property
    def dataset_root_dir(self) -> Path | None:
        """Root directory for dataset files, if configured."""
        ...

    def with_dataset_root(self, dataset_root_dir: Path | None) -> Self:
        """Return a registry with the dataset root applied."""
        ...

    def resolve_table_key(self, name: str) -> str:
        """Resolve dataset name into a fully qualified table key."""
        ...


ModuleKind = Literal["ingestion", "graphs", "analytics", "export", "views", "build"]
StepStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]


class PipelineStepRecordProtocol(Protocol):
    """Record attributes needed for pipeline step tracking."""

    @property
    def run_id(self) -> str:
        """Parent run identifier."""
        ...

    @property
    def module(self) -> ModuleKind:
        """Module name for the step."""
        ...

    @property
    def stage(self) -> str:
        """Stage name for the step."""
        ...

    @property
    def name(self) -> str:
        """Step name."""
        ...

    @property
    def status(self) -> StepStatus:
        """Step status."""
        ...

    @property
    def started_at(self) -> datetime:
        """Step start timestamp."""
        ...

    @property
    def completed_at(self) -> datetime | None:
        """Step completion timestamp."""
        ...

    @property
    def row_counts(self) -> Mapping[str, object] | None:
        """Row counts recorded for the step."""
        ...

    @property
    def extra(self) -> Mapping[str, object] | None:
        """Additional step metadata."""
        ...


class AssetLineageEdgeProtocol(Protocol):
    """Minimal lineage edge attributes for impact analysis."""

    @property
    def downstream_kind(self) -> str:
        """Downstream asset kind."""
        ...

    @property
    def downstream_key(self) -> str:
        """Downstream asset key."""
        ...

    @property
    def downstream_version(self) -> str:
        """Downstream asset version hash."""
        ...


class AssetVersionRecordProtocol(Protocol):
    """Minimal asset version record fields for persistence."""

    @property
    def asset_kind(self) -> str:
        """Asset kind identifier."""
        ...

    @property
    def asset_key(self) -> str:
        """Asset key identifier."""
        ...

    @property
    def version_hash(self) -> str:
        """Content hash for the asset version."""
        ...

    @property
    def schema_hash(self) -> str | None:
        """Schema hash for the asset version."""
        ...

    @property
    def row_count(self) -> int | None:
        """Row count for the asset version."""
        ...

    @property
    def bytes(self) -> int | None:
        """Byte size for the asset version."""
        ...

    @property
    def created_at(self) -> datetime | None:
        """Creation timestamp for the asset version."""
        ...

    @property
    def meta(self) -> Mapping[str, object] | None:
        """Optional metadata for the asset version."""
        ...


class AssetVersionEventRecordProtocol(Protocol):
    """Minimal asset version event fields for persistence."""

    @property
    def run_id(self) -> str:
        """Run identifier."""
        ...

    @property
    def repo(self) -> str:
        """Repository identifier."""
        ...

    @property
    def commit(self) -> str:
        """Commit identifier."""
        ...

    @property
    def asset_kind(self) -> str:
        """Asset kind identifier."""
        ...

    @property
    def asset_key(self) -> str:
        """Asset key identifier."""
        ...

    @property
    def version_hash(self) -> str:
        """Asset version hash."""
        ...

    @property
    def target(self) -> str | None:
        """Optional target name."""
        ...

    @property
    def impl_kind(self) -> str | None:
        """Optional implementation kind."""
        ...

    @property
    def status(self) -> str:
        """Event status."""
        ...

    @property
    def location(self) -> str | None:
        """Optional location information."""
        ...

    @property
    def input_hash(self) -> str | None:
        """Optional input hash."""
        ...

    @property
    def options_hash(self) -> str | None:
        """Optional options hash."""
        ...

    @property
    def recorded_at(self) -> datetime | None:
        """Event timestamp."""
        ...

    @property
    def meta(self) -> Mapping[str, object] | None:
        """Optional event metadata."""
        ...


class RunAssetVersionRecordProtocol(Protocol):
    """Minimal run-to-asset version linkage fields."""

    @property
    def run_id(self) -> str:
        """Run identifier."""
        ...

    @property
    def repo(self) -> str:
        """Repository identifier."""
        ...

    @property
    def commit(self) -> str:
        """Commit identifier."""
        ...

    @property
    def asset_kind(self) -> str:
        """Asset kind identifier."""
        ...

    @property
    def asset_key(self) -> str:
        """Asset key identifier."""
        ...

    @property
    def version_hash(self) -> str:
        """Asset version hash."""
        ...

    @property
    def resolution_kind(self) -> str:
        """Resolution kind label."""
        ...

    @property
    def recorded_at(self) -> datetime | None:
        """Recorded timestamp."""
        ...

    @property
    def target(self) -> str | None:
        """Optional target name."""
        ...

    @property
    def meta(self) -> Mapping[str, object] | None:
        """Optional linkage metadata."""
        ...


class AssetLineageEdgeRecordProtocol(Protocol):
    """Minimal lineage edge fields for persistence."""

    @property
    def downstream_kind(self) -> str:
        """Downstream asset kind."""
        ...

    @property
    def downstream_key(self) -> str:
        """Downstream asset key."""
        ...

    @property
    def downstream_version(self) -> str:
        """Downstream asset version hash."""
        ...

    @property
    def upstream_kind(self) -> str:
        """Upstream asset kind."""
        ...

    @property
    def upstream_key(self) -> str:
        """Upstream asset key."""
        ...

    @property
    def upstream_version(self) -> str:
        """Upstream asset version hash."""
        ...

    @property
    def edge_kind(self) -> str:
        """Edge kind label."""
        ...

    @property
    def created_at(self) -> datetime | None:
        """Edge creation timestamp."""
        ...

    @property
    def meta(self) -> Mapping[str, object] | None:
        """Optional edge metadata."""
        ...


class RunEnvironmentRecordProtocol(Protocol):
    """Minimal run environment fields for persistence."""

    @property
    def run_id(self) -> str:
        """Run identifier."""
        ...

    @property
    def python_version(self) -> str:
        """Python version string."""
        ...

    @property
    def os_name(self) -> str:
        """Operating system name."""
        ...

    @property
    def os_version(self) -> str:
        """Operating system version."""
        ...

    @property
    def tool_versions(self) -> Mapping[str, str] | None:
        """Optional tool version mapping."""
        ...

    @property
    def config_hash(self) -> str | None:
        """Optional configuration hash."""
        ...

    @property
    def git_dirty(self) -> bool:
        """Whether git state was dirty."""
        ...

    @property
    def captured_at(self) -> datetime | None:
        """Captured timestamp."""
        ...


class ScipRunIdentityProtocol(Protocol):
    """Identity fields for SCIP telemetry."""

    @property
    def run_id(self) -> str:
        """Run identifier."""
        ...

    @property
    def repo(self) -> str:
        """Repository identifier."""
        ...

    @property
    def commit(self) -> str:
        """Commit identifier."""
        ...

    @property
    def mode(self) -> str:
        """SCIP mode."""
        ...

    @property
    def options_hash(self) -> str | None:
        """Options hash."""
        ...

    @property
    def tool_version(self) -> str | None:
        """Tool version string."""
        ...


class ScipRunCountsProtocol(Protocol):
    """Count fields for SCIP telemetry."""

    @property
    def total_modules(self) -> int:
        """Total module count."""
        ...

    @property
    def changed_modules(self) -> int:
        """Changed module count."""
        ...

    @property
    def deleted_modules(self) -> int:
        """Deleted module count."""
        ...

    @property
    def changed_ratio(self) -> float | None:
        """Changed ratio."""
        ...

    @property
    def batch_size(self) -> int | None:
        """Batch size."""
        ...

    @property
    def batch_count(self) -> int:
        """Batch count."""
        ...

    @property
    def decision(self) -> str | None:
        """Decision label."""
        ...

    @property
    def ratio_gate_applied(self) -> bool | None:
        """Whether ratio gate was applied."""
        ...

    @property
    def ratio_gate_min_modules(self) -> int | None:
        """Minimum modules for ratio gate."""
        ...

    @property
    def ratio_gate_min_changed(self) -> int | None:
        """Minimum changed modules for ratio gate."""
        ...


class ScipRunHashStatsProtocol(Protocol):
    """Hashing fields for SCIP telemetry."""

    @property
    def hash_source(self) -> str | None:
        """Hash source label."""
        ...

    @property
    def hash_source_breakdown(self) -> str | None:
        """Hash source breakdown."""
        ...

    @property
    def hash_reused(self) -> int:
        """Reused hash count."""
        ...

    @property
    def hash_computed(self) -> int:
        """Computed hash count."""
        ...


class ScipRunTimingProtocol(Protocol):
    """Timing fields for SCIP telemetry."""

    @property
    def plan_ms(self) -> float | None:
        """Planning duration in ms."""
        ...

    @property
    def hash_ms(self) -> float | None:
        """Hashing duration in ms."""
        ...

    @property
    def tool_ms(self) -> float | None:
        """Tool execution duration in ms."""
        ...

    @property
    def parse_ms(self) -> float | None:
        """Parsing duration in ms."""
        ...

    @property
    def merge_ms(self) -> float | None:
        """Merge duration in ms."""
        ...

    @property
    def write_ms(self) -> float | None:
        """Write duration in ms."""
        ...

    @property
    def total_ms(self) -> float | None:
        """Total duration in ms."""
        ...


class ScipRunOutcomeProtocol(Protocol):
    """Outcome fields for SCIP telemetry."""

    @property
    def status(self) -> str:
        """Run status."""
        ...

    @property
    def error_summary(self) -> str | None:
        """Error summary."""
        ...

    @property
    def output_scip(self) -> str | None:
        """Output SCIP path."""
        ...

    @property
    def recorded_at(self) -> datetime:
        """Recorded timestamp."""
        ...


class ScipRunRecordProtocol(
    ScipRunIdentityProtocol,
    ScipRunCountsProtocol,
    ScipRunHashStatsProtocol,
    ScipRunTimingProtocol,
    ScipRunOutcomeProtocol,
    Protocol,
):
    """Minimal SCIP telemetry fields for persistence."""


class ExportAuditRecordProtocol(Protocol):
    """Export audit record attributes for logging."""

    @property
    def table_name(self) -> str:
        """Table name for the export."""
        ...

    @property
    def macro(self) -> str:
        """Macro name for the export."""
        ...

    @property
    def rows(self) -> int | None:
        """Row count for the export."""
        ...

    @property
    def duration_s(self) -> float:
        """Duration in seconds for the export."""
        ...

    @property
    def output_path(self) -> Path:
        """Output path for the export."""
        ...


class PersistSchemaManifestResultProtocol(Protocol):
    """Schema manifest persistence result attributes."""

    @property
    def catalog_hash(self) -> str:
        """Catalog hash for the manifest."""
        ...

    @property
    def tables(self) -> int:
        """Number of tables in the manifest."""
        ...

    @property
    def views(self) -> int:
        """Number of views in the manifest."""
        ...


class SchemaIndexProtocol(Protocol):
    """Minimal schema index interface for prefill operations."""

    def prefill_cache(self, schemas: Mapping[str, TableSchema]) -> None:
        """Prefill the schema index cache."""
        ...

    @property
    def inferable_table_keys(self) -> AbstractSet[str]:
        """Return inferable table keys."""
        ...


class GatewayPolicy(Protocol):
    """Protocol for policy-backed operations."""

    schema_provider: SchemaProvider | None

    def ensure_schemas_preserve(self) -> None:
        """Ensure required schemas exist without dropping tables."""
        ...

    def ensure_table(self, table_key: str, *, create_if_missing: bool = True) -> None:
        """Ensure the dataset table exists in storage."""
        ...

    def delete_for_snapshot(self, table_key: str, *, repo: str, commit: str) -> None:
        """Delete rows for a repo/commit snapshot."""
        ...

    def drop_table(
        self,
        table_key: str,
        *,
        if_exists: bool = True,
        catalog: str | None = None,
    ) -> None:
        """Drop a table by table key."""
        ...

    def bulk_insert(
        self,
        table_key: str,
        rows: Sequence[tuple[object, ...]],
        *,
        columns: Sequence[str] | None = None,
        catalog: str | None = None,
    ) -> int:
        """Insert row tuples with stable column order."""
        ...

    def bulk_insert_mappings(
        self,
        table_key: str,
        rows: Iterable[Mapping[str, object]],
        *,
        columns: Sequence[str] | None = None,
        catalog: str | None = None,
    ) -> int:
        """Insert mapping rows with stable column order."""
        ...

    def table_exists(self, *, schema: str, table: str) -> bool:
        """Return True when a table exists in the underlying catalog."""
        ...


class GatewaySchemas(Protocol):
    """Protocol for schema tracking access."""

    def persist_schema_manifest(
        self,
        manifest: SchemaManifest,
        *,
        request: SchemaCatalogRequest,
    ) -> PersistSchemaManifestResultProtocol:
        """Persist a schema manifest record."""
        ...

    def refresh_override_registry_from_manifest(
        self,
        manifest: SchemaManifest,
        *,
        request: SchemaCatalogRequest,
        catalog_hash: str | None = None,
    ) -> OverrideRegistryRefreshResult:
        """Refresh overrides based on a schema manifest."""
        ...

    def record_schema_versions_batch(self, records: Sequence[SchemaVersionRecord]) -> int:
        """Persist schema version records in batch."""
        ...

    def record_table_schema_registry_batch(
        self, records: Sequence[TableSchemaRegistryRecord]
    ) -> int:
        """Persist table schema registry records in batch."""
        ...

    def record_schema_observations_batch(self, records: Sequence[SchemaObservationRecord]) -> int:
        """Persist schema observation records in batch."""
        ...

    def load_override_registry(self) -> dict[str, TableSchema]:
        """Return the current override schema registry."""
        ...

    def prefill_schema_index(
        self,
        schema_index: SchemaIndexProtocol,
        *,
        table_keys: Sequence[str] | None = None,
    ) -> int:
        """Prefill a schema index with persisted inference data."""
        ...

    def load_latest_schema_observation(
        self,
        *,
        table_key: str,
    ) -> SchemaObservationRecord | None:
        """Return the most recent schema observation for a table."""
        ...

    def load_recent_drift_summaries(
        self,
        *,
        table_key: str,
        limit: int = 5,
    ) -> Sequence[Mapping[str, object] | None]:
        """Return recent schema drift summaries for a table."""
        ...


class GatewayAssets(Protocol):
    """Protocol for asset tracking access."""

    def record_run_environment(self, record: RunEnvironmentRecordProtocol) -> None:
        """Persist run environment telemetry."""
        ...

    def record_asset_versions_batch(self, records: Sequence[AssetVersionRecordProtocol]) -> int:
        """Persist asset version records in batch."""
        ...

    def record_asset_version_events_batch(
        self,
        records: Sequence[AssetVersionEventRecordProtocol],
    ) -> int:
        """Persist asset version event records in batch."""
        ...

    def record_run_asset_versions_batch(
        self,
        records: Sequence[RunAssetVersionRecordProtocol],
    ) -> int:
        """Persist run asset version records in batch."""
        ...

    def record_lineage_edges_batch(
        self,
        edges: Sequence[AssetLineageEdgeRecordProtocol],
    ) -> int:
        """Persist lineage edge records in batch."""
        ...

    def get_asset_target(self, asset_kind: str, asset_key: str) -> str | None:
        """Return the target that produced an asset."""
        ...

    def get_downstream_edges(
        self,
        *,
        upstream_kind: str,
        upstream_key: str,
        upstream_version: str | None,
    ) -> Sequence[AssetLineageEdgeProtocol]:
        """Return downstream lineage edges for an asset."""
        ...


class GatewayBuild(Protocol):
    """Protocol for build tracking access."""

    def start_run(self, record: BuildRunRecord) -> None:
        """Record a new build run."""
        ...

    def complete_run(
        self,
        run_id: str,
        status: BuildStatus,
        computed_targets: tuple[str, ...],
        skipped_targets: tuple[str, ...],
        error_summary: str | None = None,
    ) -> None:
        """Mark a build run as completed."""
        ...

    def save_run_targets(
        self,
        run_id: str,
        repo: str,
        commit: str,
        records: Sequence[TargetRunRecord],
    ) -> int:
        """Persist target records for a run."""
        ...

    def save_run_nodes(self, run_id: str, records: Sequence[NodeExecutionRecord]) -> int:
        """Persist node records for a run."""
        ...

    def save_manifest(self, manifest: OutputManifest) -> None:
        """Persist a build manifest payload."""
        ...

    def record_scip_run(self, record: ScipRunRecordProtocol) -> None:
        """Record a SCIP execution run."""
        ...


class GatewayRuns(Protocol):
    """Protocol for pipeline run tracking."""

    def record_step(self, record: PipelineStepRecordProtocol) -> None:
        """Persist a pipeline step record."""
        ...

    def fetch_steps(self, run_id: str) -> Sequence[PipelineStepRecordProtocol]:
        """Return pipeline step records for a run."""
        ...


class GatewayExports(Protocol):
    """Protocol for export auditing and relation building."""

    def audit_enabled(self, settings: ExportAuditSettings) -> bool:
        """Return True when export audit logging is enabled."""
        ...

    def write_export_audit(
        self,
        record: ExportAuditRecordProtocol,
        *,
        settings: ExportAuditSettings,
        sql: str | None = None,
        plan: str | None = None,
    ) -> None:
        """Persist an export audit record."""
        ...

    def build_export_relation(self, *, relation: DuckDBRelation) -> ExportRelation:
        """Return a relation wrapper for export output."""
        ...


class BuildGateway(Protocol):
    """Protocol for build-time gateway dependencies."""

    @property
    def assets(self) -> GatewayAssets:
        """Asset tracking accessor."""
        ...

    @property
    def build(self) -> GatewayBuild:
        """Build tracking accessor."""
        ...

    @property
    def config(self) -> GatewayConfig:
        """Gateway configuration."""
        ...

    @property
    def datasets(self) -> DatasetRegistryProtocol:
        """Dataset registry accessor."""
        ...

    @property
    def exports(self) -> GatewayExports:
        """Export service accessor."""
        ...

    @property
    def policy(self) -> GatewayPolicy:
        """Policy backend accessor."""
        ...

    @property
    def runs(self) -> GatewayRuns:
        """Pipeline run tracking accessor."""
        ...

    @property
    def schemas(self) -> GatewaySchemas:
        """Schema tracking accessor."""
        ...

    @property
    def con(self) -> DuckDBConnection:
        """Return the underlying DuckDB connection."""
        ...

    def execute(
        self,
        sql: str,
        params: Sequence[object] | Mapping[str, object] | None = None,
    ) -> DuckDBConnection:
        """Execute SQL against the underlying connection."""
        ...

    def register(self, name: str, obj: object) -> None:
        """Register an object with the underlying connection."""
        ...

    def unregister(self, name: str) -> None:
        """Unregister a previously registered object."""
        ...

    def relation_from_table_key(self, table_key: str) -> DuckDBRelation:
        """Return a DuckDB relation for a table key."""
        ...

    def close(self) -> None:
        """Close the underlying connection when available."""
        ...


__all__ = [
    "AssetLineageEdgeProtocol",
    "AssetLineageEdgeRecordProtocol",
    "AssetVersionEventRecordProtocol",
    "AssetVersionRecordProtocol",
    "BuildGateway",
    "DatasetRegistryProtocol",
    "GatewayAssets",
    "GatewayBuild",
    "GatewayConfig",
    "GatewayExports",
    "GatewayPolicy",
    "GatewayRuns",
    "GatewaySchemas",
    "RunAssetVersionRecordProtocol",
    "RunEnvironmentRecordProtocol",
    "ScipRunRecordProtocol",
]
