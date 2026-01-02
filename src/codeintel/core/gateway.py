"""Core gateway protocols for build-time access."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self

from codeintel.core.duckdb_types import DuckDBConnection, DuckDBRelation

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
    from codeintel.storage.exports.service import ExportAuditRecord
    from codeintel.storage.protocols import ExportRelation
    from codeintel.storage.tracking import ModuleKind, PipelineStepRecord, StepStatus
    from codeintel.storage.tracking.asset_tracking import (
        AssetLineageEdgeRecord,
        AssetVersionEventRecord,
        AssetVersionRecord,
        RunAssetVersionRecord,
        RunEnvironmentRecord,
    )
    from codeintel.storage.tracking.build_tracking import ScipRunRecord
    from codeintel.storage.tracking.schema_catalog import PersistSchemaManifestResult, SchemaIndex


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
    ) -> PersistSchemaManifestResult:
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
        schema_index: SchemaIndex,
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

    def record_run_environment(self, record: RunEnvironmentRecord) -> None:
        """Persist run environment telemetry."""
        ...

    def record_asset_versions_batch(self, records: Sequence[AssetVersionRecord]) -> int:
        """Persist asset version records in batch."""
        ...

    def record_asset_version_events_batch(self, records: Sequence[AssetVersionEventRecord]) -> int:
        """Persist asset version event records in batch."""
        ...

    def record_run_asset_versions_batch(self, records: Sequence[RunAssetVersionRecord]) -> int:
        """Persist run asset version records in batch."""
        ...

    def record_lineage_edges_batch(self, edges: Sequence[AssetLineageEdgeRecord]) -> int:
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

    def record_scip_run(self, record: ScipRunRecord) -> None:
        """Record a SCIP execution run."""
        ...


class GatewayRuns(Protocol):
    """Protocol for pipeline run tracking."""

    def record_step(self, record: PipelineStepRecord) -> None:
        """Persist a pipeline step record."""
        ...

    def fetch_steps(self, run_id: str) -> list[PipelineStepRecord]:
        """Return pipeline step records for a run."""
        ...


class GatewayExports(Protocol):
    """Protocol for export auditing and relation building."""

    def audit_enabled(self, settings: ExportAuditSettings) -> bool:
        """Return True when export audit logging is enabled."""
        ...

    def write_export_audit(
        self,
        record: ExportAuditRecord,
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
    "BuildGateway",
    "DatasetRegistryProtocol",
    "GatewayAssets",
    "GatewayBuild",
    "GatewayConfig",
    "GatewayExports",
    "GatewayPolicy",
    "GatewayRuns",
    "GatewaySchemas",
]
