"""Core gateway protocols for build-time access."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self

from codeintel.core.duckdb_types import DuckDBConnection, DuckDBRelation

if TYPE_CHECKING:
    from codeintel.core.manifests import SchemaManifest
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.core.schemas.provider import SchemaProvider


class GatewayConfig(Protocol):
    """Protocol for gateway configuration access."""

    read_only: bool
    db_path: str | Path


class DatasetRegistryProtocol(Protocol):
    """Protocol for dataset registry access."""

    by_name: Mapping[str, DatasetContract]
    jsonl_datasets: Mapping[str, str]
    parquet_datasets: Mapping[str, str]
    dataset_root_dir: Path | None

    def with_dataset_root(self, dataset_root_dir: Path | None) -> Self:
        """Return a registry with the dataset root applied."""
        ...


class GatewayPolicy(Protocol):
    """Protocol for policy-backed operations."""

    schema_provider: SchemaProvider | None

    def table_exists(self, *, schema: str, table: str) -> bool:
        """Return True when a table exists in the underlying catalog."""
        ...


class GatewaySchemas(Protocol):
    """Protocol for schema tracking access."""

    def persist_schema_manifest(
        self,
        manifest: SchemaManifest,
        *,
        request: object,
    ) -> object:
        """Persist a schema manifest record."""
        ...

    def refresh_override_registry_from_manifest(
        self,
        manifest: SchemaManifest,
        *,
        request: object,
        catalog_hash: str | None = None,
    ) -> object:
        """Refresh overrides based on a schema manifest."""
        ...

    def record_schema_versions_batch(self, records: Iterable[object]) -> None:
        """Persist schema version records in batch."""
        ...

    def record_table_schema_registry_batch(self, records: Iterable[object]) -> None:
        """Persist table schema registry records in batch."""
        ...

    def record_schema_observations_batch(self, records: Iterable[object]) -> None:
        """Persist schema observation records in batch."""
        ...

    def load_latest_schema_observation(self, *, table_key: str) -> object | None:
        """Return the most recent schema observation for a table."""
        ...

    def load_recent_drift_summaries(
        self,
        *,
        table_key: str,
        limit: int = 5,
    ) -> list[object]:
        """Return recent schema drift summaries for a table."""
        ...


class GatewayAssets(Protocol):
    """Protocol for asset tracking access."""

    def record_asset_versions_batch(self, records: Iterable[object]) -> None:
        """Persist asset version records in batch."""
        ...

    def record_asset_version_events_batch(self, records: Iterable[object]) -> None:
        """Persist asset version event records in batch."""
        ...

    def record_run_asset_versions_batch(self, records: Iterable[object]) -> None:
        """Persist run asset version records in batch."""
        ...

    def record_lineage_edges_batch(self, records: Iterable[object]) -> None:
        """Persist lineage edge records in batch."""
        ...


class GatewayBuild(Protocol):
    """Protocol for build tracking access."""

    def start_run(self, record: object) -> None:
        """Record a new build run."""
        ...

    def complete_run(self, record: object) -> None:
        """Mark a build run as completed."""
        ...

    def save_run_targets(self, run_id: str, records: Iterable[object]) -> None:
        """Persist target records for a run."""
        ...

    def save_run_nodes(self, run_id: str, records: Iterable[object]) -> None:
        """Persist node records for a run."""
        ...

    def save_manifest(self, manifest: object) -> None:
        """Persist a build manifest payload."""
        ...

    def record_scip_run(self, record: object) -> None:
        """Record a SCIP execution run."""
        ...


class GatewayRuns(Protocol):
    """Protocol for pipeline run tracking."""

    def record_step(self, record: object) -> None:
        """Persist a pipeline step record."""
        ...

    def fetch_steps(self, run_id: str) -> list[object]:
        """Return pipeline step records for a run."""
        ...


class GatewayExports(Protocol):
    """Protocol for export auditing and relation building."""

    def audit_enabled(self, settings: object) -> bool:
        """Return True when export audit logging is enabled."""
        ...

    def write_export_audit(self, record: object, *, settings: object) -> None:
        """Persist an export audit record."""
        ...

    def build_export_relation(self, *, relation: DuckDBRelation) -> DuckDBRelation:
        """Return a relation wrapper for export output."""
        ...


class BuildGateway(Protocol):
    """Protocol for build-time gateway dependencies."""

    assets: GatewayAssets
    build: GatewayBuild
    config: GatewayConfig
    datasets: DatasetRegistryProtocol
    exports: GatewayExports
    policy: GatewayPolicy
    runs: GatewayRuns
    schemas: GatewaySchemas

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
