"""Schema catalog persistence for metadata schema registries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol, TypeGuard

import pyarrow as pa

from codeintel.core.columnar.ipc import schema_from_ipc_payload
from codeintel.core.execution.ids import new_uuid_str
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.hashing import schema_hash as compute_schema_hash
from codeintel.core.schemas.serde import table_schema_from_json_obj
from codeintel.core.time import utc_now
from codeintel.storage.constants import META_CATALOG_NAME
from codeintel.storage.gateway.protocol import DuckDBError
from codeintel.storage.helpers.json import decode_json_dict, normalize_duckdb_json_value
from codeintel.storage.metadata.catalogs import build_catalog_entry, upsert_canonical_catalog
from codeintel.storage.metadata.meta_catalog import meta_table_ref
from codeintel.storage.tracking.schema_catalog_compile import (
    arrow_contract_renderer_cache,
    compile_schema_catalog_batches,
)
from codeintel.storage.tracking.schema_catalog_models import (
    ColumnStatsEntry,
    ColumnStatsPayload,
    DatasetStatsPayload,
    DerivedSettingsPayload,
    OverrideRegistryRefreshResult,
    SchemaCatalogRequest,
    SchemaManifestRunRecord,
    SchemaObservationRecord,
    SchemaVersionRecord,
    TableSchemaOverrideRegistryRecord,
    TableSchemaOverrideVersionRecord,
    TableSchemaRegistryRecord,
)
from codeintel.storage.upsert import UpsertSpec

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from collections.abc import Set as AbstractSet

    from duckdb import DuckDBPyConnection

    from codeintel.core.manifests import SchemaManifest, TableProvenance
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.storage.gateway.protocol import StorageGateway

    class SchemaIndex(Protocol):
        """Protocol for schema index consumers without build-layer imports."""

        @property
        def inferable_table_keys(self) -> AbstractSet[str]: ...

        def prefill_cache(self, schemas: Mapping[str, TableSchema]) -> None: ...


_INFERRED_REGISTRY_FILTER = (
    "AND ("
    "registry.inference_status IN ('inferred', 'override') "
    "OR registry.derivation_kind IN ('inferred_relation', 'view_inferred')"
    ")"
)
_OBSERVED_DERIVATION_FILTER = (
    "AND registry.derivation_kind IN ('inferred_relation', 'view_inferred')"
)


@dataclass(frozen=True, slots=True)
class PersistSchemaManifestResult:
    """Summary of a schema manifest persistence transaction."""

    catalog_kind: str
    catalog_hash: str
    tables: int
    views: int
    schema_versions_rows: int
    table_schema_registry_rows: int
    schema_manifest_runs_rows: int


@dataclass(frozen=True, slots=True)
class _LatestManifest:
    """Latest manifest metadata used for registry health reporting."""

    catalog_hash: str
    repo: str
    commit: str
    manifest_kind: str
    created_at: datetime | None


@dataclass(frozen=True, slots=True)
class _RegistryStats:
    """Registry row counts and freshness metadata."""

    row_count: int
    updated_at: datetime | None
    stale: bool


@dataclass(frozen=True, slots=True)
class _InferableStats:
    """Inference health summary for inferable tables."""

    total: int
    inferred: int
    errors: int
    success_rate: float | None


@dataclass(frozen=True, slots=True)
class _ContractDriftReport:
    """Summary of Arrow contract drift versus registry metadata."""

    total_tables: int
    missing_contracts: int
    missing_contract_metadata: int
    hash_mismatches: int
    digest_mismatches: int
    missing_contract_samples: tuple[str, ...]
    missing_metadata_samples: tuple[str, ...]
    mismatch_samples: tuple[str, ...]


@dataclass(slots=True)
class _ContractDriftAccumulator:
    total_tables: int = 0
    missing_contracts: int = 0
    missing_contract_metadata: int = 0
    hash_mismatches: int = 0
    digest_mismatches: int = 0
    missing_contract_samples: list[str] = field(default_factory=list)
    missing_metadata_samples: list[str] = field(default_factory=list)
    mismatch_samples: list[str] = field(default_factory=list)

    def record_missing_contract(self, table_key: str, *, limit: int) -> None:
        self.missing_contracts += 1
        _append_sample(self.missing_contract_samples, table_key, limit=limit)

    def record_missing_metadata(self, table_key: str, *, limit: int) -> None:
        self.missing_contract_metadata += 1
        _append_sample(self.missing_metadata_samples, table_key, limit=limit)

    def record_mismatch(self, table_key: str, *, limit: int) -> None:
        _append_sample(self.mismatch_samples, table_key, limit=limit)


@dataclass(frozen=True, slots=True)
class _OverrideRecordContext:
    """Context payload for building override registry records."""

    manifest: SchemaManifest
    request: SchemaCatalogRequest
    version_id: str
    catalog_hash: str | None
    now: datetime


def _load_latest_observed_schema_from_connection(
    con: DuckDBPyConnection,
    *,
    table_key: str,
) -> TableSchema | None:
    observations_ref = meta_table_ref("metadata.schema_observations")
    registry_ref = meta_table_ref("metadata.table_schema_registry")
    versions_ref = meta_table_ref("metadata.schema_versions")
    row = con.execute(
        f"""
        SELECT v.schema_json
        FROM {observations_ref} AS o
        JOIN {versions_ref} AS v
          ON o.schema_digest = v.schema_digest
        JOIN {registry_ref} AS registry
          ON registry.table_key = o.table_key
        WHERE o.table_key = ?
        {_OBSERVED_DERIVATION_FILTER}
        ORDER BY o.observed_at DESC
        LIMIT 1
        """,
        [table_key],
    ).fetchone()
    if row is None:
        return None
    schema_json = decode_json_dict(row[0])
    if not schema_json:
        return None
    return table_schema_from_json_obj(schema_json)


def _load_latest_observed_schema_rows(
    con: DuckDBPyConnection,
) -> list[tuple[object, object]]:
    observations_ref = meta_table_ref("metadata.schema_observations")
    registry_ref = meta_table_ref("metadata.table_schema_registry")
    versions_ref = meta_table_ref("metadata.schema_versions")
    rows = con.execute(
        f"""
        SELECT table_key, schema_json
        FROM (
          SELECT
            o.table_key AS table_key,
            v.schema_json AS schema_json,
            ROW_NUMBER() OVER (
              PARTITION BY o.table_key ORDER BY o.observed_at DESC
            ) AS rn
          FROM {observations_ref} AS o
          JOIN {versions_ref} AS v
            ON o.schema_digest = v.schema_digest
          JOIN {registry_ref} AS registry
            ON registry.table_key = o.table_key
          WHERE 1 = 1
          {_OBSERVED_DERIVATION_FILTER}
        )
        WHERE rn = 1
        """
    ).fetchall()
    return list(rows)


def load_table_schema_from_connection(
    con: DuckDBPyConnection,
    *,
    table_key: str,
) -> TableSchema | None:
    """Load a TableSchema from schema catalog tables.

    Parameters
    ----------
    con
        DuckDB connection with metadata catalog attached.
    table_key
        Fully qualified table key.

    Returns
    -------
    TableSchema | None
        Loaded TableSchema when present; otherwise None.
    """
    observed = _load_latest_observed_schema_from_connection(con, table_key=table_key)
    if observed is not None:
        return observed
    registry_ref = meta_table_ref("metadata.table_schema_registry")
    versions_ref = meta_table_ref("metadata.schema_versions")
    row = con.execute(
        f"""
        SELECT v.schema_json
        FROM {registry_ref} AS registry
        JOIN {versions_ref} AS v
          ON registry.schema_digest = v.schema_digest
        WHERE registry.table_key = ?
        {_INFERRED_REGISTRY_FILTER}
        """,
        [table_key],
    ).fetchone()
    if row is None:
        row = con.execute(
            f"""
            SELECT v.schema_json
            FROM {registry_ref} AS registry
            JOIN {versions_ref} AS v
              ON registry.schema_digest = v.schema_digest
            WHERE registry.table_key = ?
            """,
            [table_key],
        ).fetchone()
    if row is None:
        return None
    schema_json = decode_json_dict(row[0])
    if not schema_json:
        return None
    return table_schema_from_json_obj(schema_json)


def iter_table_schemas_from_connection(
    con: DuckDBPyConnection,
) -> Iterable[TableSchema]:
    """Iterate all registered TableSchema values from metadata tables.

    Parameters
    ----------
    con
        DuckDB connection with metadata catalog attached.

    Yields
    ------
    TableSchema
        Registered TableSchema values ordered by table key.
    """
    registry_ref = meta_table_ref("metadata.table_schema_registry")
    versions_ref = meta_table_ref("metadata.schema_versions")
    observed_rows = _load_latest_observed_schema_rows(con)
    inferred_rows = con.execute(
        f"""
        SELECT registry.table_key, v.schema_json
        FROM {registry_ref} AS registry
        JOIN {versions_ref} AS v
          ON registry.schema_digest = v.schema_digest
        WHERE 1 = 1
        {_INFERRED_REGISTRY_FILTER}
        """
    ).fetchall()
    schemas_by_key: dict[str, TableSchema] = {}
    for table_key, schema_json_raw in observed_rows:
        schema_json = decode_json_dict(schema_json_raw)
        if not schema_json:
            continue
        schemas_by_key[str(table_key)] = table_schema_from_json_obj(schema_json)
    for table_key, schema_json_raw in inferred_rows:
        schema_json = decode_json_dict(schema_json_raw)
        if not schema_json:
            continue
        schemas_by_key[str(table_key)] = table_schema_from_json_obj(schema_json)
    fallback_rows = con.execute(
        f"""
        SELECT registry.table_key, v.schema_json
        FROM {registry_ref} AS registry
        JOIN {versions_ref} AS v
          ON registry.schema_digest = v.schema_digest
        ORDER BY registry.table_key
        """
    ).fetchall()
    for table_key, schema_json_raw in fallback_rows:
        if str(table_key) in schemas_by_key:
            continue
        schema_json = decode_json_dict(schema_json_raw)
        if not schema_json:
            continue
        schemas_by_key[str(table_key)] = table_schema_from_json_obj(schema_json)
    for table_key in sorted(schemas_by_key):
        yield schemas_by_key[table_key]


@dataclass(frozen=True, slots=True)
class SchemaCatalogProvider:
    """SchemaProvider backed by metadata schema catalog tables."""

    con: DuckDBPyConnection

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        """Return the latest registered TableSchema for the table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema | None
            Latest registered TableSchema when present; otherwise None.
        """
        return load_table_schema_from_connection(self.con, table_key=table_key)

    def require_table_schema(self, table_key: str) -> TableSchema:
        """Return schema for table_key, raising when unknown.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema
            Latest registered TableSchema for the table key.

        Raises
        ------
        KeyError
            If no schema is registered for the table key.
        """
        schema = self.get_table_schema(table_key)
        if schema is None:
            msg = f"Unknown table schema: {table_key}"
            raise KeyError(msg)
        return schema

    def iter_table_schemas(self) -> Iterable[TableSchema]:
        """Iterate all registered table schemas.

        Yields
        ------
        TableSchema
            Registered TableSchema values ordered by table key.
        """
        yield from iter_table_schemas_from_connection(self.con)


class SchemaCatalogTracking:
    """Persist and read schema catalogs from metadata tables."""

    def __init__(self, gateway: StorageGateway) -> None:
        """Initialize schema catalog tracking accessor.

        Parameters
        ----------
        gateway
            Storage gateway providing database access.
        """
        self._gateway = gateway
        self._con = gateway.con
        self._backend = gateway.policy

    def record_schema_versions_batch(self, records: Sequence[SchemaVersionRecord]) -> int:
        """Insert schema versions with content-addressed deduplication.

        Returns
        -------
        int
            Number of rows processed.
        """
        if not records:
            return 0

        now = utc_now()
        rows = [
            (
                record.schema_digest,
                record.schema_hash,
                normalize_duckdb_json_value(record.schema_json),
                normalize_duckdb_json_value(record.renderer_cache)
                if record.renderer_cache is not None
                else None,
                record.created_at or now,
            )
            for record in records
        ]

        return self._backend.upsert(
            "metadata.schema_versions",
            rows,
            columns=(
                "schema_digest",
                "schema_hash",
                "schema_json",
                "renderer_cache",
                "created_at",
            ),
            catalog=META_CATALOG_NAME,
            upsert=UpsertSpec(
                conflict_columns=("schema_digest",),
                update_columns=(),
            ),
        )

    def record_schema_observations_batch(
        self,
        records: Sequence[SchemaObservationRecord],
    ) -> int:
        """Insert schema observations for inference tracking.

        Returns
        -------
        int
            Number of rows processed.
        """
        if not records:
            return 0

        now = utc_now()
        rows = [
            (
                record.observation_id or new_uuid_str(),
                record.table_key,
                record.repo,
                record.commit,
                record.target_name,
                record.schema_digest,
                record.schema_hash,
                record.arrow_schema_ipc_b64,
                normalize_duckdb_json_value(record.column_stats)
                if record.column_stats is not None
                else None,
                normalize_duckdb_json_value(record.dataset_stats)
                if record.dataset_stats is not None
                else None,
                normalize_duckdb_json_value(record.derived_settings)
                if record.derived_settings is not None
                else None,
                normalize_duckdb_json_value(record.drift_summary)
                if record.drift_summary is not None
                else None,
                record.observed_at or now,
            )
            for record in records
        ]

        return self._backend.upsert(
            "metadata.schema_observations",
            rows,
            columns=(
                "observation_id",
                "table_key",
                "repo",
                "commit",
                "target_name",
                "schema_digest",
                "schema_hash",
                "arrow_schema_ipc_b64",
                "column_stats",
                "dataset_stats",
                "derived_settings",
                "drift_summary",
                "observed_at",
            ),
            catalog=META_CATALOG_NAME,
            upsert=UpsertSpec(
                conflict_columns=("observation_id",),
                update_columns=(),
            ),
        )

    def record_override_versions_batch(
        self,
        records: Sequence[TableSchemaOverrideVersionRecord],
    ) -> int:
        """Insert override version rows for inferable table schemas.

        Returns
        -------
        int
            Number of rows processed.
        """
        if not records:
            return 0

        now = utc_now()
        rows = [
            (
                record.version_id,
                record.table_key,
                record.schema_digest,
                record.schema_hash,
                record.catalog_hash,
                record.created_at or now,
            )
            for record in records
        ]

        return self._backend.upsert(
            "metadata.table_schema_override_versions",
            rows,
            columns=(
                "version_id",
                "table_key",
                "schema_digest",
                "schema_hash",
                "catalog_hash",
                "created_at",
            ),
            catalog=META_CATALOG_NAME,
            upsert=UpsertSpec(
                conflict_columns=("version_id", "table_key"),
                update_columns=(),
            ),
        )

    def record_override_registry_batch(
        self,
        records: Sequence[TableSchemaOverrideRegistryRecord],
    ) -> int:
        """Upsert override registry pointers for inferable tables.

        Returns
        -------
        int
            Number of rows processed.
        """
        if not records:
            return 0

        now = utc_now()
        rows = [
            (
                record.table_key,
                record.schema_digest,
                record.schema_hash,
                record.version_id,
                record.updated_at or now,
            )
            for record in records
        ]

        return self._backend.upsert(
            "metadata.table_schema_override_registry",
            rows,
            columns=(
                "table_key",
                "schema_digest",
                "schema_hash",
                "version_id",
                "updated_at",
            ),
            catalog=META_CATALOG_NAME,
            upsert=UpsertSpec(
                conflict_columns=("table_key",),
                update_columns=("schema_digest", "schema_hash", "version_id", "updated_at"),
            ),
        )

    def record_table_schema_registry_batch(
        self,
        records: Sequence[TableSchemaRegistryRecord],
    ) -> int:
        """Upsert schema registry pointers for tables/views.

        Returns
        -------
        int
            Number of rows processed.
        """
        if not records:
            return 0

        now = utc_now()
        rows = [
            (
                record.table_key,
                record.schema_digest,
                record.schema_hash,
                record.derivation_kind,
                record.derivation_source,
                record.inference_status,
                record.inference_error,
                record.catalog_hash,
                record.updated_at or now,
            )
            for record in records
        ]

        return self._backend.upsert(
            "metadata.table_schema_registry",
            rows,
            columns=(
                "table_key",
                "schema_digest",
                "schema_hash",
                "derivation_kind",
                "derivation_source",
                "inference_status",
                "inference_error",
                "catalog_hash",
                "updated_at",
            ),
            catalog=META_CATALOG_NAME,
            upsert=UpsertSpec(
                conflict_columns=("table_key",),
                update_columns=(
                    "schema_digest",
                    "schema_hash",
                    "derivation_kind",
                    "derivation_source",
                    "inference_status",
                    "inference_error",
                    "catalog_hash",
                    "updated_at",
                ),
            ),
        )

    def record_schema_manifest_runs_batch(self, records: Sequence[SchemaManifestRunRecord]) -> int:
        """Upsert run -> schema manifest catalog linkages.

        Returns
        -------
        int
            Number of rows processed.
        """
        if not records:
            return 0

        now = utc_now()
        rows = [
            (
                record.run_id,
                record.repo,
                record.commit,
                record.manifest_kind,
                record.catalog_hash,
                record.created_at or now,
            )
            for record in records
        ]

        return self._backend.upsert(
            "metadata.schema_manifest_runs",
            rows,
            columns=("run_id", "repo", "commit", "manifest_kind", "catalog_hash", "created_at"),
            catalog=META_CATALOG_NAME,
            upsert=UpsertSpec(
                conflict_columns=("run_id",),
                update_columns=("repo", "commit", "manifest_kind", "catalog_hash", "created_at"),
            ),
        )

    def load_table_schema(self, table_key: str) -> TableSchema | None:
        """Load a TableSchema from the schema registry.

        Parameters
        ----------
        table_key
            Fully qualified table key.

        Returns
        -------
        TableSchema | None
            Loaded TableSchema when present; otherwise None.
        """
        return load_table_schema_from_connection(self._con, table_key=table_key)

    def load_latest_schema_observation(
        self,
        *,
        table_key: str,
    ) -> SchemaObservationRecord | None:
        """Load the latest schema observation for a table key.

        Parameters
        ----------
        table_key
            Fully qualified table key.

        Returns
        -------
        SchemaObservationRecord | None
            Latest observation record when present; otherwise None.
        """
        observations_ref = meta_table_ref("metadata.schema_observations")
        try:
            row = self._con.execute(
                f"""
                SELECT
                  observation_id,
                  table_key,
                  repo,
                  commit,
                  target_name,
                  schema_digest,
                  schema_hash,
                  arrow_schema_ipc_b64,
                  column_stats,
                  dataset_stats,
                  derived_settings,
                  drift_summary,
                  observed_at
                FROM {observations_ref}
                WHERE table_key = ?
                ORDER BY observed_at DESC
                LIMIT 1
                """,
                [table_key],
            ).fetchone()
        except DuckDBError:
            return None
        if row is None:
            return None
        return SchemaObservationRecord(
            observation_id=str(row[0]) if row[0] is not None else None,
            table_key=str(row[1]),
            repo=str(row[2]) if row[2] is not None else None,
            commit=str(row[3]) if row[3] is not None else None,
            target_name=str(row[4]) if row[4] is not None else None,
            schema_digest=str(row[5]),
            schema_hash=str(row[6]),
            arrow_schema_ipc_b64=str(row[7]),
            column_stats=_decode_optional_column_stats(row[8]),
            dataset_stats=_decode_optional_dataset_stats(row[9]),
            derived_settings=_decode_optional_derived_settings(row[10]),
            drift_summary=_decode_optional_json_dict(row[11]),
            observed_at=row[12],
        )

    def has_contract_arrow_schema(self, *, table_key: str) -> bool:
        """Return True when the registry stores Arrow schema bytes for a table.

        Parameters
        ----------
        table_key
            Fully qualified table key.

        Returns
        -------
        bool
            True when renderer_cache contains Arrow schema IPC bytes.
        """
        registry_ref = meta_table_ref("metadata.table_schema_registry")
        versions_ref = meta_table_ref("metadata.schema_versions")
        row = self._con.execute(
            f"""
            SELECT v.renderer_cache
            FROM {registry_ref} AS registry
            JOIN {versions_ref} AS v
              ON registry.schema_digest = v.schema_digest
            WHERE registry.table_key = ?
            LIMIT 1
            """,
            [table_key],
        ).fetchone()
        if row is None or row[0] is None:
            return False
        renderer_cache = decode_json_dict(row[0])
        return _renderer_cache_has_arrow_schema(renderer_cache)

    def load_recent_drift_summaries(
        self,
        *,
        table_key: str,
        limit: int = 5,
    ) -> tuple[dict[str, object] | None, ...]:
        """Return recent drift summaries for a table key.

        Parameters
        ----------
        table_key
            Fully qualified table key.
        limit
            Maximum number of summaries to return.

        Returns
        -------
        tuple[dict[str, object] | None, ...]
            Ordered drift summaries, newest first.
        """
        if limit <= 0:
            return ()
        observations_ref = meta_table_ref("metadata.schema_observations")
        rows = self._con.execute(
            f"""
            SELECT drift_summary
            FROM {observations_ref}
            WHERE table_key = ?
            ORDER BY observed_at DESC
            LIMIT ?
            """,
            [table_key, limit],
        ).fetchall()
        if not rows:
            return ()
        summaries: list[dict[str, object] | None] = []
        for (summary_raw,) in rows:
            summaries.append(_decode_optional_json_dict(summary_raw))
        return tuple(summaries)

    def drift_summary_report(self, *, limit: int = 50) -> dict[str, object]:
        """Return a summary of recent schema drift observations.

        Returns
        -------
        dict[str, object]
            Aggregate drift summary across recent observations.
        """
        observations_ref = meta_table_ref("metadata.schema_observations")
        totals_row = self._con.execute(
            f"""
            SELECT
              COUNT(DISTINCT table_key),
              COUNT(DISTINCT CASE WHEN drift_summary IS NOT NULL THEN table_key END)
            FROM {observations_ref}
            """
        ).fetchone()
        total_tables = int(totals_row[0]) if totals_row and totals_row[0] is not None else 0
        drift_tables = int(totals_row[1]) if totals_row and totals_row[1] is not None else 0

        rows = self._con.execute(
            f"""
            SELECT table_key, drift_summary, observed_at
            FROM (
              SELECT
                table_key,
                drift_summary,
                observed_at,
                ROW_NUMBER() OVER (PARTITION BY table_key ORDER BY observed_at DESC) AS rn
              FROM {observations_ref}
              WHERE drift_summary IS NOT NULL
            )
            WHERE rn = 1
            ORDER BY observed_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()

        latest: list[dict[str, object]] = []
        missing_total = 0
        extra_total = 0
        type_change_total = 0
        for table_key, drift_raw, observed_at in rows:
            summary = _decode_optional_json_dict(drift_raw) or {}
            missing = summary.get("missing_columns")
            extra = summary.get("extra_columns")
            type_changes = summary.get("type_changes")
            missing_total += len(missing) if isinstance(missing, list) else 0
            extra_total += len(extra) if isinstance(extra, list) else 0
            type_change_total += len(type_changes) if isinstance(type_changes, list) else 0
            latest.append(
                {
                    "table_key": str(table_key),
                    "drift_summary": summary,
                    "observed_at": observed_at.isoformat() if observed_at is not None else None,
                }
            )

        return {
            "total_tables": total_tables,
            "tables_with_drift": drift_tables,
            "missing_columns": missing_total,
            "extra_columns": extra_total,
            "type_changes": type_change_total,
            "latest": latest,
        }

    def load_override_registry(self) -> dict[str, TableSchema]:
        """Load active override schemas for inferable outputs.

        Returns
        -------
        dict[str, TableSchema]
            Mapping of table_key to override TableSchema entries.
        """
        registry_ref = meta_table_ref("metadata.table_schema_override_registry")
        versions_ref = meta_table_ref("metadata.schema_versions")
        rows = self._con.execute(
            f"""
            SELECT r.table_key, v.schema_json
            FROM {registry_ref} AS r
            JOIN {versions_ref} AS v
              ON r.schema_digest = v.schema_digest
            ORDER BY r.table_key
            """
        ).fetchall()
        if not rows:
            return {}

        schemas: dict[str, TableSchema] = {}
        for table_key, schema_json_raw in rows:
            schema_json = decode_json_dict(schema_json_raw)
            if not schema_json:
                continue
            schemas[str(table_key)] = table_schema_from_json_obj(schema_json)
        return schemas

    def registry_health_snapshot(self) -> dict[str, object]:
        """Return health metadata for the latest schema registry state.

        Returns
        -------
        dict[str, object]
            Health snapshot payload for the schema registry.
        """
        latest = self._load_latest_manifest()
        if latest is None:
            return {
                "status": "missing_manifest",
                "latest_manifest": None,
                "registry_rows": 0,
                "registry_updated_at": None,
                "registry_stale": True,
                "override_registry_rows": 0,
                "inferable_total": 0,
                "inferred_count": 0,
                "inference_error_count": 0,
                "inference_success_rate": None,
            }

        registry_stats = self._registry_stats(
            catalog_hash=latest.catalog_hash,
            created_at=latest.created_at,
        )
        inferable_stats = self._inferable_stats(catalog_hash=latest.catalog_hash)
        override_registry_rows = self._override_registry_rows()
        status = "ok"
        if registry_stats.stale or override_registry_rows == 0:
            status = "warn"
        if inferable_stats.errors:
            status = "warn"

        drift = self.contract_drift_report(limit=10)
        return {
            "status": status,
            "latest_manifest": {
                "catalog_hash": latest.catalog_hash,
                "repo": latest.repo,
                "commit": latest.commit,
                "manifest_kind": latest.manifest_kind,
                "created_at": latest.created_at.isoformat()
                if latest.created_at is not None
                else None,
            },
            "registry_rows": registry_stats.row_count,
            "registry_updated_at": registry_stats.updated_at.isoformat()
            if registry_stats.updated_at is not None
            else None,
            "registry_stale": registry_stats.stale,
            "override_registry_rows": override_registry_rows,
            "inferable_total": inferable_stats.total,
            "inferred_count": inferable_stats.inferred,
            "inference_error_count": inferable_stats.errors,
            "inference_success_rate": inferable_stats.success_rate,
            "contract_drift": drift,
        }

    def prefill_schema_index(
        self,
        schema_index: SchemaIndex,
        *,
        table_keys: Sequence[str] | None = None,
    ) -> int:
        """Prefill SchemaIndex cache with persisted inferred schemas.

        Parameters
        ----------
        schema_index
            SchemaIndex instance to prefill.
        table_keys
            Optional table keys to restrict the prefill query.

        Returns
        -------
        int
            Number of schemas prefetched into the cache.
        """
        inferable = schema_index.inferable_table_keys
        if not inferable:
            return 0

        if table_keys is None:
            allowed_keys = tuple(sorted(inferable))
        else:
            allowed_keys = tuple(sorted(set(table_keys) & inferable))

        if not allowed_keys:
            return 0

        placeholders = ", ".join(["?"] * len(allowed_keys))
        registry_ref = meta_table_ref("metadata.table_schema_registry")
        versions_ref = meta_table_ref("metadata.schema_versions")
        sql = (
            "SELECT r.table_key, v.schema_json "
            f"FROM {registry_ref} AS r "
            f"JOIN {versions_ref} AS v "
            "  ON r.schema_digest = v.schema_digest "
            "WHERE r.derivation_kind = ? "
            "  AND r.inference_status IN (?, ?) "
            f"  AND r.table_key IN ({placeholders})"
        )
        params: list[object] = ["inferred_relation", "inferred", "override", *allowed_keys]
        rows = self._con.execute(sql, params).fetchall()
        if not rows:
            return 0

        schemas: dict[str, TableSchema] = {}
        for table_key, schema_json_raw in rows:
            schema_json = decode_json_dict(schema_json_raw)
            if not schema_json:
                continue
            schemas[str(table_key)] = table_schema_from_json_obj(schema_json)

        if not schemas:
            return 0

        schema_index.prefill_cache(schemas)
        return len(schemas)

    def contract_drift_report(self, *, limit: int = 10) -> dict[str, object]:
        """Return Arrow contract drift metadata for the schema registry.

        Parameters
        ----------
        limit
            Maximum number of example table keys to include per category.

        Returns
        -------
        dict[str, object]
            Drift summary comparing Arrow contract metadata to registry metadata.
        """
        registry_ref = meta_table_ref("metadata.table_schema_registry")
        versions_ref = meta_table_ref("metadata.schema_versions")
        rows = self._con.execute(
            f"""
            SELECT r.table_key, r.schema_hash, r.schema_digest, v.renderer_cache
            FROM {registry_ref} AS r
            JOIN {versions_ref} AS v
              ON r.schema_digest = v.schema_digest
            ORDER BY r.table_key
            """
        ).fetchall()
        if not rows:
            return _contract_drift_payload(_ContractDriftReport(0, 0, 0, 0, 0, (), (), ()))

        accumulator = _ContractDriftAccumulator()
        for table_key, schema_hash, schema_digest, renderer_cache_raw in rows:
            contract_schema = _schema_from_renderer_cache(renderer_cache_raw)
            table_name = str(table_key)
            if contract_schema is None:
                accumulator.record_missing_contract(table_name, limit=limit)
                continue
            contract_hash = _schema_metadata_value(contract_schema, "codeintel.schema_hash")
            contract_digest = _schema_metadata_value(contract_schema, "codeintel.schema_digest")
            if contract_hash is None or contract_digest is None:
                accumulator.record_missing_metadata(table_name, limit=limit)
                continue
            registry_hash = str(schema_hash) if schema_hash is not None else None
            registry_digest = str(schema_digest) if schema_digest is not None else None
            mismatch = False
            if registry_hash != contract_hash:
                accumulator.hash_mismatches += 1
                mismatch = True
            if registry_digest != contract_digest:
                accumulator.digest_mismatches += 1
                mismatch = True
            if mismatch:
                accumulator.record_mismatch(table_name, limit=limit)

        report = _ContractDriftReport(
            total_tables=len(rows),
            missing_contracts=accumulator.missing_contracts,
            missing_contract_metadata=accumulator.missing_contract_metadata,
            hash_mismatches=accumulator.hash_mismatches,
            digest_mismatches=accumulator.digest_mismatches,
            missing_contract_samples=tuple(accumulator.missing_contract_samples),
            missing_metadata_samples=tuple(accumulator.missing_metadata_samples),
            mismatch_samples=tuple(accumulator.mismatch_samples),
        )
        return _contract_drift_payload(report)

    def backfill_renderer_cache(
        self,
        manifest: SchemaManifest,
        *,
        include_views: bool = True,
    ) -> int:
        """Backfill Arrow contract renderer_cache entries for existing schema versions.

        Parameters
        ----------
        manifest
            Schema manifest providing table schemas and provenance metadata.
        include_views
            Whether to include view schemas from the manifest in the backfill.

        Returns
        -------
        int
            Number of schema version rows updated with contract payloads.

        Raises
        ------
        RuntimeError
            If the gateway is read-only.
        """
        if getattr(self._gateway, "config", None) is not None and self._gateway.config.read_only:
            msg = "Cannot backfill renderer cache in a read-only storage gateway"
            raise RuntimeError(msg)

        payloads = self._renderer_cache_payloads(manifest, include_views=include_views)
        if not payloads:
            return 0

        digests = tuple(payloads)
        placeholders = ", ".join(["?"] * len(digests))
        versions_ref = meta_table_ref("metadata.schema_versions")
        rows = self._con.execute(
            f"""
            SELECT schema_digest, renderer_cache
            FROM {versions_ref}
            WHERE schema_digest IN ({placeholders})
            """,
            list(digests),
        ).fetchall()
        if not rows:
            return 0

        updates: list[tuple[object, str]] = []
        for schema_digest, renderer_cache_raw in rows:
            digest = str(schema_digest)
            payload = payloads.get(digest)
            if payload is None:
                continue
            renderer_cache = decode_json_dict(renderer_cache_raw)
            if _renderer_cache_has_arrow_schema(renderer_cache):
                continue
            merged = dict(renderer_cache) if renderer_cache else {}
            merged.update(payload)
            updates.append((normalize_duckdb_json_value(merged), digest))

        if not updates:
            return 0

        self._con.executemany(
            f"UPDATE {versions_ref} SET renderer_cache = ? WHERE schema_digest = ?",
            updates,
        )
        return len(updates)

    @staticmethod
    def _renderer_cache_payloads(
        manifest: SchemaManifest,
        *,
        include_views: bool,
    ) -> dict[str, dict[str, object]]:
        schemas = sorted(manifest.tables, key=lambda schema: schema.table_key)
        if include_views:
            schemas.extend(sorted(manifest.views, key=lambda schema: schema.table_key))
        if not schemas:
            return {}

        provenance_by_table_key = dict(manifest.table_provenance)
        if include_views:
            provenance_by_table_key.update(manifest.view_provenance)

        payloads: dict[str, dict[str, object]] = {}
        for table_schema in schemas:
            schema_json = table_schema.to_json_obj()
            schema_digest = fingerprint(schema_json)
            if schema_digest in payloads:
                continue
            provenance = provenance_by_table_key.get(table_schema.table_key)
            payloads[schema_digest] = arrow_contract_renderer_cache(
                table_schema,
                provenance=provenance,
            )
        return payloads

    def persist_schema_manifest(
        self,
        manifest: SchemaManifest,
        *,
        request: SchemaCatalogRequest,
    ) -> PersistSchemaManifestResult:
        """Persist SchemaManifest into canonical catalogs + schema registry tables atomically.

        Returns
        -------
        PersistSchemaManifestResult
            Summary of the persistence operation.

        Raises
        ------
        RuntimeError
            If the gateway is read-only.
        """
        if getattr(self._gateway, "config", None) is not None and self._gateway.config.read_only:
            msg = "Cannot persist schema manifest into a read-only storage gateway"
            raise RuntimeError(msg)

        batches = compile_schema_catalog_batches(manifest, request=request)

        entry = build_catalog_entry(
            catalog_kind=batches.catalog_kind,
            catalog_hash=batches.catalog_hash,
            payload=batches.catalog_payload,
            inputs=batches.catalog_inputs,
        )

        with self._backend.transaction():
            upsert_canonical_catalog(self._gateway, entry)
            n_versions = self.record_schema_versions_batch(batches.schema_versions)
            n_registry = self.record_table_schema_registry_batch(batches.table_schema_registry)
            n_runs = self.record_schema_manifest_runs_batch(batches.schema_manifest_runs)

        return PersistSchemaManifestResult(
            catalog_kind=batches.catalog_kind,
            catalog_hash=batches.catalog_hash,
            tables=len(manifest.tables),
            views=len(manifest.views) if request.include_views else 0,
            schema_versions_rows=n_versions,
            table_schema_registry_rows=n_registry,
            schema_manifest_runs_rows=n_runs,
        )

    def refresh_override_registry_from_manifest(
        self,
        manifest: SchemaManifest,
        *,
        request: SchemaCatalogRequest,
        catalog_hash: str | None = None,
    ) -> OverrideRegistryRefreshResult:
        """Update override registry when all inferable outputs are inferred.

        Returns
        -------
        OverrideRegistryRefreshResult
            Summary of the refresh attempt.

        Raises
        ------
        RuntimeError
            If the gateway is read-only.
        ValueError
            If strict provenance checks fail.
        """
        if getattr(self._gateway, "config", None) is not None and self._gateway.config.read_only:
            msg = "Cannot refresh override registry in a read-only storage gateway"
            raise RuntimeError(msg)

        now = request.now or utc_now()
        inferable_tables, blocked_tables, missing_provenance = self._select_inferable_tables(
            manifest,
            strict_provenance=request.strict_provenance,
        )
        if missing_provenance and request.strict_provenance:
            msg = (
                "Missing table provenance for override refresh: "
                f"{', '.join(sorted(missing_provenance))}"
            )
            raise ValueError(msg)

        if blocked_tables:
            reason = (
                f"inference incomplete for {len(blocked_tables)} table(s): "
                f"{', '.join(sorted(blocked_tables))}"
            )
            return OverrideRegistryRefreshResult(
                status="skipped",
                reason=reason,
                version_id=None,
                tables=len(inferable_tables),
                schema_versions_rows=0,
                override_versions_rows=0,
                override_registry_rows=0,
            )

        if not inferable_tables:
            return OverrideRegistryRefreshResult(
                status="skipped",
                reason="no inferable tables in manifest",
                version_id=None,
                tables=0,
                schema_versions_rows=0,
                override_versions_rows=0,
                override_registry_rows=0,
            )

        version_id = new_uuid_str()
        context = _OverrideRecordContext(
            manifest=manifest,
            request=request,
            version_id=version_id,
            catalog_hash=catalog_hash,
            now=now,
        )
        schema_versions, override_versions, override_registry = self._build_override_records(
            inferable_tables,
            context=context,
        )

        with self._backend.transaction():
            schema_versions_rows = self.record_schema_versions_batch(
                tuple(schema_versions.values())
            )
            override_versions_rows = self.record_override_versions_batch(override_versions)
            override_registry_rows = self.record_override_registry_batch(override_registry)

        return OverrideRegistryRefreshResult(
            status="updated",
            reason=None,
            version_id=version_id,
            tables=len(inferable_tables),
            schema_versions_rows=schema_versions_rows,
            override_versions_rows=override_versions_rows,
            override_registry_rows=override_registry_rows,
        )

    def _load_latest_manifest(self) -> _LatestManifest | None:
        manifest_runs_ref = meta_table_ref("metadata.schema_manifest_runs")
        row = self._con.execute(
            f"""
            SELECT catalog_hash, repo, commit, manifest_kind, created_at
            FROM {manifest_runs_ref}
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        catalog_hash, repo, commit, manifest_kind, created_at = row
        return _LatestManifest(
            catalog_hash=str(catalog_hash),
            repo=str(repo),
            commit=str(commit),
            manifest_kind=str(manifest_kind),
            created_at=created_at if isinstance(created_at, datetime) else None,
        )

    def _registry_stats(
        self,
        *,
        catalog_hash: str,
        created_at: datetime | None,
    ) -> _RegistryStats:
        registry_ref = meta_table_ref("metadata.table_schema_registry")
        row = self._con.execute(
            f"""
            SELECT COUNT(*), MAX(updated_at)
            FROM {registry_ref}
            WHERE catalog_hash = ?
            """,
            [catalog_hash],
        ).fetchone()
        row_count = int(row[0]) if row is not None else 0
        updated_at = row[1] if row is not None else None
        stale = updated_at is None
        if created_at is not None and updated_at is not None:
            stale = updated_at < created_at
        return _RegistryStats(row_count=row_count, updated_at=updated_at, stale=stale)

    def _inferable_stats(self, *, catalog_hash: str) -> _InferableStats:
        registry_ref = meta_table_ref("metadata.table_schema_registry")
        row = self._con.execute(
            f"""
            SELECT
                SUM(CASE WHEN derivation_kind = 'inferred_relation' THEN 1 ELSE 0 END) AS total,
                SUM(
                    CASE
                        WHEN derivation_kind = 'inferred_relation'
                         AND inference_status = 'inferred'
                        THEN 1
                        ELSE 0
                    END
                ) AS inferred_count,
                SUM(
                    CASE
                        WHEN derivation_kind = 'inferred_relation'
                         AND inference_status = 'error'
                        THEN 1
                        ELSE 0
                    END
                ) AS error_count
            FROM {registry_ref}
            WHERE catalog_hash = ?
            """,
            [catalog_hash],
        ).fetchone()
        total = int(row[0] or 0) if row is not None else 0
        inferred = int(row[1] or 0) if row is not None else 0
        errors = int(row[2] or 0) if row is not None else 0
        success_rate = inferred / total if total else None
        return _InferableStats(
            total=total, inferred=inferred, errors=errors, success_rate=success_rate
        )

    def _override_registry_rows(self) -> int:
        override_ref = meta_table_ref("metadata.table_schema_override_registry")
        row = self._con.execute(f"SELECT COUNT(*) FROM {override_ref}").fetchone()
        return int(row[0]) if row is not None else 0

    @staticmethod
    def _select_inferable_tables(
        manifest: SchemaManifest,
        *,
        strict_provenance: bool,
    ) -> tuple[list[TableSchema], list[str], list[str]]:
        inferable_tables: list[TableSchema] = []
        blocked_tables: list[str] = []
        missing_provenance: list[str] = []
        for table in manifest.tables:
            provenance = manifest.table_provenance.get(table.table_key)
            if provenance is None:
                missing_provenance.append(table.table_key)
                if strict_provenance:
                    continue
                blocked_tables.append(table.table_key)
                continue
            if provenance.derivation_kind != "inferred_relation":
                continue
            if provenance.inference_status != "inferred":
                blocked_tables.append(table.table_key)
                continue
            inferable_tables.append(table)
        return inferable_tables, blocked_tables, missing_provenance

    @staticmethod
    def _build_override_records(
        inferable_tables: Sequence[TableSchema],
        *,
        context: _OverrideRecordContext,
    ) -> tuple[
        dict[str, SchemaVersionRecord],
        list[TableSchemaOverrideVersionRecord],
        list[TableSchemaOverrideRegistryRecord],
    ]:
        schema_versions: dict[str, SchemaVersionRecord] = {}
        override_versions: list[TableSchemaOverrideVersionRecord] = []
        override_registry: list[TableSchemaOverrideRegistryRecord] = []

        for table in inferable_tables:
            provenance = context.manifest.table_provenance.get(table.table_key)
            if provenance is None:
                msg = f"Missing table provenance for override refresh: {table.table_key}"
                raise ValueError(msg)
            schema_json = table.to_json_obj()
            schema_digest = fingerprint(schema_json)
            schema_hash = _schema_hash_for_override(
                table=table,
                provenance=provenance,
                strict_hash_match=context.request.strict_hash_match,
            )
            if schema_digest not in schema_versions:
                renderer_cache = arrow_contract_renderer_cache(table, provenance=provenance)
                schema_versions[schema_digest] = SchemaVersionRecord(
                    schema_digest=schema_digest,
                    schema_hash=schema_hash,
                    schema_json=schema_json,
                    renderer_cache=renderer_cache,
                    created_at=context.now,
                )
            override_versions.append(
                TableSchemaOverrideVersionRecord(
                    version_id=context.version_id,
                    table_key=table.table_key,
                    schema_digest=schema_digest,
                    schema_hash=schema_hash,
                    catalog_hash=context.catalog_hash,
                    created_at=context.now,
                )
            )
            override_registry.append(
                TableSchemaOverrideRegistryRecord(
                    table_key=table.table_key,
                    schema_digest=schema_digest,
                    schema_hash=schema_hash,
                    version_id=context.version_id,
                    updated_at=context.now,
                )
            )
        return schema_versions, override_versions, override_registry

    def set_override_registry_version(
        self,
        *,
        table_key: str,
        schema_digest: str | None = None,
        version_id: str | None = None,
    ) -> TableSchemaOverrideRegistryRecord:
        """Pin the override registry for a table key to a prior version.

        Returns
        -------
        TableSchemaOverrideRegistryRecord
            Updated override registry record.

        Raises
        ------
        KeyError
            If the requested override version cannot be found.
        RuntimeError
            If the gateway is read-only.
        ValueError
            If schema_digest and version_id are both missing.
        """
        if schema_digest is None and version_id is None:
            msg = "schema_digest or version_id is required to update override registry"
            raise ValueError(msg)

        if getattr(self._gateway, "config", None) is not None and self._gateway.config.read_only:
            msg = "Cannot update override registry in a read-only storage gateway"
            raise RuntimeError(msg)

        now = utc_now()
        versions_ref = meta_table_ref("metadata.table_schema_override_versions")
        params: list[object] = [table_key]
        filters: list[str] = ["table_key = ?"]

        if schema_digest is not None:
            filters.append("schema_digest = ?")
            params.append(schema_digest)
        if version_id is not None:
            filters.append("version_id = ?")
            params.append(version_id)

        where_clause = " AND ".join(filters)
        row = self._con.execute(
            f"""
            SELECT table_key, schema_digest, schema_hash, version_id
            FROM {versions_ref}
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT 1
            """,
            params,
        ).fetchone()

        if row is None:
            msg = f"Override version not found for {table_key}"
            raise KeyError(msg)

        record = TableSchemaOverrideRegistryRecord(
            table_key=str(row[0]),
            schema_digest=str(row[1]),
            schema_hash=str(row[2]),
            version_id=str(row[3]),
            updated_at=now,
        )
        self.record_override_registry_batch([record])
        return record


def _schema_hash_for_override(
    *,
    table: TableSchema,
    provenance: TableProvenance,
    strict_hash_match: bool,
) -> str:
    computed_hash = compute_schema_hash(table)
    provenance_hash = provenance.schema_hash
    if strict_hash_match and provenance_hash != computed_hash:
        msg = (
            f"Schema hash mismatch for {table.table_key}: "
            f"provenance={provenance_hash} computed={computed_hash}"
        )
        raise ValueError(msg)
    return provenance_hash


def _schema_from_renderer_cache(renderer_cache_raw: object) -> pa.Schema | None:
    renderer_cache = decode_json_dict(renderer_cache_raw)
    payload = renderer_cache.get("arrow_schema_ipc_b64")
    if not isinstance(payload, str):
        return None
    return schema_from_ipc_payload(payload)


def _renderer_cache_has_arrow_schema(renderer_cache: Mapping[str, object]) -> bool:
    payload = renderer_cache.get("arrow_schema_ipc_b64")
    return isinstance(payload, str) and bool(payload)


def _schema_metadata_value(schema: pa.Schema, key: str) -> str | None:
    metadata = schema.metadata
    if not metadata:
        return None
    raw = metadata.get(key.encode("utf-8"))
    if raw is None:
        return None
    return raw.decode("utf-8")


def _decode_optional_json_dict(value: object | None) -> dict[str, object] | None:
    if value is None:
        return None
    decoded = decode_json_dict(value)
    return decoded if decoded else None


def _decode_optional_column_stats(value: object | None) -> ColumnStatsPayload | None:
    decoded = _decode_optional_json_dict(value)
    if decoded is None:
        return None
    payload: ColumnStatsPayload = {}
    for column_name, entry_raw in decoded.items():
        if not isinstance(column_name, str):
            return None
        entry = _coerce_column_stats_entry(entry_raw)
        if entry is None:
            return None
        payload[column_name] = entry
    return payload or None


def _coerce_column_stats_entry(value: object) -> ColumnStatsEntry | None:
    if not isinstance(value, Mapping):
        return None
    entry: ColumnStatsEntry = {}
    for key in ("null_count", "non_null_count", "distinct_count_max"):
        raw = value.get(key)
        if raw is None:
            continue
        if not _is_int(raw):
            return None
        entry[key] = raw
    for key in ("avg_length",):
        raw = value.get(key)
        if raw is None:
            continue
        if not _is_floatlike(raw):
            return None
        entry[key] = float(raw)
    for key in ("min", "max"):
        if key in value:
            entry[key] = value[key]
    return entry


type _DatasetStatsKey = Literal[
    "row_count",
    "batch_count",
    "total_bytes",
    "manifest_row_count",
]


_DATASET_INT_KEYS: tuple[_DatasetStatsKey, ...] = (
    "row_count",
    "batch_count",
    "total_bytes",
    "manifest_row_count",
)


def _decode_optional_dataset_stats(value: object | None) -> DatasetStatsPayload | None:
    decoded = _decode_optional_json_dict(value)
    if decoded is None:
        return None
    payload: DatasetStatsPayload = {}
    for key in _DATASET_INT_KEYS:
        if not _apply_optional_dataset_int(payload, decoded, key):
            return None
    parquet_stats = decoded.get("parquet_stats")
    if parquet_stats is not None:
        parquet_payload = _coerce_string_object_mapping(parquet_stats)
        if parquet_payload is None:
            return None
        payload["parquet_stats"] = parquet_payload
    return payload or None


type _DerivedIntKey = Literal[
    "dictionary_max_cardinality",
    "row_group_size",
    "data_page_size",
]


_DERIVED_INT_KEYS: tuple[_DerivedIntKey, ...] = (
    "dictionary_max_cardinality",
    "row_group_size",
    "data_page_size",
)


def _decode_optional_derived_settings(value: object | None) -> DerivedSettingsPayload | None:
    decoded = _decode_optional_json_dict(value)
    if decoded is None:
        return None
    payload: DerivedSettingsPayload = {}
    valid = _apply_optional_str(payload, decoded, "extras_policy") and _apply_optional_str_list(
        payload, decoded, "dictionary_encode_columns"
    )
    if valid:
        for key in _DERIVED_INT_KEYS:
            if not _apply_optional_derived_int(payload, decoded, key):
                valid = False
                break
    if valid:
        valid = _apply_optional_bool(
            payload, decoded, "unify_dictionaries"
        ) and _apply_optional_float(payload, decoded, "avg_row_bytes")
    if not valid:
        return None
    return payload or None


def _coerce_string_object_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    payload: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        payload[key] = item
    return payload


def _apply_optional_dataset_int(
    payload: DatasetStatsPayload,
    decoded: Mapping[str, object],
    key: _DatasetStatsKey,
) -> bool:
    raw = decoded.get(key)
    if raw is None:
        return True
    if not _is_int(raw):
        return False
    payload[key] = raw
    return True


def _apply_optional_str(
    payload: DerivedSettingsPayload,
    decoded: Mapping[str, object],
    key: Literal["extras_policy"],
) -> bool:
    raw = decoded.get(key)
    if raw is None:
        return True
    if not isinstance(raw, str):
        return False
    payload[key] = raw
    return True


def _apply_optional_str_list(
    payload: DerivedSettingsPayload,
    decoded: Mapping[str, object],
    key: Literal["dictionary_encode_columns"],
) -> bool:
    raw = decoded.get(key)
    if raw is None:
        return True
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        return False
    payload[key] = list(raw)
    return True


def _apply_optional_derived_int(
    payload: DerivedSettingsPayload,
    decoded: Mapping[str, object],
    key: _DerivedIntKey,
) -> bool:
    raw = decoded.get(key)
    if raw is None:
        return True
    if not _is_int(raw):
        return False
    payload[key] = raw
    return True


def _apply_optional_bool(
    payload: DerivedSettingsPayload,
    decoded: Mapping[str, object],
    key: Literal["unify_dictionaries"],
) -> bool:
    raw = decoded.get(key)
    if raw is None:
        return True
    if not isinstance(raw, bool):
        return False
    payload[key] = raw
    return True


def _apply_optional_float(
    payload: DerivedSettingsPayload,
    decoded: Mapping[str, object],
    key: Literal["avg_row_bytes"],
) -> bool:
    raw = decoded.get(key)
    if raw is None:
        return True
    if not _is_floatlike(raw):
        return False
    payload[key] = float(raw)
    return True


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_floatlike(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _append_sample(target: list[str], value: str, *, limit: int) -> None:
    if len(target) < limit:
        target.append(value)


def _contract_drift_payload(report: _ContractDriftReport) -> dict[str, object]:
    return {
        "total_tables": report.total_tables,
        "missing_contracts": report.missing_contracts,
        "missing_contract_metadata": report.missing_contract_metadata,
        "hash_mismatches": report.hash_mismatches,
        "digest_mismatches": report.digest_mismatches,
        "missing_contract_samples": list(report.missing_contract_samples),
        "missing_metadata_samples": list(report.missing_metadata_samples),
        "mismatch_samples": list(report.mismatch_samples),
    }


__all__ = [
    "OverrideRegistryRefreshResult",
    "PersistSchemaManifestResult",
    "SchemaCatalogProvider",
    "SchemaCatalogRequest",
    "SchemaCatalogTracking",
    "SchemaManifestRunRecord",
    "SchemaVersionRecord",
    "TableSchemaOverrideRegistryRecord",
    "TableSchemaOverrideVersionRecord",
    "TableSchemaRegistryRecord",
]
