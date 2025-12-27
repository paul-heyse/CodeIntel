"""Schema catalog persistence for metadata schema registries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.execution.ids import new_uuid_str
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.hashing import schema_hash as compute_schema_hash
from codeintel.core.schemas.serde import table_schema_from_json_obj
from codeintel.core.time import utc_now
from codeintel.storage.constants import META_CATALOG_NAME
from codeintel.storage.helpers.json import decode_json_dict, normalize_duckdb_json_value
from codeintel.storage.metadata.catalogs import build_catalog_entry, upsert_canonical_catalog
from codeintel.storage.metadata.meta_catalog import meta_table_ref
from codeintel.storage.tracking.schema_catalog_compile import compile_schema_catalog_batches
from codeintel.storage.tracking.schema_catalog_models import (
    OverrideRegistryRefreshResult,
    SchemaCatalogRequest,
    SchemaManifestRunRecord,
    SchemaVersionRecord,
    TableSchemaOverrideRegistryRecord,
    TableSchemaOverrideVersionRecord,
    TableSchemaRegistryRecord,
)
from codeintel.storage.upsert import UpsertSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.build.schemas.manifest import SchemaManifest
    from codeintel.build.schemas.schema_index import SchemaIndex
    from codeintel.core.manifests import TableProvenance
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.storage.gateway.protocol import StorageGateway


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
        registry_ref = meta_table_ref("metadata.table_schema_registry")
        versions_ref = meta_table_ref("metadata.schema_versions")
        row = self._con.execute(
            f"""
            SELECT v.schema_json
            FROM {registry_ref} AS r
            JOIN {versions_ref} AS v
              ON r.schema_digest = v.schema_digest
            WHERE r.table_key = ?
            """,
            [table_key],
        ).fetchone()
        if row is None:
            return None
        schema_json = decode_json_dict(row[0])
        if not schema_json:
            return None
        return table_schema_from_json_obj(schema_json)

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
        manifest_runs_ref = meta_table_ref("metadata.schema_manifest_runs")
        registry_ref = meta_table_ref("metadata.table_schema_registry")
        override_ref = meta_table_ref("metadata.table_schema_override_registry")

        latest = self._con.execute(
            f"""
            SELECT catalog_hash, repo, commit, manifest_kind, created_at
            FROM {manifest_runs_ref}
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()

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

        catalog_hash, repo, commit, manifest_kind, created_at = latest
        registry_rows = self._con.execute(
            f"""
            SELECT COUNT(*), MAX(updated_at)
            FROM {registry_ref}
            WHERE catalog_hash = ?
            """,
            [catalog_hash],
        ).fetchone()

        registry_count = int(registry_rows[0]) if registry_rows is not None else 0
        registry_updated_at = registry_rows[1] if registry_rows is not None else None
        registry_stale = registry_updated_at is None or registry_updated_at < created_at

        inferable_row = self._con.execute(
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

        inferable_total = int(inferable_row[0] or 0) if inferable_row is not None else 0
        inferred_count = int(inferable_row[1] or 0) if inferable_row is not None else 0
        inference_error_count = int(inferable_row[2] or 0) if inferable_row is not None else 0
        inference_success_rate = inferred_count / inferable_total if inferable_total else None

        override_rows = self._con.execute(f"SELECT COUNT(*) FROM {override_ref}").fetchone()
        override_registry_rows = int(override_rows[0]) if override_rows is not None else 0

        status = "ok"
        if registry_stale or override_registry_rows == 0:
            status = "warn"
        if inference_error_count:
            status = "warn"

        return {
            "status": status,
            "latest_manifest": {
                "catalog_hash": str(catalog_hash),
                "repo": str(repo),
                "commit": str(commit),
                "manifest_kind": str(manifest_kind),
                "created_at": created_at.isoformat() if created_at is not None else None,
            },
            "registry_rows": registry_count,
            "registry_updated_at": (
                registry_updated_at.isoformat() if registry_updated_at is not None else None
            ),
            "registry_stale": registry_stale,
            "override_registry_rows": override_registry_rows,
            "inferable_total": inferable_total,
            "inferred_count": inferred_count,
            "inference_error_count": inference_error_count,
            "inference_success_rate": inference_success_rate,
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
        inferable_tables: list[TableSchema] = []
        blocked_tables: list[str] = []

        for table in manifest.tables:
            provenance = manifest.table_provenance.get(table.table_key)
            if provenance is None:
                if request.strict_provenance:
                    msg = f"Missing table provenance for override refresh: {table.table_key}"
                    raise ValueError(msg)
                blocked_tables.append(table.table_key)
                continue
            if provenance.derivation_kind != "inferred_relation":
                continue
            if provenance.inference_status != "inferred":
                blocked_tables.append(table.table_key)
                continue
            inferable_tables.append(table)

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
        schema_versions: dict[str, SchemaVersionRecord] = {}
        override_versions: list[TableSchemaOverrideVersionRecord] = []
        override_registry: list[TableSchemaOverrideRegistryRecord] = []

        for table in inferable_tables:
            provenance = manifest.table_provenance.get(table.table_key)
            if provenance is None:
                msg = f"Missing table provenance for override refresh: {table.table_key}"
                raise ValueError(msg)
            schema_json = table.to_json_obj()
            schema_digest = fingerprint(schema_json)
            schema_hash = _schema_hash_for_override(
                table=table,
                provenance=provenance,
                strict_hash_match=request.strict_hash_match,
            )
            if schema_digest not in schema_versions:
                schema_versions[schema_digest] = SchemaVersionRecord(
                    schema_digest=schema_digest,
                    schema_hash=schema_hash,
                    schema_json=schema_json,
                    renderer_cache=None,
                    created_at=now,
                )
            override_versions.append(
                TableSchemaOverrideVersionRecord(
                    version_id=version_id,
                    table_key=table.table_key,
                    schema_digest=schema_digest,
                    schema_hash=schema_hash,
                    catalog_hash=catalog_hash,
                    created_at=now,
                )
            )
            override_registry.append(
                TableSchemaOverrideRegistryRecord(
                    table_key=table.table_key,
                    schema_digest=schema_digest,
                    schema_hash=schema_hash,
                    version_id=version_id,
                    updated_at=now,
                )
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


__all__ = [
    "OverrideRegistryRefreshResult",
    "PersistSchemaManifestResult",
    "SchemaCatalogRequest",
    "SchemaCatalogTracking",
    "SchemaManifestRunRecord",
    "SchemaVersionRecord",
    "TableSchemaOverrideRegistryRecord",
    "TableSchemaOverrideVersionRecord",
    "TableSchemaRegistryRecord",
]
