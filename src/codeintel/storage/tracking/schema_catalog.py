"""Schema catalog persistence for metadata schema registries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.schemas.serde import table_schema_from_json_obj
from codeintel.core.time import utc_now
from codeintel.storage.helpers.json import decode_json_dict
from codeintel.storage.metadata.catalogs import build_catalog_entry, upsert_canonical_catalog
from codeintel.storage.upsert import UpsertSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from codeintel.build.schemas.manifest import SchemaManifest
    from codeintel.build.schemas.schema_index import SchemaIndex
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.storage.gateway.protocol import StorageGateway


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
                record.schema_json,
                record.renderer_cache,
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
            upsert=UpsertSpec(
                conflict_columns=("schema_digest",),
                update_columns=(),
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
        row = self._con.execute(
            """
            SELECT v.schema_json
            FROM metadata.table_schema_registry AS r
            JOIN metadata.schema_versions AS v
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
        sql = (
            "SELECT r.table_key, v.schema_json "
            "FROM metadata.table_schema_registry AS r "
            "JOIN metadata.schema_versions AS v "
            "  ON r.schema_digest = v.schema_digest "
            "WHERE r.derivation_kind = ? "
            "  AND r.inference_status IN (?, ?) "
            f"  AND r.table_key IN ({placeholders})"
        )
        params: list[object] = ["inferred_ibis", "inferred", "override", *allowed_keys]
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
        run_id: str,
        repo: str,
        commit: str,
        catalog_inputs: Mapping[str, object] | None = None,
        include_views: bool = True,
        strict_provenance: bool = True,
        strict_hash_match: bool = True,
        now: datetime | None = None,
        catalog_kind: str = "schema_manifest_v2",
        manifest_kind: str = "schema_manifest_v2",
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
        ValueError
            If strict checks fail in batch compilation.
        """
        if getattr(self._gateway, "config", None) is not None and self._gateway.config.read_only:
            msg = "Cannot persist schema manifest into a read-only storage gateway"
            raise RuntimeError(msg)

        from codeintel.storage.tracking.schema_catalog_compile import compile_schema_catalog_batches

        batches = compile_schema_catalog_batches(
            manifest,
            run_id=run_id,
            repo=repo,
            commit=commit,
            now=now,
            catalog_kind=catalog_kind,
            manifest_kind=manifest_kind,
            include_views=include_views,
            strict_provenance=strict_provenance,
            strict_hash_match=strict_hash_match,
            catalog_inputs=dict(catalog_inputs) if catalog_inputs is not None else None,
        )

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
            views=len(manifest.views) if include_views else 0,
            schema_versions_rows=n_versions,
            table_schema_registry_rows=n_registry,
            schema_manifest_runs_rows=n_runs,
        )


__all__ = [
    "PersistSchemaManifestResult",
    "SchemaCatalogTracking",
    "SchemaManifestRunRecord",
    "SchemaVersionRecord",
    "TableSchemaRegistryRecord",
]
