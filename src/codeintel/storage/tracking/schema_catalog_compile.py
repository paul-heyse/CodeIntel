"""Compile SchemaManifest into schema catalog persistence batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.hashing import schema_hash as compute_schema_hash
from codeintel.core.time import utc_now
from codeintel.storage.tracking.schema_catalog import (
    SchemaManifestRunRecord,
    SchemaVersionRecord,
    TableSchemaRegistryRecord,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from codeintel.core.manifests import SchemaManifest, TableProvenance
    from codeintel.core.schemas.primitives import TableSchema


_SCHEMA_MANIFEST_CATALOG_KIND = "schema_manifest_v2"


@dataclass(frozen=True, slots=True)
class SchemaCatalogBatches:
    """Compiled persistence batches for a SchemaManifest."""

    catalog_kind: str
    catalog_hash: str
    catalog_payload: dict[str, object]
    catalog_inputs: dict[str, object] | None
    schema_versions: tuple[SchemaVersionRecord, ...]
    table_schema_registry: tuple[TableSchemaRegistryRecord, ...]
    schema_manifest_runs: tuple[SchemaManifestRunRecord, ...]


def compile_schema_catalog_batches(
    manifest: SchemaManifest,
    *,
    run_id: str,
    repo: str,
    commit: str,
    now: datetime | None = None,
    catalog_kind: str = _SCHEMA_MANIFEST_CATALOG_KIND,
    manifest_kind: str = _SCHEMA_MANIFEST_CATALOG_KIND,
    include_views: bool = True,
    strict_provenance: bool = True,
    strict_hash_match: bool = True,
    catalog_inputs: Mapping[str, object] | None = None,
) -> SchemaCatalogBatches:
    """Compile SchemaManifest -> schema registry persistence batches.

    Parameters
    ----------
    manifest
        Compiled schema manifest. For best results, compile with include_provenance=True.
    run_id
        Run identifier for schema_manifest_runs linkage.
    repo
        Repository identifier.
    commit
        Commit SHA.
    now
        Timestamp used consistently across all generated records.
    catalog_kind
        Catalog kind for metadata.canonical_catalogs.
    manifest_kind
        Kind label stored in metadata.schema_manifest_runs.
    include_views
        When True, include view schemas in schema registry batches.
    strict_provenance
        When True, raise if required provenance is missing for any included table/view.
    strict_hash_match
        When True, raise if provenance schema_hash differs from computed schema_hash.
    catalog_inputs
        Optional inputs metadata to store alongside the canonical catalog entry.

    Returns
    -------
    SchemaCatalogBatches
        Catalog payload/hash + record batches for persistence.

    Raises
    ------
    ValueError
        If manifest is not v2, or strict_provenance is True and provenance is missing,
        or strict_hash_match is True and a provenance hash mismatch is detected.
    """
    if not getattr(manifest, "is_v2", False):
        msg = f"Expected SchemaManifest v2; got version={getattr(manifest, 'version', None)!r}"
        raise ValueError(msg)

    ts = now or utc_now()

    payload = dict(manifest.to_json_obj())
    catalog_hash = fingerprint(payload)
    inputs_payload = dict(catalog_inputs) if catalog_inputs is not None else None

    def _sorted_by_table_key(schemas: tuple[TableSchema, ...]) -> tuple[TableSchema, ...]:
        return tuple(sorted(schemas, key=lambda schema: schema.table_key))

    def _require_provenance(
        *,
        table_key: str,
        provenance: TableProvenance | None,
        kind: str,
    ) -> TableProvenance | None:
        if provenance is not None:
            return provenance
        if strict_provenance:
            msg = (
                f"Missing {kind} provenance for {table_key}. "
                "Compile the manifest with include_provenance=True."
            )
            raise ValueError(msg)
        return None

    def _schema_digest(schema: TableSchema) -> str:
        return fingerprint(schema.to_json_obj())

    schema_versions_by_digest: dict[str, SchemaVersionRecord] = {}

    tables = _sorted_by_table_key(manifest.tables)
    views = _sorted_by_table_key(manifest.views) if include_views else ()

    for schema in tables + views:
        schema_json = schema.to_json_obj()
        digest = fingerprint(schema_json)
        if digest in schema_versions_by_digest:
            continue
        schema_versions_by_digest[digest] = SchemaVersionRecord(
            schema_digest=digest,
            schema_hash=compute_schema_hash(schema),
            schema_json=schema_json,
            renderer_cache=None,
            created_at=ts,
        )

    schema_versions = tuple(
        schema_versions_by_digest[digest] for digest in sorted(schema_versions_by_digest)
    )

    registry_records: list[TableSchemaRegistryRecord] = []

    for table in tables:
        table_key = table.table_key
        provenance = _require_provenance(
            table_key=table_key,
            provenance=manifest.table_provenance.get(table_key),
            kind="table",
        )
        computed_hash = compute_schema_hash(table)
        provenance_hash = provenance.schema_hash if provenance is not None else computed_hash
        if strict_hash_match and provenance is not None and provenance_hash != computed_hash:
            msg = (
                f"Schema hash mismatch for {table_key}: "
                f"provenance={provenance_hash} computed={computed_hash}"
            )
            raise ValueError(msg)

        registry_records.append(
            TableSchemaRegistryRecord(
                table_key=table_key,
                schema_digest=_schema_digest(table),
                schema_hash=provenance_hash,
                derivation_kind=(
                    provenance.derivation_kind if provenance is not None else "explicit_override"
                ),
                derivation_source=(
                    provenance.derivation_source if provenance is not None else "manifest"
                ),
                inference_status=(
                    provenance.inference_status if provenance is not None else None
                ),
                inference_error=(provenance.inference_error if provenance is not None else None),
                catalog_hash=catalog_hash,
                updated_at=ts,
            )
        )

    for view in views:
        view_key = view.table_key
        provenance = _require_provenance(
            table_key=view_key,
            provenance=manifest.view_provenance.get(view_key),
            kind="view",
        )
        computed_hash = compute_schema_hash(view)
        provenance_hash = provenance.schema_hash if provenance is not None else computed_hash
        if strict_hash_match and provenance is not None and provenance_hash != computed_hash:
            msg = (
                f"Schema hash mismatch for {view_key}: "
                f"provenance={provenance_hash} computed={computed_hash}"
            )
            raise ValueError(msg)

        registry_records.append(
            TableSchemaRegistryRecord(
                table_key=view_key,
                schema_digest=_schema_digest(view),
                schema_hash=provenance_hash,
                derivation_kind=(
                    provenance.derivation_kind if provenance is not None else "view_inferred"
                ),
                derivation_source=(
                    provenance.derivation_source if provenance is not None else "duckdb"
                ),
                inference_status=(
                    provenance.inference_status if provenance is not None else None
                ),
                inference_error=(provenance.inference_error if provenance is not None else None),
                catalog_hash=catalog_hash,
                updated_at=ts,
            )
        )

    table_schema_registry = tuple(sorted(registry_records, key=lambda rec: rec.table_key))

    schema_manifest_runs = (
        SchemaManifestRunRecord(
            run_id=run_id,
            repo=repo,
            commit=commit,
            manifest_kind=manifest_kind,
            catalog_hash=catalog_hash,
            created_at=ts,
        ),
    )

    return SchemaCatalogBatches(
        catalog_kind=catalog_kind,
        catalog_hash=catalog_hash,
        catalog_payload=payload,
        catalog_inputs=inputs_payload,
        schema_versions=schema_versions,
        table_schema_registry=table_schema_registry,
        schema_manifest_runs=schema_manifest_runs,
    )


__all__ = [
    "SchemaCatalogBatches",
    "compile_schema_catalog_batches",
]
