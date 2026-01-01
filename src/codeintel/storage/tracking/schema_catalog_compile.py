"""Compile SchemaManifest into schema catalog persistence batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.columnar.ipc import schema_to_ipc_payload
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.arrow_gen import (
    ARROW_SCHEMA_CONTRACT_VERSION,
    DEFAULT_EXTRAS_COLUMN,
    DEFAULT_EXTRAS_POLICY,
    ArrowSchemaMetadata,
    ArrowSchemaProvenance,
    ExtrasPolicy,
    arrow_contract_for_table_schema,
)
from codeintel.core.schemas.hashing import schema_hash as compute_schema_hash
from codeintel.core.schemas.schema_catalog_models import (
    SchemaCatalogRequest,
    SchemaManifestRunRecord,
    SchemaVersionRecord,
    TableSchemaRegistryRecord,
)
from codeintel.core.schemas.type_mappings import normalize_table_schema_types
from codeintel.core.time import utc_now

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from codeintel.core.manifests import SchemaManifest, TableProvenance
    from codeintel.core.schemas.primitives import TableSchema


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


@dataclass(frozen=True, slots=True)
class _RegistryDefaults:
    catalog_hash: str
    now: datetime
    strict_hash_match: bool


def _sorted_by_table_key(schemas: tuple[TableSchema, ...]) -> tuple[TableSchema, ...]:
    normalized = [normalize_table_schema_types(schema) for schema in schemas]
    return tuple(sorted(normalized, key=lambda schema: schema.table_key))


def _normalize_inputs(catalog_inputs: Mapping[str, object] | None) -> dict[str, object] | None:
    if catalog_inputs is None:
        return None
    return dict(catalog_inputs)


def _require_provenance(
    *,
    table_key: str,
    provenance: TableProvenance | None,
    kind: str,
    strict_provenance: bool,
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


def _schema_hash_for_schema(
    *,
    table_key: str,
    schema: TableSchema,
    provenance: TableProvenance | None,
    strict_hash_match: bool,
) -> str:
    computed_hash = compute_schema_hash(schema)
    if provenance is None:
        return computed_hash
    if strict_hash_match and provenance.schema_hash != computed_hash:
        msg = (
            f"Schema hash mismatch for {table_key}: "
            f"provenance={provenance.schema_hash} computed={computed_hash}"
        )
        raise ValueError(msg)
    return provenance.schema_hash


def _schema_versions_for_schemas(
    *,
    schemas: tuple[TableSchema, ...],
    provenance_by_table_key: Mapping[str, TableProvenance] | None,
    now: datetime,
) -> tuple[SchemaVersionRecord, ...]:
    schema_versions_by_digest: dict[str, SchemaVersionRecord] = {}
    for schema in schemas:
        schema_json = schema.to_json_obj()
        digest = fingerprint(schema_json)
        if digest in schema_versions_by_digest:
            continue
        provenance = (
            provenance_by_table_key.get(schema.table_key)
            if provenance_by_table_key is not None
            else None
        )
        renderer_cache = arrow_contract_renderer_cache(schema, provenance=provenance)
        schema_versions_by_digest[digest] = SchemaVersionRecord(
            schema_digest=digest,
            schema_hash=compute_schema_hash(schema),
            schema_json=schema_json,
            renderer_cache=renderer_cache,
            created_at=now,
        )
    return tuple(schema_versions_by_digest[digest] for digest in sorted(schema_versions_by_digest))


def _build_schema_versions(
    manifest: SchemaManifest,
    *,
    include_views: bool,
    now: datetime,
) -> tuple[SchemaVersionRecord, ...]:
    schemas = list(_sorted_by_table_key(manifest.tables))
    if include_views:
        schemas.extend(_sorted_by_table_key(manifest.views))
    provenance_by_table_key = dict(manifest.table_provenance)
    if include_views:
        provenance_by_table_key.update(manifest.view_provenance)
    return _schema_versions_for_schemas(
        schemas=tuple(schemas),
        provenance_by_table_key=provenance_by_table_key,
        now=now,
    )


def _extras_policy_for_provenance(provenance: TableProvenance | None) -> ExtrasPolicy:
    if provenance is None:
        return DEFAULT_EXTRAS_POLICY
    if provenance.derivation_kind == "declared_source":
        return "retain"
    return "reject"


def _arrow_provenance(provenance: TableProvenance | None) -> ArrowSchemaProvenance | None:
    if provenance is None:
        return None
    return ArrowSchemaProvenance(
        derivation_kind=provenance.derivation_kind,
        derivation_source=provenance.derivation_source,
        inference_status=provenance.inference_status,
        inference_error=provenance.inference_error,
        producer_target=provenance.producer_target,
        producer_module=provenance.producer_module,
        producer_version=provenance.producer_version,
    )


def arrow_contract_renderer_cache(
    schema: TableSchema,
    *,
    provenance: TableProvenance | None,
) -> dict[str, object]:
    """Build renderer cache payload for a schema's Arrow contract.

    Parameters
    ----------
    schema
        Table schema to serialize as an Arrow contract.
    provenance
        Optional provenance metadata for schema rendering.

    Returns
    -------
    dict[str, object]
        Renderer cache payload with serialized Arrow schema metadata.
    """
    extras_policy: ExtrasPolicy = _extras_policy_for_provenance(provenance)
    metadata = ArrowSchemaMetadata(
        schema_hash=provenance.schema_hash if provenance is not None else None,
        provenance=_arrow_provenance(provenance),
        contract_version=ARROW_SCHEMA_CONTRACT_VERSION,
        extras_policy=extras_policy,
        extras_column=DEFAULT_EXTRAS_COLUMN,
    )
    arrow_schema = arrow_contract_for_table_schema(table_schema=schema, metadata=metadata)
    ipc_payload = schema_to_ipc_payload(arrow_schema)
    return {
        "arrow_schema_ipc_b64": ipc_payload,
        "arrow_schema_contract_version": ARROW_SCHEMA_CONTRACT_VERSION,
        "extras_policy": extras_policy,
        "extras_column": DEFAULT_EXTRAS_COLUMN,
    }


def _build_registry_record(
    *,
    schema: TableSchema,
    provenance: TableProvenance | None,
    defaults: _RegistryDefaults,
    fallback_kind: str,
    fallback_source: str,
) -> TableSchemaRegistryRecord:
    return TableSchemaRegistryRecord(
        table_key=schema.table_key,
        schema_digest=_schema_digest(schema),
        schema_hash=_schema_hash_for_schema(
            table_key=schema.table_key,
            schema=schema,
            provenance=provenance,
            strict_hash_match=defaults.strict_hash_match,
        ),
        derivation_kind=provenance.derivation_kind if provenance is not None else fallback_kind,
        derivation_source=provenance.derivation_source
        if provenance is not None
        else fallback_source,
        inference_status=provenance.inference_status if provenance is not None else None,
        inference_error=provenance.inference_error if provenance is not None else None,
        catalog_hash=defaults.catalog_hash,
        updated_at=defaults.now,
    )


def _build_registry_records(
    manifest: SchemaManifest,
    *,
    request: SchemaCatalogRequest,
    catalog_hash: str,
    now: datetime,
) -> tuple[TableSchemaRegistryRecord, ...]:
    defaults = _RegistryDefaults(
        catalog_hash=catalog_hash,
        now=now,
        strict_hash_match=request.strict_hash_match,
    )
    records: list[TableSchemaRegistryRecord] = []

    tables = _sorted_by_table_key(manifest.tables)
    views = _sorted_by_table_key(manifest.views) if request.include_views else ()

    for table in tables:
        provenance = _require_provenance(
            table_key=table.table_key,
            provenance=manifest.table_provenance.get(table.table_key),
            kind="table",
            strict_provenance=request.strict_provenance,
        )
        records.append(
            _build_registry_record(
                schema=table,
                provenance=provenance,
                defaults=defaults,
                fallback_kind="explicit_override",
                fallback_source="manifest",
            )
        )

    for view in views:
        provenance = _require_provenance(
            table_key=view.table_key,
            provenance=manifest.view_provenance.get(view.table_key),
            kind="view",
            strict_provenance=request.strict_provenance,
        )
        records.append(
            _build_registry_record(
                schema=view,
                provenance=provenance,
                defaults=defaults,
                fallback_kind="view_inferred",
                fallback_source="duckdb",
            )
        )

    return tuple(sorted(records, key=lambda rec: rec.table_key))


def _build_schema_manifest_runs(
    *,
    request: SchemaCatalogRequest,
    catalog_hash: str,
    now: datetime,
) -> tuple[SchemaManifestRunRecord, ...]:
    return (
        SchemaManifestRunRecord(
            run_id=request.run_id,
            repo=request.repo,
            commit=request.commit,
            manifest_kind=request.manifest_kind,
            catalog_hash=catalog_hash,
            created_at=now,
        ),
    )


def compile_schema_catalog_batches(
    manifest: SchemaManifest,
    *,
    request: SchemaCatalogRequest,
) -> SchemaCatalogBatches:
    """Compile SchemaManifest -> schema registry persistence batches.

    Parameters
    ----------
    manifest
        Compiled schema manifest. For best results, compile with include_provenance=True.
    request
        Persistence request parameters (run id, repo, commit, strictness settings).

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

    ts = request.now or utc_now()

    payload = dict(manifest.to_json_obj())
    catalog_hash = fingerprint(payload)
    inputs_payload = _normalize_inputs(request.catalog_inputs)
    schema_versions = _build_schema_versions(
        manifest,
        include_views=request.include_views,
        now=ts,
    )
    table_schema_registry = _build_registry_records(
        manifest,
        request=request,
        catalog_hash=catalog_hash,
        now=ts,
    )
    schema_manifest_runs = _build_schema_manifest_runs(
        request=request,
        catalog_hash=catalog_hash,
        now=ts,
    )

    return SchemaCatalogBatches(
        catalog_kind=request.catalog_kind,
        catalog_hash=catalog_hash,
        catalog_payload=payload,
        catalog_inputs=inputs_payload,
        schema_versions=schema_versions,
        table_schema_registry=table_schema_registry,
        schema_manifest_runs=schema_manifest_runs,
    )


__all__ = [
    "SchemaCatalogBatches",
    "arrow_contract_renderer_cache",
    "compile_schema_catalog_batches",
]
