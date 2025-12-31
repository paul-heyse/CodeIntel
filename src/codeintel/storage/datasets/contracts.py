"""Dataset manifest schema helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pyarrow as pa

from codeintel.core.manifests import ArrowDatasetManifest
from codeintel.core.schemas.arrow_gen import (
    EXTRAS_POLICIES,
    ArrowSchemaMetadata,
    ArrowSchemaProvenance,
    ExtrasPolicy,
    arrow_schema_from_table_schema,
)
from codeintel.core.schemas.hashing import schema_digest, schema_hash
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.serde import table_schema_from_json_obj


def table_schema_from_manifest(manifest: ArrowDatasetManifest) -> TableSchema | None:
    """Return a TableSchema parsed from dataset manifest extras, when available.

    Returns
    -------
    TableSchema | None
        The parsed table schema, or None when not present.

    Raises
    ------
    ValueError
        If the manifest table_key does not match the parsed schema table_key.
    """
    extras = manifest.extras
    if not isinstance(extras, Mapping):
        return None
    raw_schema = extras.get("table_schema")
    if not isinstance(raw_schema, Mapping):
        return None
    table_schema = table_schema_from_json_obj(raw_schema)
    if table_schema.table_key != manifest.table_key:
        msg = (
            "Dataset manifest table_schema table_key mismatch: "
            f"{table_schema.table_key} != {manifest.table_key}"
        )
        raise ValueError(msg)
    return table_schema


def arrow_schema_from_manifest(manifest: ArrowDatasetManifest) -> pa.Schema | None:
    """Return a PyArrow schema built from manifest metadata when possible.

    Returns
    -------
    pa.Schema | None
        The schema derived from the manifest, or None when unavailable.
    """
    table_schema = table_schema_from_manifest(manifest)
    if table_schema is None:
        return None
    extras_policy = _extras_policy_from_manifest(manifest)
    provenance = _provenance_from_manifest(manifest)
    metadata = ArrowSchemaMetadata(
        schema_hash=manifest.schema_hash or schema_hash(table_schema),
        schema_digest=schema_digest(table_schema),
        extras_policy=extras_policy,
        provenance=provenance,
    )
    return arrow_schema_from_table_schema(table_schema=table_schema, metadata=metadata)


def _extras_policy_from_manifest(manifest: ArrowDatasetManifest) -> ExtrasPolicy | None:
    extras = manifest.extras
    if not isinstance(extras, Mapping):
        return None
    inferred = extras.get("inferred_settings")
    if not isinstance(inferred, Mapping):
        return None
    raw_policy = inferred.get("extras_policy")
    if isinstance(raw_policy, str) and raw_policy in EXTRAS_POLICIES:
        return cast("ExtrasPolicy", raw_policy)
    return None


def _provenance_from_manifest(manifest: ArrowDatasetManifest) -> ArrowSchemaProvenance | None:
    extras = manifest.extras
    if not isinstance(extras, Mapping):
        return None
    raw = extras.get("provenance")
    if not isinstance(raw, Mapping):
        return None
    derivation_kind = raw.get("derivation_kind")
    derivation_source = raw.get("derivation_source")
    if not isinstance(derivation_kind, str) and not isinstance(derivation_source, str):
        return None
    return ArrowSchemaProvenance(
        derivation_kind=derivation_kind if isinstance(derivation_kind, str) else None,
        derivation_source=derivation_source if isinstance(derivation_source, str) else None,
    )


__all__ = ["arrow_schema_from_manifest", "table_schema_from_manifest"]
