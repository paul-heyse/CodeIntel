"""Dataset manifest schema helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict, cast

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
from codeintel.core.schemas.schema_catalog_models import DerivedSettingsPayload
from codeintel.core.schemas.serde import table_schema_from_json_obj


class WriteSettingsPayload(TypedDict, total=False):
    """Typed payload for dataset write settings persisted in manifests."""

    compression: str
    max_rows_per_file: int
    row_group_size: int
    data_page_size: int
    dictionary_encode: bool
    dictionary_max_cardinality: int
    dictionary_encode_columns: list[str]
    unify_dictionaries: bool


@dataclass(frozen=True, slots=True)
class DatasetTuningMetadata:
    """Parsed tuning metadata from dataset manifests.

    Attributes
    ----------
    inferred_settings
        Inferred settings derived from observations, if present.
    write_settings
        Persisted write settings, if present.
    """

    inferred_settings: DerivedSettingsPayload | None
    write_settings: WriteSettingsPayload | None


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
    inferred = inferred_settings_from_manifest(manifest)
    if inferred is None:
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


def inferred_settings_from_manifest(
    manifest: ArrowDatasetManifest,
) -> DerivedSettingsPayload | None:
    """Return inferred settings payload from a dataset manifest.

    Parameters
    ----------
    manifest
        Dataset manifest to inspect.

    Returns
    -------
    DerivedSettingsPayload | None
        Parsed inferred settings payload, or None when unavailable.
    """
    extras = manifest.extras
    if not isinstance(extras, Mapping):
        return None
    raw = extras.get("inferred_settings")
    if not isinstance(raw, Mapping):
        return None
    payload: DerivedSettingsPayload = {}
    extras_policy = _read_str(raw.get("extras_policy"))
    if extras_policy is not None and extras_policy in EXTRAS_POLICIES:
        payload["extras_policy"] = extras_policy
    dictionary_columns = _read_str_list(raw.get("dictionary_encode_columns"))
    if dictionary_columns:
        payload["dictionary_encode_columns"] = dictionary_columns
    dictionary_max = _read_int(raw.get("dictionary_max_cardinality"))
    if dictionary_max is not None:
        payload["dictionary_max_cardinality"] = dictionary_max
    unify_dictionaries = _read_bool(raw.get("unify_dictionaries"))
    if unify_dictionaries is not None:
        payload["unify_dictionaries"] = unify_dictionaries
    row_group_size = _read_int(raw.get("row_group_size"))
    if row_group_size is not None:
        payload["row_group_size"] = row_group_size
    data_page_size = _read_int(raw.get("data_page_size"))
    if data_page_size is not None:
        payload["data_page_size"] = data_page_size
    avg_row_bytes = _read_float(raw.get("avg_row_bytes"))
    if avg_row_bytes is not None:
        payload["avg_row_bytes"] = avg_row_bytes
    return payload or None


def write_settings_from_manifest(
    manifest: ArrowDatasetManifest,
) -> WriteSettingsPayload | None:
    """Return write settings payload from a dataset manifest.

    Parameters
    ----------
    manifest
        Dataset manifest to inspect.

    Returns
    -------
    WriteSettingsPayload | None
        Parsed write settings payload, or None when unavailable.
    """
    extras = manifest.extras
    if not isinstance(extras, Mapping):
        return None
    raw = extras.get("write_settings")
    if not isinstance(raw, Mapping):
        return None
    payload: dict[str, object] = {}
    _set_payload_value(payload, "compression", _read_str(raw.get("compression")))
    _set_payload_value(payload, "max_rows_per_file", _read_int(raw.get("max_rows_per_file")))
    _set_payload_value(payload, "row_group_size", _read_int(raw.get("row_group_size")))
    _set_payload_value(payload, "data_page_size", _read_int(raw.get("data_page_size")))
    _set_payload_value(payload, "dictionary_encode", _read_bool(raw.get("dictionary_encode")))
    _set_payload_value(
        payload,
        "dictionary_max_cardinality",
        _read_int(raw.get("dictionary_max_cardinality")),
    )
    _set_payload_list(
        payload,
        "dictionary_encode_columns",
        _read_str_list(raw.get("dictionary_encode_columns")),
    )
    _set_payload_value(payload, "unify_dictionaries", _read_bool(raw.get("unify_dictionaries")))
    if not payload:
        return None
    return cast("WriteSettingsPayload", payload)


def tuning_metadata_from_manifest(
    manifest: ArrowDatasetManifest,
) -> DatasetTuningMetadata | None:
    """Return parsed tuning metadata from a dataset manifest.

    Parameters
    ----------
    manifest
        Dataset manifest to inspect.

    Returns
    -------
    DatasetTuningMetadata | None
        Parsed tuning metadata, or None when no settings are present.
    """
    inferred_settings = inferred_settings_from_manifest(manifest)
    write_settings = write_settings_from_manifest(manifest)
    if inferred_settings is None and write_settings is None:
        return None
    return DatasetTuningMetadata(
        inferred_settings=inferred_settings,
        write_settings=write_settings,
    )


def _read_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _read_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _read_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _read_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _read_str_list(value: object) -> list[str] | None:
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return None


def _set_payload_value(
    payload: dict[str, object],
    key: str,
    value: object | None,
) -> None:
    if value is None:
        return
    payload[key] = value


def _set_payload_list(
    payload: dict[str, object],
    key: str,
    value: list[str] | None,
) -> None:
    if not value:
        return
    payload[key] = value


__all__ = [
    "DatasetTuningMetadata",
    "WriteSettingsPayload",
    "arrow_schema_from_manifest",
    "inferred_settings_from_manifest",
    "table_schema_from_manifest",
    "tuning_metadata_from_manifest",
    "write_settings_from_manifest",
]
