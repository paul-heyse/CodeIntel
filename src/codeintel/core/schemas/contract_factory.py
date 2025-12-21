"""Shared factory for DatasetContract derivation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from codeintel.core.schemas.contract_policy import (
    default_json_schema_id,
    default_jsonl_filename,
    default_parquet_filename,
)
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.service import SchemaService

if TYPE_CHECKING:
    from codeintel.config.datasets.primitives import CompositeSchema
    from codeintel.core.schemas.row_models import GeneratedRowBinding


_OWNER_PACKAGE_BY_PREFIX: dict[str, Literal["core", "analytics", "graphs", "qa", "docs"]] = {
    "core": "core",
    "analytics": "analytics",
    "graph": "graphs",
    "docs": "docs",
    "qa": "qa",
}
_EXPECTED_TABLE_KEY_PARTS = 2


@dataclass(frozen=True, slots=True)
class DatasetContractOverrides:
    """Optional overrides applied during DatasetContract derivation."""

    json_schema_id: str | None = None
    jsonl_filename: str | None = None
    parquet_filename: str | None = None
    owner: str | None = None
    description: str | None = None
    family: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    upstream_dependencies: tuple[str, ...] = ()
    tags: frozenset[str] = field(default_factory=frozenset)
    validation_profile: Literal["strict", "lenient"] = "strict"
    stable_id: str | None = None
    schema_version: str | None = None
    deprecated: bool = False
    deprecation_message: str | None = None


def is_docs_view(table_key: str) -> bool:
    """Return True when the table key represents a docs view.

    Returns
    -------
    bool
        True when the table key is a docs view.
    """
    return table_key.startswith("docs.v_")


def _split_table_key(table_key: str) -> tuple[str, str]:
    parts = table_key.split(".", maxsplit=1)
    if len(parts) != _EXPECTED_TABLE_KEY_PARTS or not parts[0] or not parts[1]:
        msg = f"Invalid table key: {table_key!r}"
        raise ValueError(msg)
    return parts[0], parts[1]


def _owner_package_from_prefix(
    schema_prefix: str,
) -> Literal["core", "analytics", "graphs", "qa", "docs"] | None:
    return _OWNER_PACKAGE_BY_PREFIX.get(schema_prefix)


def _resolve_row_binding(
    schema_service: SchemaService, table_key: str
) -> GeneratedRowBinding | None:
    return schema_service.get_row_binding(table_key)


def build_dataset_contract(
    *,
    table_key: str,
    schema_service: SchemaService,
    overrides: DatasetContractOverrides | None = None,
    composition: CompositeSchema | None = None,
    is_view_override: bool | None = None,
) -> DatasetContract:
    """Build a DatasetContract from schema service plus optional overrides.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    schema_service
        Schema service used to resolve table schema and row bindings.
    overrides
        Optional override values for metadata fields.
    composition
        Optional composite schema metadata for profile datasets.
    is_view_override
        Force view status (used by callers when view detection is external).

    Returns
    -------
    DatasetContract
        Derived dataset contract with deterministic defaults.
    """
    schema_prefix, table_name = _split_table_key(table_key)
    is_view = is_view_override if is_view_override is not None else is_docs_view(table_key)
    schema = schema_service.get_table_schema(table_key)
    row_binding = _resolve_row_binding(schema_service, table_key)

    json_schema_id = overrides.json_schema_id if overrides is not None else None
    if json_schema_id is None:
        json_schema_id = default_json_schema_id(table_key=table_key, schema=schema)

    jsonl_filename = overrides.jsonl_filename if overrides is not None else None
    if jsonl_filename is None:
        jsonl_filename = default_jsonl_filename(table_key=table_key, schema=schema)

    parquet_filename = overrides.parquet_filename if overrides is not None else None
    if parquet_filename is None:
        parquet_filename = default_parquet_filename(table_key=table_key, schema=schema)

    description = overrides.description if overrides is not None else None
    if description is None and schema is not None:
        description = schema.description

    family = overrides.family if overrides is not None else None
    if family is None:
        family = schema_prefix

    tags = frozenset({"docs_view", "read_only"}) if is_view else frozenset({"base_table"})
    if overrides is not None and overrides.tags:
        tags |= overrides.tags

    return DatasetContract(
        table_key=table_key,
        name=table_name,
        schema=schema,
        row_binding=row_binding,
        json_schema_id=json_schema_id,
        jsonl_filename=jsonl_filename,
        parquet_filename=parquet_filename,
        is_view=is_view,
        owner_package=_owner_package_from_prefix(schema_prefix),
        tags=tags,
        description=description,
        family=family,
        owner=overrides.owner if overrides is not None else None,
        freshness_sla=overrides.freshness_sla if overrides is not None else None,
        retention_policy=overrides.retention_policy if overrides is not None else None,
        stable_id=overrides.stable_id if overrides is not None else None,
        schema_version=overrides.schema_version if overrides is not None else None,
        upstream_dependencies=overrides.upstream_dependencies if overrides is not None else (),
        validation_profile=overrides.validation_profile if overrides is not None else "strict",
        composition=composition,
        deprecated=overrides.deprecated if overrides is not None else False,
        deprecation_message=overrides.deprecation_message if overrides is not None else None,
    )


__all__ = [
    "DatasetContractOverrides",
    "build_dataset_contract",
    "is_docs_view",
]
