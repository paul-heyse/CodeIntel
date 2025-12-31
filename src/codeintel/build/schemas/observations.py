"""Schema observation helpers for inference-first pipelines."""

from __future__ import annotations

import json
import logging
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.ipc import schema_to_ipc_payload
from codeintel.core.columnar.schema_metadata import (
    decode_metadata,
    encode_metadata,
    merge_field_metadata,
    merge_metadata,
)
from codeintel.core.hamilton import tags as hamilton_tags
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.arrow_gen import (
    ARROW_SCHEMA_CONTRACT_VERSION,
    DEFAULT_EXTRAS_COLUMN,
    DEFAULT_EXTRAS_POLICY,
    ExtrasPolicy,
)
from codeintel.core.schemas.arrow_polars import table_schema_from_arrow_schema
from codeintel.core.schemas.hashing import schema_hash as compute_schema_hash
from codeintel.core.schemas.primitives import Column, ColumnType, TableSchema, normalize_column_type
from codeintel.core.time import utc_now
from codeintel.storage.tracking.schema_catalog_models import (
    ColumnStatsEntry,
    ColumnStatsPayload,
    DatasetStatsPayload,
    DerivedSettingsPayload,
    ParquetStatsPayload,
    SchemaObservationRecord,
    SchemaVersionRecord,
    TableSchemaRegistryRecord,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from codeintel.storage.gateway.protocol import StorageGateway


LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SchemaObservationBundle:
    """Bundle of records produced from a schema observation."""

    table_schema: TableSchema
    arrow_schema: pa.Schema
    observation: SchemaObservationRecord
    schema_version: SchemaVersionRecord
    registry_record: TableSchemaRegistryRecord


@dataclass(frozen=True, slots=True)
class SchemaObservationInputs:
    """Optional inputs for schema observation finalization."""

    repo: str | None = None
    commit: str | None = None
    target_name: str | None = None
    extras_policy: ExtrasPolicy | None = None
    drift_history: Sequence[Mapping[str, object] | None] | None = None
    dataset_stats: ParquetStatsPayload | None = None
    manifest_row_count: int | None = None


@dataclass(frozen=True, slots=True)
class ColumnHint:
    """Soft column-level hints merged into inferred metadata."""

    nullable: bool | None = None
    description: str | None = None
    pii_class: str | None = None


@dataclass(frozen=True, slots=True)
class SchemaHints:
    """Soft table-level hints merged into inferred metadata."""

    description: str | None = None
    columns: Mapping[str, ColumnHint] = field(default_factory=dict)


@dataclass(slots=True)
class _ColumnStatsAccumulator:
    null_count: int = 0
    non_null_count: int = 0
    min_value: object | None = None
    max_value: object | None = None
    distinct_max: int | None = None
    length_sum: int = 0
    length_count: int = 0

    def observe(self, values: pa.Array | pa.ChunkedArray) -> None:
        nulls = _null_count(values)
        self.null_count += nulls
        length = _array_length(values)
        self.non_null_count += max(length - nulls, 0)
        min_value, max_value = _min_max(values)
        if min_value is not None:
            self.min_value = _safe_min(self.min_value, min_value)
        if max_value is not None:
            self.max_value = _safe_max(self.max_value, max_value)
        distinct_value = _count_distinct(values)
        if distinct_value is not None:
            self.distinct_max = (
                distinct_value
                if self.distinct_max is None
                else max(self.distinct_max, distinct_value)
            )
        length_sum, length_count = _length_stats(values)
        self.length_sum += length_sum
        self.length_count += length_count


@dataclass(slots=True)
class SchemaObservationAccumulator:
    """Accumulate schema observation stats while streaming batches."""

    table_key: str
    declared_schema: TableSchema | None = None
    schema_hints: SchemaHints | None = None
    column_stats: dict[str, _ColumnStatsAccumulator] = field(default_factory=dict)
    row_count: int = 0
    batch_count: int = 0
    total_bytes: int = 0

    def observe_batch(self, batch: pa.RecordBatch) -> None:
        """Accumulate stats for a single record batch."""
        self.row_count += batch.num_rows
        self.batch_count += 1
        with suppress(AttributeError, TypeError):
            self.total_bytes += batch.nbytes
        for arrow_field, column in zip(batch.schema, batch.columns, strict=True):
            accumulator = self.column_stats.get(arrow_field.name)
            if accumulator is None:
                accumulator = _ColumnStatsAccumulator()
                self.column_stats[arrow_field.name] = accumulator
            accumulator.observe(column)

    def finalize(
        self,
        *,
        arrow_schema: pa.Schema,
        inputs: SchemaObservationInputs | None = None,
    ) -> SchemaObservationBundle:
        """Build observation records after streaming completes.

        Returns
        -------
        SchemaObservationBundle
            Bundle containing observation, registry, and schema version records.
        """
        resolved_inputs = inputs or SchemaObservationInputs()
        inferred = table_schema_from_arrow_schema(
            arrow_schema=arrow_schema,
            table_key=self.table_key,
        )
        merged = _merge_table_schema_hints(
            inferred,
            self.declared_schema,
            schema_hints=self.schema_hints,
            observed_nullability=_observed_nullability(self.column_stats),
        )
        schema_json = merged.to_json_obj()
        schema_digest = fingerprint(schema_json)
        schema_hash_value = compute_schema_hash(merged)
        drift_summary = _drift_summary(merged, self.declared_schema)
        resolved_extras_policy = resolved_inputs.extras_policy or _extras_policy_from_drift(
            drift_summary,
            drift_history=resolved_inputs.drift_history,
        )
        derived_settings = _derived_settings_from_stats(
            table_schema=merged,
            column_stats=self.column_stats,
            row_count=self.row_count,
            total_bytes=self.total_bytes,
            extras_policy=resolved_extras_policy,
        )
        annotation = _SchemaAnnotationContext(
            schema_hash_value=schema_hash_value,
            schema_digest=schema_digest,
            extras_policy=resolved_extras_policy,
            extras_column=DEFAULT_EXTRAS_COLUMN,
        )
        annotated_schema = _annotate_arrow_schema(
            arrow_schema,
            table_schema=merged,
            annotation=annotation,
            pii_by_column=_pii_by_column(self.schema_hints),
        )
        observed_at = utc_now()
        observation = SchemaObservationRecord(
            table_key=self.table_key,
            schema_digest=schema_digest,
            schema_hash=schema_hash_value,
            arrow_schema_ipc_b64=schema_to_ipc_payload(annotated_schema),
            repo=resolved_inputs.repo,
            commit=resolved_inputs.commit,
            target_name=resolved_inputs.target_name,
            column_stats=_column_stats_payload(self.column_stats),
            dataset_stats=_dataset_stats_payload(
                row_count=self.row_count,
                batch_count=self.batch_count,
                total_bytes=self.total_bytes,
                manifest_stats=resolved_inputs.dataset_stats,
                manifest_row_count=resolved_inputs.manifest_row_count,
            ),
            derived_settings=derived_settings,
            drift_summary=drift_summary,
            observed_at=observed_at,
        )
        schema_version = SchemaVersionRecord(
            schema_digest=schema_digest,
            schema_hash=schema_hash_value,
            schema_json=schema_json,
            renderer_cache=_renderer_cache_from_arrow_schema(
                annotated_schema,
                extras_policy=resolved_extras_policy,
                extras_column=DEFAULT_EXTRAS_COLUMN,
            ),
            created_at=observed_at,
        )
        registry_record = TableSchemaRegistryRecord(
            table_key=self.table_key,
            schema_digest=schema_digest,
            schema_hash=schema_hash_value,
            derivation_kind="inferred_relation",
            derivation_source=resolved_inputs.target_name or "observed_output",
            inference_status="inferred",
            inference_error=None,
            catalog_hash=None,
            updated_at=observed_at,
        )
        return SchemaObservationBundle(
            table_schema=merged,
            arrow_schema=annotated_schema,
            observation=observation,
            schema_version=schema_version,
            registry_record=registry_record,
        )


def instrument_reader_for_observation(
    reader: pa.RecordBatchReader,
    *,
    accumulator: SchemaObservationAccumulator,
) -> pa.RecordBatchReader:
    """Wrap a RecordBatchReader to collect observation stats while streaming.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader that updates the accumulator as batches are consumed.
    """

    def _iter_batches() -> Iterable[pa.RecordBatch]:
        for batch in reader:
            accumulator.observe_batch(batch)
            yield batch

    return pa.RecordBatchReader.from_batches(reader.schema, _iter_batches())


def observe_batches(
    batches: Iterable[pa.RecordBatch],
    *,
    accumulator: SchemaObservationAccumulator,
) -> None:
    """Consume record batches and update the observation accumulator."""
    for batch in batches:
        accumulator.observe_batch(batch)


def persist_observation_bundle(
    *,
    gateway: StorageGateway,
    bundle: SchemaObservationBundle,
) -> None:
    """Persist observation data to the schema registry tables."""
    resolved_bundle = _apply_extras_policy(
        gateway=gateway,
        bundle=bundle,
    )
    gateway.schemas.record_schema_versions_batch([resolved_bundle.schema_version])
    gateway.schemas.record_table_schema_registry_batch([resolved_bundle.registry_record])
    gateway.schemas.record_schema_observations_batch([resolved_bundle.observation])
    if resolved_bundle.observation.drift_summary:
        LOG.warning(
            "schema_drift_observed table=%s drift=%s",
            resolved_bundle.observation.table_key,
            resolved_bundle.observation.drift_summary,
        )


_SCHEMA_HINTS_TAG = "schema_hints"
_SCHEMA_DESCRIPTION_TAG = "schema_description"
_SCHEMA_COLUMNS_TAG = "schema_columns"
_SCHEMA_COLUMN_DESCRIPTIONS_TAG = "schema_column_descriptions"
_SCHEMA_NULLABLE_BY_COLUMN_TAG = "schema_nullable_by_column"
_SCHEMA_PII_BY_COLUMN_TAG = "schema_pii_by_column"
_HAMILTON_SCHEMA_OUTPUT_TAG = "hamilton.internal.schema_output"
_TAG_DESCRIPTION = hamilton_tags.TAG_DESCRIPTION

_SCHEMA_OUTPUT_TYPE_MAP: dict[str, ColumnType] = {
    "int": "BIGINT",
    "int64": "BIGINT",
    "int32": "INTEGER",
    "integer": "INTEGER",
    "bigint": "BIGINT",
    "float": "DOUBLE",
    "float64": "DOUBLE",
    "double": "DOUBLE",
    "str": "VARCHAR",
    "string": "VARCHAR",
    "varchar": "VARCHAR",
    "bool": "BOOLEAN",
    "boolean": "BOOLEAN",
    "datetime": "TIMESTAMP",
    "timestamp": "TIMESTAMP",
    "timestamptz": "TIMESTAMPTZ",
    "date": "TIMESTAMP",
    "decimal": "DECIMAL",
    "decimal(38,0)": "DECIMAL(38,0)",
    "json": "JSON",
    "dict": "JSON",
    "list": "JSON",
    "object": "JSON",
}


@dataclass(slots=True)
class _SchemaHintsBuilder:
    description: str | None = None
    columns: dict[str, ColumnHint] = field(default_factory=dict)

    def merge_description(self, description: str | None) -> None:
        if self.description is None and description:
            self.description = description

    def merge_column_hint(self, name: str, hint: ColumnHint) -> None:
        if _column_hint_empty(hint):
            return
        existing = self.columns.get(name)
        if existing is None:
            self.columns[name] = hint
            return
        self.columns[name] = ColumnHint(
            nullable=existing.nullable if existing.nullable is not None else hint.nullable,
            description=existing.description
            if existing.description is not None
            else hint.description,
            pii_class=existing.pii_class if existing.pii_class is not None else hint.pii_class,
        )

    def build(self) -> SchemaHints | None:
        if self.description is None and not self.columns:
            return None
        return SchemaHints(description=self.description, columns=dict(self.columns))


def schema_hints_from_tags(tags: Mapping[str, object]) -> SchemaHints | None:
    """Parse schema hints from Hamilton tag mappings.

    Supported tags:
    - ``schema_hints``: JSON/dict with keys ``description``, ``columns``.
    - ``schema_description``: table description string.
    - ``schema_columns``: column -> hint mapping.
    - ``schema_column_descriptions``: column -> description mapping.
    - ``schema_nullable_by_column``: column -> bool mapping.
    - ``schema_pii_by_column``: column -> pii class mapping.
    - ``description``: fallback table description tag.

    Returns
    -------
    SchemaHints | None
        Parsed schema hints when available.
    """
    builder = _SchemaHintsBuilder()
    _merge_schema_hints(builder, tags)
    return builder.build()


def schema_hints_from_tag_sets(tag_sets: Iterable[Mapping[str, object]]) -> SchemaHints | None:
    """Merge schema hints from multiple tag mappings.

    Parameters
    ----------
    tag_sets
        Iterable of tag mappings to merge.

    Returns
    -------
    SchemaHints | None
        Merged schema hints, or None when no hints are available.
    """
    builder = _SchemaHintsBuilder()
    for tags in tag_sets:
        _merge_schema_hints(builder, tags)
    return builder.build()


def table_schema_from_tag_sets(
    *,
    table_key: str,
    tag_sets: Iterable[Mapping[str, object]],
) -> TableSchema | None:
    """Return a TableSchema hint derived from @schema.output tags.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    tag_sets
        Tag mappings to scan for schema output metadata.

    Returns
    -------
    TableSchema | None
        TableSchema derived from schema.output tags, or None when unavailable.
    """
    parsed = _split_table_key(table_key)
    if parsed is None:
        return None
    schema_name, table_name = parsed
    ordered_columns: list[str] = []
    column_types: dict[str, ColumnType] = {}
    for tags in tag_sets:
        mapping = _schema_output_from_tags(tags)
        if not mapping:
            continue
        for column_name, raw_type in mapping.items():
            if column_name in column_types:
                continue
            ordered_columns.append(column_name)
            column_types[column_name] = _column_type_from_schema_output(raw_type)
    if not column_types:
        return None
    columns = [
        Column(name=column_name, type=column_types[column_name]) for column_name in ordered_columns
    ]
    return TableSchema(schema=schema_name, name=table_name, columns=columns)


def _merge_schema_hints(builder: _SchemaHintsBuilder, tags: Mapping[str, object]) -> None:
    raw_hints = _coerce_mapping(tags.get(_SCHEMA_HINTS_TAG))
    if raw_hints is not None:
        _apply_schema_hints_mapping(builder, raw_hints)

    builder.merge_description(_coerce_str(tags.get(_SCHEMA_DESCRIPTION_TAG)))
    builder.merge_description(_coerce_str(tags.get(_TAG_DESCRIPTION)))
    _merge_columns_from_mapping(builder, _coerce_mapping(tags.get(_SCHEMA_COLUMNS_TAG)))
    _merge_columns_from_scalar_mapping(
        builder,
        _coerce_mapping(tags.get(_SCHEMA_COLUMN_DESCRIPTIONS_TAG)),
        kind="description",
    )
    _merge_columns_from_scalar_mapping(
        builder,
        _coerce_mapping(tags.get(_SCHEMA_NULLABLE_BY_COLUMN_TAG)),
        kind="nullable",
    )
    _merge_columns_from_scalar_mapping(
        builder,
        _coerce_mapping(tags.get(_SCHEMA_PII_BY_COLUMN_TAG)),
        kind="pii_class",
    )


def _apply_schema_hints_mapping(
    builder: _SchemaHintsBuilder,
    raw_hints: Mapping[str, object],
) -> None:
    builder.merge_description(
        _coerce_str(raw_hints.get("description") or raw_hints.get("table_description"))
    )
    _merge_columns_from_mapping(builder, _coerce_mapping(raw_hints.get("columns")))
    _merge_columns_from_scalar_mapping(
        builder,
        _coerce_mapping(raw_hints.get("column_descriptions")),
        kind="description",
    )
    _merge_columns_from_scalar_mapping(
        builder,
        _coerce_mapping(raw_hints.get("nullable_by_column")),
        kind="nullable",
    )
    _merge_columns_from_scalar_mapping(
        builder,
        _coerce_mapping(raw_hints.get("pii_by_column")),
        kind="pii_class",
    )


def _merge_columns_from_mapping(
    builder: _SchemaHintsBuilder,
    mapping: Mapping[str, object] | None,
) -> None:
    if not mapping:
        return
    for name, raw in mapping.items():
        if not isinstance(name, str) or not name:
            continue
        hint = _column_hint_from_mapping(raw)
        builder.merge_column_hint(name, hint)


def _merge_columns_from_scalar_mapping(
    builder: _SchemaHintsBuilder,
    mapping: Mapping[str, object] | None,
    *,
    kind: str,
) -> None:
    if not mapping:
        return
    for name, raw in mapping.items():
        if not isinstance(name, str) or not name:
            continue
        hint = _column_hint_from_scalar(raw, kind=kind)
        if hint is not None:
            builder.merge_column_hint(name, hint)


def _column_hint_from_mapping(raw: object) -> ColumnHint:
    if isinstance(raw, dict):
        return ColumnHint(
            nullable=_coerce_bool(raw.get("nullable")),
            description=_coerce_str(raw.get("description") or raw.get("desc")),
            pii_class=_coerce_str(raw.get("pii") or raw.get("pii_class")),
        )
    if isinstance(raw, str):
        return ColumnHint(description=_coerce_str(raw))
    if isinstance(raw, bool):
        return ColumnHint(nullable=raw)
    return ColumnHint()


def _column_hint_from_scalar(raw: object, *, kind: str) -> ColumnHint | None:
    if kind == "description":
        description = _coerce_str(raw)
        return ColumnHint(description=description) if description is not None else None
    if kind == "nullable":
        nullable = _coerce_bool(raw)
        return ColumnHint(nullable=nullable) if nullable is not None else None
    if kind == "pii_class":
        pii_class = _coerce_str(raw)
        return ColumnHint(pii_class=pii_class) if pii_class is not None else None
    return None


def _column_hint_empty(hint: ColumnHint) -> bool:
    return hint.nullable is None and hint.description is None and hint.pii_class is None


def _coerce_mapping(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): val for key, val in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return {str(key): val for key, val in parsed.items()}
    return None


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return None


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return None


def merge_table_schema_hints(
    inferred: TableSchema,
    declared: TableSchema | None,
    *,
    schema_hints: SchemaHints | None = None,
) -> TableSchema:
    """Return inferred schema with declared hints merged in.

    Returns
    -------
    TableSchema
        Schema with declared hints merged into inferred values.
    """
    return _merge_table_schema_hints(
        inferred,
        declared,
        schema_hints=schema_hints,
        observed_nullability=None,
    )


def _merge_table_schema_hints(
    inferred: TableSchema,
    declared: TableSchema | None,
    *,
    schema_hints: SchemaHints | None,
    observed_nullability: Mapping[str, bool] | None,
) -> TableSchema:
    declared_by_name = {column.name: column for column in declared.columns} if declared else {}
    hint_columns = schema_hints.columns if schema_hints is not None else {}
    merged_columns: list[Column] = []
    for column in inferred.columns:
        hint = declared_by_name.get(column.name)
        tag_hint = hint_columns.get(column.name)
        if observed_nullability is not None and column.name in observed_nullability:
            observed_nullable = observed_nullability[column.name]
        else:
            observed_nullable = column.nullable
        nullable = observed_nullable
        if hint is not None and hint.nullable:
            nullable = True
        if tag_hint is not None and tag_hint.nullable:
            nullable = True
        description = column.description
        if description is None and tag_hint is not None:
            description = tag_hint.description
        if description is None and hint is not None:
            description = hint.description
        merged_columns.append(
            Column(
                name=column.name,
                type=column.type,
                nullable=nullable,
                description=description,
            )
        )
    primary_key = inferred.primary_key or (declared.primary_key if declared is not None else ())
    indexes = inferred.indexes or (declared.indexes if declared is not None else ())
    description = inferred.description
    if description is None and schema_hints is not None:
        description = schema_hints.description
    if description is None and declared is not None:
        description = declared.description
    write_policy = inferred.write_policy or (
        declared.write_policy if declared is not None else None
    )
    return TableSchema(
        schema=inferred.schema,
        name=inferred.name,
        columns=merged_columns,
        primary_key=primary_key,
        indexes=indexes,
        description=description,
        write_policy=write_policy,
    )


def _pii_by_column(schema_hints: SchemaHints | None) -> dict[str, str] | None:
    if schema_hints is None:
        return None
    mapping = {
        name: hint.pii_class
        for name, hint in schema_hints.columns.items()
        if hint.pii_class is not None
    }
    return mapping or None


def _observed_nullability(
    column_stats: Mapping[str, _ColumnStatsAccumulator],
) -> dict[str, bool]:
    observed: dict[str, bool] = {}
    for name, stats in column_stats.items():
        total = stats.null_count + stats.non_null_count
        if total == 0:
            observed[name] = True
            continue
        observed[name] = stats.null_count > 0
    return observed


_DEFAULT_DICT_MAX_CARDINALITY = 256
_DEFAULT_DICT_RATIO = 0.1
_TARGET_ROW_GROUP_BYTES = 64 * 1024 * 1024
_MIN_ROW_GROUP_SIZE = 10_000
_MAX_ROW_GROUP_SIZE = 1_000_000
_MIN_DATA_PAGE_SIZE = 64 * 1024
_MAX_DATA_PAGE_SIZE = 1024 * 1024
_ROW_GROUP_PAGE_DIVISOR = 128
_EXTRAS_POLICY_RETAIN_COUNT = 2


def _derived_settings_from_stats(
    *,
    table_schema: TableSchema,
    column_stats: Mapping[str, _ColumnStatsAccumulator],
    row_count: int,
    total_bytes: int,
    extras_policy: ExtrasPolicy,
) -> DerivedSettingsPayload | None:
    settings: DerivedSettingsPayload = {"extras_policy": extras_policy}
    dictionary_columns: list[str] = []
    distinct_values: list[int] = []
    for column in table_schema.columns:
        if column.type != "VARCHAR":
            continue
        stats = column_stats.get(column.name)
        if stats is None or stats.distinct_max is None:
            continue
        if stats.non_null_count <= 0:
            continue
        distinct = stats.distinct_max
        ratio = distinct / stats.non_null_count
        if distinct <= _DEFAULT_DICT_MAX_CARDINALITY and ratio <= _DEFAULT_DICT_RATIO:
            dictionary_columns.append(column.name)
            distinct_values.append(distinct)
    if dictionary_columns:
        settings["dictionary_encode_columns"] = sorted(dictionary_columns)
        settings["dictionary_max_cardinality"] = max(distinct_values)
        settings["unify_dictionaries"] = True

    if row_count > 0 and total_bytes > 0:
        avg_row_bytes = total_bytes / row_count
        if avg_row_bytes > 0:
            raw_rows = int(_TARGET_ROW_GROUP_BYTES / avg_row_bytes)
            row_group_size = max(_MIN_ROW_GROUP_SIZE, min(_MAX_ROW_GROUP_SIZE, raw_rows))
            row_group_bytes = row_group_size * avg_row_bytes
            page_bytes = int(row_group_bytes / _ROW_GROUP_PAGE_DIVISOR)
            page_bytes = max(_MIN_DATA_PAGE_SIZE, min(_MAX_DATA_PAGE_SIZE, page_bytes))
            settings["row_group_size"] = row_group_size
            settings["data_page_size"] = page_bytes
            settings["avg_row_bytes"] = avg_row_bytes

    return settings or None


def _column_stats_payload(
    column_stats: Mapping[str, _ColumnStatsAccumulator],
) -> ColumnStatsPayload | None:
    if not column_stats:
        return None
    payload: ColumnStatsPayload = {}
    for name, stats in column_stats.items():
        entry: ColumnStatsEntry = {
            "null_count": stats.null_count,
            "non_null_count": stats.non_null_count,
        }
        if stats.distinct_max is not None:
            entry["distinct_count_max"] = stats.distinct_max
        if stats.min_value is not None:
            entry["min"] = _json_safe_value(stats.min_value)
        if stats.max_value is not None:
            entry["max"] = _json_safe_value(stats.max_value)
        if stats.length_count > 0:
            entry["avg_length"] = stats.length_sum / stats.length_count
        payload[name] = entry
    return payload


def _dataset_stats_payload(
    *,
    row_count: int,
    batch_count: int,
    total_bytes: int,
    manifest_stats: ParquetStatsPayload | None,
    manifest_row_count: int | None,
) -> DatasetStatsPayload:
    payload: DatasetStatsPayload = {
        "row_count": row_count,
        "batch_count": batch_count,
        "total_bytes": total_bytes,
    }
    if manifest_row_count is not None:
        payload["manifest_row_count"] = manifest_row_count
    if manifest_stats:
        payload["parquet_stats"] = dict(manifest_stats)
    return payload


def _drift_summary(
    inferred: TableSchema,
    declared: TableSchema | None,
) -> dict[str, object] | None:
    if declared is None:
        return None
    inferred_columns = {column.name: column.type for column in inferred.columns}
    declared_columns = {column.name: column.type for column in declared.columns}
    missing = sorted(name for name in declared_columns if name not in inferred_columns)
    extra = sorted(name for name in inferred_columns if name not in declared_columns)
    type_changes: list[dict[str, str]] = []
    for name, inferred_type in inferred_columns.items():
        declared_type = declared_columns.get(name)
        if declared_type is None or declared_type == inferred_type:
            continue
        type_changes.append(
            {
                "column": name,
                "declared": declared_type,
                "observed": inferred_type,
            }
        )
    if not missing and not extra and not type_changes:
        return None
    return {
        "missing_columns": missing,
        "extra_columns": extra,
        "type_changes": type_changes,
    }


def _extras_policy_from_drift(
    drift_summary: Mapping[str, object] | None,
    *,
    drift_history: Sequence[Mapping[str, object] | None] | None = None,
) -> ExtrasPolicy:
    if _drift_has_extras(drift_summary):
        return "retain"
    if drift_history is not None:
        extra_count = sum(1 for summary in drift_history if _drift_has_extras(summary))
        if extra_count >= _EXTRAS_POLICY_RETAIN_COUNT:
            return "retain"
    return DEFAULT_EXTRAS_POLICY


def _drift_has_extras(drift_summary: Mapping[str, object] | None) -> bool:
    if drift_summary is None:
        return False
    extra = drift_summary.get("extra_columns")
    return isinstance(extra, list) and bool(extra)


@dataclass(frozen=True, slots=True)
class _SchemaAnnotationContext:
    schema_hash_value: str
    schema_digest: str
    extras_policy: ExtrasPolicy
    extras_column: str


def _annotate_arrow_schema(
    schema: pa.Schema,
    *,
    table_schema: TableSchema,
    annotation: _SchemaAnnotationContext,
    pii_by_column: Mapping[str, str] | None,
) -> pa.Schema:
    schema_metadata: dict[str, object] = {
        "codeintel.table_key": table_schema.table_key,
        "codeintel.schema_hash": annotation.schema_hash_value,
        "codeintel.schema_digest": annotation.schema_digest,
        "codeintel.primary_key": list(table_schema.primary_key),
        "codeintel.schema_contract_version": ARROW_SCHEMA_CONTRACT_VERSION,
        "codeintel.extras_policy": annotation.extras_policy,
        "codeintel.extras_column": annotation.extras_column,
    }
    if table_schema.description is not None:
        schema_metadata["codeintel.description"] = table_schema.description
    merged_metadata = merge_metadata(schema.metadata, schema_metadata)
    fields: list[pa.Field] = []
    key_roles = _key_roles(table_schema)
    for column in table_schema.columns:
        try:
            field = schema.field(column.name)
        except KeyError:
            field = pa.field(column.name, pa.string(), nullable=True)
        field_updates: dict[str, object] = {
            "codeintel.column_type": column.type,
            "codeintel.nullable": column.nullable,
            "codeintel.schema_hash": annotation.schema_hash_value,
            "codeintel.schema_digest": annotation.schema_digest,
        }
        if column.description is not None:
            field_updates["codeintel.description"] = column.description
        key_role = key_roles.get(column.name)
        if key_role is not None:
            field_updates["codeintel.key_role"] = key_role
        if pii_by_column is not None:
            pii_class = pii_by_column.get(column.name)
            if pii_class is not None:
                field_updates["codeintel.pii_class"] = pii_class
        fields.append(merge_field_metadata(field, field_updates))
    return pa.schema(fields, metadata=merged_metadata)


def _key_roles(table_schema: TableSchema) -> dict[str, str]:
    roles: dict[str, str] = dict.fromkeys(table_schema.primary_key, "primary_key")
    for index in table_schema.indexes:
        if not index.unique:
            continue
        for column in index.columns:
            roles.setdefault(column, "unique_index")
    return roles


def _renderer_cache_from_arrow_schema(
    schema: pa.Schema,
    *,
    extras_policy: ExtrasPolicy,
    extras_column: str,
) -> dict[str, object]:
    return {
        "arrow_schema_ipc_b64": schema_to_ipc_payload(schema),
        "arrow_schema_contract_version": ARROW_SCHEMA_CONTRACT_VERSION,
        "extras_policy": extras_policy,
        "extras_column": extras_column,
    }


def _apply_extras_policy(
    *,
    gateway: StorageGateway,
    bundle: SchemaObservationBundle,
) -> SchemaObservationBundle:
    desired_policy = _resolve_extras_policy(
        gateway=gateway,
        table_key=bundle.table_schema.table_key,
        drift_summary=bundle.observation.drift_summary,
    )
    current_policy = _extras_policy_from_schema(bundle.arrow_schema)
    if desired_policy == current_policy:
        return bundle
    updated_schema = _replace_schema_metadata(
        bundle.arrow_schema,
        {"codeintel.extras_policy": desired_policy},
    )
    updated_renderer_cache = _renderer_cache_from_arrow_schema(
        updated_schema,
        extras_policy=desired_policy,
        extras_column=DEFAULT_EXTRAS_COLUMN,
    )
    derived_settings = bundle.observation.derived_settings
    if derived_settings is None:
        updated_settings: DerivedSettingsPayload = {"extras_policy": desired_policy}
    else:
        updated_settings = cast("DerivedSettingsPayload", dict(derived_settings))
        updated_settings["extras_policy"] = desired_policy
    updated_observation = replace(
        bundle.observation,
        arrow_schema_ipc_b64=schema_to_ipc_payload(updated_schema),
        derived_settings=updated_settings,
    )
    updated_schema_version = replace(
        bundle.schema_version,
        renderer_cache=updated_renderer_cache,
    )
    return SchemaObservationBundle(
        table_schema=bundle.table_schema,
        arrow_schema=updated_schema,
        observation=updated_observation,
        schema_version=updated_schema_version,
        registry_record=bundle.registry_record,
    )


def _resolve_extras_policy(
    *,
    gateway: StorageGateway,
    table_key: str,
    drift_summary: Mapping[str, object] | None,
) -> ExtrasPolicy:
    if drift_summary is not None and _has_extra_columns(drift_summary):
        return "retain"
    history = gateway.schemas.load_recent_drift_summaries(table_key=table_key, limit=5)
    extras_count = sum(1 for summary in history if summary and _has_extra_columns(summary))
    if extras_count >= _EXTRAS_POLICY_RETAIN_COUNT:
        return "retain"
    return DEFAULT_EXTRAS_POLICY


def _has_extra_columns(summary: Mapping[str, object]) -> bool:
    extra = summary.get("extra_columns")
    return isinstance(extra, list) and bool(extra)


def _extras_policy_from_schema(schema: pa.Schema) -> ExtrasPolicy:
    metadata = decode_metadata(schema.metadata)
    raw = metadata.get("codeintel.extras_policy")
    if isinstance(raw, str):
        coerced = _coerce_extras_policy(raw)
        if coerced is not None:
            return coerced
    return DEFAULT_EXTRAS_POLICY


def _coerce_extras_policy(raw: str) -> ExtrasPolicy | None:
    if raw == "retain":
        return "retain"
    if raw == "reject":
        return "reject"
    if raw == "drop":
        return "drop"
    return None


def _replace_schema_metadata(
    schema: pa.Schema,
    updates: Mapping[str, object],
) -> pa.Schema:
    decoded = decode_metadata(schema.metadata)
    merged = dict(decoded)
    for key, value in updates.items():
        if value is None:
            continue
        merged[key] = value
    return schema.with_metadata(encode_metadata(merged))


def _null_count(values: pa.Array | pa.ChunkedArray) -> int:
    try:
        null_count = values.null_count
    except AttributeError:
        return 0
    return 0 if null_count is None else int(null_count)


def _array_length(values: pa.Array | pa.ChunkedArray) -> int:
    try:
        return len(values)
    except TypeError:
        return 0


def _min_max(values: pa.Array | pa.ChunkedArray) -> tuple[object | None, object | None]:
    min_value = _compute_scalar("min", values)
    max_value = _compute_scalar("max", values)
    return min_value, max_value


def _compute_scalar(name: str, values: pa.Array | pa.ChunkedArray) -> object | None:
    func = getattr(pc, name, None)
    if not callable(func):
        return None
    try:
        result = func(values)
    except (TypeError, pa.ArrowInvalid):
        return None
    return _scalar_value(result)


def _scalar_value(result: object) -> object | None:
    if result is None:
        return None
    as_py = getattr(result, "as_py", None)
    if callable(as_py):
        return as_py()
    return result


def _count_distinct(values: pa.Array | pa.ChunkedArray) -> int | None:
    func = getattr(pc, "count_distinct", None)
    if not callable(func):
        return None
    try:
        result = func(values)
    except (TypeError, pa.ArrowInvalid):
        return None
    scalar = _scalar_value(result)
    if scalar is None or isinstance(scalar, bool):
        return None
    if isinstance(scalar, int):
        return scalar
    if isinstance(scalar, float) and scalar.is_integer():
        return int(scalar)
    return None


def _length_stats(values: pa.Array | pa.ChunkedArray) -> tuple[int, int]:
    length_func = _length_func(values)
    if length_func is None:
        return 0, 0
    try:
        lengths = length_func(values)
        filled = _fill_null(lengths, 0)
        length_sum = _sum_scalar(filled)
        length_count = _count_scalar(lengths)
    except (TypeError, pa.ArrowInvalid):
        return 0, 0
    return length_sum, length_count


def _length_func(values: pa.Array | pa.ChunkedArray) -> Callable[[object], object] | None:
    data_type = values.type
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        func = getattr(pc, "utf8_length", None)
        return func if callable(func) else None
    if pa.types.is_binary(data_type) or pa.types.is_large_binary(data_type):
        func = getattr(pc, "binary_length", None)
        return func if callable(func) else None
    return None


def _fill_null(values: object, fill_value: int) -> object:
    func = getattr(pc, "fill_null", None)
    if not callable(func):
        return values
    return func(values, fill_value)


def _sum_scalar(values: object) -> int:
    func = getattr(pc, "sum", None)
    if not callable(func):
        return 0
    result = func(values)
    scalar = _scalar_value(result)
    if isinstance(scalar, bool):
        return 0
    if isinstance(scalar, int):
        return scalar
    if isinstance(scalar, float):
        return int(scalar)
    return 0


def _count_scalar(values: object) -> int:
    func = getattr(pc, "count", None)
    if not callable(func):
        return 0
    result = func(values)
    scalar = _scalar_value(result)
    if isinstance(scalar, bool):
        return 0
    if isinstance(scalar, int):
        return scalar
    if isinstance(scalar, float) and scalar.is_integer():
        return int(scalar)
    return 0


def _safe_min(current: object | None, candidate: object) -> object:
    if current is None:
        return candidate
    if isinstance(candidate, _SupportsRichComparison) and isinstance(
        current, _SupportsRichComparison
    ):
        try:
            return candidate if candidate < current else current
        except TypeError:
            return current
    return current


def _safe_max(current: object | None, candidate: object) -> object:
    if current is None:
        return candidate
    if isinstance(candidate, _SupportsRichComparison) and isinstance(
        current, _SupportsRichComparison
    ):
        try:
            return candidate if candidate > current else current
        except TypeError:
            return current
    return current


def _json_safe_value(value: object) -> object:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(value).decode("ascii")
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


@runtime_checkable
class _SupportsRichComparison(Protocol):
    def __lt__(self, other: object) -> bool: ...

    def __gt__(self, other: object) -> bool: ...


def _schema_output_from_tags(tags: Mapping[str, object]) -> dict[str, str] | None:
    raw = tags.get(_HAMILTON_SCHEMA_OUTPUT_TAG)
    if isinstance(raw, str):
        return _schema_output_from_string(raw)
    if isinstance(raw, dict):
        return _schema_output_from_mapping(raw)
    return None


def _schema_output_from_string(raw: str) -> dict[str, str] | None:
    if not raw:
        return None
    pairs = [part.strip() for part in raw.split(",") if part.strip()]
    mapping: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        name, dtype = pair.split("=", 1)
        name = name.strip()
        dtype = dtype.strip()
        if not name or not dtype:
            continue
        mapping[name] = dtype
    return mapping or None


def _schema_output_from_mapping(raw: Mapping[object, object]) -> dict[str, str] | None:
    mapping: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        mapping[key.strip()] = value.strip()
    return mapping or None


def _column_type_from_schema_output(raw: object) -> ColumnType:
    if not isinstance(raw, str):
        return "JSON"
    normalized = raw.strip()
    if not normalized:
        return "JSON"
    lower = normalized.lower()
    compact = lower.replace(" ", "")
    mapped = _SCHEMA_OUTPUT_TYPE_MAP.get(lower) or _SCHEMA_OUTPUT_TYPE_MAP.get(compact)
    if mapped is not None:
        return mapped
    try:
        return normalize_column_type(normalized)
    except ValueError:
        return "JSON"


def _split_table_key(table_key: str) -> tuple[str, str] | None:
    if "." not in table_key:
        return None
    schema, name = table_key.split(".", 1)
    if not schema or not name:
        return None
    return schema, name


__all__ = [
    "ColumnHint",
    "SchemaHints",
    "SchemaObservationAccumulator",
    "SchemaObservationBundle",
    "SchemaObservationInputs",
    "instrument_reader_for_observation",
    "merge_table_schema_hints",
    "observe_batches",
    "persist_observation_bundle",
    "schema_hints_from_tag_sets",
    "schema_hints_from_tags",
    "table_schema_from_tag_sets",
]
