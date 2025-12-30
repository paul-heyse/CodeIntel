"""Iceberg dataset saver for Hamilton materialization."""

from __future__ import annotations

import logging
import types
import typing
import uuid
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Literal, Protocol, cast, get_args, get_origin

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from hamilton.io.data_adapters import DataSaver
from polars.exceptions import PolarsError
from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError
from pyiceberg.expressions import (
    AlwaysFalse,
    AlwaysTrue,
    And,
    BooleanExpression,
    EqualTo,
    IsNull,
    Or,
    Reference,
)
from pyiceberg.io.pyarrow import bin_pack_arrow_table, pyarrow_to_schema, write_file
from pyiceberg.partitioning import (
    PARTITION_FIELD_ID_START,
    PartitionField,
    PartitionFieldValue,
    PartitionKey,
    PartitionSpec,
)
from pyiceberg.table import (
    Table,
    TableProperties,
    Transaction,
    WriteTask,
)
from pyiceberg.table.sorting import NullOrder, SortDirection, SortField, SortOrder
from pyiceberg.transforms import parse_transform
from pyiceberg.utils.config import Config as IcebergConfig
from pyiceberg.utils.properties import property_as_bool, property_as_int

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers.base import duration_ms
from codeintel.build.hamilton.materializers.base_pipeline import (
    MaterializationPipelineInput,
    run_materialization_pipeline,
)
from codeintel.build.hamilton.materializers.columnar_utils import (
    align_reader_to_contract_schema,
    arrow_schema_for_data,
    declared_schema_hint,
    schema_tag_sets_for_table,
    table_schema_for_data,
)
from codeintel.build.hamilton.materializers.materialization_validation import (
    DEFAULT_PK_UNIQUENESS_MAX_ROWS,
    ContractCheckContext,
    PrimaryKeyTracker,
    ValidationCollector,
    apply_contract_checks,
    finalize_primary_key_check,
    wrap_reader_for_validation,
)
from codeintel.build.hamilton.materializers.validation_policy import (
    ValidationPolicy,
    resolve_validation_policy,
)
from codeintel.build.schemas import get_schema_provider
from codeintel.build.schemas.observations import (
    SchemaObservationAccumulator,
    SchemaObservationInputs,
    instrument_reader_for_observation,
    persist_observation_bundle,
    schema_hints_from_tag_sets,
    table_schema_from_tag_sets,
)
from codeintel.core.columnar.polars_utils import resolve_query_opt_flags
from codeintel.core.columnar.tabular_adapter import (
    PolarsExecutionOptions,
    to_record_batch_reader,
)
from codeintel.core.config.view import SettingsView
from codeintel.core.execution.ids import new_uuid_str
from codeintel.core.execution.materialization import (
    TableMaterializationMetadata,
    failed_table_result,
    succeeded_table_result,
)
from codeintel.core.hashing.fingerprint import stable_hash
from codeintel.core.iceberg.catalog import IcebergCatalogProvider
from codeintel.core.iceberg.guardrails import require_iceberg_write
from codeintel.core.iceberg.properties import iceberg_location_properties
from codeintel.core.iceberg.schema import (
    IcebergSchemaBundle,
    iceberg_field_ids_for_table_schema,
    table_schema_to_iceberg_schema,
)
from codeintel.core.iceberg.snapshot_properties import (
    SnapshotPropertyInputs,
    snapshot_properties_for_write,
)
from codeintel.core.schemas.contracts import (
    ARROW_SCHEMA_CONTRACT_VERSION,
    DEFAULT_EXTRAS_COLUMN,
    ArrowSchemaMetadata,
    ExtrasPolicy,
    arrow_contract_for_table_schema,
)
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import Column, TableSchema, TableWritePolicy
from codeintel.core.time import utc_now
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.duckdb_types import DuckDBError, DuckDBRelation
from codeintel.storage.helpers.table_key import parse_table_key
from codeintel.storage.iceberg.cache import refresh_iceberg_metadata_cache
from codeintel.storage.iceberg.statistics_file import persist_iceberg_statistics
from codeintel.storage.iceberg.stats import iceberg_stats_for_table
from codeintel.storage.tracking.schema_catalog_models import MaterializationValidationRecord

if TYPE_CHECKING:
    from pyarrow import RecordBatchReader
    from pyiceberg.io import FileIO
    from pyiceberg.manifest import DataFile
    from pyiceberg.schema import Schema
    from pyiceberg.table.metadata import TableMetadata
    from pyiceberg.table.update.snapshot import UpdateSnapshot
    from pyiceberg.typedef import Record

    from codeintel.build.hamilton.materializers.base import MaterializationContext
    from codeintel.core.config.settings import IcebergSettings
    from codeintel.storage.tracking.schema_catalog_models import IcebergStatsPayload

    type IcebergInput = RecordBatchReader | pa.Table | pl.DataFrame | pl.LazyFrame | DuckDBRelation
else:
    type IcebergInput = object


LOG = logging.getLogger(__name__)

_RECOVERABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    OSError,
    pa.ArrowInvalid,
    PolarsError,
)

_TABULAR_TYPES: tuple[type, ...] = (
    pa.RecordBatchReader,
    pa.Table,
    pl.DataFrame,
    pl.LazyFrame,
    DuckDBRelation,
)

_DEFAULT_PARTITION_COLUMNS: tuple[str, ...] = ("repo", "commit", "target")

_PARTITION_KEY_TAG = "partition.key"
_PARTITION_TRANSFORM_TAG = "partition.transform"
_PARTITION_ORDER_TAG = "partition.order"

_SORT_KEY_TAG = "sort.key"
_SORT_DIRECTION_TAG = "sort.direction"
_SORT_NULL_ORDER_TAG = "sort.null_order"


@dataclass(frozen=True, slots=True)
class _MaterializeContext:
    env: BuildEnv
    catalog: DagCatalog
    table_key: str
    target_name: str
    partition_columns: tuple[str, ...]
    settings_view: SettingsView


@dataclass(frozen=True, slots=True)
class _IcebergPlan:
    table_schema: TableSchema
    arrow_schema: pa.Schema
    contract_schema: pa.Schema
    observation: SchemaObservationAccumulator
    iceberg_bundle: IcebergSchemaBundle
    name_mapping_digest: str
    field_ids: dict[str, int]
    extras_policy: ExtrasPolicy
    write_settings: dict[str, object]


@dataclass(frozen=True, slots=True)
class _ValidationSetup:
    policy: ValidationPolicy
    collector: ValidationCollector | None
    pk_tracker: PrimaryKeyTracker | None


@dataclass(frozen=True, slots=True)
class _ValidationOutcome:
    metadata: TableMaterializationMetadata
    error: str | None


@dataclass(frozen=True)
class IcebergDatasetSaver(DataSaver):
    """Persist tabular outputs as Iceberg tables."""

    env: BuildEnv
    catalog: DagCatalog
    target_name: str
    table_key: str
    partition_columns: tuple[str, ...] = ()
    collect_group: str | None = None
    output_role: Literal["contract", "internal"] | None = None

    @classmethod
    def name(cls) -> str:
        """Return the stable saver identifier.

        Returns
        -------
        str
            Stable saver identifier.
        """
        return "codeintel.iceberg"

    @classmethod
    def applicable_types(cls) -> list[type]:
        """Return the output types supported by this saver.

        Returns
        -------
        list[type]
            Supported output types.
        """
        return list(_TABULAR_TYPES)

    @classmethod
    def applies_to(cls, type_: type) -> bool:
        """Return True when this saver applies to the provided output type.

        Returns
        -------
        bool
            True when the saver supports the provided type.
        """
        origin = get_origin(type_)
        if origin in {types.UnionType, typing.Union}:
            args = set(get_args(type_))
            if args.issubset(set(_TABULAR_TYPES) | {type(None)}):
                return True
        return super().applies_to(type_)

    def save_data(self, data: object) -> dict[str, object]:
        """Persist tabular data to Iceberg.

        Returns
        -------
        dict[str, object]
            Materialization metadata mapping.
        """

        def _materialize(
            _context: MaterializationContext,
            value: object,
            input_hash: str | None,
            start: float,
        ) -> MaterializationResult:
            settings_view = SettingsView.from_build_env(self.env)
            if not settings_view.build.iceberg.write_enabled:
                require_iceberg_write(
                    settings=settings_view.build.iceberg,
                    table_key=self.table_key,
                )
                msg = "Iceberg writes are disabled for this build."
                raise ValueError(msg)

            ctx = _MaterializeContext(
                env=self.env,
                catalog=self.catalog,
                table_key=self.table_key,
                target_name=self.target_name,
                partition_columns=self.partition_columns,
                settings_view=settings_view,
            )
            return _materialize_iceberg(
                ctx=ctx,
                data=cast("IcebergInput", value),
                input_hash=input_hash,
                start=start,
                output_role=self.output_role,
            )

        payload = MaterializationPipelineInput(
            env=self.env,
            catalog=self.catalog,
            target_name=self.target_name,
            table_key=self.table_key,
            data=data,
            recoverable_exceptions=_RECOVERABLE_EXCEPTIONS,
            none_error="Expected tabular data but received None",
            unknown_error="Unknown Iceberg materialization failure",
        )
        result = run_materialization_pipeline(
            payload=payload,
            materialize=_materialize,
        )
        return result.to_mapping()


def _materialize_iceberg(
    *,
    ctx: _MaterializeContext,
    data: IcebergInput,
    input_hash: str | None,
    start: float,
    output_role: str | None,
) -> MaterializationResult:
    plan = _build_plan(ctx=ctx, data=data)
    validation = _prepare_validation(
        ctx=ctx,
        plan=plan,
        output_role=output_role,
    )
    reader = _build_observed_reader(
        data=data,
        settings_view=ctx.settings_view,
        plan=plan,
        validation=validation,
    )
    snapshot_id, iceberg_stats = _write_to_iceberg(
        ctx=ctx,
        plan=plan,
        reader=reader,
    )
    _persist_observation_if_ready(
        ctx=ctx,
        observation=plan.observation,
        arrow_schema=plan.arrow_schema,
        iceberg_stats=iceberg_stats,
    )
    row_count = plan.observation.row_count
    outcome = _finalize_validation(
        ctx=ctx,
        plan=plan,
        validation=validation,
        row_count=row_count,
        snapshot_id=snapshot_id,
    )
    if outcome.error is not None:
        return failed_table_result(
            table_key=ctx.table_key,
            duration_ms=duration_ms(start),
            input_hash=input_hash or "",
            error=outcome.error,
            metadata=outcome.metadata,
        )
    return succeeded_table_result(
        table_key=ctx.table_key,
        duration_ms=duration_ms(start),
        input_hash=input_hash or "",
        row_count=row_count,
        metadata=outcome.metadata,
    )


def _prepare_validation(
    *,
    ctx: _MaterializeContext,
    plan: _IcebergPlan,
    output_role: str | None,
) -> _ValidationSetup:
    policy = resolve_validation_policy(
        env=ctx.env,
        catalog=ctx.catalog,
        table_key=ctx.table_key,
        output_role=output_role,
        declared_schema=plan.observation.declared_schema,
    )
    collector: ValidationCollector | None = None
    pk_tracker: PrimaryKeyTracker | None = None
    if policy.enabled:
        collector = ValidationCollector(
            table_key=ctx.table_key,
            target_name=ctx.target_name,
            output_role=policy.output_role,
            scope=policy.scope,
            profile=policy.profile,
        )
        if policy.run_contract_checks:
            declared_schema = plan.observation.declared_schema
            primary_keys = declared_schema.primary_key if declared_schema else ()
            if primary_keys:
                pk_tracker = PrimaryKeyTracker(
                    primary_keys=primary_keys,
                    max_rows=DEFAULT_PK_UNIQUENESS_MAX_ROWS,
                )
            else:
                collector.skip_check(
                    name="primary_key_uniqueness",
                    reason="primary_key_not_defined",
                )
        else:
            reason = policy.disabled_reason or "contract_checks_disabled"
            collector.skip_check(name="contract_columns", reason=reason)
            collector.skip_check(name="contract_nullability", reason=reason)
            collector.skip_check(name="contract_min_rows", reason=reason)
            collector.skip_check(name="primary_key_uniqueness", reason=reason)
    return _ValidationSetup(policy=policy, collector=collector, pk_tracker=pk_tracker)


def _build_observed_reader(
    *,
    data: IcebergInput,
    settings_view: SettingsView,
    plan: _IcebergPlan,
    validation: _ValidationSetup,
) -> pa.RecordBatchReader:
    reader = _record_batch_reader_for_data(
        data=data,
        settings_view=settings_view,
    )
    aligned = align_reader_to_contract_schema(reader, contract_schema=plan.contract_schema)
    if validation.collector is not None:
        aligned = wrap_reader_for_validation(
            aligned,
            collector=validation.collector,
            pk_tracker=validation.pk_tracker,
        )
    return instrument_reader_for_observation(aligned, accumulator=plan.observation)


def _finalize_validation(
    *,
    ctx: _MaterializeContext,
    plan: _IcebergPlan,
    validation: _ValidationSetup,
    row_count: int,
    snapshot_id: int | None,
) -> _ValidationOutcome:
    metadata = TableMaterializationMetadata(iceberg_snapshot_id=snapshot_id)
    collector = validation.collector
    if collector is None:
        return _ValidationOutcome(metadata=metadata, error=None)
    if validation.policy.run_contract_checks:
        apply_contract_checks(
            context=ContractCheckContext(
                collector=collector,
                declared_schema=plan.observation.declared_schema,
                arrow_schema=plan.arrow_schema,
                observation=plan.observation,
                row_count=row_count,
                min_rows=0,
            )
        )
        finalize_primary_key_check(
            collector=collector,
            tracker=validation.pk_tracker,
            severity="error" if collector.profile == "strict" else "warning",
        )
    report = collector.finalize(row_count=row_count)
    validation_id = new_uuid_str()
    record = MaterializationValidationRecord(
        validation_id=validation_id,
        table_key=ctx.table_key,
        repo=ctx.env.repo,
        commit=ctx.env.commit,
        target_name=ctx.target_name,
        output_role=validation.policy.output_role,
        validation_scope=validation.policy.scope,
        validation_profile=validation.policy.profile,
        status=report.status,
        issues=report.issues_payload() or None,
        checks=report.checks or None,
        skipped_checks=report.skipped_checks or None,
        iceberg_snapshot_id=snapshot_id,
    )
    ctx.env.gateway.schemas.record_materialization_validations_batch([record])
    metadata = TableMaterializationMetadata(
        iceberg_snapshot_id=snapshot_id,
        validation_id=validation_id,
        validation_status=report.status,
    )
    if report.status != "failed":
        return _ValidationOutcome(metadata=metadata, error=None)
    error_detail = report.issues[0].message if report.issues else None
    error = error_detail or "Validation failed for materialized output"
    return _ValidationOutcome(metadata=metadata, error=error)


def _record_batch_reader_for_data(
    *,
    data: IcebergInput,
    settings_view: SettingsView,
) -> pa.RecordBatchReader:
    options = PolarsExecutionOptions(
        streaming=settings_view.build.polars_streaming,
        query_opt_flags=resolve_query_opt_flags(settings_view.build.polars_query_opt_flags),
        inspect=settings_view.build.polars_inspect,
        streaming_fallback=settings_view.build.polars_streaming_fallback,
    )
    return to_record_batch_reader(
        data,
        batch_size=DEFAULT_ARROW_BATCH_SIZE,
        options=options,
    )


def _build_plan(*, ctx: _MaterializeContext, data: IcebergInput) -> _IcebergPlan:
    tag_sets = schema_tag_sets_for_table(catalog=ctx.catalog, table_key=ctx.table_key)
    schema_hints = schema_hints_from_tag_sets(tag_sets)
    declared_schema = declared_schema_hint(ctx.table_key)
    if declared_schema is None:
        declared_schema = table_schema_from_tag_sets(
            table_key=ctx.table_key,
            tag_sets=tag_sets,
        )
    table_schema = table_schema_for_data(
        table_key=ctx.table_key,
        data=data,
        declared_schema=declared_schema,
        schema_hints=schema_hints,
    )
    arrow_schema = arrow_schema_for_data(data=data)
    extras_policy = _extras_policy_for_table(table_key=ctx.table_key)
    iceberg_bundle = table_schema_to_iceberg_schema(table_schema)
    field_ids = iceberg_field_ids_for_table_schema(table_schema)
    name_mapping_digest = _name_mapping_digest(iceberg_bundle)
    inferred_settings = _load_inferred_settings(ctx=ctx)
    write_settings = _build_write_settings(inferred_settings=inferred_settings)
    contract_metadata = ArrowSchemaMetadata(
        schema_hash=schema_hash(table_schema),
        contract_version=ARROW_SCHEMA_CONTRACT_VERSION,
        extras_policy=extras_policy,
        extras_column=DEFAULT_EXTRAS_COLUMN,
        iceberg_schema_id=iceberg_bundle.schema.schema_id,
        iceberg_name_mapping_digest=name_mapping_digest,
        iceberg_field_ids=field_ids,
    )
    contract_schema = arrow_contract_for_table_schema(
        table_schema=table_schema,
        metadata=contract_metadata,
    )
    observation = SchemaObservationAccumulator(
        table_key=ctx.table_key,
        declared_schema=declared_schema,
        schema_hints=schema_hints,
    )
    return _IcebergPlan(
        table_schema=table_schema,
        arrow_schema=arrow_schema,
        contract_schema=contract_schema,
        observation=observation,
        iceberg_bundle=iceberg_bundle,
        name_mapping_digest=name_mapping_digest,
        field_ids=field_ids,
        extras_policy=extras_policy,
        write_settings=write_settings,
    )


def _extras_policy_for_table(*, table_key: str) -> ExtrasPolicy:
    try:
        provider = get_schema_provider()
    except RuntimeError:
        return "reject"
    derivation = provider.derivation(table_key)
    if derivation is None:
        return "retain"
    if derivation.source_kind == "declared_source":
        return "retain"
    return "reject"


def _name_mapping_digest(bundle: IcebergSchemaBundle) -> str:
    payload = bundle.name_mapping.model_dump(
        by_alias=True,
        exclude_none=True,
    )
    return stable_hash(payload)


def _load_inferred_settings(*, ctx: _MaterializeContext) -> dict[str, object] | None:
    try:
        observation = ctx.env.gateway.schemas.load_latest_schema_observation(
            table_key=ctx.table_key
        )
    except (DuckDBError, RuntimeError, TypeError, ValueError):
        return None
    if observation is None or observation.derived_settings is None:
        return None
    return dict(observation.derived_settings)


def _build_write_settings(*, inferred_settings: dict[str, object] | None) -> dict[str, object]:
    compression: str | None = None
    max_rows_per_file: int | None = None
    row_group_size: int | None = None
    data_page_size: int | None = None
    dictionary_encode = False
    dictionary_max_default = 256
    dictionary_max = dictionary_max_default if dictionary_encode else None
    dictionary_columns: tuple[str, ...] | None = None
    unify_dictionaries = False
    if inferred_settings is not None:
        inferred_columns = _coerce_tuple(inferred_settings.get("dictionary_encode_columns"))
        inferred_max = _coerce_int(inferred_settings.get("dictionary_max_cardinality"))
        inferred_unify = _coerce_bool(inferred_settings.get("unify_dictionaries"))
        inferred_row_group = _coerce_int(inferred_settings.get("row_group_size"))
        inferred_page = _coerce_int(inferred_settings.get("data_page_size"))
        if inferred_columns is not None:
            dictionary_columns = inferred_columns
            dictionary_encode = True
        if inferred_max is not None:
            dictionary_max = inferred_max
        if inferred_unify is not None:
            unify_dictionaries = inferred_unify
        elif dictionary_columns is not None:
            unify_dictionaries = True
        if inferred_row_group is not None:
            row_group_size = inferred_row_group
        if inferred_page is not None:
            data_page_size = inferred_page
    if dictionary_columns is not None and dictionary_max is None:
        dictionary_max = dictionary_max_default
    return {
        "compression": compression,
        "max_rows_per_file": max_rows_per_file,
        "row_group_size": row_group_size,
        "data_page_size": data_page_size,
        "dictionary_encode": dictionary_encode,
        "dictionary_max_cardinality": dictionary_max,
        "dictionary_encode_columns": dictionary_columns,
        "unify_dictionaries": unify_dictionaries,
    }


def _coerce_int(value: object | None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _coerce_bool(value: object | None) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _coerce_tuple(value: object | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return tuple(str(item) for item in value)
    return None


def _write_to_iceberg(
    *,
    ctx: _MaterializeContext,
    plan: _IcebergPlan,
    reader: pa.RecordBatchReader,
) -> tuple[int | None, IcebergStatsPayload | None]:
    provider = IcebergCatalogProvider(ctx.settings_view.build.iceberg)
    catalog = provider.load()
    identifier = provider.resolve_identifier(ctx.table_key)
    table = _ensure_table(
        catalog=catalog,
        identifier=identifier,
        ctx=ctx,
        plan=plan,
    )
    previous_snapshot_id = table.metadata.current_snapshot_id
    snapshot_properties = _snapshot_properties(ctx=ctx, plan=plan)
    writer = _IcebergWriter(
        table=table,
        plan=plan,
        snapshot_properties=snapshot_properties,
        partition_columns=_resolve_partition_columns(
            plan.table_schema,
            ctx.partition_columns,
        ),
    )
    snapshot_id = writer.write(reader)
    table.refresh()
    snapshot_id = table.metadata.current_snapshot_id
    _update_snapshot_refs(
        ctx=ctx,
        table=table,
        snapshot_id=snapshot_id,
    )
    refresh_iceberg_metadata_cache(
        gateway=ctx.env.gateway,
        table_key=ctx.table_key,
        table=table,
    )
    tombstone_stats = None
    if ctx.settings_view.build.iceberg.tombstones_enabled:
        tombstone_stats = _maybe_append_tombstones(
            ctx=ctx,
            plan=plan,
            table=table,
            previous_snapshot_id=previous_snapshot_id,
            current_snapshot_id=snapshot_id,
        )
    iceberg_stats: IcebergStatsPayload | None = None
    try:
        iceberg_stats = iceberg_stats_for_table(table, snapshot_id=snapshot_id)
    except (RuntimeError, ValueError, TypeError, OSError):
        iceberg_stats = None
    if not iceberg_stats:
        iceberg_stats = _minimal_iceberg_stats(table=table, snapshot_id=snapshot_id)
    if iceberg_stats is not None and tombstone_stats is not None:
        _merge_tombstone_stats(iceberg_stats, tombstone_stats)
    if iceberg_stats:
        persist_iceberg_statistics(
            table=table,
            table_key=ctx.table_key,
            stats=iceberg_stats,
            snapshot_properties=snapshot_properties,
        )
    return snapshot_id, iceberg_stats


def _minimal_iceberg_stats(
    *,
    table: Table,
    snapshot_id: int | None,
) -> IcebergStatsPayload | None:
    if snapshot_id is None:
        return None
    payload: IcebergStatsPayload = {"snapshot_id": snapshot_id}
    try:
        snapshot = table.snapshot_by_id(snapshot_id)
    except (RuntimeError, TypeError, ValueError, OSError, AttributeError):
        snapshot = None
    if snapshot is not None:
        schema_id = getattr(snapshot, "schema_id", None)
        if isinstance(schema_id, int):
            payload["schema_id"] = schema_id
    return payload


def _update_snapshot_refs(
    *,
    ctx: _MaterializeContext,
    table: Table,
    snapshot_id: int | None,
) -> None:
    if snapshot_id is None:
        return
    refs: list[tuple[str, str]] = []
    if ctx.env.commit:
        refs.append(("tag", f"commit/{ctx.env.commit}"))
    if ctx.env.execution_context is not None:
        run_id = ctx.env.execution_context.run.run_id
        if run_id:
            refs.append(("tag", f"run/{run_id}"))
    if not refs:
        return
    try:
        with table.manage_snapshots() as manager:
            manager.create_branch(snapshot_id, "main")
            for ref_type, ref_name in refs:
                if ref_type == "tag":
                    manager.create_tag(snapshot_id, ref_name)
    except (RuntimeError, ValueError) as exc:
        LOG.warning("Iceberg snapshot ref update failed for %s: %s", ctx.table_key, exc)


def _snapshot_properties(
    *,
    ctx: _MaterializeContext,
    plan: _IcebergPlan,
) -> dict[str, str]:
    run_id = None
    if ctx.env.execution_context is not None:
        run_id_value = ctx.env.execution_context.run.run_id
        if run_id_value:
            run_id = run_id_value
    return snapshot_properties_for_write(
        SnapshotPropertyInputs(
            table_key=ctx.table_key,
            repo=ctx.env.repo,
            commit=ctx.env.commit,
            run_id=run_id,
            target_name=ctx.target_name,
            schema_hash=schema_hash(plan.table_schema),
            producer_version=ctx.settings_view.build.engine_version,
            write_settings=plan.write_settings,
        )
    )


def _ensure_table(
    *,
    catalog: Catalog,
    identifier: tuple[str, ...],
    ctx: _MaterializeContext,
    plan: _IcebergPlan,
) -> Table:
    tag_sets = schema_tag_sets_for_table(catalog=ctx.catalog, table_key=ctx.table_key)
    if catalog.table_exists(identifier):
        table = catalog.load_table(identifier)
        _apply_table_updates(
            table=table,
            plan=plan,
            partition_columns=_resolve_partition_columns(
                plan.table_schema,
                ctx.partition_columns,
            ),
            tag_sets=tag_sets,
            settings=ctx.settings_view.build.iceberg,
        )
        return table
    _ensure_namespace(catalog=catalog, identifier=identifier)
    properties = _table_properties(plan=plan, settings=ctx.settings_view.build.iceberg)
    partition_spec = _partition_spec(
        table_schema=plan.table_schema,
        iceberg_schema=plan.iceberg_bundle.schema,
        tag_sets=tag_sets,
        fallback_columns=_resolve_partition_columns(
            plan.table_schema,
            ctx.partition_columns,
        ),
    )
    sort_order = _sort_order(
        iceberg_schema=plan.iceberg_bundle.schema,
        tag_sets=tag_sets,
    )
    return catalog.create_table(
        identifier,
        schema=plan.iceberg_bundle.schema,
        partition_spec=partition_spec,
        sort_order=sort_order,
        properties=properties,
    )


def _ensure_namespace(*, catalog: Catalog, identifier: tuple[str, ...]) -> None:
    namespace = identifier[:-1]
    if not namespace:
        return
    try:
        catalog.create_namespace(namespace)
    except NamespaceAlreadyExistsError:
        return


def _table_properties(*, plan: _IcebergPlan, settings: IcebergSettings) -> dict[str, str]:
    properties: dict[str, str] = {
        TableProperties.DEFAULT_NAME_MAPPING: plan.iceberg_bundle.name_mapping.model_dump_json()
    }
    compression = plan.write_settings.get("compression")
    row_group_size = _coerce_int(plan.write_settings.get("row_group_size"))
    data_page_size = _coerce_int(plan.write_settings.get("data_page_size"))
    if isinstance(compression, str) and compression:
        properties[TableProperties.PARQUET_COMPRESSION] = compression
    if row_group_size is not None:
        properties[TableProperties.PARQUET_ROW_GROUP_LIMIT] = str(row_group_size)
    if data_page_size is not None:
        properties[TableProperties.PARQUET_PAGE_SIZE_BYTES] = str(data_page_size)
    properties.update(iceberg_location_properties(settings))
    return properties


def _apply_table_updates(
    *,
    table: Table,
    plan: _IcebergPlan,
    partition_columns: tuple[str, ...],
    tag_sets: Sequence[Mapping[str, object]],
    settings: IcebergSettings,
) -> None:
    with table.transaction() as tx:
        _apply_table_properties(tx=tx, properties=_table_properties(plan=plan, settings=settings))
        _apply_schema_update(tx=tx, plan=plan)
        _apply_partition_update(
            tx=tx,
            table_schema=plan.table_schema,
            iceberg_schema=plan.iceberg_bundle.schema,
            tag_sets=tag_sets,
            fallback_columns=partition_columns,
        )


def _apply_schema_update(*, tx: Transaction, plan: _IcebergPlan) -> None:
    if plan.extras_policy == "retain":
        with tx.update_schema() as update_schema:
            update_schema.union_by_name(plan.iceberg_bundle.schema)


def _apply_table_properties(*, tx: Transaction, properties: Mapping[str, str]) -> None:
    if not properties:
        return
    current = dict(tx.table_metadata.properties)
    updates = {key: value for key, value in properties.items() if current.get(key) != value}
    if updates:
        tx.set_properties(updates)


def _apply_partition_update(
    *,
    tx: Transaction,
    table_schema: TableSchema,
    iceberg_schema: Schema,
    tag_sets: Sequence[Mapping[str, object]],
    fallback_columns: tuple[str, ...],
) -> None:
    partition_spec = _partition_spec(
        table_schema=table_schema,
        iceberg_schema=iceberg_schema,
        tag_sets=tag_sets,
        fallback_columns=fallback_columns,
    )
    if partition_spec.is_unpartitioned():
        return
    current_spec = tx.table_metadata.spec()
    existing = {
        (field.source_id, str(field.transform), field.name) for field in current_spec.fields
    }
    with tx.update_spec() as update_spec:
        for field in partition_spec.fields:
            key = (field.source_id, str(field.transform), field.name)
            if key in existing:
                continue
            update_spec.add_field(
                source_column_name=iceberg_schema.find_field(field.source_id).name,
                transform=str(field.transform),
                partition_field_name=field.name,
            )


class _SnapshotProducer(Protocol):
    commit_uuid: uuid.UUID

    def __enter__(self) -> object: ...

    def __exit__(self, _: object, value: object, traceback: object) -> None: ...

    def append_data_file(self, data_file: DataFile) -> object: ...

    def delete_data_file(self, data_file: DataFile) -> object: ...


class _DuckDBArrowReader(Protocol):
    def fetch_arrow_reader(self) -> pa.RecordBatchReader: ...


class _IcebergWriter:
    def __init__(
        self,
        *,
        table: Table,
        plan: _IcebergPlan,
        snapshot_properties: Mapping[str, str],
        partition_columns: tuple[str, ...],
    ) -> None:
        self._table = table
        self._plan = plan
        self._snapshot_properties = dict(snapshot_properties)
        self._partition_columns = partition_columns

    def write(self, reader: pa.RecordBatchReader) -> int | None:
        policy = self._write_policy()
        if policy.mode == "append":
            return self._append(reader)
        if policy.mode == "replace":
            return self._overwrite(reader, policy.replace_scope)
        msg = f"Iceberg does not support write policy mode: {policy.mode}"
        raise ValueError(msg)

    def _write_policy(self) -> TableWritePolicy:
        policy = self._plan.table_schema.write_policy
        if policy is None:
            return TableWritePolicy()
        return policy

    def _append(self, reader: pa.RecordBatchReader) -> int | None:
        with self._table.transaction() as tx:
            update_snapshot = tx.update_snapshot(snapshot_properties=self._snapshot_properties)
            append_files = _append_producer(
                tx=tx,
                update_snapshot=update_snapshot,
            )
            with append_files:
                write_ctx = _build_write_context(
                    table_metadata=tx.table_metadata,
                    io=self._table.io,
                    append_files=append_files,
                    collect_partitions=False,
                )
                _append_reader_batches(reader=reader, context=write_ctx)
        return self._table.metadata.current_snapshot_id

    def _overwrite(
        self,
        reader: pa.RecordBatchReader,
        replace_scope: Literal["snapshot", "table"],
    ) -> int | None:
        partition_spec = self._table.metadata.spec()
        if replace_scope == "snapshot" and partition_spec.is_unpartitioned():
            msg = (
                "Snapshot-scoped replace requires partitioned tables "
                "with repo/commit partition columns."
            )
            raise ValueError(msg)
        with self._table.transaction() as tx:
            update_snapshot = tx.update_snapshot(snapshot_properties=self._snapshot_properties)
            overwrite_files = update_snapshot.overwrite()
            write_ctx = _build_write_context(
                table_metadata=tx.table_metadata,
                io=self._table.io,
                append_files=overwrite_files,
                collect_partitions=replace_scope == "snapshot",
            )
            partition_records = _append_reader_batches(reader=reader, context=write_ctx)
            delete_filter = _delete_filter(
                table=self._table,
                partition_records=partition_records,
                replace_scope=replace_scope,
            )
            _delete_matching_files(
                table=self._table,
                overwrite_files=overwrite_files,
                delete_filter=delete_filter,
            )
        return self._table.metadata.current_snapshot_id


@dataclass(frozen=True, slots=True)
class _IcebergWriteContext:
    table_metadata: TableMetadata
    io: FileIO
    append_files: _SnapshotProducer
    write_uuid: uuid.UUID
    counter: Iterator[int]
    target_size: int
    collect_partitions: bool


def _build_write_context(
    *,
    table_metadata: TableMetadata,
    io: FileIO,
    append_files: _SnapshotProducer,
    collect_partitions: bool,
) -> _IcebergWriteContext:
    write_uuid = append_files.commit_uuid
    counter = _task_counter()
    target_size = _target_file_size(table_metadata)
    return _IcebergWriteContext(
        table_metadata=table_metadata,
        io=io,
        append_files=append_files,
        write_uuid=write_uuid,
        counter=counter,
        target_size=target_size,
        collect_partitions=collect_partitions,
    )


def _append_reader_batches(
    *,
    reader: pa.RecordBatchReader,
    context: _IcebergWriteContext,
) -> set[Record]:
    partition_records: set[Record] = set()
    buffer: list[pa.RecordBatch] = []
    buffer_bytes = 0
    for batch in reader:
        buffer.append(batch)
        buffer_bytes += batch.nbytes
        if buffer_bytes >= context.target_size:
            partition_records.update(_flush_batches(buffer=buffer, context=context))
            buffer = []
            buffer_bytes = 0
    if buffer:
        partition_records.update(_flush_batches(buffer=buffer, context=context))
    return partition_records


def _flush_batches(
    *,
    buffer: list[pa.RecordBatch],
    context: _IcebergWriteContext,
) -> set[Record]:
    arrow_table = pa.Table.from_batches(buffer, schema=buffer[0].schema)
    data_files, partition_records = _write_data_files(
        arrow_table=arrow_table,
        context=context,
    )
    for data_file in data_files:
        context.append_files.append_data_file(data_file)
    return partition_records


def _write_data_files(
    *,
    arrow_table: pa.Table,
    context: _IcebergWriteContext,
) -> tuple[tuple[DataFile, ...], set[Record]]:
    table_meta = context.table_metadata
    file_io = context.io
    name_mapping = table_meta.schema().name_mapping
    downcast_ns_timestamp_to_us = (
        IcebergConfig().get_bool("downcast-ns-timestamp-to-us-on-write") or False
    )
    task_schema = pyarrow_to_schema(
        arrow_table.schema,
        name_mapping=name_mapping,
        downcast_ns_timestamp_to_us=downcast_ns_timestamp_to_us,
        format_version=table_meta.format_version,
    )
    data_files: list[DataFile] = []
    partition_records: set[Record] = set()
    if table_meta.spec().is_unpartitioned():
        tasks = _write_tasks_for_table(
            arrow_table=arrow_table,
            context=_write_task_context(
                task_schema=task_schema,
                write_uuid=context.write_uuid,
                counter=context.counter,
                target_size=context.target_size,
                partition_key=None,
            ),
        )
        data_files.extend(
            write_file(
                io=file_io,
                table_metadata=table_meta,
                tasks=iter(tasks),
            )
        )
        return tuple(data_files), partition_records
    partitions = _determine_partitions(
        spec=table_meta.spec(),
        schema=table_meta.schema(),
        arrow_table=arrow_table,
    )
    for partition in partitions:
        if context.collect_partitions:
            partition_records.add(partition.partition_key.partition)
        tasks = _write_tasks_for_table(
            arrow_table=partition.arrow_table_partition,
            context=_write_task_context(
                task_schema=task_schema,
                write_uuid=context.write_uuid,
                counter=context.counter,
                target_size=context.target_size,
                partition_key=partition.partition_key,
            ),
        )
        data_files.extend(
            write_file(
                io=file_io,
                table_metadata=table_meta,
                tasks=iter(tasks),
            )
        )
    return tuple(data_files), partition_records


@dataclass(frozen=True, slots=True)
class _WriteTaskContext:
    task_schema: object
    write_uuid: uuid.UUID
    counter: Iterator[int]
    target_size: int
    partition_key: PartitionKey | None


def _write_task_context(
    *,
    task_schema: object,
    write_uuid: uuid.UUID,
    counter: Iterator[int],
    target_size: int,
    partition_key: PartitionKey | None,
) -> _WriteTaskContext:
    return _WriteTaskContext(
        task_schema=task_schema,
        write_uuid=write_uuid,
        counter=counter,
        target_size=target_size,
        partition_key=partition_key,
    )


def _write_tasks_for_table(
    *,
    arrow_table: pa.Table,
    context: _WriteTaskContext,
) -> list[WriteTask]:
    task_schema_cast = cast("Schema", context.task_schema)
    return [
        WriteTask(
            write_uuid=context.write_uuid,
            task_id=next(context.counter),
            record_batches=batches,
            schema=task_schema_cast,
            partition_key=context.partition_key,
        )
        for batches in bin_pack_arrow_table(arrow_table, context.target_size)
    ]


def _target_file_size(table_metadata: object) -> int:
    meta = cast("TableMetadata", table_metadata)
    target_size = property_as_int(
        meta.properties,
        TableProperties.WRITE_TARGET_FILE_SIZE_BYTES,
        TableProperties.WRITE_TARGET_FILE_SIZE_BYTES_DEFAULT,
    )
    if target_size is None:
        return TableProperties.WRITE_TARGET_FILE_SIZE_BYTES_DEFAULT
    return target_size


def _task_counter() -> Iterator[int]:
    return count(1)


@dataclass(frozen=True)
class _TablePartition:
    partition_key: PartitionKey
    arrow_table_partition: pa.Table


def _determine_partitions(
    *,
    spec: PartitionSpec,
    schema: object,
    arrow_table: pa.Table,
) -> list[_TablePartition]:
    iceberg_schema = cast("Schema", schema)
    partition_fields = [f"_partition_{field.name}" for field in spec.fields]
    for partition, name in zip(spec.fields, partition_fields, strict=True):
        source_field = iceberg_schema.find_field(partition.source_id)
        full_field_name = iceberg_schema.find_column_name(partition.source_id)
        if full_field_name is None:
            msg = f"Could not find column name for field ID: {partition.source_id}"
            raise ValueError(msg)
        field_array = _get_field_from_arrow_table(arrow_table, full_field_name)
        arrow_table = arrow_table.append_column(
            name,
            partition.transform.pyarrow_transform(source_field.field_type)(field_array),
        )
    unique_partition_fields = (
        arrow_table.select(partition_fields).group_by(partition_fields).aggregate([])
    )
    table_partitions: list[_TablePartition] = []
    for unique_partition in unique_partition_fields.to_pylist():
        partition_key = PartitionKey(
            field_values=[
                PartitionFieldValue(field=field, value=unique_partition[name])
                for field, name in zip(spec.fields, partition_fields, strict=True)
            ],
            partition_spec=spec,
            schema=iceberg_schema,
        )
        filtered_table = arrow_table.filter(
            _partition_predicate(
                partition_fields=partition_fields,
                partition_values=unique_partition,
            )
        )
        filtered_table = filtered_table.drop_columns(partition_fields)
        table_partitions.append(
            _TablePartition(
                partition_key=partition_key,
                arrow_table_partition=filtered_table.combine_chunks(),
            )
        )
    return table_partitions


def _compute_field(field_name: str) -> pc.Expression:
    field_fn = cast("Callable[[str], pc.Expression]", pc.field)
    return field_fn(field_name)


def _resolve_compute_fn(name: str) -> Callable[..., object]:
    func = getattr(pc, name, None)
    if not callable(func):
        msg = f"pyarrow.compute.{name} is not available"
        raise TypeError(msg)
    return func


def _compute_is_null(expr: pc.Expression) -> pc.Expression:
    is_null_fn = _resolve_compute_fn("is_null")
    return cast("pc.Expression", is_null_fn(expr))


def _compute_equal(expr: pc.Expression, value: object) -> pc.Expression:
    equal_fn = _resolve_compute_fn("equal")
    return cast("pc.Expression", equal_fn(expr, value))


def _compute_and(left: pc.Expression, right: pc.Expression) -> pc.Expression:
    and_fn = _resolve_compute_fn("and_")
    return cast("pc.Expression", and_fn(left, right))


def _compute_struct_field(field_array: pa.Array, path: Sequence[str]) -> pa.Array:
    struct_field_fn = _resolve_compute_fn("struct_field")
    return cast("pa.Array", struct_field_fn(field_array, path))


def _partition_predicate(
    *,
    partition_fields: Sequence[str],
    partition_values: Mapping[str, object],
) -> pc.Expression:
    predicates: list[pc.Expression] = []
    for field_name in partition_fields:
        value = partition_values.get(field_name)
        if value is None:
            predicates.append(_compute_is_null(_compute_field(field_name)))
        else:
            predicates.append(_compute_equal(_compute_field(field_name), value))
    combined = predicates[0]
    for predicate in predicates[1:]:
        combined = _compute_and(combined, predicate)
    return combined


def _get_field_from_arrow_table(arrow_table: pa.Table, field_path: str) -> pa.Array:
    if field_path in arrow_table.column_names:
        return arrow_table[field_path]
    path_parts = field_path.split(".")
    field_array = arrow_table[path_parts[0]]
    return _compute_struct_field(field_array, path_parts[1:])


def _append_producer(
    *,
    tx: Transaction,
    update_snapshot: UpdateSnapshot,
) -> _SnapshotProducer:
    manifest_merge_enabled = property_as_bool(
        tx.table_metadata.properties,
        TableProperties.MANIFEST_MERGE_ENABLED,
        TableProperties.MANIFEST_MERGE_ENABLED_DEFAULT,
    )
    if manifest_merge_enabled:
        return update_snapshot.merge_append()
    return update_snapshot.fast_append()


def _delete_filter(
    *,
    table: Table,
    partition_records: Collection[Record],
    replace_scope: Literal["snapshot", "table"],
) -> BooleanExpression:
    if replace_scope == "table":
        return AlwaysTrue()
    if not partition_records:
        return AlwaysFalse()
    spec = table.metadata.spec()
    schema = table.metadata.schema()
    partition_fields = [schema.find_field(field.source_id).name for field in spec.fields]
    expr: BooleanExpression = AlwaysFalse()
    for record in partition_records:
        match_expr: BooleanExpression = AlwaysTrue()
        for pos, field_name in enumerate(partition_fields):
            predicate = (
                EqualTo(Reference(field_name), record[pos])
                if record[pos] is not None
                else IsNull(Reference(field_name))
            )
            match_expr = And(match_expr, predicate)
        expr = Or(expr, match_expr)
    return expr


def _delete_matching_files(
    *,
    table: Table,
    overwrite_files: _SnapshotProducer,
    delete_filter: BooleanExpression,
) -> None:
    scan = table.scan(row_filter=delete_filter)
    tasks = scan.plan_files()
    deleted: set[object] = set()
    for task in tasks:
        data_file = task.file
        if data_file in deleted:
            continue
        deleted.add(data_file)
        overwrite_files.delete_data_file(data_file)


def _resolve_partition_columns(
    table_schema: TableSchema,
    requested: tuple[str, ...],
) -> tuple[str, ...]:
    if requested:
        _validate_partition_columns(table_schema, requested)
        return requested
    return tuple(
        column for column in _DEFAULT_PARTITION_COLUMNS if column in table_schema.column_names()
    )


def _validate_partition_columns(table_schema: TableSchema, columns: tuple[str, ...]) -> None:
    if not columns:
        return
    column_set = set(table_schema.column_names())
    missing = [column for column in columns if column not in column_set]
    if missing:
        msg = f"Partition columns missing from {table_schema.table_key}: {missing}"
        raise ValueError(msg)


def _partition_spec(
    *,
    table_schema: TableSchema,
    iceberg_schema: object,
    tag_sets: Sequence[Mapping[str, object]],
    fallback_columns: tuple[str, ...],
) -> PartitionSpec:
    schema = cast("Schema", iceberg_schema)
    columns = _partition_columns_from_tags(tag_sets, fallback_columns)
    if not columns:
        return PartitionSpec()
    _validate_partition_columns(table_schema, tuple(name for name, _ in columns))
    fields: list[PartitionField] = []
    field_id = PARTITION_FIELD_ID_START
    for name, transform in columns:
        source_id = schema.find_field(name).field_id
        field_name = name if transform == "identity" else f"{name}_{transform}"
        fields.append(
            PartitionField(
                source_id=source_id,
                field_id=field_id,
                transform=parse_transform(transform),
                name=field_name,
            )
        )
        field_id += 1
    return PartitionSpec(*fields)


def _partition_columns_from_tags(
    tag_sets: Sequence[Mapping[str, object]],
    fallback_columns: tuple[str, ...],
) -> list[tuple[str, str]]:
    for tags in tag_sets:
        keys = _tag_list(tags, _PARTITION_KEY_TAG)
        if not keys:
            continue
        transforms = _tag_list(tags, _PARTITION_TRANSFORM_TAG)
        if transforms and len(transforms) != len(keys):
            msg = "partition.transform must match partition.key length"
            raise ValueError(msg)
        resolved_transforms = transforms or ["identity"] * len(keys)
        orders = _tag_order_list(tags, _PARTITION_ORDER_TAG)
        if orders is not None:
            if len(orders) != len(keys):
                msg = "partition.order must match partition.key length"
                raise ValueError(msg)
            ordered = list(zip(keys, resolved_transforms, orders, strict=True))
            ordered.sort(key=lambda item: item[2])
            return [(key, transform) for key, transform, _ in ordered]
        return list(zip(keys, resolved_transforms, strict=True))
    return [(column, "identity") for column in fallback_columns]


def _sort_order(
    *,
    iceberg_schema: object,
    tag_sets: Sequence[Mapping[str, object]],
) -> SortOrder:
    schema = cast("Schema", iceberg_schema)
    for tags in tag_sets:
        keys = _tag_list(tags, _SORT_KEY_TAG)
        if not keys:
            continue
        directions = _tag_list(tags, _SORT_DIRECTION_TAG)
        null_orders = _tag_list(tags, _SORT_NULL_ORDER_TAG)
        sort_fields: list[SortField] = []
        for index, key in enumerate(keys):
            source_id = schema.find_field(key).field_id
            direction = _parse_sort_direction(directions, index=index)
            null_order = _parse_null_order(null_orders, direction=direction, index=index)
            sort_fields.append(
                SortField(
                    source_id=source_id,
                    transform=parse_transform("identity"),
                    direction=direction,
                    null_order=null_order,
                )
            )
        return SortOrder(*sort_fields)
    return SortOrder(order_id=0)


def _parse_sort_direction(
    raw: list[str] | None,
    *,
    index: int,
) -> SortDirection:
    if raw and index < len(raw):
        value = raw[index].lower()
        if value == "desc":
            return SortDirection.DESC
    return SortDirection.ASC


def _parse_null_order(
    raw: list[str] | None,
    *,
    direction: SortDirection,
    index: int,
) -> NullOrder:
    if raw and index < len(raw):
        value = raw[index].lower()
        if value in {"nulls_last", "nulls-last"}:
            return NullOrder.NULLS_LAST
        if value in {"nulls_first", "nulls-first"}:
            return NullOrder.NULLS_FIRST
    return NullOrder.NULLS_LAST if direction == SortDirection.DESC else NullOrder.NULLS_FIRST


def _tag_list(tags: Mapping[str, object], key: str) -> list[str] | None:
    raw = tags.get(key)
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return raw
    return None


def _tag_order_list(tags: Mapping[str, object], key: str) -> list[int] | None:
    raw = tags.get(key)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, str):
        return _coerce_order_item(raw)
    if isinstance(raw, list):
        return _coerce_order_list(raw)
    return None


def _coerce_order_item(value: str) -> list[int] | None:
    stripped = value.strip()
    if stripped.isdigit():
        return [int(stripped)]
    return None


def _coerce_order_list(values: list[object]) -> list[int] | None:
    if not values:
        return None
    parsed: list[int] = []
    for item in values:
        parsed_item = _parse_order_item(item)
        if parsed_item is None:
            return None
        parsed.append(parsed_item)
    return parsed


def _parse_order_item(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _maybe_append_tombstones(
    *,
    ctx: _MaterializeContext,
    plan: _IcebergPlan,
    table: Table,
    previous_snapshot_id: int | None,
    current_snapshot_id: int | None,
) -> IcebergStatsPayload | None:
    if previous_snapshot_id is None or current_snapshot_id is None:
        return None
    primary_key = plan.table_schema.primary_key
    if not primary_key:
        LOG.warning("Skipping tombstones for %s: missing primary key", ctx.table_key)
        return None
    tombstone_table = _ensure_tombstone_table(ctx=ctx, plan=plan)
    deleted_reader = _tombstone_diff_reader(
        ctx=ctx,
        table=table,
        primary_key=primary_key,
        previous_snapshot_id=previous_snapshot_id,
        current_snapshot_id=current_snapshot_id,
    )
    if deleted_reader is None:
        return None
    _append_tombstones(
        table=tombstone_table,
        reader=deleted_reader,
        snapshot_properties=_snapshot_properties(ctx=ctx, plan=plan),
    )
    try:
        tombstone_table.refresh()
        tombstone_stats = iceberg_stats_for_table(tombstone_table)
    except (RuntimeError, ValueError, TypeError, OSError):
        return None
    if tombstone_stats is not None:
        persist_iceberg_statistics(
            table=tombstone_table,
            table_key=_tombstone_table_key(ctx.table_key),
            stats=tombstone_stats,
            snapshot_properties=_snapshot_properties(ctx=ctx, plan=plan),
        )
    return tombstone_stats


def _ensure_tombstone_table(*, ctx: _MaterializeContext, plan: _IcebergPlan) -> Table:
    tombstone_key = _tombstone_table_key(ctx.table_key)
    tombstone_schema = _tombstone_table_schema(plan.table_schema)
    provider = IcebergCatalogProvider(ctx.settings_view.build.iceberg)
    catalog = provider.load()
    identifier = provider.resolve_identifier(tombstone_key)
    if catalog.table_exists(identifier):
        return catalog.load_table(identifier)
    bundle = table_schema_to_iceberg_schema(tombstone_schema)
    properties = {
        TableProperties.DEFAULT_NAME_MAPPING: bundle.name_mapping.model_dump_json(),
    }
    properties.update(iceberg_location_properties(ctx.settings_view.build.iceberg))
    return catalog.create_table(
        identifier,
        schema=bundle.schema,
        partition_spec=PartitionSpec(),
        sort_order=SortOrder(order_id=0),
        properties=properties,
    )


def _tombstone_table_key(table_key: str) -> str:
    parsed = parse_table_key(table_key)
    return f"{parsed.schema}.{parsed.name}__tombstones"


def _tombstone_table_schema(table_schema: TableSchema) -> TableSchema:
    tombstone_columns: list[Column] = []
    columns_by_name = {column.name: column for column in table_schema.columns}
    for key in table_schema.primary_key:
        column = columns_by_name.get(key)
        if column is None:
            continue
        tombstone_columns.append(Column(column.name, column.type, nullable=False))
    tombstone_columns.extend(
        [
            Column("deleted_at", "TIMESTAMPTZ", nullable=False),
            Column("snapshot_id", "BIGINT", nullable=False),
            Column("run_id", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("reason", "VARCHAR"),
        ]
    )
    parsed = parse_table_key(table_schema.table_key)
    return TableSchema(
        schema=parsed.schema,
        name=f"{parsed.name}__tombstones",
        columns=tombstone_columns,
        primary_key=tuple(table_schema.primary_key),
    )


def _merge_tombstone_stats(
    base_stats: IcebergStatsPayload,
    tombstone_stats: IcebergStatsPayload,
) -> None:
    tombstone_rows = tombstone_stats.get("total_records")
    if not isinstance(tombstone_rows, int):
        return
    base_stats["tombstone_rows"] = tombstone_rows
    base_stats["deleted_rows"] = tombstone_rows
    base_rows = base_stats.get("total_records")
    if isinstance(base_rows, int) and base_rows > 0:
        base_stats["tombstone_ratio"] = tombstone_rows / base_rows


def _tombstone_diff_reader(
    *,
    ctx: _MaterializeContext,
    table: Table,
    primary_key: Sequence[str],
    previous_snapshot_id: int,
    current_snapshot_id: int,
) -> pa.RecordBatchReader | None:
    if not primary_key:
        return None
    prev_reader = table.scan(snapshot_id=previous_snapshot_id).to_arrow_batch_reader()
    curr_reader = table.scan(snapshot_id=current_snapshot_id).to_arrow_batch_reader()
    con = ctx.env.gateway.con
    prev_name = f"prev_{uuid.uuid4().hex}"
    curr_name = f"curr_{uuid.uuid4().hex}"
    try:
        con.register(prev_name, prev_reader)
        con.register(curr_name, curr_reader)
        pk_clause = " AND ".join([f"p.{col} = c.{col}" for col in primary_key])
        select_cols = ", ".join([f"p.{col}" for col in primary_key])
        deleted_at = utc_now()
        query = f"""
            SELECT
                {select_cols},
                ?::TIMESTAMPTZ AS deleted_at,
                ?::BIGINT AS snapshot_id,
                ?::VARCHAR AS run_id,
                ?::VARCHAR AS commit
            FROM {prev_name} AS p
            LEFT ANTI JOIN {curr_name} AS c
            ON {pk_clause}
        """
        run_id = ctx.env.execution_context.run.run_id if ctx.env.execution_context else ""
        relation = cast(
            "_DuckDBArrowReader",
            con.execute(
                query,
                [
                    deleted_at,
                    current_snapshot_id,
                    run_id,
                    ctx.env.commit,
                ],
            ),
        )
        reader = relation.fetch_arrow_reader()
    finally:
        con.unregister(prev_name)
        con.unregister(curr_name)
    return reader


def _append_tombstones(
    *,
    table: Table,
    reader: pa.RecordBatchReader,
    snapshot_properties: Mapping[str, str],
) -> None:
    with table.transaction() as tx:
        update_snapshot = tx.update_snapshot(snapshot_properties=dict(snapshot_properties))
        append_files = _append_producer(tx=tx, update_snapshot=update_snapshot)
        with append_files:
            write_ctx = _build_write_context(
                table_metadata=tx.table_metadata,
                io=table.io,
                append_files=append_files,
                collect_partitions=False,
            )
            _append_reader_batches(reader=reader, context=write_ctx)


def _persist_observation_if_ready(
    *,
    ctx: _MaterializeContext,
    observation: SchemaObservationAccumulator,
    arrow_schema: pa.Schema,
    iceberg_stats: IcebergStatsPayload | None,
) -> None:
    try:
        drift_history: tuple[Mapping[str, object] | None, ...] | None = None
        try:
            drift_history = ctx.env.gateway.schemas.load_recent_drift_summaries(
                table_key=ctx.table_key
            )
        except (DuckDBError, RuntimeError, TypeError, ValueError):
            drift_history = None
        inputs = SchemaObservationInputs(
            repo=ctx.env.repo,
            commit=ctx.env.commit,
            target_name=ctx.target_name,
            drift_history=drift_history,
            iceberg_stats=iceberg_stats,
        )
        bundle = observation.finalize(arrow_schema=arrow_schema, inputs=inputs)
        persist_observation_bundle(gateway=ctx.env.gateway, bundle=bundle)
    except (TypeError, ValueError, pa.ArrowInvalid) as exc:
        LOG.warning("Schema observation persistence failed for %s: %s", ctx.table_key, exc)


__all__ = ["IcebergDatasetSaver"]
