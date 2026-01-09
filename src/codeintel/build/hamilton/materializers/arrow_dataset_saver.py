"""Arrow dataset saver for Hamilton materialization."""

from __future__ import annotations

import inspect
import logging
import shutil
import threading
import types
import typing
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal, TypeAliasType, cast, get_args, get_origin

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
from hamilton.io.data_adapters import DataSaver
from polars.exceptions import PolarsError

from codeintel.build.contracts.registry import contract_descriptor_for_table_schema
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.build_log import record_build_event
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers.base import (
    MaterializationContextError,
    duration_ms,
    resolve_materialization_context,
)
from codeintel.build.schemas import get_schema_provider
from codeintel.build.schemas.observation_pipeline import (
    ObservationPersistContext,
    ObservationPersistPayload,
    build_observation_inputs,
    build_observation_setup,
    persist_observation,
)
from codeintel.build.schemas.observation_provider import observation_provider_for_env
from codeintel.build.schemas.observations import (
    SchemaObservationAccumulator,
    SchemaObservationInputs,
    instrument_reader_for_observation,
    observe_batches,
    schema_drift_summary,
)
from codeintel.build.tabular.conversion import (
    reader_to_table,
    record_batch_reader_from_iterable,
    table_to_reader,
)
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar import (
    LazyFrameStream,
    align_reader_to_contract,
    extras_policy_from_schema,
)
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table
from codeintel.core.columnar.polars_utils import resolve_query_opt_flags
from codeintel.core.columnar.schema import DEFAULT_SCHEMA_PROMOTE_OPTIONS, SchemaPromoteOptions
from codeintel.core.config.settings import BuildSettings
from codeintel.core.datasets.arrow_store import (
    ArrowDatasetManifestRequest,
    ArrowDatasetWriteOptions,
    build_dataset_manifest,
    write_dataset,
)
from codeintel.core.datasets.manifests import (
    dataset_manifest_path,
    read_dataset_manifest,
    write_dataset_manifest,
)
from codeintel.core.datasets.paths import dataset_snapshot_dir
from codeintel.core.datasets.scanner_ops import ScannerParams, build_scanner
from codeintel.core.execution.materialization import failed_table_result, succeeded_table_result
from codeintel.core.hamilton import tags as hamilton_tags
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.arrow_gen import (
    EXTRAS_POLICIES,
    ArrowSchemaMetadata,
    ArrowSchemaProvenance,
    ExtrasPolicy,
    arrow_schema_from_table_schema,
)
from codeintel.core.schemas.arrow_polars import (
    table_schema_from_arrow_schema,
    table_schema_from_polars_lazyframe,
)
from codeintel.core.schemas.hashing import schema_digest, schema_hash
from codeintel.core.schemas.primitives import TableSchema, resolve_stable_sort_keys
from codeintel.core.schemas.resolution import resolve_table_schema

if TYPE_CHECKING:
    from pyarrow import RecordBatchReader

    from codeintel.core.config.settings import ArrowDatasetSettings
    from codeintel.core.manifests import ArrowDatasetManifest

    type ArrowDatasetInput = RecordBatchReader
else:
    type ArrowDatasetInput = object

type TabularData = InferableTabularInput

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
)

_DEFAULT_PARTITION_COLUMNS: tuple[str, ...] = ("repo", "commit")
_COLLECT_GROUP_TAG = "ci.collect_group"
_COLLECT_ALL_WAIT_S = 0.5
_PROFILE_RESULT_TUPLE_LENGTH = 2
_SCHEMA_OUTPUT_TAG = "hamilton.internal.schema_output"
_ALLOWED_SCHEMA_DRIFT_MODES: frozenset[str] = frozenset({"off", "warn", "strict"})

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CollectGroupState:
    expected_keys: tuple[str, ...]
    frames: dict[str, pl.LazyFrame] = field(default_factory=dict)
    results: dict[str, pl.DataFrame] = field(default_factory=dict)
    condition: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def collect(
        self,
        *,
        table_key: str,
        frame: pl.LazyFrame,
        allow_wait: bool,
    ) -> pl.DataFrame:
        with self.condition:
            cached = self.results.get(table_key)
            if cached is not None:
                return cached
            self.frames[table_key] = frame
            if self._ready_to_collect():
                return self._collect_all_locked(table_key)
            if allow_wait:
                self.condition.wait_for(self._ready_to_collect, timeout=_COLLECT_ALL_WAIT_S)
                if self._ready_to_collect():
                    return self._collect_all_locked(table_key)

        collected = frame.collect()
        with self.condition:
            self.results[table_key] = collected
            self.frames.pop(table_key, None)
            self.condition.notify_all()
        return collected

    def _ready_to_collect(self) -> bool:
        return bool(self.frames) and (
            len(self.frames) + len(self.results) >= len(self.expected_keys)
        )

    def _collect_all_locked(self, table_key: str) -> pl.DataFrame:
        pending_keys = [key for key in self.expected_keys if key in self.frames]
        frames = [self.frames[key] for key in pending_keys]
        if not frames:
            cached = self.results.get(table_key)
            if cached is None:
                msg = f"Collect group missing frame for {table_key}"
                raise RuntimeError(msg)
            return cached
        dataframes = [frames[0].collect()] if len(frames) == 1 else pl.collect_all(frames)
        for key, data in zip(pending_keys, dataframes, strict=True):
            self.results[key] = data
            self.frames.pop(key, None)
        self.condition.notify_all()
        cached = self.results.get(table_key)
        if cached is None:
            msg = f"Collect group missing result for {table_key}"
            raise RuntimeError(msg)
        return cached


_COLLECT_GROUPS: dict[str, _CollectGroupState] = {}
_COLLECT_GROUP_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class _MaterializeContext:
    env: BuildEnv
    catalog: DagCatalog
    table_key: str
    target_name: str
    partition_columns: tuple[str, ...]
    collect_group: str | None


@dataclass(frozen=True, slots=True)
class _DatasetWriteContext:
    dataset_root: Path
    table_key: str
    snapshot_id: str
    options: ArrowDatasetWriteOptions
    arrow_settings: ArrowDatasetSettings
    build_settings: BuildSettings


@dataclass(frozen=True, slots=True)
class _MaterializationPlan:
    arrow_schema: pa.Schema
    observation: SchemaObservationAccumulator
    contract_schema: pa.Schema | None
    options: ArrowDatasetWriteOptions
    snapshot_id: str
    dataset_root: Path


@dataclass(frozen=True, slots=True)
class _MaterializationInputs:
    table_schema: TableSchema
    drift_summary: Mapping[str, object] | None
    settings_fingerprint: str
    arrow_schema: pa.Schema
    observation: SchemaObservationAccumulator
    resolved_partitions: tuple[str, ...]
    schema_hash_value: str
    schema_digest_value: str
    inferred_settings: dict[str, object] | None
    provenance: Mapping[str, str] | None
    contract_schema: pa.Schema
    write_settings: dict[str, object]


@dataclass(frozen=True, slots=True)
class _ManifestExtrasInputs:
    table_schema: TableSchema
    table_key: str
    inferred_settings: dict[str, object] | None
    write_settings: dict[str, object]
    drift_summary: Mapping[str, object] | None
    settings_fingerprint: str
    provenance: Mapping[str, str] | None


@dataclass(frozen=True, slots=True)
class _ParquetMetadataInputs:
    ctx: _MaterializeContext
    table_schema: TableSchema
    schema_hash_value: str
    schema_digest_value: str
    partition_columns: tuple[str, ...]
    settings_fingerprint: str


@dataclass(frozen=True)
class ArrowDatasetSaver(DataSaver):
    """Persist tabular outputs as Arrow datasets with manifest metadata."""

    env: BuildEnv
    catalog: DagCatalog
    target_name: str
    table_key: str
    partition_columns: tuple[str, ...] = ()
    collect_group: str | None = None
    output_role: Literal["contract", "internal"] | None = None

    @classmethod
    def name(cls) -> str:
        """Return a stable name for this saver adapter.

        Returns
        -------
        str
            Stable saver identifier.
        """
        return "codeintel.arrow_dataset"

    @classmethod
    def applicable_types(cls) -> list[type]:
        """Return types this saver can persist.

        Returns
        -------
        list[type]
            Supported output types.
        """
        return list(_TABULAR_TYPES)

    @classmethod
    def applies_to(cls, type_: type) -> bool:
        """Return True when this saver can handle the Hamilton node output type.

        Returns
        -------
        bool
            True when the saver applies to the provided type.
        """
        resolved = _resolve_type_alias(type_)
        origin = get_origin(resolved)
        if origin in {types.UnionType, typing.Union}:
            args = set(get_args(resolved))
            non_null = {arg for arg in args if arg is not type(None)}
            if non_null and all(_is_tabular_annotation(arg) for arg in non_null):
                return True
        if _is_record_batch_iterable_type(resolved):
            return True
        if isinstance(resolved, type):
            return super().applies_to(resolved)
        return False

    def save_data(self, data: object) -> dict[str, object]:
        """Save the provided data and return metadata describing the write.

        Returns
        -------
        dict[str, object]
            Materialization metadata mapping.
        """
        start = perf_counter()
        input_hash: str | None = None
        result: MaterializationResult | None = None

        try:
            prepared = resolve_materialization_context(
                env=self.env,
                catalog=self.catalog,
                target_name=self.target_name,
            )
            if isinstance(prepared, MaterializationContextError):
                result = failed_table_result(
                    table_key=self.table_key,
                    duration_ms=duration_ms(start),
                    input_hash=prepared.input_hash or "",
                    error=prepared.message,
                )
            else:
                context = prepared
                input_hash = context.input_hash
                if data is None:
                    result = failed_table_result(
                        table_key=self.table_key,
                        duration_ms=duration_ms(start),
                        input_hash=input_hash or "",
                        error="Expected tabular data but received None",
                    )
                else:
                    context = _MaterializeContext(
                        env=self.env,
                        catalog=self.catalog,
                        table_key=self.table_key,
                        target_name=self.target_name,
                        partition_columns=self.partition_columns,
                        collect_group=self.collect_group,
                    )
                    manifest, manifest_path = _materialize_dataset(
                        ctx=context,
                        data=cast("TabularData", data),
                    )
                    row_count = manifest.row_count if manifest.row_count is not None else 0
                    result = succeeded_table_result(
                        table_key=self.table_key,
                        duration_ms=duration_ms(start),
                        input_hash=input_hash or "",
                        row_count=row_count,
                        dataset_manifest_path=str(manifest_path),
                    )
        except _RECOVERABLE_EXCEPTIONS as exc:
            result = failed_table_result(
                table_key=self.table_key,
                duration_ms=duration_ms(start),
                input_hash=input_hash or "",
                error=str(exc),
            )

        if result is None:
            result = failed_table_result(
                table_key=self.table_key,
                duration_ms=duration_ms(start),
                input_hash=input_hash or "",
                error="Unknown Arrow dataset materialization failure",
            )
        return result.to_mapping()


def _materialize_dataset(
    *,
    ctx: _MaterializeContext,
    data: TabularData,
) -> tuple[ArrowDatasetManifest, Path]:
    normalized = _normalize_tabular_data(data)
    plan = _build_materialization_plan(ctx=ctx, data=normalized)
    write_ctx = _DatasetWriteContext(
        dataset_root=plan.dataset_root,
        table_key=ctx.table_key,
        snapshot_id=plan.snapshot_id,
        options=plan.options,
        arrow_settings=ctx.env.settings.arrow_dataset,
        build_settings=ctx.env.settings,
    )

    if isinstance(normalized, pl.LazyFrame):
        manifest = _write_lazyframe_dataset(
            ctx=write_ctx,
            data=normalized,
            contract_schema=plan.contract_schema,
            observation=plan.observation,
        )
        manifest_path = dataset_manifest_path(
            dataset_root=plan.dataset_root,
            table_key=ctx.table_key,
            snapshot_id=plan.snapshot_id,
        )
        _persist_observation_if_ready(
            ctx=ctx,
            observation=plan.observation,
            arrow_schema=plan.arrow_schema,
            manifest=manifest,
        )
        return manifest, manifest_path

    arrow_input = _coerce_arrow_input(normalized)
    aligned_reader = _align_reader_to_contract(
        arrow_input,
        table_key=ctx.table_key,
        contract_schema=plan.contract_schema,
        schema_promote_options=ctx.env.settings.schema_promote_options,
    )
    manifest = _write_dataset_from_reader(
        ctx=write_ctx,
        reader=aligned_reader,
        observation=plan.observation,
    )
    manifest_path = dataset_manifest_path(
        dataset_root=plan.dataset_root,
        table_key=ctx.table_key,
        snapshot_id=plan.snapshot_id,
    )
    _persist_observation_if_ready(
        ctx=ctx,
        observation=plan.observation,
        arrow_schema=plan.arrow_schema,
        manifest=manifest,
    )
    return manifest, manifest_path


def _normalize_tabular_data(data: TabularData) -> TabularData:
    if isinstance(data, pl.LazyFrame):
        return data
    if isinstance(data, pl.DataFrame):
        return data.lazy()
    if isinstance(data, pa.RecordBatchReader):
        return data
    if isinstance(data, pa.Table):
        return table_to_reader(cast("pa.Table", data))
    if isinstance(data, Iterable):
        reader = record_batch_reader_from_iterable(
            data,
            empty_policy="error",
        )
        if reader is None:
            msg = "Record batch iterable is empty; schema cannot be inferred"
            raise ValueError(msg)
        return reader
    msg = f"Unsupported Arrow dataset input type: {type(data).__name__}"
    raise TypeError(msg)


def _build_materialization_plan(
    *,
    ctx: _MaterializeContext,
    data: TabularData,
) -> _MaterializationPlan:
    inputs = _resolve_materialization_inputs(ctx=ctx, data=data)
    extras = _manifest_extras(
        _ManifestExtrasInputs(
            table_schema=inputs.table_schema,
            table_key=ctx.table_key,
            inferred_settings=inputs.inferred_settings,
            write_settings=inputs.write_settings,
            drift_summary=inputs.drift_summary,
            settings_fingerprint=inputs.settings_fingerprint,
            provenance=inputs.provenance,
        )
    )
    parquet_metadata = _parquet_metadata_payload(
        _ParquetMetadataInputs(
            ctx=ctx,
            table_schema=inputs.table_schema,
            schema_hash_value=inputs.schema_hash_value,
            schema_digest_value=inputs.schema_digest_value,
            partition_columns=inputs.resolved_partitions,
            settings_fingerprint=inputs.settings_fingerprint,
        )
    )
    options = _build_write_options(
        inputs=inputs,
        extras=extras,
        parquet_metadata=parquet_metadata,
    )
    return _MaterializationPlan(
        arrow_schema=inputs.arrow_schema,
        observation=inputs.observation,
        contract_schema=inputs.contract_schema,
        options=options,
        snapshot_id=_snapshot_id(ctx.env),
        dataset_root=ctx.env.paths.dataset_root_dir,
    )


def _resolve_materialization_inputs(
    *,
    ctx: _MaterializeContext,
    data: TabularData,
) -> _MaterializationInputs:
    tag_sets = _schema_tag_sets_for_table(catalog=ctx.catalog, table_key=ctx.table_key)
    table_schema = _authoritative_table_schema(ctx.table_key)
    setup = build_observation_setup(
        table_key=ctx.table_key,
        tag_sets=tag_sets,
        declared_schema=table_schema,
    )
    observed_schema = _observed_schema_for_data(
        table_key=ctx.table_key,
        data=data,
    )
    drift_summary = _handle_schema_drift(
        ctx=ctx,
        inferred=observed_schema,
        baseline=table_schema,
    )
    settings_fingerprint = _settings_fingerprint(ctx.env)
    record_build_event(
        "build.dataset.settings_fingerprint",
        table_key=ctx.table_key,
        target=ctx.target_name,
        settings_fingerprint=settings_fingerprint,
    )
    arrow_schema = _arrow_schema_for_data(data=data)
    observation = setup.accumulator
    resolved_partitions = _resolve_partition_columns(
        ctx=ctx,
        table_schema=table_schema,
        observed_schema=observed_schema,
        requested=ctx.partition_columns,
    )
    schema_hash_value = schema_hash(table_schema)
    schema_digest_value = schema_digest(table_schema)
    inferred_settings = _load_inferred_settings(ctx=ctx)
    provenance = _schema_provenance(ctx.table_key)
    contract_schema = _contract_schema_for_table(
        table_schema=table_schema,
        schema_hash_value=schema_hash_value,
        schema_digest_value=schema_digest_value,
        inferred_settings=inferred_settings,
        provenance=provenance,
    )
    write_settings = _build_write_settings(
        ctx=ctx,
        inferred_settings=inferred_settings,
    )
    return _MaterializationInputs(
        table_schema=table_schema,
        drift_summary=drift_summary,
        settings_fingerprint=settings_fingerprint,
        arrow_schema=arrow_schema,
        observation=observation,
        resolved_partitions=resolved_partitions,
        schema_hash_value=schema_hash_value,
        schema_digest_value=schema_digest_value,
        inferred_settings=inferred_settings,
        provenance=provenance,
        contract_schema=contract_schema,
        write_settings=write_settings,
    )


def _build_write_options(
    *,
    inputs: _MaterializationInputs,
    extras: dict[str, object],
    parquet_metadata: Mapping[str, object] | None,
) -> ArrowDatasetWriteOptions:
    return ArrowDatasetWriteOptions(
        partition_columns=inputs.resolved_partitions,
        schema_hash=inputs.schema_hash_value,
        manifest_extras=extras,
        schema_metadata=parquet_metadata,
        stable_sort_keys=resolve_stable_sort_keys(inputs.table_schema),
        max_rows_per_file=_int_setting(inputs.write_settings, "max_rows_per_file"),
        row_group_size=_int_setting(inputs.write_settings, "row_group_size"),
        data_page_size=_int_setting(inputs.write_settings, "data_page_size"),
        compression=_str_setting(inputs.write_settings, "compression"),
        dictionary_encode=_bool_setting(inputs.write_settings, "dictionary_encode") or False,
        dictionary_max_cardinality=_int_setting(
            inputs.write_settings, "dictionary_max_cardinality"
        ),
        dictionary_encode_columns=_tuple_setting(
            inputs.write_settings, "dictionary_encode_columns"
        ),
        unify_dictionaries=_bool_setting(inputs.write_settings, "unify_dictionaries") or False,
    )


def _write_lazyframe_dataset(
    *,
    ctx: _DatasetWriteContext,
    data: pl.LazyFrame,
    contract_schema: pa.Schema | None,
    observation: SchemaObservationAccumulator | None,
) -> ArrowDatasetManifest:
    snapshot_dir = dataset_snapshot_dir(
        ctx.dataset_root,
        table_key=ctx.table_key,
        snapshot_id=ctx.snapshot_id,
    )
    _prepare_snapshot_dir(snapshot_dir, behavior=ctx.options.existing_data_behavior)
    query_opt_flags = resolve_query_opt_flags(ctx.build_settings.polars_query_opt_flags)
    _log_lazyframe_plan(
        data,
        table_key=ctx.table_key,
        streaming=ctx.build_settings.polars_streaming,
        query_opt_flags=query_opt_flags,
        inspect_enabled=ctx.build_settings.polars_inspect,
    )
    profiled = _profile_lazyframe(
        data,
        table_key=ctx.table_key,
        streaming=ctx.build_settings.polars_streaming,
        query_opt_flags=query_opt_flags,
        profile_enabled=ctx.build_settings.polars_profile,
    )
    if profiled is not None:
        LOG.warning("Polars profile enabled; materializing %s before write", ctx.table_key)
        return _write_profiled_dataset(
            ctx=ctx,
            frame=profiled,
            contract_schema=contract_schema,
            observation=observation,
            query_opt_flags=query_opt_flags,
        )
    if contract_schema is not None:
        return _write_contract_dataset(
            ctx=ctx,
            data=data,
            contract_schema=contract_schema,
            observation=observation,
            query_opt_flags=query_opt_flags,
        )
    partition_by = list(ctx.options.partition_columns) if ctx.options.partition_columns else None
    if ctx.options.schema_metadata:
        return _write_partitioned_dataset(
            ctx=ctx,
            data=data,
            observation=observation,
            query_opt_flags=query_opt_flags,
        )
    if partition_by or not ctx.arrow_settings.enable_sink_parquet:
        return _write_partitioned_dataset(
            ctx=ctx,
            data=data,
            observation=observation,
            query_opt_flags=query_opt_flags,
        )

    return _write_sink_or_dataset(
        ctx=ctx,
        data=data,
        snapshot_dir=snapshot_dir,
        observation=observation,
        query_opt_flags=query_opt_flags,
    )


def _lazyframe_reader(
    *,
    ctx: _DatasetWriteContext,
    data: pl.LazyFrame,
    query_opt_flags: object | None,
) -> pa.RecordBatchReader:
    batch_size = ctx.build_settings.arrow_scan.batch_size
    return _lazyframe_stream(
        data,
        streaming=ctx.build_settings.polars_streaming,
        streaming_fallback=ctx.build_settings.polars_streaming_fallback,
        query_opt_flags=query_opt_flags,
        inspect=ctx.build_settings.polars_inspect,
    ).to_reader(batch_size=batch_size)


def _write_dataset_from_reader(
    *,
    ctx: _DatasetWriteContext,
    reader: ArrowDatasetInput,
    observation: SchemaObservationAccumulator | None,
) -> ArrowDatasetManifest:
    table = reader_to_table(reader)
    result = finalize_table(table, spec=FinalizeSpec(table_key=ctx.table_key, mode="tolerant"))
    if result.errors.num_rows:
        LOG.warning(
            "Finalize produced %d error rows for %s; persisting good rows only",
            result.errors.num_rows,
            ctx.table_key,
        )
    batch_size = ctx.build_settings.arrow_scan.batch_size
    reader = table_to_reader(result.good, batch_size=batch_size)
    if observation is not None:
        reader = instrument_reader_for_observation(reader, accumulator=observation)
    snapshot_dir = dataset_snapshot_dir(
        ctx.dataset_root,
        table_key=ctx.table_key,
        snapshot_id=ctx.snapshot_id,
    )
    _prepare_snapshot_dir(snapshot_dir, behavior=ctx.options.existing_data_behavior)
    return write_dataset(
        dataset_root=ctx.dataset_root,
        table_key=ctx.table_key,
        snapshot_id=ctx.snapshot_id,
        data=reader,
        options=ctx.options,
    )


def _write_profiled_dataset(
    *,
    ctx: _DatasetWriteContext,
    frame: pl.DataFrame,
    contract_schema: pa.Schema | None,
    observation: SchemaObservationAccumulator | None,
    query_opt_flags: object | None,
) -> ArrowDatasetManifest:
    reader = _reader_from_frame(
        frame,
        ctx=ctx,
        query_opt_flags=query_opt_flags,
    )
    aligned = _align_reader_to_contract(
        reader,
        table_key=ctx.table_key,
        contract_schema=contract_schema,
        schema_promote_options=ctx.build_settings.schema_promote_options,
    )
    return _write_dataset_from_reader(ctx=ctx, reader=aligned, observation=observation)


def _write_contract_dataset(
    *,
    ctx: _DatasetWriteContext,
    data: pl.LazyFrame,
    contract_schema: pa.Schema | None,
    observation: SchemaObservationAccumulator | None,
    query_opt_flags: object | None,
) -> ArrowDatasetManifest:
    reader = _lazyframe_reader(ctx=ctx, data=data, query_opt_flags=query_opt_flags)
    aligned = _align_reader_to_contract(
        reader,
        table_key=ctx.table_key,
        contract_schema=contract_schema,
        schema_promote_options=ctx.build_settings.schema_promote_options,
    )
    return _write_dataset_from_reader(ctx=ctx, reader=aligned, observation=observation)


def _write_partitioned_dataset(
    *,
    ctx: _DatasetWriteContext,
    data: pl.LazyFrame,
    observation: SchemaObservationAccumulator | None,
    query_opt_flags: object | None,
) -> ArrowDatasetManifest:
    reader = _lazyframe_reader(ctx=ctx, data=data, query_opt_flags=query_opt_flags)
    return _write_dataset_from_reader(ctx=ctx, reader=reader, observation=observation)


def _write_sink_or_dataset(
    *,
    ctx: _DatasetWriteContext,
    data: pl.LazyFrame,
    snapshot_dir: Path,
    observation: SchemaObservationAccumulator | None,
    query_opt_flags: object | None,
) -> ArrowDatasetManifest:
    sink_path = snapshot_dir / "data.parquet"
    try:
        _sink_parquet_lazyframe(
            data,
            output_path=sink_path,
            options=ctx.options,
            query_opt_flags=query_opt_flags,
        )
    except (PolarsError, TypeError, ValueError) as exc:
        LOG.warning("LazyFrame sink_parquet failed; falling back to dataset write: %s", exc)
        reader = _lazyframe_reader(ctx=ctx, data=data, query_opt_flags=query_opt_flags)
        return _write_dataset_from_reader(ctx=ctx, reader=reader, observation=observation)
    dataset = ds.dataset(str(snapshot_dir), format="parquet")
    _observe_sink_dataset(
        dataset=dataset,
        table_key=ctx.table_key,
        observation=observation,
        batch_size=ctx.build_settings.arrow_scan.batch_size,
    )
    return _manifest_from_sink(ctx=ctx, dataset=dataset, snapshot_dir=snapshot_dir)


def _observe_sink_dataset(
    *,
    dataset: ds.Dataset,
    table_key: str,
    observation: SchemaObservationAccumulator | None,
    batch_size: int,
) -> None:
    if observation is None:
        return
    try:
        params = ScannerParams(batch_size=batch_size, unify_schemas=True)
        scanner = build_scanner(dataset, params=params)
        observe_batches(scanner.to_reader(), accumulator=observation)
    except (TypeError, ValueError, pa.ArrowInvalid, OSError):
        LOG.warning("Schema observation scan failed for %s", table_key)


def _manifest_from_sink(
    *,
    ctx: _DatasetWriteContext,
    dataset: ds.Dataset,
    snapshot_dir: Path,
) -> ArrowDatasetManifest:
    request = ArrowDatasetManifestRequest(
        table_key=ctx.table_key,
        snapshot_id=ctx.snapshot_id,
        partition_columns=ctx.options.partition_columns,
        schema_hash=ctx.options.schema_hash,
        extras=ctx.options.manifest_extras,
    )
    manifest = build_dataset_manifest(
        dataset=dataset,
        snapshot_dir=snapshot_dir,
        request=request,
    )
    if ctx.options.persist_manifest:
        path = dataset_manifest_path(
            dataset_root=ctx.dataset_root,
            table_key=ctx.table_key,
            snapshot_id=ctx.snapshot_id,
        )
        write_dataset_manifest(path, manifest)
    return manifest


def _sink_parquet_lazyframe(
    frame: pl.LazyFrame,
    *,
    output_path: Path,
    options: ArrowDatasetWriteOptions,
    query_opt_flags: object | None,
) -> None:
    sink_fn = getattr(frame, "sink_parquet", None)
    if not callable(sink_fn):
        msg = "LazyFrame.sink_parquet is unavailable"
        raise TypeError(msg)
    kwargs = _sink_parquet_kwargs(sink_fn, options=options, query_opt_flags=query_opt_flags)
    sink_fn(str(output_path), **kwargs)


def _add_optional_sink_kwargs(
    kwargs: dict[str, object],
    *,
    parameters: Mapping[str, inspect.Parameter],
    items: Sequence[tuple[str, object | None]],
) -> None:
    kwargs.update(
        {name: value for name, value in items if value is not None and name in parameters}
    )


def _add_dictionary_sink_kwargs(
    kwargs: dict[str, object],
    *,
    parameters: Mapping[str, inspect.Parameter],
    enable_dictionary: bool,
) -> None:
    if not enable_dictionary:
        return
    for name in ("use_dictionary", "dictionary"):
        if name in parameters:
            kwargs[name] = True
            return


def _add_query_opt_sink_kwargs(
    kwargs: dict[str, object],
    *,
    parameters: Mapping[str, inspect.Parameter],
    query_opt_flags: object | None,
) -> None:
    if query_opt_flags is None:
        return
    for name in ("optimization_flags", "query_opt_flags", "optimizations"):
        if name in parameters:
            kwargs[name] = query_opt_flags
            return


def _sink_parquet_kwargs(
    sink_fn: object,
    *,
    options: ArrowDatasetWriteOptions,
    query_opt_flags: object | None,
) -> dict[str, object]:
    try:
        signature = inspect.signature(sink_fn)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}
    kwargs: dict[str, object] = {}
    _add_optional_sink_kwargs(
        kwargs,
        parameters=signature.parameters,
        items=(
            ("compression", options.compression),
            ("row_group_size", options.row_group_size),
            ("data_page_size", options.data_page_size),
        ),
    )
    _add_dictionary_sink_kwargs(
        kwargs,
        parameters=signature.parameters,
        enable_dictionary=options.dictionary_encode,
    )
    _add_query_opt_sink_kwargs(
        kwargs,
        parameters=signature.parameters,
        query_opt_flags=query_opt_flags,
    )
    return kwargs


def _lazyframe_stream(
    frame: pl.LazyFrame,
    *,
    streaming: bool,
    streaming_fallback: bool,
    query_opt_flags: object | None,
    inspect: bool,
) -> LazyFrameStream:
    return LazyFrameStream(
        frame,
        query_opt_flags=query_opt_flags,
        streaming=streaming,
        streaming_fallback=streaming_fallback,
        inspect=inspect,
    )


def _reader_from_frame(
    frame: pl.DataFrame,
    *,
    ctx: _DatasetWriteContext,
    query_opt_flags: object | None,
) -> pa.RecordBatchReader:
    stream = LazyFrameStream(
        frame.lazy(),
        query_opt_flags=query_opt_flags,
        streaming=ctx.build_settings.polars_streaming,
        streaming_fallback=ctx.build_settings.polars_streaming_fallback,
        inspect=ctx.build_settings.polars_inspect,
    )
    return stream.to_reader(batch_size=ctx.build_settings.arrow_scan.batch_size)


def _log_lazyframe_plan(
    frame: pl.LazyFrame,
    *,
    table_key: str,
    streaming: bool,
    query_opt_flags: object | None,
    inspect_enabled: bool,
) -> None:
    if not inspect_enabled:
        return
    explain = _polars_explain(
        frame,
        streaming=streaming,
        query_opt_flags=query_opt_flags,
    )
    if explain is not None:
        LOG.debug("polars_explain table=%s plan=%s", table_key, explain)
    graph = _polars_show_graph(
        frame,
        streaming=streaming,
        query_opt_flags=query_opt_flags,
    )
    if graph is not None:
        LOG.debug("polars_graph table=%s graph=%s", table_key, graph)


def _polars_explain(
    frame: pl.LazyFrame,
    *,
    streaming: bool,
    query_opt_flags: object | None,
) -> str | None:
    explain_fn = getattr(frame, "explain", None)
    if not callable(explain_fn):
        return None
    kwargs = _polars_plan_kwargs(
        explain_fn,
        streaming=streaming,
        query_opt_flags=query_opt_flags,
        optimized=True,
    )
    try:
        result = explain_fn(**kwargs)
    except PolarsError:
        return None
    return result if isinstance(result, str) else None


def _polars_show_graph(
    frame: pl.LazyFrame,
    *,
    streaming: bool,
    query_opt_flags: object | None,
) -> str | None:
    show_graph = getattr(frame, "show_graph", None)
    if not callable(show_graph):
        return None
    try:
        signature = inspect.signature(show_graph)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        signature = None
    kwargs = _polars_plan_kwargs(
        show_graph,
        streaming=streaming,
        query_opt_flags=query_opt_flags,
        optimized=True,
    )
    if signature is not None:
        if "show" in signature.parameters:
            kwargs["show"] = False
        if "raw_output" in signature.parameters:
            kwargs["raw_output"] = True
    try:
        result = show_graph(**kwargs)
    except PolarsError:
        return None
    return result if isinstance(result, str) else None


def _profile_lazyframe(
    frame: pl.LazyFrame,
    *,
    table_key: str,
    streaming: bool,
    query_opt_flags: object | None,
    profile_enabled: bool,
) -> pl.DataFrame | None:
    if not profile_enabled:
        return None
    profile_fn = getattr(frame, "profile", None)
    if not callable(profile_fn):
        LOG.warning("Polars profile is unavailable for %s", table_key)
        return None
    kwargs = _polars_plan_kwargs(
        profile_fn,
        streaming=streaming,
        query_opt_flags=query_opt_flags,
        optimized=True,
    )
    try:
        result = profile_fn(**kwargs)
    except PolarsError as exc:
        LOG.warning("Polars profile failed for %s: %s", table_key, exc)
        return None
    if isinstance(result, tuple) and len(result) == _PROFILE_RESULT_TUPLE_LENGTH:
        frame_result, profile = result
        _log_polars_profile(table_key, profile)
        return frame_result if isinstance(frame_result, pl.DataFrame) else None
    if isinstance(result, pl.DataFrame):
        return result
    return None


def _polars_plan_kwargs(
    func: object,
    *,
    streaming: bool,
    query_opt_flags: object | None,
    optimized: bool,
) -> dict[str, object]:
    try:
        signature = inspect.signature(func)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}
    kwargs: dict[str, object] = {}
    if "optimized" in signature.parameters:
        kwargs["optimized"] = optimized
    if "engine" in signature.parameters and streaming:
        kwargs["engine"] = "streaming"
    elif "streaming" in signature.parameters:
        kwargs["streaming"] = streaming
    if query_opt_flags is not None:
        if "optimization_flags" in signature.parameters:
            kwargs["optimization_flags"] = query_opt_flags
        elif "query_opt_flags" in signature.parameters:
            kwargs["query_opt_flags"] = query_opt_flags
        elif "optimizations" in signature.parameters:
            kwargs["optimizations"] = query_opt_flags
    return kwargs


def _log_polars_profile(table_key: str, profile: object) -> None:
    if profile is None:
        return
    to_string = getattr(profile, "to_string", None)
    if callable(to_string):
        try:
            profile_repr = to_string()
        except (TypeError, ValueError):
            profile_repr = None
    else:
        profile_repr = str(profile)
    if profile_repr:
        LOG.info("polars_profile table=%s profile=%s", table_key, profile_repr)


def _collect_partitioned_frame(
    *,
    ctx: _MaterializeContext,
    frame: pl.LazyFrame,
) -> pl.DataFrame:
    if ctx.collect_group is None:
        return frame.collect()
    expected_keys = _collect_group_members(
        catalog=ctx.catalog,
        target_name=ctx.target_name,
        collect_group=ctx.collect_group,
    )
    if ctx.table_key not in expected_keys or len(expected_keys) <= 1:
        return frame.collect()
    allow_wait = _allow_collect_wait(env=ctx.env, expected_count=len(expected_keys))
    state = _collect_group_state(
        group_key=_collect_group_key(
            env=ctx.env,
            target_name=ctx.target_name,
            collect_group=ctx.collect_group,
        ),
        expected_keys=expected_keys,
    )
    return state.collect(table_key=ctx.table_key, frame=frame, allow_wait=allow_wait)


def _collect_group_members(
    *,
    catalog: DagCatalog,
    target_name: str,
    collect_group: str,
) -> tuple[str, ...]:
    outputs = catalog.table_outputs_by_target.get(target_name, ())
    members = [
        output.key for output in outputs if output.tags.get(_COLLECT_GROUP_TAG) == collect_group
    ]
    return tuple(sorted(members))


def _allow_collect_wait(*, env: BuildEnv, expected_count: int) -> bool:
    if expected_count <= 1:
        return False
    backend = env.execution_settings.parallel_backend.lower()
    if backend == "sequential":
        return False
    max_workers = env.execution_settings.max_workers
    return max_workers is None or max_workers >= expected_count


def _collect_group_key(*, env: BuildEnv, target_name: str, collect_group: str) -> str:
    return f"{env.snapshot.repo}:{env.snapshot.commit}:{target_name}:{collect_group}"


def _collect_group_state(*, group_key: str, expected_keys: tuple[str, ...]) -> _CollectGroupState:
    with _COLLECT_GROUP_LOCK:
        state = _COLLECT_GROUPS.get(group_key)
        if state is None:
            state = _CollectGroupState(expected_keys=expected_keys)
            _COLLECT_GROUPS[group_key] = state
        return state


def _prepare_snapshot_dir(snapshot_dir: Path, *, behavior: object) -> None:
    if snapshot_dir.exists():
        if behavior == "error":
            msg = f"Dataset snapshot already exists: {snapshot_dir}"
            raise FileExistsError(msg)
        if behavior in {"delete_matching", "overwrite_or_ignore"}:
            shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)


def _partitioning_from_schema(
    *,
    schema: pa.Schema,
    partition_columns: Sequence[str],
) -> ds.Partitioning:
    try:
        fields = [schema.field(str(column)) for column in partition_columns]
    except KeyError as exc:
        msg = f"Partition columns missing from schema: {partition_columns}"
        raise ValueError(msg) from exc
    return ds.partitioning(schema=pa.schema(fields))


def _observed_schema_for_data(
    *,
    table_key: str,
    data: TabularData,
) -> TableSchema:
    if isinstance(data, pl.LazyFrame):
        return table_schema_from_polars_lazyframe(frame=data, table_key=table_key)
    if isinstance(data, pa.RecordBatchReader):
        arrow_reader = cast("RecordBatchReader", data)
        return table_schema_from_arrow_schema(
            arrow_schema=arrow_reader.schema,
            table_key=table_key,
        )
    msg = f"Unsupported Arrow dataset input type: {type(data).__name__}"
    raise TypeError(msg)


def _authoritative_table_schema(table_key: str) -> TableSchema:
    provider = get_schema_provider()
    schema = provider.get_table_schema(table_key)
    if schema is None:
        msg = f"Missing TableSchema for output table {table_key}"
        raise ValueError(msg)
    return schema


def _resolve_partition_columns(
    *,
    ctx: _MaterializeContext,
    table_schema: TableSchema,
    observed_schema: TableSchema,
    requested: tuple[str, ...],
) -> tuple[str, ...]:
    if requested:
        missing_declared = _missing_partition_columns(table_schema, requested)
        if missing_declared:
            record_build_event(
                "build.schema.partition_columns_missing",
                table_key=ctx.table_key,
                target=ctx.target_name,
                stage="declared",
                missing=missing_declared,
            )
            LOG.warning(
                "build.schema.partition_columns_missing table_key=%s stage=declared missing=%s",
                ctx.table_key,
                missing_declared,
            )
        resolved = tuple(column for column in requested if column not in missing_declared)
    else:
        resolved = tuple(
            column for column in _DEFAULT_PARTITION_COLUMNS if column in table_schema.column_names()
        )
    missing_observed = _missing_partition_columns(observed_schema, resolved)
    if missing_observed:
        record_build_event(
            "build.schema.partition_columns_missing",
            table_key=ctx.table_key,
            target=ctx.target_name,
            stage="observed",
            missing=missing_observed,
        )
        LOG.warning(
            "build.schema.partition_columns_missing table_key=%s stage=observed missing=%s",
            ctx.table_key,
            missing_observed,
        )
        resolved = tuple(column for column in resolved if column not in missing_observed)
    return resolved


def _missing_partition_columns(
    table_schema: TableSchema,
    columns: Sequence[str],
) -> list[str]:
    column_set = set(table_schema.column_names())
    return [column for column in columns if column not in column_set]


def _normalize_settings_payload(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_settings_payload(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_settings_payload(item) for item in value]
    return value


def _settings_fingerprint(env: BuildEnv) -> str:
    payload = {
        "build_settings": _normalize_settings_payload(asdict(env.settings)),
        "execution_settings": _normalize_settings_payload(asdict(env.execution_settings)),
        "variants": env.variants.variant_fingerprint,
    }
    return fingerprint(payload)


def _schema_drift_mode(env: BuildEnv) -> str:
    mode = env.config.schema_drift_mode()
    if mode in _ALLOWED_SCHEMA_DRIFT_MODES:
        return mode
    msg = f"Unsupported schema drift mode: {mode}"
    raise ValueError(msg)


def _handle_schema_drift(
    *,
    ctx: _MaterializeContext,
    inferred: TableSchema,
    baseline: TableSchema | None,
) -> dict[str, object] | None:
    drift_summary = schema_drift_summary(inferred, baseline)
    if drift_summary is None:
        return None
    mode = _schema_drift_mode(ctx.env)
    missing = drift_summary.get("missing_columns")
    extra = drift_summary.get("extra_columns")
    type_changes = drift_summary.get("type_changes")
    missing_count = len(missing) if isinstance(missing, list) else 0
    extra_count = len(extra) if isinstance(extra, list) else 0
    type_change_count = len(type_changes) if isinstance(type_changes, list) else 0
    if mode != "off":
        record_build_event(
            "build.schema.drift.detected",
            table_key=ctx.table_key,
            mode=mode,
            details=drift_summary,
            missing_columns=missing,
            extra_columns=extra,
            type_changes=type_changes,
        )
        LOG.warning(
            "build.schema.drift.detected table_key=%s mode=%s missing=%d extra=%d type_changes=%d",
            ctx.table_key,
            mode,
            missing_count,
            extra_count,
            type_change_count,
        )
    else:
        LOG.info(
            "build.schema.drift.detected table_key=%s mode=off missing=%d extra=%d type_changes=%d",
            ctx.table_key,
            missing_count,
            extra_count,
            type_change_count,
        )
    if mode == "strict":
        record_build_event(
            "build.schema.drift.blocked",
            table_key=ctx.table_key,
            mode=mode,
            details=drift_summary,
        )
        LOG.warning(
            "build.schema.drift.strict_override table_key=%s missing=%d extra=%d type_changes=%d",
            ctx.table_key,
            missing_count,
            extra_count,
            type_change_count,
        )
    return drift_summary


def _manifest_extras(inputs: _ManifestExtrasInputs) -> dict[str, object]:
    extras: dict[str, object] = {"table_schema": inputs.table_schema.to_json_obj()}
    descriptor = contract_descriptor_for_table_schema(inputs.table_schema)
    extras["contract_version"] = descriptor.contract_version
    extras["contract_hash"] = descriptor.contract_hash
    resolved_provenance = inputs.provenance or _schema_provenance(inputs.table_key)
    if resolved_provenance:
        extras["provenance"] = resolved_provenance
    if inputs.drift_summary is not None:
        extras["schema_drift_summary"] = dict(inputs.drift_summary)
    extras["settings_fingerprint"] = inputs.settings_fingerprint
    if inputs.inferred_settings:
        extras["inferred_settings"] = dict(inputs.inferred_settings)
    settings_payload = _write_settings_payload(inputs.write_settings)
    if settings_payload:
        extras["write_settings"] = settings_payload
    return extras


def _parquet_metadata_payload(inputs: _ParquetMetadataInputs) -> dict[str, object]:
    columns_json = {col.name: col.type for col in inputs.table_schema.columns}
    nullability_json = {col.name: col.nullable for col in inputs.table_schema.columns}
    output = inputs.ctx.catalog.table_outputs.get(inputs.ctx.table_key)
    target = inputs.ctx.catalog.get_target(inputs.ctx.target_name)
    run_context = inputs.ctx.env.run_context
    build_id = run_context.run_id if run_context is not None else inputs.ctx.env.commit
    descriptor = contract_descriptor_for_table_schema(inputs.table_schema)
    return {
        "codeintel.table_key": inputs.table_schema.table_key,
        "codeintel.domain": inputs.table_schema.schema,
        "codeintel.target": inputs.ctx.target_name,
        "codeintel.schema_hash": inputs.schema_hash_value,
        "codeintel.schema_digest": inputs.schema_digest_value,
        "codeintel.schema_contract_version": descriptor.contract_version,
        "codeintel.settings_fingerprint": inputs.settings_fingerprint,
        "codeintel.columns_json": columns_json,
        "codeintel.nullability_json": nullability_json,
        "codeintel.primary_keys_json": list(inputs.table_schema.primary_key),
        "codeintel.partition_columns_json": list(inputs.partition_columns),
        "codeintel.build_id": build_id,
        "codeintel.repo": inputs.ctx.env.repo,
        "codeintel.commit": inputs.ctx.env.commit,
        "codeintel.snapshot_id": _snapshot_id(inputs.ctx.env),
        "codeintel.generated_at": datetime.now(UTC).isoformat(),
        "codeintel.hamilton.node": (
            output.saver_node if output is not None else inputs.ctx.target_name
        ),
        "codeintel.hamilton.graph_version": target.spec_version if target is not None else None,
        "codeintel.inputs_json": _input_lineage(ctx=inputs.ctx),
    }


def _input_lineage(*, ctx: _MaterializeContext) -> list[dict[str, object]]:
    surface = ctx.catalog.io_surfaces.get(ctx.target_name)
    if surface is None or not surface.reads:
        return []
    dataset_root = ctx.env.paths.dataset_root_dir
    snapshot_id = _snapshot_id(ctx.env)
    entries: list[dict[str, object]] = []
    for read in surface.reads:
        schema_hash = None
        if dataset_root is not None:
            schema_hash = _schema_hash_for_input(
                dataset_root=dataset_root,
                table_key=read.table_key,
                snapshot_id=snapshot_id,
            )
        entries.append({"table_key": read.table_key, "schema_hash": schema_hash})
    return entries


def _schema_hash_for_input(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> str | None:
    try:
        manifest_path = dataset_manifest_path(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
        manifest = read_dataset_manifest(manifest_path)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return manifest.schema_hash


def _contract_schema_for_table(
    *,
    table_schema: TableSchema,
    schema_hash_value: str,
    schema_digest_value: str,
    inferred_settings: dict[str, object] | None,
    provenance: Mapping[str, str] | None,
) -> pa.Schema:
    extras_policy = _extras_policy_from_settings(inferred_settings)
    metadata = ArrowSchemaMetadata(
        schema_hash=schema_hash_value,
        schema_digest=schema_digest_value,
        extras_policy=extras_policy,
        provenance=_arrow_schema_provenance(provenance),
    )
    return arrow_schema_from_table_schema(
        table_schema=table_schema,
        metadata=metadata,
    )


def _extras_policy_from_settings(
    inferred_settings: Mapping[str, object] | None,
) -> ExtrasPolicy | None:
    if inferred_settings is None:
        return None
    raw = inferred_settings.get("extras_policy")
    if isinstance(raw, str) and raw in EXTRAS_POLICIES:
        return cast("ExtrasPolicy", raw)
    return None


def _arrow_schema_provenance(
    provenance: Mapping[str, str] | None,
) -> ArrowSchemaProvenance | None:
    if provenance is None:
        return None
    derivation_kind = provenance.get("derivation_kind")
    derivation_source = provenance.get("derivation_source")
    if not isinstance(derivation_kind, str) and not isinstance(derivation_source, str):
        return None
    return ArrowSchemaProvenance(
        derivation_kind=derivation_kind if isinstance(derivation_kind, str) else None,
        derivation_source=derivation_source if isinstance(derivation_source, str) else None,
    )


def _schema_provenance(table_key: str) -> dict[str, str] | None:
    provider = get_schema_provider()
    derivation_fn = getattr(provider, "derivation", None)
    if not callable(derivation_fn):
        return None
    derivation = derivation_fn(table_key)
    if derivation is None:
        return None
    source_kind = getattr(derivation, "source_kind", None)
    source_ref = getattr(derivation, "source_ref", None)
    if isinstance(source_kind, str) and isinstance(source_ref, str):
        return {"derivation_kind": source_kind, "derivation_source": source_ref}
    return None


def _snapshot_id(env: BuildEnv) -> str:
    snapshot_id = env.snapshot.commit.strip()
    if snapshot_id:
        return snapshot_id
    msg = "Snapshot commit is required for Arrow dataset materialization"
    raise ValueError(msg)


def _resolve_type_alias(type_: object) -> object:
    if isinstance(type_, TypeAliasType):
        return _resolve_type_alias(type_.__value__)
    return type_


def _is_record_batch_iterable_type(type_: object) -> bool:
    resolved = _resolve_type_alias(type_)
    origin = get_origin(resolved)
    if origin is None:
        return False
    if not isinstance(origin, type):
        return False
    if not issubclass(origin, Iterable):
        return False
    args = get_args(resolved)
    return len(args) == 1 and args[0] is pa.RecordBatch


def _is_tabular_annotation(type_: object) -> bool:
    resolved = _resolve_type_alias(type_)
    if isinstance(resolved, type) and resolved in _TABULAR_TYPES:
        return True
    if _is_record_batch_iterable_type(resolved):
        return True
    origin = get_origin(resolved)
    if origin in {types.UnionType, typing.Union}:
        args = [arg for arg in get_args(resolved) if arg is not type(None)]
        return bool(args) and all(_is_tabular_annotation(arg) for arg in args)
    return False


def _coerce_arrow_input(data: TabularData) -> ArrowDatasetInput:
    if isinstance(data, pa.RecordBatchReader):
        return cast("RecordBatchReader", data)
    msg = f"Unsupported Arrow dataset input type: {type(data).__name__}"
    raise TypeError(msg)


def _align_reader_to_contract(
    reader: ArrowDatasetInput,
    *,
    table_key: str,
    contract_schema: pa.Schema | None,
    schema_promote_options: SchemaPromoteOptions = DEFAULT_SCHEMA_PROMOTE_OPTIONS,
) -> ArrowDatasetInput:
    if contract_schema is None:
        return reader
    extras_policy = extras_policy_from_schema(contract_schema)
    try:
        return align_reader_to_contract(
            reader,
            contract_schema,
            extras_policy=extras_policy,
            schema_promote_options=schema_promote_options,
        )
    except (
        ValueError,
        TypeError,
        pa.ArrowInvalid,
        pa.ArrowTypeError,
        pa.ArrowNotImplementedError,
    ) as exc:
        record_build_event(
            "build.schema.contract_alignment_failed",
            table_key=table_key,
            extras_policy=extras_policy,
            error=str(exc),
        )
        LOG.warning(
            "build.schema.contract_alignment_failed table_key=%s extras_policy=%s error=%s",
            table_key,
            extras_policy,
            exc,
        )
        return reader


def _schema_tag_sets_for_table(
    *,
    catalog: DagCatalog,
    table_key: str,
) -> tuple[Mapping[str, object], ...]:
    tag_sets: list[Mapping[str, object]] = []
    output = catalog.table_outputs.get(table_key)
    if output is not None:
        tag_sets.append(output.tags)
        tag_sets.extend(_schema_output_tag_sets(catalog=catalog, saver_node=output.saver_node))
    tag_sets.extend(
        node.tags
        for node in catalog.nodes.values()
        if node.tags.get(hamilton_tags.TAG_TABLE_KEY) == table_key
    )
    return tuple(tag_sets)


def _schema_output_tag_sets(
    *,
    catalog: DagCatalog,
    saver_node: str,
) -> list[Mapping[str, object]]:
    node = catalog.nodes.get(saver_node)
    if node is None:
        return []
    visited: set[str] = set()
    stack = list(node.deps)
    tag_sets: list[Mapping[str, object]] = []
    while stack:
        node_name = stack.pop()
        if node_name in visited:
            continue
        visited.add(node_name)
        candidate = catalog.nodes.get(node_name)
        if candidate is None:
            continue
        if _SCHEMA_OUTPUT_TAG in candidate.tags:
            tag_sets.append(candidate.tags)
        stack.extend(candidate.deps)
    return tag_sets


def _arrow_schema_for_data(*, data: TabularData) -> pa.Schema:
    if isinstance(data, pl.DataFrame):
        return data.lazy().collect_schema().to_arrow()
    if isinstance(data, pl.LazyFrame):
        return data.collect_schema().to_arrow()
    if isinstance(data, pa.Table):
        table = cast("pa.Table", data)
        return table.schema
    if isinstance(data, pa.RecordBatchReader):
        reader = cast("pa.RecordBatchReader", data)
        return reader.schema
    if isinstance(data, Iterable):
        batches = list(data)
        if not batches:
            msg = "Record batch iterable is empty; schema cannot be inferred"
            raise ValueError(msg)
        if not all(isinstance(batch, pa.RecordBatch) for batch in batches):
            msg = "Record batch iterable contains non-RecordBatch values"
            raise TypeError(msg)
        first_batch = cast("pa.RecordBatch", batches[0])
        return first_batch.schema
    msg = f"Unsupported Arrow dataset input type: {type(data).__name__}"
    raise TypeError(msg)


def _load_inferred_settings(*, ctx: _MaterializeContext) -> dict[str, object] | None:
    resolution = resolve_table_schema(
        ctx.table_key,
        observation_provider=observation_provider_for_env(ctx.env),
    )
    observation = resolution.observation
    if observation is None or observation.derived_settings is None:
        return None
    return dict(observation.derived_settings)


def _build_write_settings(
    *,
    ctx: _MaterializeContext,
    inferred_settings: dict[str, object] | None,
) -> dict[str, object]:
    settings = ctx.env.settings.arrow_dataset
    dictionary_encode = settings.dictionary_encode
    dictionary_max = settings.dictionary_max_cardinality if settings.dictionary_encode else None
    dictionary_columns: tuple[str, ...] | None = None
    unify_dictionaries = settings.unify_dictionaries
    row_group_size = settings.row_group_size
    data_page_size = settings.data_page_size
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
        dictionary_max = settings.dictionary_max_cardinality
    return {
        "compression": settings.compression,
        "max_rows_per_file": settings.max_rows_per_file,
        "row_group_size": row_group_size,
        "data_page_size": data_page_size,
        "dictionary_encode": dictionary_encode,
        "dictionary_max_cardinality": dictionary_max,
        "dictionary_encode_columns": dictionary_columns,
        "unify_dictionaries": unify_dictionaries,
    }


def _write_settings_payload(write_settings: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in write_settings.items():
        if value is None:
            continue
        if isinstance(value, tuple):
            payload[key] = list(value)
        else:
            payload[key] = value
    return payload


def _int_setting(settings: dict[str, object], key: str) -> int | None:
    return _coerce_int(settings.get(key))


def _str_setting(settings: dict[str, object], key: str) -> str | None:
    value = settings.get(key)
    if isinstance(value, str):
        return value
    return None


def _bool_setting(settings: dict[str, object], key: str) -> bool | None:
    return _coerce_bool(settings.get(key))


def _tuple_setting(settings: dict[str, object], key: str) -> tuple[str, ...] | None:
    return _coerce_tuple(settings.get(key))


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


def _persist_observation_if_ready(
    *,
    ctx: _MaterializeContext,
    observation: SchemaObservationAccumulator,
    arrow_schema: pa.Schema,
    manifest: ArrowDatasetManifest | None,
) -> None:
    base_inputs = SchemaObservationInputs(
        repo=ctx.env.repo,
        commit=ctx.env.commit,
        target_name=ctx.target_name,
        dataset_stats=manifest.stats if manifest is not None else None,
        manifest_row_count=manifest.row_count if manifest is not None else None,
    )
    gateway = None if ctx.env.metadata_bundle is not None else ctx.env.gateway
    inputs = build_observation_inputs(
        gateway=gateway,
        table_key=ctx.table_key,
        base=base_inputs,
    )
    persist_observation(
        context=ObservationPersistContext(
            gateway=gateway,
            metadata_bundle=ctx.env.metadata_bundle,
        ),
        payload=ObservationPersistPayload(
            observation=observation,
            arrow_schema=arrow_schema,
            inputs=inputs,
        ),
    )


__all__ = ["ArrowDatasetSaver"]
