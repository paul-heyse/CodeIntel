"""Arrow dataset saver for Hamilton materialization."""

from __future__ import annotations

import inspect
import logging
import shutil
import threading
import types
import typing
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Literal, cast, get_args, get_origin

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
from hamilton.io.data_adapters import DataSaver
from polars.exceptions import PolarsError

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers.base import (
    MaterializationContextError,
    duration_ms,
    resolve_materialization_context,
)
from codeintel.build.schemas import get_schema_provider
from codeintel.core.columnar import (
    LazyFrameStream,
)
from codeintel.core.execution.materialization import failed_table_result, succeeded_table_result
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.schemas.arrow_polars import (
    table_schema_from_arrow_schema,
    table_schema_from_polars_lazyframe,
)
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import TableSchema
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.datasets.arrow_store import (
    ArrowDatasetManifestRequest,
    ArrowDatasetWriteOptions,
    build_dataset_manifest,
    write_dataset,
)
from codeintel.storage.datasets.manifests import dataset_manifest_path, write_dataset_manifest
from codeintel.storage.datasets.paths import dataset_snapshot_dir

if TYPE_CHECKING:
    from pathlib import Path

    from pyarrow import RecordBatchReader

    from codeintel.core.config.settings import ArrowDatasetSettings
    from codeintel.core.manifests import ArrowDatasetManifest

    type ArrowDatasetInput = RecordBatchReader
else:
    type ArrowDatasetInput = object

type TabularData = ArrowDatasetInput | pl.LazyFrame

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
    pl.LazyFrame,
)

_DEFAULT_PARTITION_COLUMNS: tuple[str, ...] = ("repo", "commit", "target")
_COLLECT_GROUP_TAG = "ci.collect_group"
_COLLECT_ALL_WAIT_S = 0.5

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
        origin = get_origin(type_)
        if origin in {types.UnionType, typing.Union}:
            args = set(get_args(type_))
            if args.issubset(set(_TABULAR_TYPES) | {type(None)}):
                return True
        return super().applies_to(type_)

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
    table_schema = _table_schema_for_data(table_key=ctx.table_key, data=data)
    contract_schema = arrow_contract_for_table_schema(table_schema=table_schema)
    resolved_partitions = _resolve_partition_columns(
        table_schema=table_schema,
        requested=ctx.partition_columns,
    )
    schema_hash_value = schema_hash(table_schema)
    extras = _manifest_extras(table_schema=table_schema, table_key=ctx.table_key)
    options = _build_write_options(
        ctx=ctx,
        partition_columns=resolved_partitions,
        schema_hash_value=schema_hash_value,
        extras=extras,
    )
    snapshot_id = _snapshot_id(ctx.env)
    dataset_root = ctx.env.paths.dataset_root_dir

    if isinstance(data, pl.LazyFrame):
        write_ctx = _DatasetWriteContext(
            dataset_root=dataset_root,
            table_key=ctx.table_key,
            snapshot_id=snapshot_id,
            options=options,
            arrow_settings=ctx.env.settings.arrow_dataset,
        )
        manifest = _write_lazyframe_dataset(
            ctx=write_ctx,
            data=data,
            contract_schema=contract_schema,
        )
        manifest_path = dataset_manifest_path(
            dataset_root=dataset_root,
            table_key=ctx.table_key,
            snapshot_id=snapshot_id,
        )
        return manifest, manifest_path

    arrow_input = _coerce_arrow_input(data)
    aligned_reader = _align_reader_to_contract(arrow_input, contract_schema=contract_schema)
    manifest = write_dataset(
        dataset_root=dataset_root,
        table_key=ctx.table_key,
        snapshot_id=snapshot_id,
        data=aligned_reader,
        options=options,
    )
    manifest_path = dataset_manifest_path(
        dataset_root=dataset_root,
        table_key=ctx.table_key,
        snapshot_id=snapshot_id,
    )
    return manifest, manifest_path


def _build_write_options(
    *,
    ctx: _MaterializeContext,
    partition_columns: tuple[str, ...],
    schema_hash_value: str,
    extras: dict[str, object],
) -> ArrowDatasetWriteOptions:
    settings = ctx.env.settings.arrow_dataset
    dictionary_max = settings.dictionary_max_cardinality if settings.dictionary_encode else None
    return ArrowDatasetWriteOptions(
        partition_columns=partition_columns,
        schema_hash=schema_hash_value,
        manifest_extras=extras,
        max_rows_per_file=settings.max_rows_per_file,
        row_group_size=settings.row_group_size,
        data_page_size=settings.data_page_size,
        compression=settings.compression,
        dictionary_encode=settings.dictionary_encode,
        dictionary_max_cardinality=dictionary_max,
        unify_dictionaries=settings.unify_dictionaries,
    )


def _write_lazyframe_dataset(
    *,
    ctx: _DatasetWriteContext,
    data: pl.LazyFrame,
    contract_schema: pa.Schema | None,
) -> ArrowDatasetManifest:
    snapshot_dir = dataset_snapshot_dir(
        ctx.dataset_root,
        table_key=ctx.table_key,
        snapshot_id=ctx.snapshot_id,
    )
    _prepare_snapshot_dir(snapshot_dir, behavior=ctx.options.existing_data_behavior)
    if contract_schema is not None:
        reader = LazyFrameStream(data).to_reader(batch_size=DEFAULT_ARROW_BATCH_SIZE)
        aligned = _align_reader_to_contract(reader, contract_schema=contract_schema)
        return write_dataset(
            dataset_root=ctx.dataset_root,
            table_key=ctx.table_key,
            snapshot_id=ctx.snapshot_id,
            data=aligned,
            options=ctx.options,
        )
    partition_by = list(ctx.options.partition_columns) if ctx.options.partition_columns else None
    if partition_by or not ctx.arrow_settings.enable_sink_parquet:
        reader = LazyFrameStream(data).to_reader(batch_size=DEFAULT_ARROW_BATCH_SIZE)
        return write_dataset(
            dataset_root=ctx.dataset_root,
            table_key=ctx.table_key,
            snapshot_id=ctx.snapshot_id,
            data=reader,
            options=ctx.options,
        )

    sink_path = snapshot_dir / "data.parquet"
    try:
        _sink_parquet_lazyframe(
            data,
            output_path=sink_path,
            options=ctx.options,
        )
    except (PolarsError, TypeError, ValueError) as exc:
        LOG.warning("LazyFrame sink_parquet failed; falling back to dataset write: %s", exc)
        reader = LazyFrameStream(data).to_reader(batch_size=DEFAULT_ARROW_BATCH_SIZE)
        return write_dataset(
            dataset_root=ctx.dataset_root,
            table_key=ctx.table_key,
            snapshot_id=ctx.snapshot_id,
            data=reader,
            options=ctx.options,
        )
    dataset = ds.dataset(str(snapshot_dir), format="parquet")
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
) -> None:
    sink_fn = getattr(frame, "sink_parquet", None)
    if not callable(sink_fn):
        msg = "LazyFrame.sink_parquet is unavailable"
        raise TypeError(msg)
    kwargs = _sink_parquet_kwargs(sink_fn, options=options)
    sink_fn(str(output_path), **kwargs)


def _sink_parquet_kwargs(
    sink_fn: object,
    *,
    options: ArrowDatasetWriteOptions,
) -> dict[str, object]:
    try:
        signature = inspect.signature(sink_fn)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}
    kwargs: dict[str, object] = {}
    if options.compression and "compression" in signature.parameters:
        kwargs["compression"] = options.compression
    if options.row_group_size and "row_group_size" in signature.parameters:
        kwargs["row_group_size"] = options.row_group_size
    if options.data_page_size and "data_page_size" in signature.parameters:
        kwargs["data_page_size"] = options.data_page_size
    if options.dictionary_encode:
        if "use_dictionary" in signature.parameters:
            kwargs["use_dictionary"] = True
        elif "dictionary" in signature.parameters:
            kwargs["dictionary"] = True
    return kwargs


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


def _table_schema_for_data(*, table_key: str, data: TabularData) -> TableSchema:
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


def _resolve_partition_columns(
    *, table_schema: TableSchema, requested: tuple[str, ...]
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


def _manifest_extras(*, table_schema: TableSchema, table_key: str) -> dict[str, object]:
    extras: dict[str, object] = {"table_schema": table_schema.to_json_obj()}
    provenance = _schema_provenance(table_key)
    if provenance:
        extras["provenance"] = provenance
    return extras


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


def _coerce_arrow_input(data: TabularData) -> ArrowDatasetInput:
    if isinstance(data, pa.RecordBatchReader):
        return cast("RecordBatchReader", data)
    msg = f"Unsupported Arrow dataset input type: {type(data).__name__}"
    raise TypeError(msg)


def _align_reader_to_contract(
    reader: ArrowDatasetInput,
    *,
    contract_schema: pa.Schema | None,
) -> ArrowDatasetInput:
    if contract_schema is None:
        return reader
    return align_reader_to_contract(
        reader,
        contract_schema,
        extras_policy=extras_policy_from_schema(contract_schema),
    )


__all__ = ["ArrowDatasetSaver"]
