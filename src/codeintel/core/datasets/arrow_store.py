"""Arrow dataset store for snapshot-scoped tables."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from codeintel.core.columnar.arrowdsl import (
    ExecutionContext,
    ExecutionPlan,
    apply_deterministic_order,
)
from codeintel.core.columnar.compute_helpers import call_compute, combine_table_chunks
from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.dedupe_ops import DedupeTier
from codeintel.core.columnar.normalization import normalize_table_for_compute
from codeintel.core.columnar.plan_ops import ScanPlanOptions, build_scan_plan
from codeintel.core.columnar.readers import record_batch_reader_from_batches
from codeintel.core.columnar.schema_metadata import decode_metadata, merge_metadata
from codeintel.core.columnar.streaming import (
    DatasetScanOptions,
    configure_arrow_threading,
    dataset_for_manifest,
    dataset_for_path,
)
from codeintel.core.constants import DEFAULT_ARROW_USE_THREADS
from codeintel.core.datasets.manifests import (
    dataset_manifest_path,
    read_dataset_manifest,
    write_dataset_manifest,
)
from codeintel.core.datasets.paths import dataset_snapshot_dir
from codeintel.core.datasets.scanner_ops import build_scanner
from codeintel.core.manifests import ArrowDatasetManifest
from codeintel.core.schemas.arrow_metadata import arrow_schema_hash
from codeintel.core.schemas.service import get_schema_service
from codeintel.core.validation.schema_constraints import schema_metadata_errors

if TYPE_CHECKING:
    from pyarrow import RecordBatchReader, Table
    from pyarrow.dataset import FileWriteOptions

    from codeintel.core.schemas.primitives import TableSchema

    type ArrowDatasetInput = Table | RecordBatchReader
else:
    type ArrowDatasetInput = object
ExistingDataBehavior = Literal["delete_matching", "error", "overwrite_or_ignore"]

LOG = logging.getLogger(__name__)


@runtime_checkable
class _SupportsAsPy(Protocol):
    def as_py(self) -> object: ...


@runtime_checkable
class _SupportsItem(Protocol):
    def item(self) -> object: ...


@runtime_checkable
class _SupportsRichComparison(Protocol):
    def __lt__(self, other: object) -> bool: ...

    def __gt__(self, other: object) -> bool: ...


@dataclass(frozen=True, slots=True)
class ArrowDatasetStats:
    """Summary statistics for an Arrow dataset snapshot."""

    row_count: int | None
    row_group_count: int | None = None
    file_count: int | None = None
    rows_from_metadata: int | None = None
    total_bytes: int | None = None
    sort_keys: tuple[str, ...] | None = None
    column_min_max: Mapping[str, Mapping[str, object]] | None = None
    row_group_stats: Mapping[str, Mapping[str, float]] | None = None
    dictionary_encoding: Mapping[str, object] | None = None

    def to_mapping(self) -> dict[str, object] | None:
        """Return a stats mapping suitable for manifest storage.

        Returns
        -------
        dict[str, object] | None
            Manifest-ready stats mapping or None when no stats are present.
        """
        stats: dict[str, object] = {}
        if self.row_group_count is not None:
            stats["row_groups"] = self.row_group_count
        if self.file_count is not None:
            stats["file_count"] = self.file_count
        if self.rows_from_metadata is not None:
            stats["rows_from_metadata"] = self.rows_from_metadata
        if self.total_bytes is not None:
            stats["total_bytes"] = self.total_bytes
        if self.sort_keys:
            stats["sort_keys"] = list(self.sort_keys)
        if self.column_min_max:
            stats["min_max"] = {
                column: dict(values) for column, values in self.column_min_max.items()
            }
        if self.row_group_stats:
            stats["row_group_stats"] = {
                key: dict(values) for key, values in self.row_group_stats.items()
            }
        if self.dictionary_encoding:
            stats["dictionary_encoding"] = dict(self.dictionary_encoding)
        return stats or None


@dataclass(frozen=True, slots=True)
class ArrowDatasetWriteOptions:
    """Options for writing Arrow dataset snapshots."""

    partition_columns: tuple[str, ...] = ()
    existing_data_behavior: ExistingDataBehavior = "delete_matching"
    persist_manifest: bool = True
    schema_hash: str | None = None
    manifest_extras: Mapping[str, object] | None = None
    schema_metadata: Mapping[str, object] | None = None
    stable_sort_keys: tuple[str, ...] | None = None
    combine_chunks: bool | None = None
    max_rows_per_file: int | None = None
    row_group_size: int | None = None
    data_page_size: int | None = None
    compression: str | None = None
    dictionary_encode: bool = False
    dictionary_max_cardinality: int | None = None
    dictionary_encode_columns: tuple[str, ...] | None = None
    unify_dictionaries: bool = False


ArrowDatasetScanOptions = DatasetScanOptions


@dataclass(frozen=True, slots=True)
class ScanPushdown:
    """Projection and filter overrides for dataset scans."""

    columns: Sequence[str] | Mapping[str, ds.Expression] | None = None
    filter_expression: ds.Expression | None = None


@dataclass(frozen=True, slots=True)
class ArrowDatasetManifestRequest:
    """Arguments for building Arrow dataset manifests."""

    table_key: str
    snapshot_id: str
    partition_columns: tuple[str, ...]
    schema_hash: str | None = None
    extras: Mapping[str, object] | None = None
    created_at: str | None = None


def write_dataset(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    data: ArrowDatasetInput,
    options: ArrowDatasetWriteOptions | None = None,
) -> ArrowDatasetManifest:
    """Write a dataset snapshot and return its manifest.

    Parameters
    ----------
    dataset_root
        Root directory where Arrow datasets are stored.
    table_key
        Fully qualified table key (schema.table).
    snapshot_id
        Snapshot identifier used to scope the dataset.
    data
        Tabular Arrow data to write.
    options
        Optional write options for partitioning and manifest persistence.

    Returns
    -------
    ArrowDatasetManifest
        Manifest describing the written dataset snapshot.
    """
    start = perf_counter()
    resolved = options or ArrowDatasetWriteOptions()
    stable_sort_keys = _resolve_stable_sort_keys(table_key, options=resolved)
    combine_chunks = _resolve_combine_chunks(table_key, options=resolved)
    if (
        stable_sort_keys is not resolved.stable_sort_keys
        or combine_chunks is not resolved.combine_chunks
    ):
        resolved = replace(
            resolved,
            stable_sort_keys=stable_sort_keys,
            combine_chunks=combine_chunks,
        )
    execution_ctx = _execution_context_for_write(table_key=table_key, options=resolved)
    configure_arrow_threading()
    prepared = _prepare_write_data(
        data,
        table_key=table_key,
        options=resolved,
        execution_ctx=execution_ctx,
    )
    _validate_schema_metadata(prepared.schema, table_key=table_key)
    snapshot_dir = dataset_snapshot_dir(
        dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    partitioning = _partitioning(resolved.partition_columns, schema=prepared.schema)
    parquet_format, file_options = _parquet_write_options(resolved)
    ds.write_dataset(
        prepared,
        str(snapshot_dir),
        format=parquet_format,
        partitioning=partitioning,
        existing_data_behavior=resolved.existing_data_behavior,
        file_options=file_options,
        max_rows_per_file=resolved.max_rows_per_file,
        max_rows_per_group=resolved.row_group_size,
    )
    dataset = dataset_for_path(snapshot_dir, schema=prepared.schema)
    request = ArrowDatasetManifestRequest(
        table_key=table_key,
        snapshot_id=snapshot_id,
        partition_columns=resolved.partition_columns,
        schema_hash=resolved.schema_hash,
        extras=resolved.manifest_extras,
    )
    manifest = build_dataset_manifest(
        dataset=dataset,
        snapshot_dir=snapshot_dir,
        request=request,
    )
    _write_parquet_metadata_sidecars(
        snapshot_dir=snapshot_dir,
        schema=prepared.schema,
        files=manifest.files,
    )
    if resolved.persist_manifest:
        manifest_path = dataset_manifest_path(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
        write_dataset_manifest(manifest_path, manifest)
    if LOG.isEnabledFor(logging.INFO):
        duration_ms = (perf_counter() - start) * 1000
        LOG.info(
            "Arrow dataset write: table=%s rows=%s files=%s duration_ms=%.2f",
            table_key,
            manifest.row_count,
            len(manifest.files),
            duration_ms,
        )
    return manifest


def _write_parquet_metadata_sidecars(
    *,
    snapshot_dir: Path,
    schema: pa.Schema,
    files: Sequence[str],
) -> None:
    if not files:
        _write_common_metadata(snapshot_dir=snapshot_dir, schema=schema)
        return
    metadata_path = snapshot_dir / "_metadata"
    metadata_collector: list[pq.FileMetaData] = []
    for entry in files:
        path = _resolve_parquet_path(snapshot_dir, entry)
        try:
            parquet_file = pq.ParquetFile(str(path))
        except (OSError, ValueError, pa.ArrowInvalid):
            continue
        metadata = parquet_file.metadata
        if metadata is not None:
            metadata_collector.append(metadata)
    if metadata_collector:
        metadata_schema = _metadata_schema_for_sidecar(
            base_schema=schema,
            metadata_collector=metadata_collector,
        )
        try:
            pq.write_metadata(
                metadata_schema,
                metadata_path,
                metadata_collector=metadata_collector,
            )
        except (OSError, ValueError, pa.ArrowInvalid):
            LOG.debug("Failed to write _metadata sidecar for %s", snapshot_dir)
    _write_common_metadata(snapshot_dir=snapshot_dir, schema=schema)


def _metadata_schema_for_sidecar(
    *,
    base_schema: pa.Schema,
    metadata_collector: Sequence[pq.FileMetaData],
) -> pa.Schema:
    if not metadata_collector:
        return base_schema
    metadata_schema = metadata_collector[0].schema.to_arrow_schema()
    if metadata_schema.metadata == base_schema.metadata:
        return metadata_schema
    merged = merge_metadata(metadata_schema.metadata, decode_metadata(base_schema.metadata))
    if merged is None:
        return metadata_schema
    return metadata_schema.with_metadata(merged)


def _write_common_metadata(*, snapshot_dir: Path, schema: pa.Schema) -> None:
    common_path = snapshot_dir / "_common_metadata"
    try:
        pq.write_metadata(schema, common_path)
    except (OSError, ValueError, pa.ArrowInvalid):
        LOG.debug("Failed to write _common_metadata sidecar for %s", snapshot_dir)


def _resolve_parquet_path(snapshot_dir: Path, entry: str) -> Path:
    path = Path(entry)
    if path.is_absolute():
        return path
    return snapshot_dir / path


def scan_dataset(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> ds.Dataset:
    """Return a dataset handle for a snapshot.

    Parameters
    ----------
    dataset_root
        Root directory where Arrow datasets are stored.
    table_key
        Fully qualified table key (schema.table).
    snapshot_id
        Snapshot identifier used to scope the dataset.

    Returns
    -------
    pyarrow.dataset.Dataset
        Dataset handle for scanning.

    Raises
    ------
    FileNotFoundError
        If the dataset snapshot directory does not exist.
    """
    snapshot_dir = dataset_snapshot_dir(
        dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    if not snapshot_dir.is_dir():
        msg = f"Dataset snapshot not found: {snapshot_dir}"
        raise FileNotFoundError(msg)
    manifest_path = dataset_manifest_path(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    if manifest_path.is_file():
        try:
            manifest = read_dataset_manifest(manifest_path)
            return dataset_for_manifest(manifest=manifest, manifest_path=manifest_path)
        except (OSError, ValueError, pa.ArrowInvalid):
            LOG.debug("Falling back to raw dataset scan for %s", manifest_path)
    return dataset_for_path(snapshot_dir)


def scan_dataset_scanner(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    options: ArrowDatasetScanOptions | None = None,
    pushdown: ScanPushdown | None = None,
) -> ds.Scanner:
    """Return a dataset scanner for streaming reads.

    Parameters
    ----------
    dataset_root
        Root directory where Arrow datasets are stored.
    table_key
        Fully qualified table key (schema.table).
    snapshot_id
        Snapshot identifier used to scope the dataset.
    options
        Batch size and scan options for the scanner.
    pushdown
        Optional projection/filter overrides for scan pushdown.

    Returns
    -------
    pyarrow.dataset.Scanner
        Scanner configured for streaming reads.
    """
    resolved_options = _merge_scan_pushdown_options(
        options or DatasetScanOptions(),
        pushdown=pushdown,
    )
    _log_missing_scan_pushdown(
        table_key=table_key,
        snapshot_id=snapshot_id,
        options=resolved_options,
    )
    dataset = scan_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    return build_scanner(dataset, options=resolved_options)


def scan_dataset_reader(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    options: ArrowDatasetScanOptions | None = None,
    pushdown: ScanPushdown | None = None,
) -> pa.RecordBatchReader:
    """Return a RecordBatchReader for a dataset snapshot.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader streaming record batches from the dataset.
    """
    resolved_options = _merge_scan_pushdown_options(
        options or DatasetScanOptions(),
        pushdown=pushdown,
    )
    _log_missing_scan_pushdown(
        table_key=table_key,
        snapshot_id=snapshot_id,
        options=resolved_options,
    )
    dataset = scan_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    plan_reader = _plan_scan_reader(dataset, resolved_options)
    if plan_reader is not None:
        return plan_reader
    scanner = build_scanner(dataset, options=resolved_options)
    return scanner.to_reader()


def _plan_scan_reader(
    dataset: ds.Dataset,
    options: ArrowDatasetScanOptions,
) -> pa.RecordBatchReader | None:
    use_threads = options.use_threads
    resolved_use_threads = use_threads if use_threads is not None else True
    try:
        plan = build_scan_plan(
            dataset,
            options=ScanPlanOptions(
                columns=options.projection_columns(),
                filter_expr=options.filter_expression,
                implicit_ordering=options.implicit_ordering,
                require_sequenced_output=options.require_sequenced_output,
            ),
        )
        return plan.to_reader(use_threads=resolved_use_threads)
    except (
        pa.ArrowInvalid,
        pa.ArrowNotImplementedError,
        pa.ArrowTypeError,
        TypeError,
        ValueError,
    ):
        return None


def _merge_scan_pushdown_options(
    options: ArrowDatasetScanOptions,
    *,
    pushdown: ScanPushdown | None,
) -> ArrowDatasetScanOptions:
    if pushdown is None:
        return options
    resolved = options
    if pushdown.columns is not None:
        resolved = replace(resolved, columns=pushdown.columns)
    if pushdown.filter_expression is not None:
        resolved = replace(resolved, filter_expression=pushdown.filter_expression)
    return resolved


def _log_missing_scan_pushdown(
    *,
    table_key: str,
    snapshot_id: str,
    options: ArrowDatasetScanOptions,
) -> None:
    if options.columns is None and options.filter_expression is None:
        LOG.debug("Dataset scan without pushdown for %s@%s", table_key, snapshot_id)


def dataset_stats(dataset: ds.Dataset) -> ArrowDatasetStats:
    """Return lightweight dataset statistics.

    Parameters
    ----------
    dataset
        Arrow dataset handle.

    Returns
    -------
    ArrowDatasetStats
        Statistics derived from the dataset.
    """
    files = tuple(dataset.files)
    parquet_stats = _parquet_stats(files)
    parquet_rows = parquet_stats.rows_from_metadata if parquet_stats else None
    sort_keys = parquet_stats.sort_keys if parquet_stats and parquet_stats.sort_keys else None
    column_min_max = (
        parquet_stats.column_min_max if parquet_stats and parquet_stats.column_min_max else None
    )
    row_group_stats = (
        parquet_stats.row_group_stats if parquet_stats and parquet_stats.row_group_stats else None
    )
    dictionary_encoding = (
        parquet_stats.dictionary_encoding
        if parquet_stats and parquet_stats.dictionary_encoding
        else None
    )
    return ArrowDatasetStats(
        row_count=_count_rows(dataset, parquet_rows=parquet_rows),
        row_group_count=parquet_stats.row_group_count if parquet_stats else None,
        file_count=parquet_stats.file_count if parquet_stats else None,
        rows_from_metadata=parquet_stats.rows_from_metadata if parquet_stats else None,
        total_bytes=parquet_stats.total_bytes if parquet_stats else None,
        sort_keys=sort_keys,
        column_min_max=column_min_max,
        row_group_stats=row_group_stats,
        dictionary_encoding=dictionary_encoding,
    )


def build_dataset_manifest(
    *,
    dataset: ds.Dataset,
    snapshot_dir: Path,
    request: ArrowDatasetManifestRequest,
) -> ArrowDatasetManifest:
    """Return a dataset manifest for an on-disk dataset snapshot.

    Parameters
    ----------
    dataset
        Arrow dataset handle for the snapshot.
    snapshot_dir
        Snapshot directory containing the dataset files.
    request
        Manifest parameters for the snapshot.

    Returns
    -------
    ArrowDatasetManifest
        Manifest describing the on-disk dataset snapshot.
    """
    stats = dataset_stats(dataset)
    files = _relative_files(dataset.files, base_dir=snapshot_dir)
    resolved_hash = request.schema_hash or arrow_schema_hash(dataset.schema)
    extras = dict(request.extras) if request.extras else {}
    parquet_extras = _parquet_extras(stats)
    if parquet_extras:
        existing = extras.get("parquet_stats")
        if isinstance(existing, Mapping):
            merged = dict(existing)
            merged.update(parquet_extras)
            extras["parquet_stats"] = merged
        else:
            extras["parquet_stats"] = parquet_extras
    return ArrowDatasetManifest(
        dataset_id=request.table_key,
        snapshot_id=request.snapshot_id,
        table_key=request.table_key,
        schema_hash=resolved_hash,
        partition_columns=tuple(str(column) for column in request.partition_columns),
        files=files,
        row_count=stats.row_count,
        stats=stats.to_mapping(),
        created_at=request.created_at or datetime.now(tz=UTC).isoformat(),
        extras=extras or None,
    )


def _parquet_extras(stats: ArrowDatasetStats) -> dict[str, object] | None:
    payload: dict[str, object] = {}
    if stats.row_group_stats:
        payload["row_group_stats"] = {
            key: dict(values) for key, values in stats.row_group_stats.items()
        }
    if stats.dictionary_encoding:
        payload["dictionary_encoding"] = dict(stats.dictionary_encoding)
    return payload or None


def _partitioning(
    columns: Sequence[str],
    *,
    schema: pa.Schema | None,
) -> ds.Partitioning | None:
    if not columns:
        return None
    if schema is None:
        msg = "Partitioning requires a schema when partition columns are provided"
        raise ValueError(msg)
    try:
        fields = [schema.field(str(column)) for column in columns]
    except KeyError as exc:
        msg = f"Partition columns missing from schema: {columns}"
        raise ValueError(msg) from exc
    return ds.partitioning(schema=pa.schema(fields))


def _parquet_write_options(
    options: ArrowDatasetWriteOptions,
) -> tuple[ds.FileFormat, FileWriteOptions | None]:
    parquet_format = ds.ParquetFileFormat()
    make_options = getattr(parquet_format, "make_write_options", None)
    if not callable(make_options):
        return parquet_format, None
    signature = inspect.signature(make_options)
    kwargs: dict[str, object] = {}
    if options.compression and "compression" in signature.parameters:
        kwargs["compression"] = options.compression
    if options.data_page_size and "data_page_size" in signature.parameters:
        kwargs["data_page_size"] = options.data_page_size
    if "use_dictionary" in signature.parameters:
        if options.dictionary_encode_columns:
            kwargs["use_dictionary"] = list(options.dictionary_encode_columns)
        elif options.dictionary_encode:
            kwargs["use_dictionary"] = True
    if kwargs:
        return parquet_format, cast("FileWriteOptions", make_options(**kwargs))
    return parquet_format, cast("FileWriteOptions", make_options())


def _apply_dictionary_options(
    data: ArrowDatasetInput,
    options: ArrowDatasetWriteOptions,
) -> ArrowDatasetInput:
    if (
        not options.dictionary_encode
        and not options.dictionary_encode_columns
        and not (options.unify_dictionaries)
    ):
        return data
    encode_columns = (
        set(options.dictionary_encode_columns) if options.dictionary_encode_columns else None
    )
    encode_enabled = options.dictionary_encode or encode_columns is not None
    if isinstance(data, pa.Table):
        table = data
        if encode_enabled:
            table = _dictionary_encode_table(
                table,
                max_cardinality=options.dictionary_max_cardinality,
                encode_columns=encode_columns,
            )
        if options.unify_dictionaries:
            table = normalize_table_for_compute(table)
        return table
    if isinstance(data, pa.RecordBatchReader):
        if not encode_enabled and not options.unify_dictionaries:
            return data
        try:
            table = reader_to_table(data)
        except (OSError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError):
            LOG.debug("Dictionary encode skipped for stream input")
            return data
        if encode_enabled:
            table = _dictionary_encode_table(
                table,
                max_cardinality=options.dictionary_max_cardinality,
                encode_columns=encode_columns,
            )
        if options.unify_dictionaries:
            table = normalize_table_for_compute(table)
        return table
    return data


def _apply_schema_metadata(
    data: ArrowDatasetInput,
    metadata: Mapping[str, object] | None,
) -> ArrowDatasetInput:
    if not metadata:
        return data
    if isinstance(data, pa.Table):
        merged = merge_metadata(data.schema.metadata, metadata, overwrite=True)
        return data.replace_schema_metadata(merged)
    if isinstance(data, pa.RecordBatchReader):
        merged = merge_metadata(data.schema.metadata, metadata, overwrite=True)
        if merged == data.schema.metadata:
            return data
        schema = data.schema.with_metadata(merged)
        return record_batch_reader_from_batches(schema, data)
    return data


def _execution_plan_for_input(data: ArrowDatasetInput) -> ExecutionPlan | None:
    if isinstance(data, pa.Table):
        return ExecutionPlan.from_table(data)
    if isinstance(data, pa.RecordBatchReader):
        return ExecutionPlan.from_reader(data)
    return None


def _requires_table_transform(
    data: ArrowDatasetInput,
    *,
    options: ArrowDatasetWriteOptions,
    execution_ctx: ExecutionContext,
) -> bool:
    if isinstance(data, pa.Table):
        return True
    if options.dictionary_encode or options.dictionary_encode_columns or options.unify_dictionaries:
        return True
    if options.stable_sort_keys and options.stable_sort_keys != ():
        return True
    if execution_ctx.determinism == "canonical":
        return True
    return execution_ctx.combine_chunks


def _prepare_write_data(
    data: ArrowDatasetInput,
    *,
    table_key: str,
    options: ArrowDatasetWriteOptions,
    execution_ctx: ExecutionContext,
) -> ArrowDatasetInput:
    prepared = _apply_schema_metadata(data, options.schema_metadata)
    if not _requires_table_transform(prepared, options=options, execution_ctx=execution_ctx):
        return prepared
    plan = _execution_plan_for_input(prepared)
    if plan is None:
        msg = f"Unsupported write input for table {table_key}"
        raise TypeError(msg)

    def _apply_dictionary(table: pa.Table) -> pa.Table:
        return cast("pa.Table", _apply_dictionary_options(table, options))

    post: list[Callable[[pa.Table], pa.Table]] = [
        _apply_dictionary,
        lambda table: _apply_stable_sort(
            table,
            sort_keys=options.stable_sort_keys,
            determinism=execution_ctx.determinism,
        ),
        lambda table: _apply_chunk_consolidation(
            table,
            combine_chunks=execution_ctx.combine_chunks,
        ),
    ]
    table = plan.to_table(ctx=execution_ctx)
    for step in post:
        table = step(table)
    return table


def _apply_stable_sort(
    table: pa.Table,
    *,
    sort_keys: Sequence[str] | None,
    determinism: DedupeTier,
) -> pa.Table:
    if not sort_keys:
        return apply_deterministic_order(
            table,
            sort_keys=(),
            determinism=determinism,
        )
    available = [key for key in sort_keys if key in table.column_names]
    resolved_keys: list[tuple[str, Literal["ascending", "descending"]]] = [
        (key, "ascending") for key in available
    ]
    return apply_deterministic_order(
        table,
        sort_keys=resolved_keys,
        determinism=determinism,
    )


def _apply_chunk_consolidation(
    table: pa.Table,
    *,
    combine_chunks: bool,
) -> pa.Table:
    if not combine_chunks:
        return table
    return combine_table_chunks(table)


def _resolve_stable_sort_keys(
    table_key: str,
    *,
    options: ArrowDatasetWriteOptions,
) -> tuple[str, ...] | None:
    if options.stable_sort_keys is not None:
        return options.stable_sort_keys
    schema = _lookup_table_schema(table_key)
    if schema is None:
        return None
    policy = schema.write_policy
    if policy is not None and policy.stable_sort_keys is not None:
        return policy.stable_sort_keys
    return schema.primary_key or None


def _resolve_combine_chunks(
    table_key: str,
    *,
    options: ArrowDatasetWriteOptions,
) -> bool:
    if options.combine_chunks is not None:
        return options.combine_chunks
    schema = _lookup_table_schema(table_key)
    if schema is None:
        return True
    policy = schema.write_policy
    if policy is not None and policy.combine_chunks is not None:
        return policy.combine_chunks
    return True


def _write_determinism(stable_sort_keys: tuple[str, ...] | None) -> DedupeTier:
    return "throughput" if stable_sort_keys == () else "canonical"


def _execution_context_for_write(
    *,
    table_key: str,
    options: ArrowDatasetWriteOptions,
) -> ExecutionContext:
    determinism = _write_determinism(options.stable_sort_keys)
    if determinism == "canonical" and not options.stable_sort_keys:
        msg = f"Canonical dataset writes require stable_sort_keys or primary key: {table_key}"
        raise ValueError(msg)
    combine_chunks = True if options.combine_chunks is None else options.combine_chunks
    return ExecutionContext(
        use_threads=DEFAULT_ARROW_USE_THREADS,
        determinism=determinism,
        combine_chunks=combine_chunks,
    )


def _lookup_table_schema(table_key: str) -> TableSchema | None:
    try:
        service = get_schema_service()
    except RuntimeError:
        return None
    return service.get_table_schema(table_key)


def _validate_schema_metadata(schema: pa.Schema, *, table_key: str) -> None:
    errors = schema_metadata_errors(schema)
    if not errors:
        return
    message = f"Arrow schema metadata validation failed for {table_key}: " + "; ".join(errors)
    raise ValueError(message)


def _dictionary_encode_table(
    table: pa.Table,
    *,
    max_cardinality: int | None,
    encode_columns: set[str] | None,
) -> pa.Table:
    if max_cardinality is None or max_cardinality <= 0:
        return table
    arrays: list[pa.Array | pa.ChunkedArray] = []
    fields: list[pa.Field] = []
    for name in table.schema.names:
        column = table.column(name)
        if encode_columns is not None and name not in encode_columns:
            encoded = column
        else:
            encoded = _maybe_dictionary_encode_array(column, max_cardinality=max_cardinality)
        arrays.append(encoded)
        fields.append(pa.field(name, encoded.type))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _maybe_dictionary_encode_array(
    array: pa.Array | pa.ChunkedArray,
    *,
    max_cardinality: int,
) -> pa.Array | pa.ChunkedArray:
    data_type = array.type
    if not (pa.types.is_string(data_type) or pa.types.is_large_string(data_type)):
        return array
    distinct = _count_distinct(array)
    if distinct is None or distinct > max_cardinality:
        return array
    return _dictionary_encode(array)


def _count_distinct(array: pa.Array | pa.ChunkedArray) -> int | None:
    result = call_compute("count_distinct", [array])
    return _coerce_int(_normalize_stat_value(result))


def _dictionary_encode(array: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    result = call_compute("dictionary_encode", [array])
    if isinstance(result, (pa.Array, pa.ChunkedArray)):
        return result
    return array


def _count_rows(dataset: ds.Dataset, *, parquet_rows: int | None) -> int | None:
    if parquet_rows is not None:
        return parquet_rows
    counter = getattr(dataset, "count_rows", None)
    if callable(counter):
        try:
            coerced = _coerce_int(counter())
            if coerced is not None:
                return coerced
        except (TypeError, ValueError, pa.ArrowInvalid):
            pass
    scanner = dataset.scanner()
    scanner_counter = getattr(scanner, "count_rows", None)
    if callable(scanner_counter):
        try:
            coerced = _coerce_int(scanner_counter())
            if coerced is not None:
                return coerced
        except (TypeError, ValueError, pa.ArrowInvalid):
            pass
    return None


def _relative_files(files: Iterable[str], *, base_dir: Path) -> tuple[str, ...]:
    base_dir = base_dir.resolve()
    normalized: list[str] = []
    for file in files:
        path = Path(file)
        try:
            normalized.append(str(path.resolve().relative_to(base_dir)))
        except (ValueError, RuntimeError):
            normalized.append(str(path))
    return tuple(normalized)


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


@dataclass(frozen=True, slots=True)
class _ParquetStats:
    row_group_count: int
    file_count: int
    rows_from_metadata: int
    total_bytes: int
    sort_keys: tuple[str, ...]
    column_min_max: dict[str, dict[str, object]]
    row_group_stats: dict[str, dict[str, float]] | None
    dictionary_encoding: dict[str, object] | None


_DICTIONARY_ENCODINGS = frozenset({"PLAIN_DICTIONARY", "RLE_DICTIONARY"})


def _parquet_stats(files: tuple[str, ...]) -> _ParquetStats | None:
    if not files:
        return None
    row_groups = 0
    rows = 0
    total_bytes = 0
    sort_keys: list[str] = []
    sort_key_seen: set[str] = set()
    min_max: dict[str, tuple[object, object]] = {}
    row_group_rows: list[int] = []
    row_group_bytes: list[int] = []
    dictionary_counts: dict[str, int] = {}
    for path in files:
        file_path = Path(path)
        try:
            total_bytes += file_path.stat().st_size
        except OSError:
            continue
        try:
            parquet_file = pq.ParquetFile(file_path)
        except (OSError, pa.ArrowInvalid):
            continue
        row_groups += parquet_file.num_row_groups
        metadata = parquet_file.metadata
        if metadata is not None:
            rows += metadata.num_rows
            _extend_sort_keys(
                sort_keys,
                sort_key_seen,
                metadata=metadata,
                schema=parquet_file.schema_arrow,
            )
            _merge_min_max(metadata, min_max)
            _extend_row_group_stats(
                metadata,
                row_group_rows=row_group_rows,
                row_group_bytes=row_group_bytes,
                dictionary_counts=dictionary_counts,
            )
    row_group_stats = _row_group_stats(row_group_rows, row_group_bytes)
    dictionary_encoding = _dictionary_encoding(dictionary_counts, row_group_count=row_groups)
    return _ParquetStats(
        row_group_count=row_groups,
        file_count=len(files),
        rows_from_metadata=rows,
        total_bytes=total_bytes,
        sort_keys=tuple(sort_keys),
        column_min_max=_min_max_to_mapping(min_max),
        row_group_stats=row_group_stats,
        dictionary_encoding=dictionary_encoding,
    )


def _extend_sort_keys(
    keys: list[str],
    seen: set[str],
    *,
    metadata: pq.FileMetaData,
    schema: pa.Schema,
) -> None:
    sort_columns = getattr(metadata, "sorting_columns", None)
    if not sort_columns:
        return
    names = schema.names
    for column in sort_columns:
        index = getattr(column, "column_idx", None)
        if index is None or index < 0 or index >= len(names):
            continue
        name = names[index]
        if name in seen:
            continue
        seen.add(name)
        keys.append(name)


def _extend_row_group_stats(
    metadata: pq.FileMetaData,
    *,
    row_group_rows: list[int],
    row_group_bytes: list[int],
    dictionary_counts: dict[str, int],
) -> None:
    for group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(group_index)
        row_count = _coerce_int(row_group.num_rows)
        if row_count is not None:
            row_group_rows.append(row_count)
        byte_size = _coerce_int(row_group.total_byte_size)
        if byte_size is not None:
            row_group_bytes.append(byte_size)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            if not _has_dictionary_encoding(column):
                continue
            column_name = column.path_in_schema
            dictionary_counts[column_name] = dictionary_counts.get(column_name, 0) + 1


def _has_dictionary_encoding(column: pq.ColumnChunkMetaData) -> bool:
    encodings = getattr(column, "encodings", None)
    if not encodings:
        return False
    return any(str(encoding).upper() in _DICTIONARY_ENCODINGS for encoding in encodings)


def _row_group_stats(
    row_group_rows: Sequence[int],
    row_group_bytes: Sequence[int],
) -> dict[str, dict[str, float]] | None:
    stats: dict[str, dict[str, float]] = {}
    row_stats = _summary_stats(row_group_rows)
    if row_stats is not None:
        stats["rows"] = row_stats
    byte_stats = _summary_stats(row_group_bytes)
    if byte_stats is not None:
        stats["bytes"] = byte_stats
    return stats or None


def _summary_stats(values: Sequence[int]) -> dict[str, float] | None:
    if not values:
        return None
    count = len(values)
    total = sum(values)
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "avg": float(total / count),
    }


def _dictionary_encoding(
    dictionary_counts: Mapping[str, int],
    *,
    row_group_count: int,
) -> dict[str, object] | None:
    if not dictionary_counts:
        return None
    return {
        "row_groups": row_group_count,
        "columns": dict(sorted(dictionary_counts.items())),
    }


def _merge_min_max(
    metadata: pq.FileMetaData,
    accumulator: dict[str, tuple[object, object]],
) -> None:
    for group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(group_index)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            stats = column.statistics
            if stats is None or not getattr(stats, "has_min_max", False):
                continue
            min_value, max_value = _extract_min_max(stats)
            if min_value is None or max_value is None:
                continue
            column_name = column.path_in_schema
            accumulator[column_name] = _merge_min_max_pair(
                accumulator.get(column_name),
                min_value=min_value,
                max_value=max_value,
            )


def _extract_min_max(stats: object) -> tuple[object | None, object | None]:
    min_value = _normalize_stat_value(getattr(stats, "min", None))
    max_value = _normalize_stat_value(getattr(stats, "max", None))
    if min_value is None or max_value is None:
        return None, None
    return min_value, max_value


def _normalize_stat_value(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, _SupportsAsPy):
        return value.as_py()
    if isinstance(value, _SupportsItem):
        try:
            return value.item()
        except (TypeError, ValueError, OverflowError):
            return value
    return value


def _merge_min_max_pair(
    current: tuple[object, object] | None,
    *,
    min_value: object,
    max_value: object,
) -> tuple[object, object]:
    if current is None:
        return min_value, max_value
    current_min, current_max = current
    return _safe_min(current_min, min_value), _safe_max(current_max, max_value)


def _safe_min(current: object, candidate: object) -> object:
    if isinstance(candidate, _SupportsRichComparison) and isinstance(
        current, _SupportsRichComparison
    ):
        try:
            return candidate if candidate < current else current
        except TypeError:
            return current
    return current


def _safe_max(current: object, candidate: object) -> object:
    if isinstance(candidate, _SupportsRichComparison) and isinstance(
        current, _SupportsRichComparison
    ):
        try:
            return candidate if candidate > current else current
        except TypeError:
            return current
    return current


def _min_max_to_mapping(
    min_max: dict[str, tuple[object, object]],
) -> dict[str, dict[str, object]]:
    return {
        column: {
            "min": _json_safe_value(values[0]),
            "max": _json_safe_value(values[1]),
        }
        for column, values in min_max.items()
    }


def _json_safe_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        result: object = value
    elif isinstance(value, bytes):
        result = value.hex()
    elif isinstance(value, (datetime, date)):
        result = value.isoformat()
    elif isinstance(value, Decimal):
        result = str(value)
    elif isinstance(value, _SupportsAsPy):
        result = _json_safe_value(value.as_py())
    elif isinstance(value, _SupportsItem):
        try:
            result = _json_safe_value(value.item())
        except (TypeError, ValueError, OverflowError):
            result = str(value)
    else:
        result = str(value)
    return result


__all__ = [
    "ArrowDatasetInput",
    "ArrowDatasetManifestRequest",
    "ArrowDatasetScanOptions",
    "ArrowDatasetStats",
    "ArrowDatasetWriteOptions",
    "ExistingDataBehavior",
    "ScanPushdown",
    "build_dataset_manifest",
    "dataset_stats",
    "scan_dataset",
    "scan_dataset_reader",
    "scan_dataset_scanner",
    "write_dataset",
]
