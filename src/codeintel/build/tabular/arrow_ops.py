"""Arrow-first join and materialization helpers for build pipelines."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.conversion import (
    lazyframe_to_reader,
    reader_to_table,
    table_to_frame,
    tabular_to_arrow_reader,
)
from codeintel.build.tabular.frames import (
    JoinSpec,
    JoinStrategy,
    JoinValidation,
    join_validated,
)
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.schema_alignment import align_reader_to_contract as _align_reader
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.datasets.arrow_store import scan_dataset
from codeintel.core.datasets.scanning import DatasetScanOptions, build_scanner
from codeintel.core.schemas.arrow_gen import (
    ArrowSchemaMetadata,
    ExtrasPolicy,
    arrow_contract_for_table_schema,
)
from codeintel.core.schemas.service import SchemaService

_ARROW_JOIN_TYPES = {
    "left": "left outer",
    "inner": "inner",
    "right": "right outer",
    "full": "full outer",
    "outer": "full outer",
}

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ArrowJoinSpec:
    """Arrow join configuration for materialized joins."""

    on: Sequence[str] | None = None
    left_on: Sequence[str] | None = None
    right_on: Sequence[str] | None = None
    how: JoinStrategy = "left"
    validate: JoinValidation | None = None
    suffix: str = ""
    coalesce_keys: bool = True
    left_suffix: str | None = None
    right_suffix: str | None = None


@dataclass(frozen=True, slots=True)
class ParquetScanOptions:
    columns: Sequence[str] | None = None
    repo: str | None = None
    commit: str | None = None
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE


@dataclass(frozen=True, slots=True)
class ParquetScanSpec:
    """Parquet scan settings for snapshot retrieval."""

    dataset_root: Path
    table_key: str
    snapshot_id: str
    columns: Sequence[str] | None = None
    repo: str | None = None
    commit: str | None = None
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE


def _arrow_spec_from_join_spec(spec: JoinSpec) -> ArrowJoinSpec:
    """Create an ArrowJoinSpec from a JoinSpec.

    Returns
    -------
    ArrowJoinSpec
        Arrow join configuration.
    """
    return ArrowJoinSpec(
        on=spec.on,
        left_on=spec.left_on,
        right_on=spec.right_on,
        how=spec.how,
        validate=spec.validate,
        suffix=spec.suffix,
    )


def _join_spec_from_arrow_spec(spec: ArrowJoinSpec) -> JoinSpec:
    """Build a JoinSpec fallback from an ArrowJoinSpec.

    Returns
    -------
    JoinSpec
        Join configuration for LazyFrame joins.
    """
    suffix = spec.suffix
    if not suffix and spec.right_suffix:
        suffix = spec.right_suffix
    return JoinSpec(
        on=spec.on,
        left_on=spec.left_on,
        right_on=spec.right_on,
        how=spec.how,
        validate=spec.validate,
        suffix=suffix,
    )


def _polars_join_fallback(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    *,
    spec: ArrowJoinSpec | JoinSpec,
) -> pl.DataFrame:
    join_spec = _join_spec_from_arrow_spec(spec) if isinstance(spec, ArrowJoinSpec) else spec
    left_lazy = left.lazy() if isinstance(left, pl.DataFrame) else left
    right_lazy = right.lazy() if isinstance(right, pl.DataFrame) else right
    left_lazy, right_lazy = _coerce_null_join_keys(left_lazy, right_lazy, spec=join_spec)
    joined = join_validated(left_lazy, right_lazy, spec=join_spec)
    joined = _ensure_right_join_keys(joined, spec=join_spec)
    return joined.collect()


def _coerce_null_join_keys(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    *,
    spec: JoinSpec,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    left_keys, right_keys = _resolve_join_keys(
        on=spec.on,
        left_on=spec.left_on,
        right_on=spec.right_on,
    )
    if not left_keys:
        return left, right
    resolved_right = right_keys if right_keys is not None else left_keys
    if len(left_keys) != len(resolved_right):
        return left, right
    left_schema = left.collect_schema()
    right_schema = right.collect_schema()
    for left_key, right_key in zip(left_keys, resolved_right, strict=True):
        left_dtype = left_schema.get(left_key)
        right_dtype = right_schema.get(right_key)
        if left_dtype is None or right_dtype is None:
            continue
        if left_dtype == pl.Null and right_dtype != pl.Null:
            left = left.with_columns(pl.col(left_key).cast(right_dtype))
        elif right_dtype == pl.Null and left_dtype != pl.Null:
            right = right.with_columns(pl.col(right_key).cast(left_dtype))
    return left, right


def _ensure_right_join_keys(
    frame: pl.LazyFrame,
    *,
    spec: JoinSpec,
) -> pl.LazyFrame:
    left_keys, right_keys = _resolve_join_keys(
        on=spec.on,
        left_on=spec.left_on,
        right_on=spec.right_on,
    )
    if not right_keys:
        return frame
    resolved_right = right_keys
    if left_keys == resolved_right:
        return frame
    schema_names = set(frame.collect_schema().names())
    expressions: list[pl.Expr] = []
    for left_key, right_key in zip(left_keys, resolved_right, strict=True):
        if right_key in schema_names or left_key == right_key:
            continue
        expressions.append(pl.col(left_key).alias(right_key))
    if not expressions:
        return frame
    return frame.with_columns(expressions)


def _resolve_join_keys(
    *,
    on: Sequence[str] | None,
    left_on: Sequence[str] | None,
    right_on: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    if on:
        return tuple(on), tuple(right_on) if right_on else None
    if left_on:
        return tuple(left_on), tuple(right_on) if right_on else None
    if right_on:
        return tuple(right_on), tuple(right_on)
    return (), None


def _call_compute(
    name: str,
    args: Sequence[object],
    *,
    options: pc.FunctionOptions | None = None,
) -> object | None:
    try:
        return pc.call_function(name, list(args), options=options)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return None


def _compute_scalar(
    name: str,
    args: Sequence[object],
    *,
    options: pc.FunctionOptions | None = None,
) -> object | None:
    result = _call_compute(name, args, options=options)
    if result is None:
        return None
    if isinstance(result, pa.Scalar):
        return cast("pa.Scalar", result).as_py()
    return result


def _ensure_unique_keys(table: pa.Table, keys: Sequence[str], *, label: str) -> None:
    if not keys:
        return
    missing = [key for key in keys if key not in table.column_names]
    if missing:
        msg = f"Missing join keys on {label}: {', '.join(missing)}"
        raise ValueError(msg)
    count_source = keys[0]
    grouped = table.group_by(list(keys)).aggregate([(count_source, "count")])
    count_name = f"{count_source}_count"
    if grouped.num_rows == 0 or count_name not in grouped.column_names:
        return
    max_value = _compute_scalar(
        "max",
        [grouped[count_name]],
        options=pc.ScalarAggregateOptions(skip_nulls=True),
    )
    if isinstance(max_value, (int, float)) and not isinstance(max_value, bool) and max_value > 1:
        msg = f"Join validation failed for {label}: keys not unique"
        raise ValueError(msg)
    if max_value is not None:
        return
    try:
        counts = grouped[count_name].combine_chunks().to_numpy(zero_copy_only=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return
    if counts.size > 0 and counts.max() > 1:
        msg = f"Join validation failed for {label}: keys not unique"
        raise ValueError(msg)


def _validate_join(
    left: pa.Table,
    right: pa.Table,
    *,
    left_keys: Sequence[str],
    right_keys: Sequence[str] | None,
    validate: JoinValidation | None,
) -> None:
    if validate is None or validate == "m:m":
        return
    right_key_values = right_keys if right_keys is not None else left_keys
    if validate in {"m:1", "1:1"}:
        _ensure_unique_keys(right, right_key_values, label="right")
    if validate in {"1:m", "1:1"}:
        _ensure_unique_keys(left, left_keys, label="left")
    if validate not in {"m:1", "1:m", "1:1", "m:m"}:
        msg = f"Unsupported join validation: {validate}"
        raise ValueError(msg)


def arrow_table_from_tabular(value: InferableTabularInput) -> pa.Table:
    """Convert a tabular input into a fully materialized Arrow Table.

    Returns
    -------
    pa.Table
        Materialized Arrow table.
    """
    reader = tabular_to_arrow_reader(value)
    return reader_to_table(reader)


def arrow_table_from_lazyframe(frame: pl.LazyFrame) -> pa.Table:
    """Collect a LazyFrame into an Arrow Table.

    Returns
    -------
    pa.Table
        Materialized Arrow table.
    """
    return reader_to_table(lazyframe_to_reader(frame))


def _arrow_table_from_frame(frame: pl.DataFrame | pl.LazyFrame) -> pa.Table:
    if isinstance(frame, pl.DataFrame):
        return frame.to_arrow()
    return arrow_table_from_lazyframe(frame)


def arrow_join_tables(
    left: pa.Table,
    right: pa.Table,
    *,
    spec: ArrowJoinSpec,
) -> pa.Table:
    """Join two Arrow tables using the provided keys.

    Parameters
    ----------
    left
        Left-hand table.
    right
        Right-hand table.
    spec
        Join configuration for Arrow joins.

    Raises
    ------
    ValueError
        If join keys are missing or if validation fails.

    Returns
    -------
    pa.Table
        Joined Arrow table.
    """
    keys, right_keys = _resolve_join_keys(
        on=spec.on,
        left_on=spec.left_on,
        right_on=spec.right_on,
    )
    if not keys:
        msg = "Arrow join requires join keys"
        raise ValueError(msg)
    right_suffix = spec.right_suffix
    if spec.suffix and right_suffix is None:
        right_suffix = spec.suffix
    if right_suffix in {None, ""}:
        resolved_right_keys = keys if right_keys is None else right_keys
        overlapping = set(left.column_names) & set(right.column_names)
        coalesced = set(keys) & set(resolved_right_keys) if spec.coalesce_keys else set()
        if overlapping - coalesced:
            right_suffix = "_right"
    _validate_join(
        left,
        right,
        left_keys=keys,
        right_keys=right_keys,
        validate=spec.validate,
    )
    join_type = _ARROW_JOIN_TYPES.get(spec.how, spec.how)
    return left.join(
        right,
        keys=tuple(keys),
        right_keys=tuple(right_keys) if right_keys is not None else None,
        join_type=join_type,
        left_suffix=spec.left_suffix,
        right_suffix=right_suffix,
        coalesce_keys=spec.coalesce_keys,
    )


def _arrow_schema_for_table(
    table_key: str,
    *,
    extras_policy: ExtrasPolicy | None,
) -> pa.Schema:
    schema_service = get_schema_service()
    if extras_policy is None:
        arrow_schema = schema_service.get_arrow_schema(table_key)
        if arrow_schema is not None:
            return arrow_schema
    table_schema = schema_service.require_table_schema(table_key)
    metadata = None if extras_policy is None else ArrowSchemaMetadata(extras_policy=extras_policy)
    return arrow_contract_for_table_schema(table_schema=table_schema, metadata=metadata)


def _schema_service() -> SchemaService:
    return get_schema_service()


def align_reader_to_contract(
    table_key: str,
    reader: pa.RecordBatchReader,
    *,
    extras_policy: ExtrasPolicy | None = None,
) -> pa.RecordBatchReader:
    """Align an Arrow reader to the contract schema for a table.

    Returns
    -------
    pa.RecordBatchReader
        Reader aligned to the contract schema.
    """
    contract_schema = _arrow_schema_for_table(table_key, extras_policy=extras_policy)
    return _align_reader(reader, contract_schema, extras_policy=extras_policy)


def align_table_to_contract(
    table_key: str,
    table: pa.Table,
    *,
    extras_policy: ExtrasPolicy | None = None,
) -> pa.Table:
    """Align an Arrow table to the contract schema for a table.

    Returns
    -------
    pa.Table
        Arrow table aligned to the contract schema.
    """
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())
    aligned = align_reader_to_contract(table_key, reader, extras_policy=extras_policy)
    return reader_to_table(aligned)


def dedupe_table_for_table(
    table_key: str,
    table: pa.Table,
    *,
    prefer_columns: Sequence[str] | None = None,
) -> pa.Table:
    """Return a table with duplicate primary-key rows removed.

    Returns
    -------
    pa.Table
        Table with duplicate primary-key rows removed.
    """
    schema_service = _schema_service()
    schema = schema_service.get_table_schema(table_key)
    if schema is None or not schema.primary_key:
        return table
    key_columns = list(schema.primary_key)
    if prefer_columns:
        prefer = [name for name in prefer_columns if name in set(table.column_names)]
        if prefer:
            table = _sort_table_for_preference(table, prefer)
    try:
        return table.drop_duplicates(key_columns)
    except (AttributeError, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        seen: set[tuple[object, ...]] = set()
        rows: list[dict[str, object]] = []
        for row in table.to_pylist():
            key = tuple(row.get(col) for col in key_columns)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
        if not rows:
            return pa.Table.from_batches([], schema=table.schema)
        return pa.Table.from_pylist(rows, schema=table.schema)


def _sort_table_for_preference(table: pa.Table, prefer_columns: Sequence[str]) -> pa.Table:
    sort_keys = [(name, "descending") for name in prefer_columns]
    options = pc.SortOptions(sort_keys=sort_keys)
    try:
        options = pc.SortOptions(sort_keys=sort_keys, null_placement="at_end")
        indices = _call_compute("sort_indices", [table], options=options)
    except (TypeError, pa.ArrowNotImplementedError):
        indices = None
    if indices is None:
        indices = _call_compute("sort_indices", [table], options=options)
    if indices is None:
        return table
    return table.take(indices)


def scan_parquet_dataset(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    options: ParquetScanOptions | None = None,
) -> pa.RecordBatchReader | None:
    """Return a RecordBatchReader for a parquet dataset snapshot.

    Returns
    -------
    pa.RecordBatchReader | None
        RecordBatchReader when a dataset snapshot is available, otherwise None.
    """
    resolved = options or ParquetScanOptions()
    try:
        dataset = scan_dataset(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
    except FileNotFoundError:
        LOG.warning("Dataset snapshot missing for %s@%s", table_key, snapshot_id)
        return None
    except (OSError, ValueError, pa.ArrowInvalid) as exc:
        LOG.warning("Dataset scan failed for %s@%s: %s", table_key, snapshot_id, exc)
        return None

    names = set(dataset.schema.names)
    expression: ds.Expression | None = None
    if resolved.repo is not None and "repo" in names:
        expression = ds.field("repo") == resolved.repo
    if resolved.commit is not None and "commit" in names:
        commit_expr = ds.field("commit") == resolved.commit
        expression = commit_expr if expression is None else expression & commit_expr

    scan_options = DatasetScanOptions(
        batch_size=resolved.batch_size,
        filter_expression=expression,
        columns=tuple(resolved.columns) if resolved.columns is not None else None,
        unify_schemas=True,
    )
    scanner = build_scanner(dataset, options=scan_options)
    return scanner.to_reader()


def scan_parquet_table(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    options: ParquetScanOptions | None = None,
) -> pa.Table | None:
    """Return a materialized Arrow Table for a parquet dataset snapshot.

    Returns
    -------
    pa.Table | None
        Materialized Arrow table when available, otherwise None.
    """
    reader = scan_parquet_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        options=options,
    )
    if reader is None:
        return None
    return reader_to_table(reader)


def arrow_join_frames(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    *,
    spec: ArrowJoinSpec | JoinSpec,
) -> pl.DataFrame:
    """Collect, join in Arrow, and return a Polars DataFrame.

    Returns
    -------
    pl.DataFrame
        Joined Polars DataFrame.
    """
    resolved_spec = spec if isinstance(spec, ArrowJoinSpec) else _arrow_spec_from_join_spec(spec)
    left_table = _arrow_table_from_frame(left)
    right_table = _arrow_table_from_frame(right)
    try:
        joined = arrow_join_tables(
            left_table,
            right_table,
            spec=resolved_spec,
        )
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        return _polars_join_fallback(left, right, spec=resolved_spec)
    return table_to_frame(joined)


def arrow_join_lazyframes(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    *,
    spec: JoinSpec | ArrowJoinSpec | None = None,
) -> pl.LazyFrame:
    """Join two LazyFrames via Arrow, returning a LazyFrame.

    Returns
    -------
    pl.LazyFrame
        LazyFrame wrapping the Arrow join result.
    """
    if spec is None:
        resolved = _arrow_spec_from_join_spec(JoinSpec())
    elif isinstance(spec, ArrowJoinSpec):
        resolved = spec
    else:
        resolved = _arrow_spec_from_join_spec(spec)
    joined = arrow_join_frames(
        left,
        right,
        spec=resolved,
    )
    return joined.lazy()


__all__ = [
    "ArrowJoinSpec",
    "align_reader_to_contract",
    "align_table_to_contract",
    "arrow_join_frames",
    "arrow_join_lazyframes",
    "arrow_join_tables",
    "arrow_table_from_lazyframe",
    "arrow_table_from_tabular",
    "dedupe_table_for_table",
    "scan_parquet_dataset",
    "scan_parquet_table",
]
