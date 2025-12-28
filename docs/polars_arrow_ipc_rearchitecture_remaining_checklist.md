# Polars/Arrow IPC Re-architecture Remaining Checklist (tracked)

This checklist captures the remaining implementation scope from
`docs/polars_arrow_ipc_rearchitecture_implementation_plan.md`. Each item includes an
exact file-level diff to apply. Check items as they land.

## 1. Quality gate fixes (blocking)

- [ ] Fix columnar row helper docstrings and typing.
  File: `src/codeintel/core/columnar/rows.py`
  ```diff
  diff --git a/src/codeintel/core/columnar/rows.py b/src/codeintel/core/columnar/rows.py
  --- a/src/codeintel/core/columnar/rows.py
  +++ b/src/codeintel/core/columnar/rows.py
  @@
   def columnar_buffer_for_table_key(table_key: str) -> ColumnarRowBuffer:
       """Create a ColumnarRowBuffer using the table schema registry.
   
  +    Parameters
  +    ----------
  +    table_key
  +        Fully qualified table key (schema.table).
  +
       Returns
       -------
       ColumnarRowBuffer
           Buffer seeded with table columns and types.
  +
  +    Raises
  +    ------
  +    KeyError
  +        If the table key is not registered in the schema service.
       """
       schema = get_schema_service().require_table_schema(table_key)
       columns = tuple(schema.column_names())
  -    column_types = tuple(column.type for column in schema.columns)
  +    column_types: tuple[ColumnType, ...] = tuple(column.type for column in schema.columns)
       return ColumnarRowBuffer(
           table_key=table_key,
           columns=columns,
           column_types=column_types,
           data={name: [] for name in columns},
       )
   
   
   def columnar_row_count(columns: Mapping[str, Sequence[object]]) -> int:
       """Return row count for a columnar mapping, validating lengths.
   
  +    Parameters
  +    ----------
  +    columns
  +        Columnar mapping of column names to sequences of values.
  +
       Returns
       -------
       int
           Number of rows represented by the columnar mapping.
  ```

- [ ] Remove nested if in the SCIP incremental skip path.
  File: `src/codeintel/build/hamilton/native/ingestion/scip.py`
  ```diff
  diff --git a/src/codeintel/build/hamilton/native/ingestion/scip.py b/src/codeintel/build/hamilton/native/ingestion/scip.py
  --- a/src/codeintel/build/hamilton/native/ingestion/scip.py
  +++ b/src/codeintel/build/hamilton/native/ingestion/scip.py
  @@
       output = run_tool_step(context=context, run=_execute)
       scip_output = _coerce_scip_run_output(output, run_id=run_id, mode="unknown")
  -    if scip_output.result.skipped:
  -        if output_scip.is_file():
  -            return ScipRunResult(
  -                result=ExecutionResult.skip("SCIP target skipped"),
  -                outputs={SCIP_ARTIFACT_INDEX: output_scip},
  -                run_id=run_id,
  -                mode="skipped",
  -            )
  -        log.warning("SCIP target marked up-to-date but index.scip is missing; rebuilding")
  -        return _execute()
  +    if scip_output.result.skipped and output_scip.is_file():
  +        return ScipRunResult(
  +            result=ExecutionResult.skip("SCIP target skipped"),
  +            outputs={SCIP_ARTIFACT_INDEX: output_scip},
  +            run_id=run_id,
  +            mode="skipped",
  +        )
  +    if scip_output.result.skipped:
  +        log.warning("SCIP target marked up-to-date but index.scip is missing; rebuilding")
  +        return _execute()
  ```

- [ ] Replace pyarrow.compute null counting with a streaming-safe fallback.
  File: `src/codeintel/storage/validation/columnar.py`
  ```diff
  diff --git a/src/codeintel/storage/validation/columnar.py b/src/codeintel/storage/validation/columnar.py
  --- a/src/codeintel/storage/validation/columnar.py
  +++ b/src/codeintel/storage/validation/columnar.py
  @@
  -import pyarrow.compute as pc
  @@
   def _has_nulls(values: pa.Array | pa.ChunkedArray) -> bool:
       null_count = values.null_count
       if null_count is not None and null_count >= 0:
           return null_count > 0
  -    return bool(pc.count_null(values).as_py() or 0)
  +    return any(item is None for item in values.to_pylist())
  ```

- [ ] Tighten columnar row narrowing for snapshot materialization helpers.
  File: `tests/_helpers/ingestion.py`
  ```diff
  diff --git a/tests/_helpers/ingestion.py b/tests/_helpers/ingestion.py
  --- a/tests/_helpers/ingestion.py
  +++ b/tests/_helpers/ingestion.py
  @@
   def _is_columnar_rows(
       rows: Sequence[tuple[object, ...]] | ColumnarRows,
   ) -> TypeGuard[ColumnarRows]:
  -    return isinstance(rows, Mapping)
  +    return isinstance(rows, dict)
  ```

## 2. Enforce in-process tabular contracts (RecordBatchReader + LazyFrame only)

- [ ] Remove pa.Table from tabular type aliases.
  File: `src/codeintel/build/tabular/types.py`
  ```diff
  diff --git a/src/codeintel/build/tabular/types.py b/src/codeintel/build/tabular/types.py
  --- a/src/codeintel/build/tabular/types.py
  +++ b/src/codeintel/build/tabular/types.py
  @@
   type TabularRelation = DuckDBRelation
   type TabularFrame = pl.LazyFrame
  -type TabularInput = DuckDBRelation | pa.RecordBatchReader | pa.Table | TabularFrame
  -type InferableTabularInput = pa.RecordBatchReader | pa.Table | TabularFrame
  +type TabularInput = DuckDBRelation | pa.RecordBatchReader | TabularFrame
  +type InferableTabularInput = pa.RecordBatchReader | TabularFrame
  ```

- [ ] Drop pa.Table conversion helper and update consumers.
  File: `src/codeintel/build/tabular/conversion.py`
  ```diff
  diff --git a/src/codeintel/build/tabular/conversion.py b/src/codeintel/build/tabular/conversion.py
  --- a/src/codeintel/build/tabular/conversion.py
  +++ b/src/codeintel/build/tabular/conversion.py
  @@
  -def table_to_lazyframe(table: pa.Table) -> pl.LazyFrame:
  -    """Convert an Arrow Table into a Polars LazyFrame.
  -
  -    Returns
  -    -------
  -    pl.LazyFrame
  -        LazyFrame constructed from the Arrow table.
  -    """
  -    frame = pl.from_arrow(table)
  -    if isinstance(frame, pl.Series):
  -        return frame.to_frame().lazy()
  -    return frame.lazy()
  -
  @@
   __all__ = [
       "arrow_reader_to_lazyframe",
       "lazyframe_from_rows",
       "relation_to_arrow_reader",
       "relation_to_polars_lazy",
  -    "table_to_lazyframe",
   ]
  ```

- [ ] Remove pa.Table handling from DuckDB relation conversion.
  File: `src/codeintel/build/tabular/duckdb_relation.py`
  ```diff
  diff --git a/src/codeintel/build/tabular/duckdb_relation.py b/src/codeintel/build/tabular/duckdb_relation.py
  --- a/src/codeintel/build/tabular/duckdb_relation.py
  +++ b/src/codeintel/build/tabular/duckdb_relation.py
  @@
  -from codeintel.build.tabular.conversion import arrow_reader_to_lazyframe, table_to_lazyframe
  +from codeintel.build.tabular.conversion import arrow_reader_to_lazyframe
  @@
       if isinstance(relation, pl.LazyFrame):
           return relation
  -    if isinstance(relation, pa.Table):
  -        return table_to_lazyframe(relation)
       if isinstance(relation, pa.RecordBatchReader):
           reader = cast("pa.RecordBatchReader", relation)
           return arrow_reader_to_lazyframe(reader)
  ```

- [ ] Build empty LazyFrames from RecordBatchReader (no pa.Table).
  File: `src/codeintel/build/hamilton/native/ingestion/frame_utils.py`
  ```diff
  diff --git a/src/codeintel/build/hamilton/native/ingestion/frame_utils.py b/src/codeintel/build/hamilton/native/ingestion/frame_utils.py
  --- a/src/codeintel/build/hamilton/native/ingestion/frame_utils.py
  +++ b/src/codeintel/build/hamilton/native/ingestion/frame_utils.py
  @@
  -from codeintel.build.tabular.conversion import lazyframe_from_rows, table_to_lazyframe
  +from codeintel.build.tabular.conversion import arrow_reader_to_lazyframe, lazyframe_from_rows
  @@
       schema = get_schema_service().require_table_schema(table_key)
       arrow_schema = arrow_schema_from_table_schema(table_schema=schema)
  -    table = pa.Table.from_batches([], schema=arrow_schema)
  -    return table_to_lazyframe(table)
  +    reader = pa.RecordBatchReader.from_batches(arrow_schema, [])
  +    return arrow_reader_to_lazyframe(reader)
  @@
   def lazyframe_for_table_columns(
       table_key: str,
       columns: Mapping[str, Sequence[object]],
   ) -> pl.LazyFrame:
       """Build a LazyFrame for columnar data using the schema's column order.
  
  +    Parameters
  +    ----------
  +    table_key
  +        Fully qualified table key (schema.table).
  +    columns
  +        Columnar mapping of column names to sequences of values.
  +
       Returns
       -------
       pl.LazyFrame
           LazyFrame with columns aligned to the schema order.
  
       Raises
       ------
  +    KeyError
  +        If the table key is not registered in the schema service.
       ValueError
           If input columns contain unexpected names.
       """
  ```

- [ ] Seed harness empty LazyFrames via RecordBatchReader.
  File: `src/codeintel/build/schemas/seed_harness.py`
  ```diff
  diff --git a/src/codeintel/build/schemas/seed_harness.py b/src/codeintel/build/schemas/seed_harness.py
  --- a/src/codeintel/build/schemas/seed_harness.py
  +++ b/src/codeintel/build/schemas/seed_harness.py
  @@
  -from codeintel.build.tabular.conversion import table_to_lazyframe
  +from codeintel.build.tabular.conversion import arrow_reader_to_lazyframe
  @@
       table_schema = self.schema_provider.require_table_schema(table_key)
       arrow_schema = arrow_schema_from_table_schema(table_schema=table_schema)
  -    table = pa.Table.from_batches([], schema=arrow_schema)
  -    frame = table_to_lazyframe(table)
  +    reader = pa.RecordBatchReader.from_batches(arrow_schema, [])
  +    frame = arrow_reader_to_lazyframe(reader)
       self._seeded[table_key] = frame
       return frame
  ```

- [ ] Remove pa.Table from DuckDB relation saver.
  File: `src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py`
  ```diff
  diff --git a/src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py b/src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py
  --- a/src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py
  +++ b/src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py
  @@
   _TABULAR_TYPES: tuple[type, ...] = (
       DuckDBRelation,
  -    pa.Table,
       pa.RecordBatchReader,
       pl.LazyFrame,
   )
  ```

- [ ] Remove pa.Table from Arrow dataset materializer inputs.
  File: `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
  ```diff
  diff --git a/src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py b/src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py
  --- a/src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py
  +++ b/src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py
  @@
       from pathlib import Path
  
  -    from pyarrow import RecordBatchReader, Table
  +    from pyarrow import RecordBatchReader
  
       from codeintel.core.manifests import ArrowDatasetManifest
  
  -    type ArrowDatasetInput = Table | RecordBatchReader
  +    type ArrowDatasetInput = RecordBatchReader
   else:
       type ArrowDatasetInput = object
  @@
   _TABULAR_TYPES: tuple[type, ...] = (
  -    pa.Table,
       pa.RecordBatchReader,
       pl.LazyFrame,
   )
  @@
   def _table_schema_for_data(*, table_key: str, data: TabularData) -> TableSchema:
       if isinstance(data, pl.LazyFrame):
           return table_schema_from_polars_lazyframe(frame=data, table_key=table_key)
  -    if isinstance(data, pa.Table):
  -        arrow_table = cast("Table", data)
  -        return table_schema_from_arrow_schema(arrow_schema=arrow_table.schema, table_key=table_key)
       if isinstance(data, pa.RecordBatchReader):
           arrow_reader = cast("RecordBatchReader", data)
           return table_schema_from_arrow_schema(
               arrow_schema=arrow_reader.schema,
               table_key=table_key,
           )
  @@
   def _coerce_arrow_input(data: TabularData) -> ArrowDatasetInput:
  -    if isinstance(data, pa.Table):
  -        return cast("Table", data)
       if isinstance(data, pa.RecordBatchReader):
           return cast("RecordBatchReader", data)
       msg = f"Unsupported Arrow dataset input type: {type(data).__name__}"
       raise TypeError(msg)
  ```

- [ ] Remove pa.Table from file artifact saver acceptance path.
  File: `src/codeintel/build/hamilton/materializers/artifact_saver.py`
  ```diff
  diff --git a/src/codeintel/build/hamilton/materializers/artifact_saver.py b/src/codeintel/build/hamilton/materializers/artifact_saver.py
  --- a/src/codeintel/build/hamilton/materializers/artifact_saver.py
  +++ b/src/codeintel/build/hamilton/materializers/artifact_saver.py
  @@
  -from typing import TYPE_CHECKING, Literal, cast, get_args, get_origin
  +from typing import TYPE_CHECKING, Literal, get_args, get_origin
  @@
           bytes,
           str,
           Path,
           DuckDBRelation,
  -        pa.Table,
           pa.RecordBatchReader,
       ]
  @@
  -    if type_ in {bytes, str, Path, DuckDBRelation, pa.Table, pa.RecordBatchReader}:
  +    if type_ in {bytes, str, Path, DuckDBRelation, pa.RecordBatchReader}:
           return True
       if type_ is ArtifactWritePlan:
           return True
  @@
                   bytes,
                   str,
                   Path,
                   DuckDBRelation,
  -                pa.Table,
                   pa.RecordBatchReader,
                   type(None),
               }
           ):
               return True
  @@
  -        data
  -            Artifact payload. Supported types are bytes, str (encoded as UTF-8),
  -            Path (reads bytes from the referenced file), DuckDB relations, or
  -            Arrow tables/readers.
  +        data
  +            Artifact payload. Supported types are bytes, str (encoded as UTF-8),
  +            Path (reads bytes from the referenced file), DuckDB relations, or
  +            Arrow record batch readers.
  @@
       if isinstance(data, DuckDBRelation):
           return _write_relation_artifact(output_path, data)
       if isinstance(data, pa.RecordBatchReader):
           return _write_arrow_reader(output_path, data)
  -    if isinstance(data, pa.Table):
  -        table = cast("pa.Table", data)
  -        return _write_arrow_reader(output_path, table.to_reader())
       if isinstance(data, Path) and _same_path(data, output_path):
           return output_path.stat().st_size
  ```

## 3. Dataset stats streaming + type safety

- [ ] Use Parquet metadata for row_count and normalize min/max values safely.
  File: `src/codeintel/storage/datasets/arrow_store.py`
  ```diff
  diff --git a/src/codeintel/storage/datasets/arrow_store.py b/src/codeintel/storage/datasets/arrow_store.py
  --- a/src/codeintel/storage/datasets/arrow_store.py
  +++ b/src/codeintel/storage/datasets/arrow_store.py
  @@
   def dataset_stats(dataset: ds.Dataset) -> ArrowDatasetStats:
       """Return lightweight dataset statistics.
  @@
       ArrowDatasetStats
           Statistics derived from the dataset.
       """
       files = tuple(dataset.files)
       parquet_stats = _parquet_stats(files)
  +    parquet_rows = parquet_stats.rows_from_metadata if parquet_stats else None
  +    row_count = _count_rows(dataset, parquet_rows=parquet_rows)
       sort_keys = parquet_stats.sort_keys if parquet_stats and parquet_stats.sort_keys else None
       column_min_max = (
           parquet_stats.column_min_max if parquet_stats and parquet_stats.column_min_max else None
       )
       return ArrowDatasetStats(
  -        row_count=_count_rows(dataset),
  +        row_count=row_count,
           row_group_count=parquet_stats.row_group_count if parquet_stats else None,
           file_count=parquet_stats.file_count if parquet_stats else None,
           rows_from_metadata=parquet_stats.rows_from_metadata if parquet_stats else None,
           total_bytes=parquet_stats.total_bytes if parquet_stats else None,
           sort_keys=sort_keys,
           column_min_max=column_min_max,
       )
  @@
  -def _count_rows(dataset: ds.Dataset) -> int | None:
  -    counter = getattr(dataset, "count_rows", None)
  -    if callable(counter):
  -        try:
  -            coerced = _coerce_int(counter())
  -            if coerced is not None:
  -                return coerced
  -        except (TypeError, ValueError, pa.ArrowInvalid):
  -            pass
  -    scanner = dataset.scanner()
  -    scanner_counter = getattr(scanner, "count_rows", None)
  -    if callable(scanner_counter):
  -        try:
  -            coerced = _coerce_int(scanner_counter())
  -            if coerced is not None:
  -                return coerced
  -        except (TypeError, ValueError, pa.ArrowInvalid):
  -            pass
  -    table = scanner.to_table()
  -    return table.num_rows
  +def _count_rows(dataset: ds.Dataset, *, parquet_rows: int | None) -> int | None:
  +    if parquet_rows is not None:
  +        return parquet_rows
  +    counter = getattr(dataset, "count_rows", None)
  +    if callable(counter):
  +        try:
  +            coerced = _coerce_int(counter())
  +            if coerced is not None:
  +                return coerced
  +        except (TypeError, ValueError, pa.ArrowInvalid):
  +            pass
  +    scanner = dataset.scanner()
  +    scanner_counter = getattr(scanner, "count_rows", None)
  +    if callable(scanner_counter):
  +        try:
  +            coerced = _coerce_int(scanner_counter())
  +            if coerced is not None:
  +                return coerced
  +        except (TypeError, ValueError, pa.ArrowInvalid):
  +            pass
  +    return None
  @@
   def _normalize_stat_value(value: object) -> object | None:
       if value is None:
           return None
  -    if isinstance(value, pa.Scalar):
  -        return value.as_py()
  -    item = getattr(value, "item", None)
  -    if callable(item):
  -        try:
  -            return item()
  -        except (TypeError, ValueError, OverflowError):
  -            return value
  +    if isinstance(value, _SupportsAsPy):
  +        return value.as_py()
  +    if isinstance(value, _SupportsItem):
  +        try:
  +            return value.item()
  +        except (TypeError, ValueError, OverflowError):
  +            return value
       return value
  @@
   def _safe_min(current: object, candidate: object) -> object:
  -    try:
  -        return candidate if candidate < current else current
  -    except TypeError:
  -        return current
  +    if not isinstance(current, _SupportsRichComparison):
  +        return current
  +    if not isinstance(candidate, _SupportsRichComparison):
  +        return current
  +    try:
  +        return candidate if candidate < current else current
  +    except TypeError:
  +        return current
  @@
   def _safe_max(current: object, candidate: object) -> object:
  -    try:
  -        return candidate if candidate > current else current
  -    except TypeError:
  -        return current
  +    if not isinstance(current, _SupportsRichComparison):
  +        return current
  +    if not isinstance(candidate, _SupportsRichComparison):
  +        return current
  +    try:
  +        return candidate if candidate > current else current
  +    except TypeError:
  +        return current
  @@
   def _json_safe_value(value: object) -> object:
       if value is None or isinstance(value, (bool, int, float, str)):
           result: object = value
       elif isinstance(value, bytes):
           result = value.hex()
       elif isinstance(value, (datetime, date)):
           result = value.isoformat()
       elif isinstance(value, Decimal):
           result = str(value)
  -    else:
  -        as_py = getattr(value, "as_py", None)
  -        if callable(as_py):
  -            result = _json_safe_value(as_py())
  -        else:
  -            item = getattr(value, "item", None)
  -            if callable(item):
  -                try:
  -                    result = _json_safe_value(item())
  -                except (TypeError, ValueError, OverflowError):
  -                    result = str(value)
  -            else:
  -                result = str(value)
  +    elif isinstance(value, _SupportsAsPy):
  +        result = _json_safe_value(value.as_py())
  +    elif isinstance(value, _SupportsItem):
  +        try:
  +            result = _json_safe_value(value.item())
  +        except (TypeError, ValueError, OverflowError):
  +            result = str(value)
  +    else:
  +        result = str(value)
       return result
  ```
