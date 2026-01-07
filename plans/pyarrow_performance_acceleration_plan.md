# PyArrow Performance Acceleration Plan

This plan turns the performance opportunities from the PyArrow review into concrete, code-scoped
work. Each scope item lists the exact files to touch and a representative code pattern to apply.

## Scope Item 1: Scanner tuning with cache_metadata and fragment_scan_options

Goal: maximize scan throughput by enabling Parquet fragment scan options and metadata caching.

Approach: extend dataset scan settings and plumb them into scanner construction so every scan can
enable `cache_metadata` and `ParquetFragmentScanOptions` with performance-first defaults.

Status: In progress (core plumbing complete; remaining call-site override exposure).

Files (completed):
- `src/codeintel/core/constants.py`
- `src/codeintel/core/config/settings.py`
- `src/codeintel/core/runtime/loader.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/core/datasets/scanner_ops.py`
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
- `src/codeintel/build/hamilton/diagnostics.py`

Files (remaining):
- `src/codeintel/build/tabular/arrow_ops.py` (expose cache/fragment overrides in `ParquetScanOptions`)
- `src/codeintel/build/graphs/engine/datasets.py` (surface scan overrides for snapshot readers)

Code pattern:
```python
parquet_opts = ds.ParquetFragmentScanOptions(
    pre_buffer=settings.pre_buffer,
    use_buffered_stream=settings.use_buffered_stream,
    buffer_size=settings.buffer_size,
)
scanner = dataset.scanner(
    columns=columns,
    filter=filter_expression,
    batch_size=settings.batch_size,
    batch_readahead=settings.batch_readahead,
    fragment_readahead=settings.fragment_readahead,
    cache_metadata=settings.cache_metadata,
    use_threads=settings.use_threads,
    fragment_scan_options=parquet_opts,
)
return scanner.to_reader()
```

## Scope Item 2: Performance-first defaults for batch sizes and threading

Goal: bias default scan behavior toward high-end hardware and maximal parallelism.

Approach: raise default batch sizes and readahead settings, and align Arrow CPU/IO thread settings
with available cores while keeping env-based overrides in place.

Status: Completed.

Files (completed):
- `src/codeintel/core/constants.py`
- `src/codeintel/core/config/settings.py`
- `src/codeintel/core/columnar/streaming.py`

Code pattern:
```python
DEFAULT_ARROW_BATCH_SIZE = 131_072
DEFAULT_ARROW_BATCH_READAHEAD = 64
DEFAULT_ARROW_FRAGMENT_READAHEAD = 16
DEFAULT_ARROW_USE_THREADS = True

@dataclass(frozen=True, slots=True)
class ArrowScanSettings:
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE
    batch_readahead: int = DEFAULT_ARROW_BATCH_READAHEAD
    fragment_readahead: int = DEFAULT_ARROW_FRAGMENT_READAHEAD
    use_threads: bool | None = True
    cpu_count: int | None = None
    io_thread_count: int | None = None
```

## Scope Item 3: Streaming-first pipelines (avoid materializing Tables)

Goal: keep data in streaming readers/batches as long as possible to reduce memory pressure and
accelerate throughput.

Approach: replace eager `to_table`/`reader_to_table` usage with batch iterators, only materializing
when required by downstream APIs.

Status: In progress (streaming defaults adopted in key paths; remaining call-site audit).

Files (completed):
- `src/codeintel/build/tabular/conversion.py`
- `src/codeintel/build/hamilton/diagnostics.py`
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`

Files (remaining):
- `src/codeintel/build/tabular/arrow_ops.py` (evaluate `scan_parquet_table`/`reader_to_table` usage)
- `src/codeintel/build/graphs/engine/datasets.py` (convert call sites to reader/batch iteration)

Code pattern:
```python
scanner = build_scanner(dataset, options=scan_options)
for batch in scanner.to_batches():
    batch_table = pa.Table.from_batches([batch])
    # compute per-batch here
    ...
return scanner.to_reader()
```

## Scope Item 4: Pushdown filters/projections and Arrow-first compute

Goal: move filtering and aggregation into Arrow kernels or dataset pushdown to avoid Python row
iteration and maximize vectorization.

Approach: build `pc.Expression` filters for dataset scans and use `Table.group_by().aggregate()`
or Acero for filter-project-aggregate pipelines.

Status: In progress (call-graph and config analytics pushdown in place).

Files (completed):
- `src/codeintel/build/graphs/validation/checks/database.py`
- `src/codeintel/build/analytics/graphs/config_graph_metrics.py`
- `src/codeintel/build/analytics/graphs/config_data_flow.py`
- `src/codeintel/build/analytics/functions/function_effects.py`

Files (remaining):
- `src/codeintel/build/analytics/semantic_roles/core.py`

Code pattern:
```python
expr = equal_expr("repo", repo) & equal_expr("commit", commit)
scanner = dataset.scanner(
    columns=["module", "key", "repo", "commit"],
    filter=expr,
    use_threads=True,
)
for batch in scanner.to_batches():
    grouped = pa.Table.from_batches([batch]).group_by("module").aggregate([("key", "count")])
```

## Scope Item 5: Consolidate row iteration helpers

Goal: reduce duplicate implementations and standardize row-iteration behavior and performance.

Approach: route all row iteration through `codeintel.core.columnar.iter.iter_rows` and remove
duplicate definitions in build helpers.

Status: Completed.

Files (completed):
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/tabular/table_ops.py`
- `src/codeintel/core/columnar/iter.py` (verified canonical helper; no code change required)

Code pattern:
```python
from codeintel.core.columnar.iter import iter_rows

def table_rows(table: pa.Table) -> list[dict[str, object]]:
    return list(iter_rows(table))
```

## Scope Item 6: Parquet cache reads with ParquetFile.iter_batches

Goal: improve cache read throughput by using ParquetFile batching, memory mapping, and dictionary
encoding at read time.

Approach: prefer `pq.ParquetFile(...).iter_batches(...)` for cache reads and unify dictionaries
after load when dictionary columns are present.

Status: Completed.

Files (completed):
- `src/codeintel/build/hamilton/materializers/arrow_parquet_cache.py`

Code pattern:
```python
parquet_file = pq.ParquetFile(path, memory_map=True, pre_buffer=True)
batch_iter = parquet_file.iter_batches(
    batch_size=DEFAULT_ARROW_BATCH_SIZE,
    columns=columns,
    use_threads=True,
    read_dictionary=read_dictionary,
)
table = pa.Table.from_batches(batch_iter)
if read_dictionary:
    table = table.unify_dictionaries()
```

## Caching Alignment Notes (Hamilton DAG)

This scope is compatible with Hamilton caching because it only changes IO/compute performance,
not the data semantics that the cache adapter versions.

Key guardrails to preserve cache stability and avoid surprises:
- Hash stability: `src/codeintel/build/hamilton/arrow_hashing.py` uses `DEFAULT_ARROW_BATCH_SIZE`
  when hashing tables. If scan defaults are raised for performance, consider keeping a fixed hash
  batch size to avoid churn in `data_version` hashes.
- Deterministic ordering: multithreaded joins/aggregations can reorder rows. For cached outputs
  where ordering is expected to be stable, apply a deterministic sort before caching (only where
  needed).
- Helper changes and invalidation: Hamilton `code_version` ignores helper dependencies. If a helper
  change alters output semantics, plan to invalidate caches or set affected nodes to RECOMPUTE for
  the rollout.
- Streaming is safe: streaming readers/batches still materialize final node outputs the cache sees;
  performance changes do not bypass cache boundaries.
