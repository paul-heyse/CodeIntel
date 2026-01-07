# PyArrow Best-in-Class Unification Plan

## Goals
- Standardize Arrow compute + dataset scanning patterns across `src/codeintel/build`.
- Reduce bespoke compute code by consolidating shared helpers.
- Enforce a consistent, expression-first Arrow methodology.

## Status Summary
- 1) Expression-first filtering: **completed** (all remaining `table.filter` paths moved to
  `safe_filter` + expression helpers).
- 2) Scanner-first ingestion: **completed** (streaming defaults + shared scanner wiring).
- 3) Fragment + row-group pruning: **completed** (helper exposed and used by scanner).
- 4) Schema alignment helper: **completed** (shared align helper + concat cleanup).
- 5) Join/sort normalization: **completed** (join normalization now unifies dictionaries + chunks).
- 6) Dictionary encoding strategy: **completed** (cache Parquet reads now apply dictionary
  columns + dictionary unification).
- 7) Compute options helpers: **completed** (centralized cast/sort/take helpers + call-site updates).
- 8) IPC streaming + metadata: **completed** (Arrow IPC export path uses shared IPC options
  with per-batch metadata).
- 9) Arrow C Data Interface: **completed** (stream/array adapter support in interop).
- 10) Memory/threading policy: **completed** (runtime Arrow scan settings + threading overrides
  wired into scan defaults).

## Implementation Details (Completed)

### A) Expression-first filtering (remaining table.filter paths)
**Complete file target list**
- `src/codeintel/build/hamilton/native/graphs/cdg.py`
- `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/goids.py`

**Representative code pattern**
```python
from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import is_valid_expr, non_empty_string_expr

expr = is_valid_expr("function_goid_h128") & non_empty_string_expr("edge_kind")
return safe_filter(table, expr)
```

## Remaining Scope
- None.

### B) Dictionary encoding for Hamilton Parquet cache reads
**Complete file target list**
- `src/codeintel/build/hamilton/materializers/arrow_parquet_cache.py`

**Representative code pattern**
```python
dictionary_columns = _resolve_dictionary_columns(self.path)
table = pq.read_table(
    self.path,
    read_dictionary=dictionary_columns if dictionary_columns else False,
)
if dictionary_columns:
    table = table.unify_dictionaries()
```

### C) IPC streaming + per-batch metadata in export writers
**Complete file target list**
- `src/codeintel/build/exports/writers.py`
- `src/codeintel/build/exports/engine.py`

**Representative code pattern**
```python
from codeintel.core.exports.arrow_ipc import default_ipc_write_options, iter_ipc_stream

options = default_ipc_write_options()
for chunk in iter_ipc_stream(
    reader,
    metadata=metadata,
    batch_metadata=batch_metadata,
    options=options,
):
    sink.write(chunk)
```

### D) Arrow threading + scan defaults from runtime settings
**Complete file target list**
- `src/codeintel/core/config/settings.py`
- `src/codeintel/core/runtime/loader.py`
- `src/codeintel/build/settings.py`
- `src/codeintel/core/columnar/streaming.py`

**Representative code pattern**
```python
settings = get_build_settings()
arrow = settings.arrow_scan
configure_arrow_threading(
    cpu_count=arrow.cpu_count,
    io_thread_count=arrow.io_thread_count,
)
scan_options = DatasetScanOptions(
    batch_size=arrow.batch_size,
    batch_readahead=arrow.batch_readahead,
    fragment_readahead=arrow.fragment_readahead,
    use_threads=arrow.use_threads,
)
```

## Scope Items and Code Patterns

### 1) Expression-first filtering and projection
**Intent**: Unify filter/projection logic around `pc.Expression` so we avoid mixing `pa.scalar` and `pc.scalar`.

**Target files**
- `src/codeintel/build/tabular/compute_masks.py`
- `src/codeintel/build/tabular/compute_helpers.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/exports/exprs.py`
- `src/codeintel/build/analytics/cfg_dfg/helpers.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/graphs/compute/callgraph/collection.py`
- `src/codeintel/build/graphs/compute/callgraph/resolution.py`
- `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
- `src/codeintel/build/hamilton/native/graphs/cdg.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/goids.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/syntax.py`
- `src/codeintel/build/hamilton/native/views/view_outputs.py`
- `src/codeintel/build/assets/emitter.py`

**Pattern**
```python
import pyarrow.compute as pc

expr = (pc.field("repo") == pc.scalar(repo)) & pc.is_valid(pc.field("commit"))
filtered = safe_filter(table, expr)
```

### 2) Scanner-first ingestion (avoid `to_table()` in hot paths)
**Intent**: Stream batches via `Scanner.to_batches()` or `to_reader()` to reduce memory spikes.

**Target files**
- `src/codeintel/core/datasets/scanner_ops.py`
- `src/codeintel/core/datasets/arrow_store.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
- `src/codeintel/build/hamilton/diagnostics.py`
- `src/codeintel/build/exports/common.py`
- `src/codeintel/build/exports/parquet.py`
- `src/codeintel/build/exports/engine.py`

**Pattern**
```python
import pyarrow.dataset as ds

scanner = dataset.scanner(
    columns=["repo", "commit", "rel_path"],
    filter=ds.field("repo") == repo,
    batch_size=DEFAULT_ARROW_BATCH_SIZE,
)
for batch in scanner.to_batches():
    consume(batch)
```

### 3) Fragment + row-group pruning helper
**Intent**: Centralize row-group selection using Parquet statistics and fragment APIs.

**Target files**
- `src/codeintel/core/datasets/scanner_ops.py`
- `src/codeintel/core/datasets/arrow_store.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
- `src/codeintel/build/schemas/observations.py`
- `src/codeintel/build/exports/common.py`
- `src/codeintel/build/exports/parquet.py`

**Pattern**
```python
import pyarrow.dataset as ds

pred = ds.field("commit") == commit
for frag in dataset.get_fragments(filter=pred):
    if isinstance(frag, ds.ParquetFileFragment):
        for rg_frag in frag.split_by_row_group(filter=pred):
            for batch in rg_frag.scanner(filter=pred).to_batches():
                consume(batch)
```

### 4) Schema alignment and concatenation helper
**Intent**: Use `pa.unify_schemas` + `cast` + `concat_tables_unified` consistently.

**Target files**
- `src/codeintel/core/columnar/schema_ops.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/graphs/pdg.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/assemble.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/overlays_inspect.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/hamilton/diagnostics.py`

**Pattern**
```python
import pyarrow as pa

schema = pa.unify_schemas([left.schema, right.schema], promote_options="permissive")
left_aligned = left.cast(schema)
right_aligned = right.cast(schema)
combined = pa.concat_tables([left_aligned, right_aligned])
```

### 5) Join/sort normalization helper
**Intent**: Normalize join inputs with dictionary unification + chunk compaction.

**Target files**
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/analytics/subsystems/cache.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/syntax.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/goids.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py`

**Pattern**
```python
normalized = table.unify_dictionaries().combine_chunks()
joined = normalized.join(other, keys=["repo", "commit"], join_type="left outer")
```

### 6) Dictionary encoding strategy (string-heavy columns)
**Intent**: Use dictionary encoding at read time and unify dictionaries before IPC writes.

**Target files**
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/exports/common.py`
- `src/codeintel/build/exports/parquet.py`
- `src/codeintel/build/exports/writers.py`
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
- `src/codeintel/build/schemas/observations.py`
- `src/codeintel/core/datasets/arrow_store.py`

**Pattern**
```python
import pyarrow.parquet as pq

table = pq.read_table(path, read_dictionary=["repo", "commit", "rel_path"])
table = table.unify_dictionaries()
```

### 7) Compute options helpers (`CastOptions`, `SortOptions`, `TakeOptions`)
**Intent**: Centralize option construction to reduce bespoke option wiring.

**Target files**
- `src/codeintel/build/tabular/compute_helpers.py`
- `src/codeintel/build/tabular/array_ops.py`
- `src/codeintel/build/tabular/dedupe_ops.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/anchors.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py`

**Pattern**
```python
import pyarrow.compute as pc

opts = pc.CastOptions(target_type=pa.int64(), allow_int_overflow=False)
casted = pc.cast(values, options=opts)
```

### 8) IPC streaming + metadata helpers
**Intent**: Use IPC stream/file writers with consistent options and metadata.

**Target files**
- `src/codeintel/build/exports/writers.py`
- `src/codeintel/build/exports/common.py`
- `src/codeintel/build/exports/engine.py`
- `src/codeintel/build/exports/parquet.py`
- `src/codeintel/build/exports/jsonl.py`
- `src/codeintel/build/hamilton/materializers/artifact_saver.py`
- `src/codeintel/build/hamilton/arrow_hashing.py`
- `src/codeintel/build/tabular/arrow_ops.py`

**Pattern**
```python
import pyarrow as pa
import pyarrow.ipc as ipc

opts = ipc.IpcWriteOptions(compression="zstd", unify_dictionaries=True)
with ipc.new_stream(sink, table.schema, options=opts) as writer:
    for batch in table.to_batches(max_chunksize=DEFAULT_ARROW_BATCH_SIZE):
        writer.write_batch(batch, custom_metadata={"table_key": table_key})
```

### 9) Arrow C Data Interface for interop (avoid `__dataframe__` limits)
**Intent**: Accept `__arrow_c_stream__`/`__arrow_c_array__` producers directly.

**Target files**
- `src/codeintel/build/tabular/conversion.py`
- `src/codeintel/build/graphs/assembly/readers.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/tabular/dedupe_ops.py`
- `src/codeintel/build/tabular/scoping.py`
- `src/codeintel/build/analytics/functions/metrics.py`
- `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`
- `src/codeintel/build/hamilton/transforms/tabular_steps.py`
- `src/codeintel/build/hamilton/native/ingestion/pipelines.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/file_line_index.py`
- `src/codeintel/build/hamilton/native/analytics/data_models.py`
- `src/codeintel/build/hamilton/native/analytics/semantic_roles.py`
- `src/codeintel/build/hamilton/native/analytics/function_ast_features.py`
- `src/codeintel/build/hamilton/native/analytics/function_contracts.py`
- `src/codeintel/build/hamilton/native/analytics/function_effects.py`
- `src/codeintel/build/hamilton/native/analytics/entrypoints.py`
- `src/codeintel/build/hamilton/native/analytics/subsystems.py`
- `src/codeintel/build/hamilton/native/analytics/subsystem_cache.py`
- `src/codeintel/build/hamilton/native/analytics/subsystem_agreement.py`
- `src/codeintel/build/hamilton/native/analytics/tables_dependencies.py`
- `src/codeintel/build/hamilton/native/analytics/cfg_dfg_metrics.py`
- `src/codeintel/build/hamilton/native/analytics/py_cpg_quality_report.py`
- `src/codeintel/build/hamilton/native/graphs/cdg.py`
- `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
- `src/codeintel/build/hamilton/native/graphs/call_graph.py`
- `src/codeintel/build/hamilton/native/graphs/goids.py`
- `src/codeintel/build/hamilton/native/views/view_outputs.py`

**Pattern**
```python
import pyarrow as pa

reader = pa.RecordBatchReader.from_stream(obj_with_arrow_c_stream)
table = reader.read_all()
```

### 10) Memory/threading policy centralization
**Intent**: Single place to configure Arrow thread counts and scanner readahead defaults.

**Target files**
- `src/codeintel/core/constants.py`
- `src/codeintel/build/settings.py`
- `src/codeintel/core/datasets/scanner_ops.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
- `src/codeintel/build/hamilton/diagnostics.py`
- `src/codeintel/build/exports/common.py`
- `src/codeintel/build/exports/engine.py`

**Pattern**
```python
import pyarrow as pa

pa.set_cpu_count(DEFAULT_CPU_COUNT)
pa.set_io_thread_count(DEFAULT_IO_THREADS)
```

## Implementation Steps
1. Add expression helpers in `src/codeintel/build/tabular/compute_masks.py` (or a new
   `compute_exprs.py`) to normalize `pc.field`/`pc.scalar` usage.
2. Extend scanner helpers in `src/codeintel/core/datasets/scanner_ops.py` to provide
   streaming-first defaults and expose readahead/batch size tuning.
3. Introduce a Parquet fragment pruning helper in dataset utilities to standardize
   row-group filtering based on stats.
4. Add a schema alignment helper in `src/codeintel/core/columnar/schema_ops.py`.
5. Add a join/sort normalization helper in `src/codeintel/build/tabular/arrow_ops.py`.
6. Add dictionary encoding guidance helpers near dataset readers/writers and IPC writers.
7. Add compute options helpers in `src/codeintel/build/tabular/compute_helpers.py`.
8. Add IPC stream/file wrappers in `src/codeintel/build/exports` or tabular helpers to
   standardize compression + dictionary unification + per-batch metadata.
9. Add an interop adapter in `src/codeintel/build/tabular/conversion.py` for Arrow C
   stream/array producers.
10. Centralize thread/memory settings in `src/codeintel/core/constants.py` or
    `src/codeintel/build/settings.py` and thread them through scanner helpers.

## Validation
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted `pytest` for ingestion + dataset scanning paths touched by helper changes.
