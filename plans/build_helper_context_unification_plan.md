# Build Helper and Context Unification Plan

This plan consolidates helper, utility, and context patterns across `src/codeintel/build` into
shared, reusable modules. Each scope item below includes goals, target files, and representative
code patterns.

## Scope Item 1: Snapshot-scoped scan context

Goal: centralize dataset scan option resolution, snapshot filtering, and projection so every scan
uses a single, consistent helper with override hooks.

Targets:
- `src/codeintel/build/scopes/snapshot.py`
- `src/codeintel/build/tabular/scoping.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/graphs/validation/checks/database.py`
- `src/codeintel/build/analytics/functions/function_effects.py`
- `src/codeintel/build/analytics/semantic_roles/core.py`

Pattern:
```python
@dataclass(frozen=True, slots=True)
class SnapshotScanContext:
    repo: str
    commit: str
    settings: ArrowScanSettings

    def filter_expr(self, schema: pa.Schema) -> ds.Expression | None:
        if "repo" not in schema.names or "commit" not in schema.names:
            return None
        repo_mask = equal_expr("repo", self.repo)
        commit_mask = equal_expr("commit", self.commit)
        return repo_mask & commit_mask

    def scan_options(
        self,
        *,
        columns: Sequence[str] | None,
        batch_size: int | None = None,
    ) -> DatasetScanOptions:
        return DatasetScanOptions(
            batch_size=batch_size or self.settings.batch_size,
            columns=columns,
            filter_expression=None,
            cache_metadata=self.settings.cache_metadata,
            use_threads=self.settings.use_threads,
            batch_readahead=self.settings.batch_readahead,
            fragment_readahead=self.settings.fragment_readahead,
            parquet_pre_buffer=self.settings.parquet_pre_buffer,
            parquet_use_buffered_stream=self.settings.parquet_use_buffered_stream,
            parquet_buffer_size=self.settings.parquet_buffer_size,
            unify_schemas=True,
        )
```

## Scope Item 2: Reader and batch normalization helper

Goal: normalize string/binary view types once per batch for all reader-driven pipelines.

Targets:
- `src/codeintel/core/columnar/type_normalization.py`
- `src/codeintel/core/columnar/iter.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/hamilton/native/views/view_outputs.py`

Pattern:
```python
def normalize_record_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    arrays: list[pa.Array] = []
    fields: list[pa.Field] = []
    for idx, field in enumerate(batch.schema):
        array = batch.column(idx)
        array = normalize_string_view_array(array)
        array = normalize_binary_view_array(array)
        arrays.append(array)
        fields.append(pa.field(field.name, array.type, nullable=field.nullable))
    return pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))


def normalize_reader(reader: pa.RecordBatchReader) -> Iterator[pa.RecordBatch]:
    for batch in reader:
        if batch.num_rows:
            yield normalize_record_batch(batch)
```

## Scope Item 3: Tabular conversion with projection and scope

Goal: unify `tabular_to_arrow_table(...).select(...)` and scoped filtering into a single helper.

Targets:
- `src/codeintel/build/tabular/conversion.py`
- `src/codeintel/build/tabular/scoping.py`
- `src/codeintel/build/hamilton/native/analytics/*`
- `src/codeintel/build/hamilton/native/ingestion/*`
- `src/codeintel/build/hamilton/native/graphs/*`

Pattern:
```python
def tabular_to_scoped_table(
    value: InferableTabularInput,
    *,
    columns: Sequence[str] | None,
    scope: SnapshotScope | None,
    require_scope_columns: bool,
) -> pa.Table:
    table = tabular_to_arrow_table(value)
    if columns is not None:
        table = table.select(list(columns))
    if scope is None:
        return table
    return scope.filter_arrow_table(table, require_columns=require_scope_columns)
```

## Scope Item 4: Unified tuple iteration helper

Goal: reduce duplicate row-iteration logic by delegating tuple iteration to a single helper.

Targets:
- `src/codeintel/core/columnar/iter.py`
- `src/codeintel/core/query_results.py`
- `src/codeintel/build/graphs/engine/views.py`

Pattern:
```python
def iter_tuples(
    reader: pa.RecordBatchReader,
    *,
    columns: Sequence[str] | None = None,
) -> Iterator[tuple[object, ...]]:
    for batch in reader:
        if batch.num_rows == 0:
            continue
        column_names = list(batch.schema.names) if columns is None else list(columns)
        data = batch.to_pydict()
        values = [data[name] for name in column_names]
        yield from zip(*values, strict=True)
```

## Scope Item 5: Graph input normalization helper

Goal: provide a single conversion point for `GraphInput` to NetworkX or `RxGraphStore`.

Targets:
- `src/codeintel/build/graphs/rx/convert.py`
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/analytics/graphs/graph_metrics_ext.py`
- `src/codeintel/build/graphs/validation/checks/structure.py`
- `src/codeintel/build/graphs/engine/views.py`

Pattern:
```python
def graph_to_networkx(graph: GraphInput) -> nx.Graph:
    if isinstance(graph, RxGraphStore):
        return rx_to_networkx(graph.graph)
    if isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        return rx_to_networkx(graph)
    return cast("nx.Graph", graph)


def graph_to_store(graph: GraphInput) -> RxGraphStore:
    return ensure_store(graph)
```

## Scope Item 6: Graph view factory

Goal: standardize dataset scan -> normalized batches -> graph store -> NetworkX conversion.

Targets:
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/graphs/assembly/*`

Pattern:
```python
@dataclass(frozen=True, slots=True)
class GraphViewFactory:
    scan_ctx: SnapshotScanContext

    def load_edges(
        self,
        *,
        dataset_root: Path,
        table_key: str,
        columns: Sequence[str],
    ) -> pa.RecordBatchReader | None:
        request = SnapshotScanRequest(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=self.scan_ctx.commit,
            columns=columns,
            repo=self.scan_ctx.repo,
            commit=self.scan_ctx.commit,
        )
        return scan_snapshot_reader(request)
```

## Scope Item 7: Unified graph metrics orchestration

Goal: route all graph metrics computation through a single config-driven pipeline.

Targets:
- `src/codeintel/build/analytics/graphs/orchestrator.py`
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/analytics/graphs/graph_stats.py`
- `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`
- `src/codeintel/build/analytics/graphs/subsystem_graph_metrics.py`

Pattern:
```python
@dataclass(frozen=True)
class MetricsPipelineConfig[TSlices, TRow: Mapping[str, object]]:
    table_key: str
    filter_graph: Callable[[GraphMetricFilters, GraphInput], GraphInput]
    build_context: Callable[[GraphRuntimeOptions, str, str], GraphContext]
    build_views: Callable[[GraphInput], GraphViews]
    build_slices: Callable[[GraphViews, GraphContext], TSlices]
    build_rows: Callable[[str, str, GraphContext, GraphViews, TSlices], list[TRow]]
```

## Scope Item 8: Hamilton table target spec factory

Goal: consolidate repeated `MultiTableTargetContext.build_*_spec` patterns into a factory.

Targets:
- `src/codeintel/build/hamilton/native/patterns/table_target.py`
- `src/codeintel/build/hamilton/native/analytics/*`
- `src/codeintel/build/hamilton/native/ingestion/*`
- `src/codeintel/build/hamilton/native/graphs/*`

Pattern:
```python
def build_table_target_specs(
    *,
    context: TableTargetContext,
    table_keys: Sequence[str],
    relation: bool,
) -> list[TableTargetSpec]:
    specs: list[TableTargetSpec] = []
    for table_key in table_keys:
        if relation:
            spec = TableTargetContext.build_relation_table_spec(context=context, table_key=table_key)
        else:
            spec = TableTargetContext.build_dataset_table_spec(context=context, table_key=table_key)
        specs.append(spec)
    return specs
```

## Scope Item 9: Contracted table context helper

Goal: unify contract lookup, schema alignment, and metadata propagation.

Targets:
- `src/codeintel/build/contracts/registry.py`
- `src/codeintel/build/contracts/types.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/hamilton/transforms/table_contract.py`

Pattern:
```python
@dataclass(frozen=True, slots=True)
class ContractedTableContext:
    contract: TableContract
    policy: ContractPolicy

    def align(self, reader: pa.RecordBatchReader) -> pa.RecordBatchReader:
        aligned = align_reader_to_contract(
            reader,
            self.contract.schema,
            extras_policy=self.policy.extras_policy,
        )
        return aligned
```

## Scope Item 10: Row decoding utility

Goal: centralize per-row decoding, null handling, and json/payload parsing.

Targets:
- `src/codeintel/build/analytics/utilities/ast.py`
- `src/codeintel/build/analytics/semantic_roles/core.py`
- `src/codeintel/build/analytics/functions/*`
- `src/codeintel/build/analytics/graphs/*`
- `src/codeintel/build/analytics/entrypoints/*`

Pattern:
```python
@dataclass(frozen=True, slots=True)
class RowDecoder:
    columns: Sequence[str]

    def decode(self, row: Mapping[str, object]) -> dict[str, object]:
        return {name: decode_payload(row.get(name)) for name in self.columns}
```

## Scope Item 11: Semantic role classification consolidation

Goal: ensure semantic role scoring logic lives in one canonical classification module.

Targets:
- `src/codeintel/build/analytics/semantic_roles/core.py`
- `src/codeintel/build/analytics/compute/semantic_roles/classification.py`

Pattern:
```python
def classify_function_role(context: FunctionContext) -> tuple[str, float, str | None, dict[str, object]]:
    return classification.classify_function_role(context)
```

## Scope Item 12: Compute mask and expression context

Goal: standardize `safe_filter`, boolean mask logic, and expression building across build code.

Targets:
- `src/codeintel/build/tabular/compute_masks.py`
- `src/codeintel/build/tabular/compute_helpers.py`
- `src/codeintel/build/tabular/scoping.py`
- `src/codeintel/build/analytics/*`
- `src/codeintel/build/graphs/validation/*`

Pattern:
```python
@dataclass(frozen=True, slots=True)
class FilterExprContext:
    repo: str | None = None
    commit: str | None = None

    def apply(self, table: pa.Table) -> pa.Table:
        mask = None
        if self.repo is not None and "repo" in table.column_names:
            mask = equal_mask(table["repo"], pa.scalar(self.repo))
        if self.commit is not None and "commit" in table.column_names:
            commit_mask = equal_mask(table["commit"], pa.scalar(self.commit))
            mask = commit_mask if mask is None else and_kleene(mask, commit_mask)
        return table if mask is None else safe_filter(table, mask)
```

## Scope Item 13: Dataset metadata context

Goal: unify metadata reading, row-group pruning, and caching behavior for dataset scans.

Targets:
- `src/codeintel/build/analytics/utilities/datasets.py`
- `src/codeintel/core/datasets/parquet_metadata.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/hamilton/materializers/arrow_parquet_cache.py`

Pattern:
```python
@dataclass(frozen=True, slots=True)
class DatasetMetadataContext:
    dataset_root: Path
    table_key: str

    def read_metadata(self) -> pq.FileMetaData | None:
        metadata_path = self.dataset_root / "_metadata"
        if not metadata_path.is_file():
            return None
        parquet_file = pq.ParquetFile(metadata_path)
        return parquet_file.metadata
```
