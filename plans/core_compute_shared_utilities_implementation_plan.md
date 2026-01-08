# Core Compute Shared Utilities Implementation Plan

## Decisions
- Implement core-only compute utilities under `src/codeintel/core/columnar` for non-build
  pipelines (ingestion, storage, serving, validation).
- Keep the core surface small and typed; use Arrow compute kernels and Acero plans instead of
  Python row loops.
- Treat finalize gates as the single boundary for schema alignment, invariants, dedupe, and
  error artifacts (strict vs tolerant).
- Prefer dataset scan pushdown + telemetry by default for any Parquet-backed workload.
- Avoid Acero hash joins with list payload columns; explode lists before joins or drop list
  payloads.

## Phase sequencing
- Phase 0 (completed): Core helper modules (expr vocab, kernels, plan ops, explode ops, finalize, nested).
- Phase 1 (not started): Core scan/telemetry standardization + threading/chunking policy.
- Phase 2 (not started): Adoption in ingestion + validation (extras_json migration, finalize gates).
- Phase 3 (partial): Adoption in storage + serving (scan pushdown, deterministic ordering, exports).
- Phase 4 (not started): Optional escape hatches (Substrait / DataFusion) when Acero is insufficient.

---

## Scope items

### 1) Core expression vocabulary + kernel vocabulary

Status: Partial (expr vocab + kernels implemented; masks alignment pending).

Completed files
- `src/codeintel/core/columnar/expr_vocab.py` (new)
- `src/codeintel/core/columnar/kernels.py` (new)
- `src/codeintel/core/columnar/__init__.py` (reexports)

Remaining files
- `src/codeintel/core/columnar/masks.py` (align to new kernel helpers)

Representative pattern
```python
import pyarrow as pa

from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.kernels import case_when, stable_sort_indices

expr = E.and_(E.is_valid("repo"), E.field("kind").isin(["call", "import"]))

error_code = case_when(
    (
        E.is_null("repo").to_expression().as_array(),
        "NULL_REPO",
    ),
    else_="OK",
)

ordered = table.take(
    stable_sort_indices(table, sort_keys=[("repo", "ascending"), ("commit", "ascending")])
)
```

Distinctive pattern to standardize
- Use `E.*` for plan-time expressions; use `kernels.*` for eager array/table transforms.
- Keep core helpers minimal and typed; avoid introducing Any.

---

### 2) Core plan ops (Acero DSL)

Status: Completed.

Completed files
- `src/codeintel/core/columnar/plan_ops.py` (new)
- `src/codeintel/core/columnar/acero_ops.py` (refactor to reexport Plan/HashJoinSpec)
- `src/codeintel/core/columnar/__init__.py` (exports)

Representative pattern
```python
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.plan_ops import Plan

plan = (
    Plan.scan(
        dataset,
        columns={
            "repo": E.field("repo"),
            "commit": E.field("commit"),
            "kind": E.field("kind"),
        },
        filter_expr=E.field("kind") == E.scalar("call"),
    )
    .project(
        {
            "repo": E.field("repo"),
            "commit": E.field("commit"),
            "kind": E.field("kind"),
        }
    )
    .filter(E.is_valid("repo"))
)

reader = plan.to_reader(use_threads=True)
```

Distinctive pattern to standardize
- Express scan -> project -> filter -> join/aggregate -> order as a Plan chain.
- Use `to_reader()` for streaming and `to_table()` only at explicit boundaries.

---

### 3) HashJoin policy for core plans

Status: Completed.

Completed files
- `src/codeintel/core/columnar/plan_ops.py` (HashJoinSpec, join adapter)
- `src/codeintel/core/columnar/kernels.py` (stable sort indices)

Representative pattern
```python
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.kernels import stable_sort_indices
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan

left = (
    Plan.table(left_table)
    .filter(E.is_valid("key"))
    .project({"key": E.field("key").cast("int64"), "payload_left": E.field("payload_left")})
)

right = (
    Plan.table(right_table)
    .filter(E.is_valid("key"))
    .project({"key": E.field("key").cast("int64"), "payload_right": E.field("payload_right")})
)

joined = left.hash_join(
    right=right,
    spec=HashJoinSpec(
        left_keys=["key"],
        right_keys=["key"],
        how="left outer",
        left_output=["key", "payload_left"],
        right_output=["payload_right"],
    ),
)

result = joined.to_table(use_threads=True)
result = result.take(stable_sort_indices(result, sort_keys=[("key", "ascending")]))
```

Distinctive pattern to standardize
- Pre-project and cast join keys; enforce non-null keys before join.
- Do not include list payload columns in join outputs.
- Apply deterministic ordering after join when outputs are cached or exported.

---

### 4) Core scan ops + telemetry standardization

Status: Partial (scan options preserved in manifest planning; core scan helpers pending).

Completed files
- `src/codeintel/storage/datasets/manifest_index.py` (preserve scan options for planning)

Remaining files
- `src/codeintel/core/datasets/scanning.py` (centralized telemetry helpers)
- `src/codeintel/core/datasets/scanner_ops.py` (ScannerParams extensions)
- `src/codeintel/core/columnar/streaming.py` (DatasetScanOptions defaults)
- `src/codeintel/core/datasets/arrow_store.py` (enforce scan pushdown)

Representative pattern
```python
from codeintel.core.datasets.scanner_ops import ScannerParams, build_scanner

params = ScannerParams(
    columns={
        "repo": E.field("repo"),
        "commit": E.field("commit"),
        "rel_path": E.field("rel_path"),
        "__filename": E.field("__filename"),
    },
    filter_expression=E.field("repo") == E.scalar(repo),
    batch_size=131_072,
    batch_readahead=16,
    fragment_readahead=4,
    cache_metadata=True,
    parquet_pre_buffer=True,
    implicit_ordering=True,
    require_sequenced_output=True,
)

reader = build_scanner(dataset, params=params).to_reader()
```

Distinctive pattern to standardize
- Always push projection + filters into scan options.
- Emit planning telemetry (fragment count, row count) before heavy work.
- Expose implicit ordering and sequenced output options consistently.

---

### 5) Core explode ops (list explode + alignment)

Status: Partial (explode ops implemented; validation alignment checks pending).

Completed files
- `src/codeintel/core/columnar/explode_ops.py` (new)
- `src/codeintel/core/columnar/kernels.py` (list helper kernels)

Remaining files
- `src/codeintel/core/validation/schema_constraints.py` (optional list alignment checks)

Representative pattern
```python
from codeintel.core.columnar.explode_ops import ExplodeSpec, explode_edges

result = explode_edges(
    table,
    spec=ExplodeSpec(
        src_col="src_id",
        dst_list_col="dst_ids",
        repeat_cols=("repo", "commit"),
        aligned_list_cols=("edge_spans",),
        null_list_policy="error",
        null_child_policy="drop",
        enforce_parent_valid=True,
    ),
)

good_edges = result.good
error_rows = result.errors
```

Distinctive pattern to standardize
- Validate aligned list lengths before explode; keep errors at parent-row granularity.
- Enforce explicit null-list and null-child policy.
- Avoid storing list_view types in persisted contracts.

---

### 6) Core finalize gate (strict/tolerant + artifacts)

Status: Partial (finalize ops implemented; serving kernel uses finalize; validation/export pending).

Completed files
- `src/codeintel/core/columnar/finalize_ops.py` (new)
- `src/codeintel/serving/semantic/kernel.py` (finalize boundary in serving)

Remaining files
- `src/codeintel/core/validation/engine.py` (use finalize before validation)
- `src/codeintel/storage/validation/columnar.py` (integrate finalize results)
- `src/codeintel/serving/export/ndjson.py` (finalize boundary for export)

Representative pattern
```python
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table

result = finalize_table(
    table,
    spec=FinalizeSpec(
        table_key="core.ast_nodes",
        mode="tolerant",
        required_non_null=("repo", "commit", "node_id"),
        invariants=(),
        emit_artifacts=True,
    ),
)

good = result.good
errors = result.errors
alignment = result.alignment
stats = result.stats
```

Distinctive pattern to standardize
- Finalize gate is the only boundary for schema alignment, invariants, and dedupe.
- Tolerant mode never raises; strict mode fails fast with error artifacts available.

---

### 7) Core nested ops (extras struct + extras_kv + deep cast)

Status: Partial (nested ops implemented; extras_json removal pending).

Completed files
- `src/codeintel/core/columnar/nested_ops.py` (new)

Remaining files
- `src/codeintel/core/columnar/type_normalization.py` (reuse view-cast helpers)
- `src/codeintel/core/schemas/output_registry.py` (extras_json removal)

Representative pattern
```python
import pyarrow as pa

from codeintel.core.columnar.nested_ops import (
    deep_cast_table_to_contract,
    make_extras_kv_map,
    make_extras_struct,
    unify_schemas_with_contract_first,
)

extras = make_extras_struct(
    table,
    fields={
        "repo": pa.string(),
        "commit": pa.string(),
        "parse_version": pa.int32(),
    },
)
extras_kv = make_extras_kv_map(table, keys="extras_keys", values="extras_values")

with_extras = table.append_column("extras", extras).append_column("extras_kv", extras_kv)
contract = unify_schemas_with_contract_first(contract_schema, [with_extras.schema])
casted = deep_cast_table_to_contract(with_extras, contract)
```

Distinctive pattern to standardize
- Extras are typed struct + optional map; no extras_json in contracts.
- Deep casting is centralized and recursive for list/struct/map types.

---

### 8) Deterministic ordering + ID hashing

Status: Partial (kernels implemented; ordering adoption pending).

Completed files
- `src/codeintel/core/columnar/kernels.py` (stable_sort_indices, hash_struct_ordinal)

Remaining files
- `src/codeintel/serving/semantic/kernel.py` (deterministic export ordering)
- `src/codeintel/storage/datasets/arrow_store.py` (optional stable sort before write)

Representative pattern
```python
from codeintel.core.columnar.kernels import hash_struct_ordinal, stable_sort_indices

ordinal = hash_struct_ordinal(
    table,
    columns=("repo", "commit", "node_id"),
    modulus=2**31 - 1,
)

sorted_table = table.take(
    stable_sort_indices(table, sort_keys=[("repo", "ascending"), ("commit", "ascending")])
)
```

Distinctive pattern to standardize
- Hash-based ordinals live in a single helper.
- Deterministic ordering is explicit and applied near output boundaries.

---

### 9) Threading + chunking policy integration

Status: Not started.

Target files (pending)
- `src/codeintel/core/columnar/streaming.py` (configure_arrow_threading)
- `src/codeintel/core/columnar/compute_helpers.py` (combine_chunks helpers)
- `src/codeintel/core/datasets/arrow_store.py` (pre-write chunk consolidation)

Representative pattern
```python
import pyarrow as pa

from codeintel.core.columnar.streaming import configure_arrow_threading

configure_arrow_threading(cpu_count=32, io_thread_count=32)
table = table.combine_chunks()
```

Distinctive pattern to standardize
- Combine small chunks before heavy compute stages.
- Threading defaults are set once and reused across scans and compute.

---

### 10) Ingestion adoption: extras_json migration + finalize gates

Status: Not started.

Target files (pending)
- `src/codeintel/ingestion/compute/tree_sitter_index.py`
- `src/codeintel/ingestion/compute/cst_extract.py`
- `src/codeintel/ingestion/compute/ast_extract.py`
- `src/codeintel/ingestion/tree_sitter/runner.py`
- `src/codeintel/core/schemas/output_registry.py`

Representative pattern
```python
extras = {
    "node_type": node.node_type,
    "parse_state": node.parse_state,
}

row = {
    "repo": repo,
    "commit": commit,
    "node_id": node.node_id,
    "extras": extras,
    "extras_kv": None,
}
```

Distinctive pattern to standardize
- Emit typed extras structs at creation time; avoid encoding JSON blobs.
- Route ingestion outputs through finalize gates before storage or export.

---

### 11) Serving + storage adoption: scan pushdown + finalize boundary

Status: Partial (scan options + finalize boundary added; streaming + maintenance pending).

Completed files
- `src/codeintel/serving/semantic/duckdb_relation_builder.py`
- `src/codeintel/serving/semantic/kernel.py`
- `src/codeintel/storage/datasets/manifest_index.py`

Remaining files
- `src/codeintel/serving/http/streaming.py`
- `src/codeintel/storage/datasets/maintenance.py`

Representative pattern
```python
reader, telemetry = scan_parquet_dataset_with_telemetry(
    dataset_root=root,
    table_key=table_key,
    snapshot_id=snapshot_id,
    options=ParquetScanOptions(columns=["repo", "commit"], repo=repo, commit=commit),
)

result = finalize_table(
    reader_to_table(reader),
    spec=FinalizeSpec(table_key=table_key, mode="tolerant", emit_artifacts=True),
)
```

Distinctive pattern to standardize
- Always apply pushdown filters/projections at scan time.
- Apply finalize gates before serving/exporting data.

---

## Quality gates
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- `uv run pytest -q` for targeted subsets, then segmented by major directories.
