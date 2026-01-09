# Compute Build Shared Utilities Implementation Plan

## Decisions
- Deprecate extras_json immediately and replace it with typed extras structs at creation time.
- Add optional extras_kv map for long-tail metadata; flatten only at export boundaries.
- Use Acero hashjoin as the join primitive; do not conflate joins with hash kernels.
- Enforce join key non-null policy and route null keys to finalize errors before hashjoin.
- Explode list payloads before hashjoin; do not join list payload columns directly.
- Enforce deterministic ordering after hashjoin via stable sort indices or order_by.
- Use scan pushdown with projection expressions and planning telemetry by default.
- Finalize gate emits alignment + error stats artifacts and enforces nested invariants.
- Redefine ordinals and goids using Arrow hash kernels when available, with no compatibility.
- First pilot pipeline is graph.cpg_edges_calls in call_wiring.py.

## HashJoin Policy Summary
- Always pre-project and cast join keys before hashjoin; never rely on implicit casting.
- Enforce non-null join keys and route null-key rows to finalize errors before hashjoin.
- Use explicit left_output/right_output to avoid accidental payload bloat.
- Do not include list payload columns in hashjoin inputs; explode or drop them first.
- Apply deterministic ordering after hashjoin (stable sort indices preferred).

## Implementation Status (Action Sets 1-11 Complete; Backlog Open)
- Completed utilities: `src/codeintel/build/tabular/plan_ops.py` (Plan + HashJoinSpec),
  `src/codeintel/build/tabular/expr_vocab.py`, `src/codeintel/build/tabular/kernels.py`,
  `src/codeintel/build/tabular/explode_ops.py` (ExplodeSpec + error artifacts),
  `src/codeintel/build/tabular/finalize_ops.py` (FinalizeSpec),
  `src/codeintel/build/tabular/nested_ops.py`, `src/codeintel/build/tabular/compute_helpers.py`
  (exports require_array/require_scalar).
- Scan/streaming upgrades: `src/codeintel/core/columnar/streaming.py` (projection expressions
  + ordering flags), `src/codeintel/core/datasets/scanner_ops.py` (mapping columns + ordering),
  `src/codeintel/core/datasets/scanning.py` (ParquetScanTelemetry +
  scan_parquet_dataset_with_telemetry + normalize_table_for_compute),
  `src/codeintel/build/graphs/engine/datasets.py` (GraphViewScanOptions + telemetry),
  `src/codeintel/storage/datasets/scanning.py` (re-exports).
- Schema + row buffers: `src/codeintel/core/schemas/output_registry.py`
  (graph extras/extras_kv + graph.cpg_call_candidates), `config/schema_breaks.yaml` approvals,
  `src/codeintel/core/columnar/rows.py` (fill missing nullable columns).
- Pilot migration: call wiring now emits graph.cpg_call_candidates and builds graph.cpg_edges_calls
  via explode + hash_join + finalize; call_wiring targets and registry inventory updated; cpg2
  call_wiring plane reads extras_kv.
- Graph outputs finalize adoption: cfg_dfg/cdg/call_graph/import_graph/symbol_use/pdg now use
  finalize_table and no longer emit extras_json; cpg2 assembly finalized with strict gates; and
  analytics/validation outputs now finalize (native analytics modules + validation findings).
- CPG2 plane migration: link/symbol/flow planes moved to Plan + HashJoinSpec with deterministic
  ordering; ordinals now use hash_struct_ordinal; cpg_edge_ordinal uses Arrow hash kernels; and
  GOID hashing uses hash_struct_goid.
- Edge builder conversions: CPG2 call_wiring + overlay/SCIP/bytecode/inspect/symtable edges now
  finalize via finalize_cpg_edge_rows (explode + finalize).
- Threading/chunking integration: configure_arrow_threading now runs in
  normalize_table_for_compute (and normalize_table_for_join); materialize_plan normalizes plan
  outputs; scan_parquet_table normalizes tables for compute kernels.
- Action Set 5 completed: extras/extras_kv migrations for remaining CPG2/legacy planes
  (ast/bytecode/inspect/overlays/py_sym/treesitter/scip/edges), hashjoin adoption in ingestion
  (syntax_enrich/syntax_augment/extraction_targets/scip_resolution) + analytics/subsystems/cache,
  Plan/HashJoin reexports in acero_ops/arrow_ops, and streaming boundaries in join pipelines
  (materialize_plan + normalize_table_for_compute).
- Action Set 6 completed: scan pushdown + telemetry defaults wired through build/storage/serving
  readers (ParquetScanOptions metrics_enabled + ordering defaults, GraphViewScanOptions defaults,
  export/quality/causal scan usage, storage query scans, and serving DuckDB relation scans).
- Action Set 7 completed: deterministic IDs beyond graph/CPG2 now use Arrow hash kernels for
  compute_goid and SCIP occurrence IDs.
- Action Set 8 completed: finalize-on-read gates added for graph snapshot tables, post-run
  analytics readers, and Parquet scan helpers; serving DuckDB relation scans finalize aligned
  inputs when enabled.
- Action Set 9 completed: call wiring IDs now use Arrow hash kernels (stable_decimal_id), and
  core short hash helpers route non-security hashes through Arrow kernels; join-heavy pipeline
  sweep outside CPG2/ingestion found no additional conversions required.
- Action Set 10 completed: guardrail cleanup + validation/serving finalize consistency
  (streaming-safe iteration + finalize boundaries), plus removal of remaining to_pylist usage
  in export logging.
- Action Set 11 completed: plan materialization normalized across join-heavy pipelines and
  edge builder audit found no remaining non-CPG2 list payloads requiring explode conversion.
- Pending migrations: schema snapshot refresh (deferred) + guardrails blocked by missing
  analytics.config_references schema, optional escape hatches, and any new pipelines that
  bypass shared scan/plan helpers.

## Phase sequencing
- Phase 0: Schema and contract updates (extras_json removal, extras/extras_kv additions).
- Phase 1: Shared utilities (plan ops, hashjoin policy, scan ops, explode ops, finalize gate,
  nested ops, expr/kernels, deterministic IDs, threading/chunking, streaming boundaries).
- Phase 2: Pilot migration (graph.cpg_edges_calls in call_wiring.py).
- Phase 3: Systematic conversion of remaining graph and analytics pipelines.
- Phase 4: Optional escape hatches (Substrait/DataFusion) when Acero is insufficient.

## Action Set 5 (Completed)

Scope
- Finished extras/extras_kv migrations for remaining CPG2 and legacy CPG planes, removed
  extras_json usage entirely for graph.cpg_nodes/graph.cpg_edges producers, and updated
  row builders to emit typed extras structs.

Status
- Completed.

Targets
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/ast.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/bytecode.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/inspect.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/overlays_symtable.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/overlays_bytecode.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/overlays_inspect.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/py_sym.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/treesitter.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py
- src/codeintel/build/hamilton/native/graphs/cpg/edges.py

Deliverables
- All remaining CPG producers emit extras/extras_kv (typed struct + optional map), no
  extras_json bytes anywhere in graph outputs.
- Any payload encoding helpers are removed or replaced with nested_ops helpers.
- Finalize gates remain the sole output boundary for graph tables.

## Action Set 6 (Completed)

Scope
- Adopt scan pushdown defaults (projection expressions + ordering flags) and telemetry logging
  across build/storage/serving dataset readers.

Status
- Completed.

Targets
- src/codeintel/core/datasets/scanning.py
- src/codeintel/build/exports/common.py
- src/codeintel/build/hamilton/post_run_quality_outputs.py
- src/codeintel/build/causal_analysis/scan_utils.py
- src/codeintel/build/graphs/engine/datasets.py
- src/codeintel/storage/queries/parquet.py
- src/codeintel/storage/serving/snapshot_service.py
- src/codeintel/serving/semantic/duckdb_relation_builder.py

Deliverables
- metrics_enabled/implicit_ordering/require_sequenced_output defaults applied in readers.
- Telemetry emitted for Parquet scans when enabled (debug logging in scan helper).
- Scan paths now use consistent ordering flags for deterministic processing.

## Action Set 7 (Completed)

Scope
- Extend deterministic ID policy beyond graph/CPG2 to remaining build identifiers.

Status
- Completed.

Targets
- src/codeintel/build/graphs/compute/goid.py
- src/codeintel/build/hamilton/native/ingestion/scip_resolution.py

Deliverables
- compute_goid now uses Arrow hash kernels with DECIMAL(38,0) normalization.
- SCIP occurrence IDs use Arrow hash kernels and normalized string IDs.

## Action Set 8 (Completed)

Scope
- Enforce finalize-on-read gates for dataset readers (graph snapshot tables, post-run analytics,
  and serving/storage scans) to ensure contracts are respected at consumption boundaries.

Status
- Completed.

Targets
- src/codeintel/core/datasets/scanning.py
- src/codeintel/build/graphs/engine/datasets.py
- src/codeintel/build/hamilton/post_run_quality_outputs.py
- src/codeintel/storage/queries/parquet.py
- src/codeintel/serving/semantic/duckdb_relation_builder.py

Deliverables
- ParquetScanOptions supports finalize_mode for full-table reads.
- Graph snapshot table scans and post-run analytics reads finalize with tolerant gates.
- Serving DuckDB relation scans finalize aligned inputs when enabled.

## Action Set 9 (Completed)

Scope
- Migrate remaining deterministic IDs to Arrow hash kernels and confirm join-heavy pipeline
  coverage outside CPG2/ingestion.

Status
- Completed.

Targets
- src/codeintel/build/hamilton/native/graphs/call_wiring.py
- src/codeintel/core/hashing/short.py

Deliverables
- call_id stable IDs use Arrow hash kernels via stable_decimal_id.
- short_hash/sha1_short/sha256_short use Arrow kernels for non-security hashes.
- Join-heavy pipeline sweep found no additional conversions required.

## Action Set 10 (Completed)

Scope
- Guardrail cleanup + validation/serving finalize consistency: remove remaining to_pylist and
  streaming violations in validation/export paths and ensure finalize boundaries are respected
  where validation consumes tables.

Status
- Completed (guardrails still blocked by missing analytics.config_references schema).

Targets
- src/codeintel/core/validation/engine.py
- src/codeintel/storage/validation/columnar.py
- src/codeintel/serving/http/export_dispatch.py

Deliverables
- Validation/export paths use streaming-safe iteration helpers.
- Finalize boundaries stay consistent for validation consumption points.
- Guardrail to_pylist checks pass; remaining guardrail failures are schema-related.

## Action Set 11 (Completed)

Scope
- Remaining edge builders + threading/chunking policy rollout: normalize plan materialization
  and ensure chunk consolidation/threading across join-heavy pipelines; audit for non-CPG2 list
  payload edge builders that require explode conversions.

Status
- Completed (audit found no additional non-CPG2 edge builders requiring explode migration).

Targets
- src/codeintel/build/tabular/plan_ops.py (materialize_plan)
- src/codeintel/core/columnar/normalization.py
- join-heavy plan materializations (call_wiring, CPG2 planes, ingestion joins, analytics cache)

Deliverables
- Plan materialization uses shared normalization (no streaming_to_table guardrail hits).
- Chunk consolidation + Arrow threading configured at compute boundaries.
- Edge builder audit completed; no remaining non-CPG2 list payload migrations required.

## Scope items

### 1) Plan Ops (Acero DSL)

Status
- Implemented in `src/codeintel/build/tabular/plan_ops.py`; reexports added in acero_ops and
  arrow_ops; adopted in CPG2 join planes, ingestion joins, analytics cache, and call_wiring
  joins with materialize_plan for guardrail-safe materialization.

Target files
- src/codeintel/build/tabular/plan_ops.py (new)
- src/codeintel/core/columnar/acero_ops.py (refactor or fold into plan ops)
- src/codeintel/build/tabular/arrow_ops.py (bridge helpers or reexports)

Representative pattern
```python
import pyarrow.compute as pc
import pyarrow.dataset as ds

from codeintel.build.tabular.plan_ops import Plan, materialize_plan

scan = Plan.scan(
    dataset,
    columns={
        "repo": ds.field("repo"),
        "commit": ds.field("commit"),
        "call_id": ds.field("call_id"),
        "callee_ids": ds.field("callee_ids"),
    },
    filter_expr=(ds.field("repo") == "r") & (ds.field("commit") == "c"),
)

plan = (
    scan.project(
        {
            "repo": pc.field("repo"),
            "commit": pc.field("commit"),
            "call_id": pc.field("call_id"),
            "callee_ids": pc.field("callee_ids"),
        }
    )
    .filter(pc.field("call_id").is_valid())
    .aggregate(
        keys=[pc.field("repo"), pc.field("commit")],
        aggregates=[(pc.field("call_id"), "count", None, "call_count")],
    )
)

table = materialize_plan(plan, use_threads=True)
```

Distinctive pattern to standardize
- Always express scan -> project -> filter -> hashjoin -> aggregate -> order in plan form.
- Use Declaration.from_sequence for linear pipelines and table_source for in-memory tables.
- Use scan for datasets and table_source for in-memory tables.


### 2) HashJoin-First Join Plan (Acero)

Status
- HashJoinSpec is implemented; adopted in call_wiring, CPG2 link/symbol/flow, ingestion join
  pipelines (syntax_enrich/syntax_augment/extraction_targets/scip_resolution), and
  analytics/subsystems/cache; audit found no additional join-heavy pipelines pending.

Target files
- src/codeintel/build/tabular/plan_ops.py (new)
- src/codeintel/core/columnar/acero_ops.py (refactor or fold into plan ops)
- src/codeintel/build/tabular/arrow_ops.py (bridge helpers or reexports)

Representative pattern
```python
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.kernels import stable_sort_indices
from codeintel.build.tabular.plan_ops import HashJoinSpec, Plan, materialize_plan

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

join_spec = HashJoinSpec(
    left_keys=["key"],
    right_keys=["key"],
    how="left outer",
    left_output=["key", "payload_left"],
    right_output=["payload_right"],
    filter_expression=E.is_valid("payload_right"),
)

joined = left.hash_join(right=right, spec=join_spec)

result = materialize_plan(joined, use_threads=True)
result = result.take(stable_sort_indices(result, sort_keys=[("key", "ascending")]))
```

Distinctive pattern to standardize
- Pre-project and cast join keys before hashjoin.
- Explicitly specify left_output/right_output and suffixes.
- Residual filters run after key match, not as a substitute for join logic.
- Do not join with list payload columns (explode first or drop list payloads).
- Always apply deterministic ordering after hashjoin when output stability matters.


### 3) Scan Ops (pushdown + telemetry)

Status
- Implemented scan options (projection mapping + ordering flags) and telemetry helper; adopted
  across build/storage/serving readers (graph views, exports, causal scans, storage queries,
  serving scans). New readers should use shared scan helpers.

Target files
- src/codeintel/core/datasets/scanner_ops.py (extend parameters)
- src/codeintel/core/datasets/scanning.py (surface pushdown and metrics)
- src/codeintel/build/tabular/arrow_ops.py (read helpers for build pipelines)
- src/codeintel/build/graphs/engine/datasets.py (GraphViewScanOptions + scan wiring)
- src/codeintel/build/graphs/engine/views.py (scan option usage)

Representative pattern
```python
import pyarrow.dataset as ds

from codeintel.core.datasets.scanner_ops import ScannerParams, build_scanner

expr = (ds.field("repo") == "r") & (ds.field("commit") == "c")

_ = list(dataset.get_fragments(filter=expr))
_ = dataset.count_rows(filter=expr)

params = ScannerParams(
    columns={
        "repo": ds.field("repo"),
        "commit": ds.field("commit"),
        "filename": ds.field("__filename"),
    },
    filter_expression=expr,
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
- Push down filter and projection at scan time.
- Include planning telemetry (get_fragments/count_rows) before heavy work.
- Expose implicit_ordering/require_sequenced_output knobs for ordered scans.
- Use scan_parquet_dataset_with_telemetry for snapshot readers when needed.


### 4) Streaming boundaries (to_reader vs to_table)

Status
- Implemented `Plan.to_reader` + materialize_plan; join pipelines now materialize via shared
  normalization to satisfy streaming guardrails; `Plan.to_table` reserved for non-build paths.

Target files
- src/codeintel/build/tabular/plan_ops.py (new)
- src/codeintel/build/tabular/arrow_ops.py (bridge helpers)

Representative pattern
```python
reader = plan.to_reader(use_threads=True)

# Streaming path
for batch in reader:
    process_batch(batch)

# Materialization boundary (guardrail-safe)
from codeintel.build.tabular.plan_ops import materialize_plan

table = materialize_plan(plan, use_threads=True)
```

Distinctive pattern to standardize
- Use to_reader for streaming pipelines and materialize_plan at finalize boundaries.
- Reserve Plan.to_table for non-build contexts that are not guardrailed.


### 5) Explode Ops (list explode + alignment)

Status
- Implemented in `src/codeintel/build/tabular/explode_ops.py`; adopted in call_wiring and CPG2
  edge builders via finalize_cpg_edge_rows; audit found no remaining non-CPG2 list payload
  edge builders requiring explode conversion.

Target files
- src/codeintel/build/tabular/explode_ops.py (new)
- src/codeintel/build/tabular/kernels.py (shared kernel wrappers)

Representative pattern
```python
from codeintel.build.tabular.explode_ops import ExplodeSpec, explode_edges

result = explode_edges(
    table,
    spec=ExplodeSpec(
        src_col="call_id",
        dst_list_col="callee_ids",
        repeat_cols=("repo", "commit"),
        aligned_list_cols=("callsite_spans",),
        null_list_policy="error",
        null_child_policy="drop",
        enforce_parent_valid=True,
    ),
)

good_edges = result.good
error_rows = result.errors
```

Distinctive pattern to standardize
- Use ExplodeSpec to define policies (null list/child, aligned list columns).
- Use pc.list_parent_indices + pc.list_flatten + pc.take.
- Validate list alignment before explode (pc.list_value_length).
- Enforce null list vs empty list policy explicitly.
- Handle list_view inputs, but avoid storing list_view in contracts.


### 6) Finalize Gate (strict/tolerant contracts + artifacts)

Status
- Implemented in `src/codeintel/build/tabular/finalize_ops.py`; adopted for graph outputs and
  CPG2 assembly; analytics/validation outputs now finalize via helpers and findings.

Target files
- src/codeintel/build/tabular/finalize_ops.py (new)
- src/codeintel/build/tabular/arrow_ops.py (optional reexports)
- src/codeintel/build/hamilton/native/analytics/finalize_helpers.py
- src/codeintel/build/graphs/validation/findings.py

Representative pattern
```python
from codeintel.build.tabular.finalize_ops import (
    FinalizeDedupe,
    FinalizeInvariant,
    FinalizeResult,
    FinalizeSpec,
    finalize_table,
)

result: FinalizeResult = finalize_table(
    edges,
    spec=FinalizeSpec(
        table_key="graph.cpg_edges_calls",
        mode="tolerant",
        required_non_null=("repo", "commit", "call_id"),
        invariants=(
            FinalizeInvariant.list_alignment("callee_ids", ("callsite_spans",)),
            FinalizeInvariant.struct_required("extras", ("repo", "commit")),
        ),
        dedupe=FinalizeDedupe(prefer_columns=("confidence",)),
        emit_artifacts=True,
    ),
)

good = result.good
errors = result.errors
alignment = result.alignment
stats = result.stats
```

Distinctive pattern to standardize
- Align to contract via align_table_to_contract.
- Compute vectorized invariants with compute masks.
- Emit alignment report and error stats tables.
- Dedupe after invariants using deterministic sort keys.
- Tolerant mode never throws and returns error artifacts.
- FinalizeSpec carries table_key/mode/invariants consistently.


### 7) Nested + Schema Evolution Kit (typed extras + extras_kv)

Status
- Implemented utilities + graph schema updates in `output_registry.py`; producer migrations
  completed for graph + remaining CPG2/legacy planes; non-graph producers pending.

Target files
- src/codeintel/build/tabular/nested_ops.py (new)
- src/codeintel/core/schemas/output_registry.py (replace extras_json with extras/extras_kv)
- src/codeintel/build/hamilton/contracts/schemas/pandera_schemas.py (if registered)

Representative pattern
```python
import pyarrow as pa

from codeintel.build.tabular.nested_ops import (
    deep_cast_table_to_contract,
    make_extras_struct,
    make_extras_kv_map,
    unify_schemas_with_contract_first,
)

extras = make_extras_struct(
    table,
    fields={
        "repo": pa.string(),
        "commit": pa.string(),
        "parse_version": pa.int32(),
        "confidence": pa.float64(),
    },
)
extras_kv = make_extras_kv_map(table, keys="extras_keys", values="extras_values")

with_extras = table.append_column("extras", extras).append_column("extras_kv", extras_kv)

unified = unify_schemas_with_contract_first(contract_schema, [with_extras.schema])
casted = deep_cast_table_to_contract(with_extras, contract_schema)
```

Distinctive pattern to standardize
- Extras stored as struct<...> at creation time plus optional extras_kv map.
- No extras_json in graph and analytics tables.
- Deep casts for list<struct> or struct evolution are centralized.
- Use list_ over large_list unless required; avoid list_view in persisted contracts.


### 8) Expr/Kernels vocabulary + determinism

Status
- Implemented expr/kernels helpers; adopted in join pipelines and ordering; remaining pipeline
  adoption pending.

Target files
- src/codeintel/build/tabular/expr_vocab.py (new)
- src/codeintel/build/tabular/kernels.py (new)

Representative pattern
```python
from codeintel.build.tabular.compute_helpers import call_compute, require_array
from codeintel.build.tabular.kernels import (
    case_when,
    stable_sort_indices,
)

repo_null = require_array(call_compute("is_null", [table["repo"]]), name="is_null")
commit_null = require_array(call_compute("is_null", [table["commit"]]), name="is_null")
error_code = case_when(
    (repo_null, "NULL_REPO"),
    (commit_null, "NULL_COMMIT"),
    else_="OK",
)

ordered = table.take(stable_sort_indices(table, sort_keys=[("repo", "ascending")]))
```

Distinctive pattern to standardize
- expr_* helpers return pc.Expression for scan/filter/join.
- k_* helpers operate on arrays/tables (filter, take, list, struct, sort).
- Deterministic ordering uses stable sort indices and explicit tie-breakers.


### 9) Deterministic IDs (Arrow hash kernels)

Status
- Completed: hash kernel wrapper implemented; assembly IDs + CPG2 ordinals migrated; GOID hashing,
  call wiring IDs, and short hashes now use Arrow kernels.

Target files
- src/codeintel/build/tabular/kernels.py (new)
- src/codeintel/build/graphs/assembly/ids.py (migrate to vectorized hash)
- src/codeintel/build/hamilton/native/graphs/cpg2/ids.py (migrate to vectorized hash)
- src/codeintel/build/hamilton/native/graphs/goids.py

Representative pattern
```python
from codeintel.build.tabular.kernels import hash_struct_ordinal

ordinal = hash_struct_ordinal(
    table,
    columns=["repo", "commit", "call_id", "callsite_line", "callsite_col"],
    modulus=2**31 - 1,
)
```

Distinctive pattern to standardize
- ID hashing is separate from joins and uses Arrow compute kernels when available.
- Centralize hashing in kernels to avoid Python row loops.
- Expose a single hashing strategy per ID family (ordinal, goid).


### 10) Threading + chunking

Status
- Completed: normalize_table_for_compute configures Arrow threading (via shared settings) and
  materialize_plan consolidates chunks across plan materialization; scan_parquet_table
  normalizes tables for compute kernels.

Target files
- src/codeintel/core/columnar/normalization.py
- src/codeintel/build/tabular/plan_ops.py
- src/codeintel/build/tabular/arrow_ops.py
- src/codeintel/core/datasets/scanning.py

Representative pattern
```python
from codeintel.core.columnar.normalization import normalize_table_for_compute

table = normalize_table_for_compute(table)
```

Distinctive pattern to standardize
- Combine small chunks before heavy compute stages.
- Configure CPU and IO thread pools for predictable throughput.


### 11) Edge builders -> list-of-struct + explode + finalize

Status
- Completed for graph.cpg_edges_calls and CPG2 edge builders (call_wiring, overlays, scip);
  audit found no remaining non-CPG2 list payload edge builders requiring explode migration.

Target files
- src/codeintel/build/hamilton/native/graphs/call_wiring.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/call_wiring.py
- src/codeintel/build/hamilton/native/graphs/cpg2/edge_helpers.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/overlays_bytecode.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/overlays_inspect.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/overlays_symtable.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py
- src/codeintel/build/hamilton/native/graphs/cpg/edges.py
- src/codeintel/core/schemas/output_registry.py (new graph.cpg_call_candidates table)
- config/registry/dag_output_inventory.yaml

Representative pattern
```python
from codeintel.build.tabular.explode_ops import ExplodeSpec, explode_edges
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_table

rows = [
    {
        "repo": repo,
        "commit": commit,
        "call_id": call_id,
        "call_node_id": call_node_id,
        "extras": extras_struct,
        "extras_kv": extras_kv,
        "candidates": [
            {
                "callee_goid_h128": callee_goid,
                "binding_kind": binding_kind,
                "confidence": confidence,
                "extras_kv": candidate_extras,
            }
        ],
    }
]

candidates = table_for_rows("graph.cpg_call_candidates", rows)[0]

exploded = explode_edges(
    candidates,
    spec=ExplodeSpec(
        src_col="call_id",
        dst_list_col="candidates",
        repeat_cols=("repo", "commit", "call_node_id", "extras"),
        aligned_list_cols=("callsite_spans",),
    ),
)

result = finalize_table(
    exploded.good,
    spec=FinalizeSpec(table_key="graph.cpg_edges_calls", mode="strict"),
)
```

Distinctive pattern to standardize
- Per-call rows carry list<struct> candidates with list-aligned metadata.
- Explode at a single step using shared explode ops.
- Finalize gate standardizes schema, invariants, and dedupe.


### 12) Join pipelines -> plan ops + expr vocab

Status
- Completed for CPG2 link/symbol/flow and ingestion join pipelines; analytics/subsystems/cache
  migrated; audit found no remaining join-heavy pipelines pending.

Target files
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py

Representative pattern
```python
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.kernels import stable_sort_indices
from codeintel.build.tabular.plan_ops import HashJoinSpec, Plan, materialize_plan

plan = (
    Plan.table(left)
    .filter(E.is_valid("caller_goid_h128"))
    .hash_join(
        right=right,
        spec=HashJoinSpec(
            left_keys=["caller_goid_h128"],
            right_keys=["goid_h128"],
            how="left outer",
            right_output=["cpg_node_id"],
        ),
    )
    .filter(E.is_valid("cpg_node_id"))
    .project({"src_cpg_node_id": E.field("cpg_node_id")})
)

joined = materialize_plan(plan, use_threads=True)
joined = joined.take(stable_sort_indices(joined, sort_keys=[("src_cpg_node_id", "ascending")]))
```

Distinctive pattern to standardize
- Acero hash joins with explicit projections and filters.
- Expressions for filters and projections are always built with expr vocab.
- Deterministic ordering applied after hashjoin when required.


### 13) Graph analysis outputs -> finalize gate + kernel masks

Status
- Completed for cfg_dfg/cdg/call_graph/import_graph/symbol_use/pdg + CPG2 assemble.

Target files
- src/codeintel/build/hamilton/native/graphs/cfg_dfg.py
- src/codeintel/build/hamilton/native/graphs/cdg.py
- src/codeintel/build/hamilton/native/graphs/call_graph.py
- src/codeintel/build/hamilton/native/graphs/import_graph.py
- src/codeintel/build/hamilton/native/graphs/symbol_use.py

Representative pattern
```python
from codeintel.build.tabular.finalize_ops import finalize_table

edges_table, _ = table_for_rows("graph.cfg_edges", rows)

result = finalize_table(
    "graph.cfg_edges",
    edges_table,
    mode="tolerant",
    required_non_null=("repo", "commit", "src_block_id", "dst_block_id"),
    emit_artifacts=True,
)

return result.good
```

Distinctive pattern to standardize
- Domain logic can remain in Python when necessary.
- Output shaping, alignment, invariants, and dedupe are centralized.
- Alignment and error stats are emitted for diagnostics.


### 14) Analytics and validation -> scan ops + finalize

Status
- Completed: hashjoin/plan conversion in analytics/subsystems/cache; finalize adoption completed;
  scan ops pushdown/telemetry wired through shared dataset readers for analytics/validation.

Target files
- src/codeintel/build/analytics/**
- src/codeintel/build/graphs/validation/**

Representative pattern
```python
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_table
from codeintel.core.datasets.scanning import ParquetScanOptions, scan_parquet_dataset

reader = scan_parquet_dataset(
    dataset_root=root,
    table_key="graph.cpg_edges_calls",
    snapshot_id=snapshot_id,
    options=ParquetScanOptions(columns=["repo", "commit", "call_id"]),
)

result = finalize_table(
    computed_table,
    spec=FinalizeSpec(
        table_key="analytics.graph_metrics",
        mode="tolerant",
        required_non_null=("repo", "commit"),
        emit_artifacts=True,
    ),
)
```

Distinctive pattern to standardize
- Use scan pushdown and planning telemetry for analytics inputs.
- Finalize gate produces structured errors and alignment reports.


### 15) Escape hatches (Substrait / DataFusion)

Status
- Not started (optional).

Target files
- src/codeintel/build/tabular/substrait_ops.py (new, optional)
- src/codeintel/build/tabular/datafusion_ops.py (new, optional)

Representative pattern
```python
import pyarrow.substrait as ps

reader = ps.run_query(plan_bytes)
```

Distinctive pattern to standardize
- Use Substrait for standardized plan interchange where Acero plans are generated externally.
- Use DataFusion for complex relational cases Acero cannot express.


## Pilot plan: graph.cpg_edges_calls (call_wiring.py)

Status
- Completed.

Target files
- src/codeintel/build/hamilton/native/graphs/call_wiring.py
- src/codeintel/build/hamilton/native/graphs/graph_targets.py
- src/codeintel/core/schemas/output_registry.py
- config/registry/dag_output_inventory.yaml
- src/codeintel/build/tabular/explode_ops.py
- src/codeintel/build/tabular/finalize_ops.py

Representative pattern
```python
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.explode_ops import ExplodeSpec, explode_edges
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_table
from codeintel.build.tabular.plan_ops import HashJoinSpec, Plan, materialize_plan

candidates = table_for_rows("graph.cpg_call_candidates", rows)[0]

exploded = explode_edges(
    candidates,
    spec=ExplodeSpec(
        src_col="call_id",
        dst_list_col="candidates",
        repeat_cols=("repo", "commit", "call_node_id", "extras"),
    ),
)

plan = (
    Plan.table(exploded.good)
    .project(
        {
            "repo": E.field("repo"),
            "commit": E.field("commit"),
            "call_id": E.field("call_id"),
            "call_node_id": E.field("call_node_id"),
            "extras": E.field("extras"),
            "callee_goid_h128": E.field(("candidates", "callee_goid_h128")),
            "confidence": E.field(("candidates", "confidence")),
            "extras_kv": E.field(("candidates", "extras_kv")),
        }
    )
    .filter(E.is_valid("callee_goid_h128"))
    .hash_join(
        right=entry_blocks,
        spec=HashJoinSpec(
            left_keys=["callee_goid_h128"],
            right_keys=["function_goid_h128"],
            how="left outer",
            right_output=["entry_block_id"],
        ),
    )
    .filter(E.is_valid("entry_block_id"))
    .project(
        {
            "repo": E.field("repo"),
            "commit": E.field("commit"),
            "call_id": E.field("call_id"),
            "call_node_id": E.field("call_node_id"),
            "callee_entry_block_id": E.field("entry_block_id"),
            "edge_kind": E.scalar("CALLS"),
            "confidence": E.field("confidence"),
            "extras": E.field("extras"),
            "extras_kv": E.field("extras_kv"),
        }
    )
)

edges = materialize_plan(plan, use_threads=True)
edges = edges.take(
    stable_sort_indices(
        edges,
        sort_keys=[
            ("repo", "ascending"),
            ("commit", "ascending"),
            ("call_id", "ascending"),
            ("call_node_id", "ascending"),
        ],
    )
)
result = finalize_table(
    edges,
    spec=FinalizeSpec(table_key="graph.cpg_edges_calls", mode="strict"),
)
```

Distinctive pattern to standardize
- list<struct> candidates are materialized once and exploded via kernels.
- Acero handles join/filter/project for edge assembly.
- Extras are typed struct, not JSON.
- Finalize gate is the only output boundary.
- Deterministic ordering is enforced after hashjoin.


## Schema changes (extras_json removal)

Status
- Graph outputs migrated for cfg_dfg/cdg/call_graph/import_graph/symbol_use/pdg; call wiring
  producers migrated for graph.cpg_call_candidates/cpg_edges_calls/cpg_edges_ret_to_call and cpg2
  call_wiring plane reads extras_kv; remaining CPG2/legacy producers migrated to extras/extras_kv;
  schema snapshot refresh deferred.

Target files
- src/codeintel/core/schemas/output_registry.py
- config/schema_breaks.yaml
- src/codeintel/build/hamilton/contracts/schemas/pandera_schemas.py (if table registered)
- src/codeintel/build/hamilton/native/graphs/* (remove encode/decode payload usage)

Representative pattern
```python
# Example column definition in output_registry.py
Column("extras", "STRUCT(repo VARCHAR, commit VARCHAR, confidence DOUBLE)")
Column("extras_kv", "MAP(VARCHAR, VARCHAR)")
```

Distinctive pattern to standardize
- extras_json is removed from schemas and from producer/consumer code.
- extras struct and optional extras_kv are the only metadata channels.


## Quality gates per phase

- Phase 0/1/2/3 follow the AOP gates:
  - uv run python -m tools.quality_report --output build/quality-results/quality_report.json
  - uv run pytest -q for targeted subsets, then segmented by major directories
