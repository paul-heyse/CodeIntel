# Core Compute Acero DSL Best-in-Class Implementation Plan

## Goal
Consolidate Arrow/Acero compute into a single shared DSL that is Acero-first, streaming-safe,
deterministic when required, and centered around "plan -> execute -> finalize" with typed
extras and schema-evolution guardrails.

## Scope Items

### 1) Unified Arrow DSL surface + ExecutionContext

Pattern (Plan + runtime policy in one place)
```python
from dataclasses import dataclass
from typing import Callable, Protocol

import pyarrow as pa
from pyarrow import acero


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    use_threads: bool
    combine_chunks: bool
    determinism: str  # "canonical" | "stable_set" | "best_effort"


TableThunk = Callable[[], pa.Table]


@dataclass(frozen=True, slots=True)
class Plan:
    declaration: acero.Declaration | None
    fallback: TableThunk | None = None

    def execute(self, *, ctx: ExecutionContext) -> pa.Table:
        if self.declaration is not None:
            return self.declaration.to_table(use_threads=ctx.use_threads)
        if self.fallback is None:
            msg = "Plan execution requires a declaration or fallback."
            raise ValueError(msg)
        return self.fallback()
```

Target files
- src/codeintel/core/columnar/plan_ops.py (canonical Plan/HashJoinSpec)
- src/codeintel/core/columnar/runtime.py (new ExecutionContext)
- src/codeintel/build/tabular/plan_ops.py (re-export or thin wrapper)
- src/codeintel/build/tabular/arrow_ops.py (re-export runtime helpers)

Checklist
- [ ] Add ExecutionContext (threads, determinism, chunking policy).
- [ ] Ensure Plan supports Acero Declaration + optional fallback thunk.
- [ ] Re-export in build/tabular to remove parallel APIs.
- [ ] Migrate callers to import the canonical core Plan/ExecutionContext.


### 2) Acero scan-first + streaming discipline

Pattern (scan -> project/filter -> reader; materialize only at finalize)
```python
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.plan_ops import Plan

plan = (
    Plan.scan(dataset, columns=["repo", "commit", "kind", "src", "dst"])
    .filter(E.and_(E.is_valid("repo"), E.is_valid("commit")))
    .project({"repo": E.field("repo"), "dst": E.field("dst")})
)

reader = plan.to_reader(use_threads=True)
```

Target files
- src/codeintel/core/columnar/plan_ops.py (scan defaults, order_by optional)
- src/codeintel/core/datasets/scanning.py (prefer plan reader where supported)
- src/codeintel/build/tabular/plan_ops.py (remove divergent scan semantics)
- src/codeintel/build/hamilton/native/** (plan-first readers in nodes)

Checklist
- [ ] Make scan-first the default entry for dataset reads (pushdown + project).
- [ ] Prefer RecordBatchReader until finalize boundaries.
- [ ] Standardize implicit_ordering/require_sequenced_output usage.
- [ ] Add plan-first helper to replace ad hoc scanner usage.


### 3) Determinism + dedupe policy in finalize contracts

Pattern (determinism tier and tie-breakers)
```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DedupeSpec:
    keys: tuple[str, ...]
    tie_breakers: tuple[tuple[str, str], ...]  # ("col", "ascending"|"descending")
    mode: str  # "keep_first" | "keep_best" | "collapse_list" | "none"


@dataclass(frozen=True, slots=True)
class ContractPolicy:
    determinism: str  # "canonical" | "stable_set" | "best_effort"
    dedupe: DedupeSpec | None
```

Target files
- src/codeintel/build/tabular/finalize_ops.py (DedupeSpec + determinism tiers)
- src/codeintel/build/tabular/dedupe_ops.py (keep_best strategy + order-independent path)
- src/codeintel/build/schemas/** (contract policy metadata)

Checklist
- [ ] Add DedupeSpec with explicit tie-breakers and determinism tier.
- [ ] Implement order-independent winner selection (min/max join-back).
- [ ] Enforce deterministic sort before keep_first when determinism=canonical.
- [ ] Persist determinism policy in contract metadata.


### 4) Nested schema evolution integrated into finalize

Pattern (deep cast + allowed promotions)
```python
from codeintel.build.tabular.nested_ops import deep_cast_table_to_contract
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_table

casted = deep_cast_table_to_contract(table, contract_schema)
result = finalize_table(casted, spec=FinalizeSpec(table_key="analytics.config_references", mode="tolerant"))
```

Target files
- src/codeintel/build/tabular/nested_ops.py (promotion policy + list/list_view rules)
- src/codeintel/build/tabular/finalize_ops.py (use deep_cast when nested present)
- src/codeintel/build/tabular/arrow_ops.py (align->deep_cast pipeline)

Checklist
- [ ] Define allowed type promotions for nested fields.
- [ ] Route nested alignment through deep_cast_table_to_contract.
- [ ] Enforce list (not list_view) in persisted contracts.
- [ ] Add extras_version checks in finalize invariants.


### 5) Kernel/expr vocabulary expansion + enforcement

Pattern (shared helpers only)
```python
from codeintel.build.tabular.kernels import (
    case_when,
    stable_sort_indices,
)
from codeintel.build.tabular.expr_vocab import E

mask = E.and_(E.is_valid("repo"), E.is_valid("commit"))
ordered = table.take(stable_sort_indices(table, sort_keys=[("repo", "ascending")]))
```

Target files
- src/codeintel/build/tabular/expr_vocab.py (complete expression surface)
- src/codeintel/build/tabular/kernels.py (list/struct/map, regex, safe casts)
- src/codeintel/core/columnar/expr_vocab.py (canonical copy)
- src/codeintel/core/columnar/kernels.py (canonical copy)

Checklist
- [ ] Expand kernels: list_len, list_slice, struct_field, indices_nonzero, safe_cast.
- [ ] Expand expr vocab: in_, coalesce, cast, is_null/is_valid.
- [ ] Add a "no raw pc.* in nodes" guardrail (lint or scripted check).
- [ ] Update nodes to use expr_vocab/kernels exclusively.


### 6) HashJoin policy module + guardrails

Pattern (pre-project/cast + null-key gate)
```python
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan

plan = (
    Plan.table(left)
    .project({"key": E.cast(E.field("key"), "int64"), "payload": E.field("payload")})
    .filter(E.is_valid("key"))
    .hash_join(
        right=Plan.table(right),
        spec=HashJoinSpec(
            left_keys=["key"],
            right_keys=["key"],
            left_output=["key", "payload"],
            right_output=["value"],
        ),
    )
)
```

Target files
- src/codeintel/core/columnar/plan_ops.py (HashJoinSpec policy helpers)
- src/codeintel/build/tabular/plan_ops.py (re-export)
- src/codeintel/build/tabular/finalize_ops.py (join precheck errors)

Checklist
- [ ] Implement HashJoin policy helper enforcing pre-cast, non-null keys.
- [ ] Add list-payload guardrail before hashjoin (explode or drop).
- [ ] Centralize residual filter semantics for post-join gating.
- [ ] Emit join precheck errors for null or mismatched keys.


### 7) Non-graph producers: extras/extras_kv audit + migration

Pattern (typed extras at creation time)
```python
rows = [
    {
        "repo": repo,
        "commit": commit,
        "extras": {"reference_paths": paths, "reference_modules": modules},
        "created_at": now,
    }
]
```

Target files
- src/codeintel/build/analytics/** (non-graph producers and consumers)
- src/codeintel/build/exports/** (flatten extras at export boundaries only)
- src/codeintel/build/schemas/** (extras-only contracts)

Checklist
- [ ] Audit non-graph tables for any extras_json / untyped metadata.
- [ ] Migrate remaining producers to extras + extras_kv.
- [ ] Remove all JSON payload helpers in compute paths.
- [ ] Update consumers to read extras directly.


### 8) Scan provenance + partitioning policy adoption

Pattern (provenance columns + telemetry)
```python
from codeintel.core.datasets.scanning import ParquetScanOptions, scan_parquet_dataset

reader = scan_parquet_dataset(
    dataset_root=root,
    table_key="graph.cpg_edges_calls",
    snapshot_id=snapshot_id,
    options=ParquetScanOptions(
        columns=["repo", "commit", "call_id"],
        provenance_columns=("__filename", "__fragment_index"),
        metrics_enabled=True,
    ),
)
```

Target files
- src/codeintel/core/datasets/scanning.py (provenance columns default policy)
- src/codeintel/core/columnar/streaming.py (partitioning + scanner config)
- src/codeintel/build/graphs/validation/** (include provenance in error artifacts)

Checklist
- [ ] Standardize provenance columns for error tables.
- [ ] Ensure partitioning metadata is honored for pruning.
- [ ] Emit fragment/file telemetry for scan plans.
- [ ] Document partitioning policy in schema registry metadata.


### 9) Escape hatch integration into the DSL (Substrait/DataFusion)

Pattern (ExternalPlanSpec wired into Plan execution)
```python
from codeintel.core.columnar.plan_ops import ExternalPlanSpec, run_external_plan

spec = ExternalPlanSpec(engine="substrait", payload=plan_bytes)
reader = run_external_plan(
    spec=spec,
    dataset=dataset,
    filter_expr=filter_expr,
    columns=columns,
    use_threads=True,
)
```

Target files
- src/codeintel/core/columnar/plan_ops.py (ExternalPlanSpec wiring)
- src/codeintel/build/tabular/substrait_ops.py (optional)
- src/codeintel/build/tabular/datafusion_ops.py (optional)
- src/codeintel/build/tabular/plan_ops.py (re-export external runner)

Checklist
- [ ] Implement a single external plan runner interface.
- [ ] Wire Substrait/DataFusion wrappers into run_external_plan.
- [ ] Ensure external outputs pass through finalize gates.
- [ ] Document when to choose external engines vs Acero.


### 10) Chunking/materialization boundaries

Pattern (single boundary)
```python
from codeintel.core.columnar.normalization import normalize_table_for_compute

table = normalize_table_for_compute(table, combine_chunks=True)
```

Target files
- src/codeintel/core/columnar/normalization.py (policy-based chunking)
- src/codeintel/build/tabular/finalize_ops.py (combine_chunks at boundary)
- src/codeintel/core/datasets/scanning.py (batch_size alignment)

Checklist
- [ ] Centralize combine_chunks policy in normalization helpers.
- [ ] Only materialize to Table at finalize or global ops (sort/dedupe).
- [ ] Align batch_size with downstream compute expectations.
- [ ] Add metrics for chunk counts pre/post normalization.


## Deliverable Summary
- A single, canonical Arrow DSL and runtime context shared across build/core.
- Explicit determinism and dedupe policy baked into finalize.
- Consistent Acero scan-first plans with streaming readers and planned materialization.
- Typed extras everywhere, no JSON payloads except at export boundaries.
- Escape hatch engines integrated but contained behind a single interface.

