# Core Columnar Acero DSL Best-in-Class Implementation Plan

## Purpose
Deliver a single, contract-driven Arrow/Acero execution surface for core compute that
preserves streaming, enforces deterministic outputs, and makes observability artifacts
first-class. This plan integrates the identified findings and the agreed design decisions
into a single, phased implementation roadmap.

## Non-Negotiable Design Decisions (Locked)
- `run_pipeline` returns `FinalizeResult` by default; a `.good` convenience helper
  is provided for legacy call sites.
- Canonical ordering derives from `stable_sort_keys` when explicitly configured.
  `stable_sort_keys = ()` means "no canonical order" and must error or downgrade.
  `stable_sort_keys is None` falls back to primary keys or requires explicit `order_by`.
- List alignment, null list policies, and dedupe specs are contract-driven and derived
  from schema metadata; `FinalizeSpec` overrides are allowed but rare.
- Runtime profiles are first-class defaults: `DEV_FAST`, `DEV_DETERMINISTIC`,
  `CI_STABLE`, `PROD_THROUGHPUT`.

---

## Scope 00 - IR v2 + Runner Returns FinalizeResult (Plan Sum-Type)
**Findings addressed**
- `run_pipeline` drops artifacts and forces `pa.Table` materialization.
- IR split between `ExecutionPlan` and `Plan` with no reader/thunk path.

**Pattern**
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import pyarrow as pa
from pyarrow import acero

from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.finalize_ops import FinalizeResult, FinalizeSpec, finalize_reader
from codeintel.core.columnar.ordering import OrderingSpec

TableThunk = Callable[[], pa.Table]
ReaderThunk = Callable[[], pa.RecordBatchReader]

@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    decl: acero.Declaration | None = None
    table_thunk: TableThunk | None = None
    reader_thunk: ReaderThunk | None = None
    ordering: OrderingSpec | None = None

    def to_reader(self, *, ctx: ExecutionContext) -> pa.RecordBatchReader:
        if self.decl is not None:
            return self.decl.to_reader(use_threads=ctx.use_threads)
        if self.reader_thunk is not None:
            return self.reader_thunk()
        if self.table_thunk is not None:
            return self.table_thunk().to_reader()
        raise RuntimeError("ExecutionPlan has no backend")


def run_pipeline(
    *,
    plan: ExecutionPlan,
    finalize: FinalizeSpec,
    ctx: ExecutionContext,
) -> FinalizeResult:
    reader = plan.to_reader(ctx=ctx)
    return finalize_reader(reader, spec=finalize)


def run_pipeline_good(
    *,
    plan: ExecutionPlan,
    finalize: FinalizeSpec,
    ctx: ExecutionContext,
) -> pa.Table:
    return run_pipeline(plan=plan, finalize=finalize, ctx=ctx).good
```

**Target files**
- src/codeintel/core/columnar/arrowdsl.py
- src/codeintel/core/columnar/plan_ops.py
- src/codeintel/core/columnar/stream.py
- src/codeintel/core/columnar/__init__.py

**Checklist**
- [ ] Collapse ExecutionPlan into a sum-type with Declaration/TableThunk/ReaderThunk.
- [ ] Make `run_pipeline` return `FinalizeResult` and add `run_pipeline_good`.
- [ ] Update call sites that need artifacts to use `FinalizeResult` directly.
- [ ] Integrate `ExternalPlanSpec`/external runners into `ExecutionPlan`.
- [ ] Ensure streaming paths stay reader-first until finalize.

---

## Scope 01 - Contract-Driven Finalize Spec (Stable Sort + Invariants)
**Findings addressed**
- Canonical ordering ignores `stable_sort_keys`.
- List alignment/null list policies drift in hidden maps.

**Pattern**
```python
from dataclasses import dataclass

from codeintel.core.columnar.finalize_ops import (
    FinalizeDedupe,
    FinalizeInvariant,
    FinalizeListPolicy,
    FinalizeSpec,
)
from codeintel.core.schemas.primitives import resolve_stable_sort_keys
from codeintel.core.schemas.service import get_schema_service

@dataclass(frozen=True)
class FinalizePolicy:
    required_non_null: tuple[str, ...] = ()
    list_policies: tuple[FinalizeListPolicy, ...] = ()
    invariants: tuple[FinalizeInvariant, ...] = ()
    dedupe: FinalizeDedupe | None = None
    canonical_sort_keys: tuple[str, ...] | None = None


def finalize_spec_for_table(
    table_key: str,
    *,
    mode: str,
    target_name: str | None = None,
) -> FinalizeSpec:
    schema = get_schema_service().require_table_schema(table_key)
    policy = schema.finalize_policy
    canonical = policy.canonical_sort_keys
    if canonical is None:
        canonical = resolve_stable_sort_keys(schema)
    order_by = tuple((name, "ascending") for name in canonical or ())
    return FinalizeSpec(
        table_key=table_key,
        mode=mode,
        required_non_null=policy.required_non_null,
        list_policies=policy.list_policies,
        invariants=policy.invariants,
        dedupe=policy.dedupe,
        order_by=order_by,
        target_name=target_name,
    )
```

**Target files**
- src/codeintel/core/schemas/primitives.py
- src/codeintel/core/schemas/serde.py
- src/codeintel/core/schemas/contract_serde.py
- src/codeintel/core/columnar/finalize_ops.py
- src/codeintel/core/validation/schema_constraints.py
- src/codeintel/core/schemas/table_registry.py

**Checklist**
- [ ] Add `FinalizePolicy` to TableSchema metadata with serde support.
- [ ] Replace hidden list-alignment maps with schema-level policy definitions.
- [ ] Derive `FinalizeSpec` from schema policy by default.
- [ ] Enforce `stable_sort_keys` precedence for canonical ordering.
- [ ] Require explicit `order_by` when `stable_sort_keys = ()` and determinism is canonical.

---

## Scope 02 - Order-Independent Dedupe (`keep_best_by_score`)
**Findings addressed**
- `keep_best_by_score` is still order-dependent.

**Pattern**
```python
from codeintel.core.columnar.kernels import hash_struct_ordinal
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan

SCORE_COL = "__dedupe_score"


def dedupe_keep_best_by_score(
    table: pa.Table,
    *,
    keys: tuple[str, ...],
    score_columns: tuple[str, ...],
) -> pa.Table:
    score = hash_struct_ordinal(table, columns=score_columns, modulus=2**31 - 1)
    scored = table.append_column(SCORE_COL, score)
    winners = scored.group_by(list(keys)).aggregate([(SCORE_COL, "max")])
    plan = Plan.table(scored).hash_join(
        right=Plan.table(winners),
        spec=HashJoinSpec(
            left_keys=list(keys),
            right_keys=list(keys),
            left_output=list(scored.column_names),
            right_output=[f"{SCORE_COL}_max"],
        ),
    )
    joined = plan.to_table(use_threads=True)
    return joined.filter(joined[SCORE_COL] == joined[f"{SCORE_COL}_max"]).drop(
        [SCORE_COL, f"{SCORE_COL}_max"]
    )
```

**Target files**
- src/codeintel/core/columnar/dedupe_ops.py
- src/codeintel/core/columnar/kernels.py
- src/codeintel/core/columnar/arrowdsl.py
- tests/columnar/test_dedupe_keep_best_by_score.py

**Checklist**
- [ ] Implement score-based, order-independent winner selection.
- [ ] Ensure join-safe projection before using hash join.
- [ ] Add canonical determinism tests (threaded vs unthreaded).
- [ ] Add collision fallback handling when score ties occur.

---

## Scope 03 - Runtime Profiles and Explicit Threading Ownership
**Findings addressed**
- Threading policy is split and implicit; profiles are not first-class.

**Pattern**
```python
from dataclasses import dataclass

from codeintel.core.columnar.dedupe_ops import DedupeTier

@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    name: str
    scan_profile: str
    cpu_threads: int | None
    io_threads: int | None
    plan_use_threads: bool
    determinism: DedupeTier
    provenance: bool

# Examples
DEV_FAST = RuntimeProfile(
    name="DEV_FAST",
    scan_profile="dev_fast",
    cpu_threads=None,
    io_threads=None,
    plan_use_threads=True,
    determinism="stable_set",
    provenance=True,
)
```

**Target files**
- src/codeintel/core/columnar/execution_context.py
- src/codeintel/core/columnar/streaming.py
- src/codeintel/core/config/settings.py
- src/codeintel/core/columnar/normalization.py

**Checklist**
- [ ] Extend `RuntimeProfile` with CPU/I/O thread counts and scan profile.
- [ ] Apply profile defaults at entrypoints (scanner and plan runner).
- [ ] Remove implicit threading configuration from normalization.
- [ ] Add profile registry and surface it in runtime settings.

---

## Scope 04 - Observability: Run Manifest + Finalize Artifacts
**Findings addressed**
- Run manifests are not wired into execution; artifacts are not guaranteed.

**Pattern**
```python
from codeintel.core.columnar.run_manifest import RunManifestOptions, write_run_manifest
from codeintel.core.columnar.streaming import scan_telemetry_for_queryspec

result = run_pipeline(plan=plan, finalize=spec, ctx=ctx)
write_run_manifest(
    output_dir,
    options=RunManifestOptions(
        determinism=ctx.determinism,
        ordering=plan.ordering,
        scan_telemetry=telemetry,
        profile_name=ctx.runtime_profile.name if ctx.runtime_profile else None,
        scan_profile=ctx.runtime_profile.scan_profile if ctx.runtime_profile else None,
    ),
)
```

**Target files**
- src/codeintel/core/columnar/arrowdsl.py
- src/codeintel/core/columnar/run_manifest.py
- src/codeintel/core/columnar/finalize_ops.py
- src/codeintel/core/columnar/streaming.py
- tools/arrowdsl/run_manifest.py

**Checklist**
- [ ] Emit run manifests from pipeline execution.
- [ ] Ensure tolerant finalize always emits artifacts.
- [ ] Attach provenance columns to errors when enabled.
- [ ] Persist scan telemetry in artifacts and manifests.

---

## Scope 05 - Retire Legacy Surfaces + Guardrails
**Findings addressed**
- `acero_ops.py` uses legacy signatures and bypasses ordering metadata.
- Ad hoc materialization and raw `pc.*` usage risk sprawl.

**Pattern**
```python
# tools/lint_no_raw_pyarrow_compute_in_nodes.py
if "pyarrow.compute" in file_text and "columnar" not in path:
    raise SystemExit("Use columnar.expr_vocab or columnar.kernels only")
```

**Target files**
- src/codeintel/core/columnar/acero_ops.py
- src/codeintel/core/datasets/scanner_ops.py
- tools/lint_no_raw_pyarrow_compute_in_nodes.py
- tools/lint_no_materialize_in_nodes.py
- tools/quality_report.py

**Checklist**
- [ ] Replace `acero_ops.build_exec_plan` with IR v2 usage or deprecate it.
- [ ] Consolidate scanner construction around QuerySpec + streaming helpers.
- [ ] Add lints for "no raw pc in nodes" and "no to_table outside finalize".
- [ ] Wire lints into quality report gating.

---

## Scope 06 - Filter Compiler + QuerySpec Integration
**Findings addressed**
- Filter compilation is isolated from QuerySpec, risking drift.

**Pattern**
```python
from codeintel.core.queries.filter_compiler import (
    compile_filter_predicates,
    arrow_filter_expression,
)
from codeintel.core.columnar.queryspec import ProjectionSpec, QuerySpec


def queryspec_from_filters(
    *,
    filters,
    projection: ProjectionSpec,
    allowed_columns: frozenset[str],
    column_types,
) -> QuerySpec:
    predicates = compile_filter_predicates(filters, allowed_columns=allowed_columns,
                                           column_types=column_types)
    predicate = arrow_filter_expression(predicates)
    return QuerySpec(
        predicate=predicate,
        pushdown_predicate=predicate,
        projection=projection,
    )
```

**Target files**
- src/codeintel/core/queries/filter_compiler.py
- src/codeintel/core/columnar/queryspec.py
- src/codeintel/core/columnar/plan_ops.py
- tests/queries/test_queryspec_filter_compiler.py

**Checklist**
- [ ] Provide QuerySpec helpers that consume filter compiler output.
- [ ] Ensure predicate/projection parity across Arrow/DuckDB/Polars.
- [ ] Add tests validating QuerySpec equivalence across engines.

---

## Sequencing Recommendation
1) Scope 00 (IR v2 + FinalizeResult runner)
2) Scope 01 (contract-driven finalize)
3) Scope 02 (order-independent dedupe)
4) Scope 03 (runtime profiles)
5) Scope 04 (observability)
6) Scope 05 (legacy retirement + guardrails)
7) Scope 06 (filter compiler integration)

## Validation Gates
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted pytest runs for modified modules (columnar, schemas, queries)
- Determinism tests for dedupe and canonical ordering
- Artifact presence tests for tolerant finalize

