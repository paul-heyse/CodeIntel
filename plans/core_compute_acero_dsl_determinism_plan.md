# Core Compute Acero + DSL Determinism Plan

## Decision record (approved)
- Keep `build` and `core` compute layers separate for now.
- Standardize determinism tiers for dedupe and finalize behavior across the core layer.
- Use Acero where possible; fallback to table‑based kernels only when necessary.
- Centralize compute idioms in a single core kernel surface and limit direct `pc.*` usage.

## Determinism tiers (policy to enforce)

### Tier 0: Canonical (deterministic outputs)
- Required for persisted datasets, exports, validation artifacts, cache boundaries, and tests.
- Winner selection must be order‑independent by default.
- If `first/last/list` is used, enforce stable ordering first and run aggregation with
  `use_threads=False`.
- Must enforce deterministic output ordering at finalize boundaries (stable sort keys or
  explicit order_by).
- Must reject `hash_one`/arbitrary aggregations.

### Tier 1: Throughput (fast path)
- Used for intermediate/ephemeral tables where ordering is not a contract requirement.
- Dedupe only when correctness requires it.
- Prefer order‑independent winner selection to keep the set stable while allowing parallelism.
- Sorting is optional and only required if downstream semantics need it.

## Implementation plan

### 1) Single execution surface (plan → execute → finalize)

**Goal**: Centralize plan construction and execution, with a single entrypoint that applies
determinism + finalize policies.

Target files
- `src/codeintel/core/columnar/plan_ops.py` (augment Plan or add execution wrapper)
- `src/codeintel/core/columnar/finalize_ops.py` (ensure finalize integrates determinism policy)
- Optional new module: `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/__init__.py` (exports)

Code patterns
```python
from dataclasses import dataclass
from typing import Callable, Union

import pyarrow as pa
from pyarrow import acero

TableThunk = Callable[[], pa.Table]


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    use_threads: bool
    determinism: str  # "canonical" | "throughput"


@dataclass(frozen=True, slots=True)
class ExecPlan:
    inner: Union[acero.Declaration, TableThunk]

    def execute(self, *, ctx: ExecutionContext) -> pa.Table:
        if isinstance(self.inner, acero.Declaration):
            return self.inner.to_table(use_threads=ctx.use_threads)
        return self.inner()


def run_pipeline(
    *,
    plan: ExecPlan,
    post: list[Callable[[pa.Table], pa.Table]],
    finalize: Callable[[pa.Table], pa.Table],
    ctx: ExecutionContext,
) -> pa.Table:
    table = plan.execute(ctx=ctx)
    for fn in post:
        table = fn(table)
    return finalize(table)
```

Checklist
- [ ] Add `ExecutionContext` with determinism tier and threading knobs.
- [ ] Add an `ExecPlan` wrapper (Declaration or table thunk).
- [ ] Provide a single `run_pipeline` entrypoint used by core pipelines.
- [ ] Ensure finalize is the only boundary that enforces schema + dedupe + ordering.

---

### 2) Consolidate compute helpers into a core kernel surface

**Goal**: Make `src/codeintel/core/columnar/kernels.py` the single approved kernel surface and
limit direct `pc.*` usage outside core columnar modules.

Target files
- `src/codeintel/core/columnar/kernels.py` (expand surface)
- `src/codeintel/core/columnar/__init__.py` (reexports)
- `src/codeintel/core/columnar/masks.py` (thin wrappers if needed)
- `src/codeintel/core/columnar/compute.py` (fold into kernels or reexport)
- `src/codeintel/build/tabular/*` (thin‑wrap to core equivalents, keep build layer separate)

Code patterns
```python
def stable_sort_table(table: pa.Table, *, sort_keys: Sequence[SortKey]) -> pa.Table:
    indices = stable_sort_indices(table, sort_keys=sort_keys)
    return table.take(indices)


def safe_divide(
    numerator: pa.Array | pa.ChunkedArray,
    denominator: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    zero = pc.equal(denominator, pa.scalar(0))
    safe = pc.if_else(zero, pa.scalar(None), pc.divide(numerator, denominator))
    return safe
```

Checklist
- [ ] Add missing kernels (regex match/replace, list_len, struct_field, safe_divide).
- [ ] Reexport kernels from `src/codeintel/core/columnar/__init__.py`.
- [ ] Replace ad‑hoc `pc.*` in core compute paths with kernel helpers.
- [ ] Add build tabular wrappers that forward to core kernels (no behavior changes).

---

### 3) Deep‑cast + promotion policy inside finalize

**Goal**: Ensure finalize enforces nested schema evolution with explicit promotion and list_view
normalization.

Target files
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/core/columnar/nested_ops.py`
- `src/codeintel/core/columnar/type_normalization.py`
- Optional: `src/codeintel/core/columnar/schema_alignment.py`

Code patterns
```python
from codeintel.core.columnar.nested_ops import (
    deep_cast_table_to_contract,
    unify_schemas_with_contract_first,
)
from codeintel.core.columnar.type_normalization import normalize_string_view_schema

def _align_for_finalize(table: pa.Table, contract_schema: pa.Schema) -> pa.Table:
    unified = unify_schemas_with_contract_first(contract_schema, [table.schema])
    normalized = normalize_string_view_schema(unified)
    return deep_cast_table_to_contract(table, normalized)
```

Checklist
- [ ] Normalize list_view/binary_view/string_view at finalize boundary.
- [ ] Apply `unify_schemas_with_contract_first` before casting.
- [ ] Use `deep_cast_table_to_contract` for nested list/struct/map casting.
- [ ] Emit alignment artifacts when nested promotions occur.

---

### 4) Dedupe policy: spec‑driven, tier‑aware

**Goal**: Replace `drop_duplicates` defaults with a spec that encodes keys, tie‑breakers, and
determinism tier.

Target files
- `src/codeintel/core/columnar/dedupe_ops.py`
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/core/schemas/primitives.py` (TableSchema / TableWritePolicy extensions)

Code patterns
```python
@dataclass(frozen=True, slots=True)
class DedupeSpec:
    keys: tuple[str, ...]
    tie_breakers: tuple[tuple[str, str], ...] = ()
    tier: Literal["canonical", "throughput"] = "canonical"
    strategy: Literal["order_independent", "first"] = "order_independent"


def dedupe_table(table: pa.Table, *, spec: DedupeSpec) -> pa.Table:
    if spec.strategy == "order_independent":
        # choose winner by min/max of a stable score column
        return _dedupe_with_min_score(table, keys=spec.keys, tie_breakers=spec.tie_breakers)
    sorted_table = stable_sort_table(table, sort_keys=spec.tie_breakers)
    return _dedupe_keep_first(sorted_table, keys=spec.keys, use_threads=False)
```

Checklist
- [ ] Define `DedupeSpec` with keys, tie‑breakers, tier, and strategy.
- [ ] Canonical tier: enforce order‑independent winner selection by default.
- [ ] Canonical tier: require deterministic output ordering at finalize boundary.
- [ ] Throughput tier: allow non‑ordered dedupe only when correctness requires it.
- [ ] Reject `hash_one`/arbitrary aggregations in canonical tier.

---

### 5) Join‑safety helpers + post‑join ordering

**Goal**: Encode join safety (no list payloads in hash joins) and deterministic ordering policy.

Target files
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/kernels.py`
- `src/codeintel/core/columnar/expr_vocab.py`

Code patterns
```python
def ensure_join_safe(table: pa.Table, *, allowed_columns: Sequence[str]) -> pa.Table:
    list_cols = [name for name in table.column_names if is_list_like(table[name].type)]
    if list_cols:
        return table.select([name for name in allowed_columns if name in table.column_names])
    return table


def canonical_post_join_order(
    table: pa.Table,
    *,
    sort_keys: Sequence[tuple[str, str]],
) -> pa.Table:
    return stable_sort_table(table, sort_keys=sort_keys)
```

Checklist
- [ ] Add join‑safety validation (list payload detection).
- [ ] Provide a helper that drops/forbids list payloads before join.
- [ ] Canonical tier: enforce post‑join stable sort.
- [ ] Throughput tier: allow unordered join output when allowed by contract.

---

### 6) Scan telemetry + provenance into finalize artifacts

**Goal**: Make provenance columns and scan telemetry visible in error artifacts and logs.

Target files
- `src/codeintel/core/datasets/scanning.py`
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/serving/http/export_dispatch.py`
- `src/codeintel/serving/http/streaming.py`

Code patterns
```python
# Scan: always include provenance columns when metrics_enabled
options = ParquetScanOptions(
    columns=columns,
    provenance_columns=("__filename", "__fragment_index", "__batch_index"),
    metrics_enabled=True,
)

# Finalize: include provenance in error tables
FinalizeSpec(
    table_key=table_key,
    mode="tolerant",
    context_fields=("__filename", "__fragment_index"),
    emit_artifacts=True,
)
```

Checklist
- [ ] Ensure scan options include provenance columns when metrics are enabled.
- [ ] Thread provenance through `FinalizeSpec.context_fields`.
- [ ] Log provenance in finalize error summaries.
- [ ] Include scan telemetry in export logs for reproducible debugging.

---

## Migration checklist (core only, build stays separate)
- [ ] Add or augment `arrowdsl.py`/`plan_ops.py` with `ExecutionContext` and `ExecPlan`.
- [ ] Expand `kernels.py` surface and reexport from `core/columnar/__init__.py`.
- [ ] Replace direct `pc.*` in core compute modules with kernel helpers.
- [ ] Wire deep‑cast + schema unification into finalize.
- [ ] Implement `DedupeSpec` + tier‑aware dedupe in `dedupe_ops.py`.
- [ ] Add join‑safety helpers and canonical post‑join ordering.
- [ ] Thread scan telemetry + provenance into finalize artifacts and export logging.

## Validation checklist
- [ ] Add unit tests for canonical vs throughput dedupe behavior.
- [ ] Add unit tests for list payload join safety and enforced ordering.
- [ ] Add unit tests for deep‑cast with nested list/struct/map changes.
- [ ] Add integration tests for finalize artifacts including provenance fields.
- [ ] Verify canonical tier outputs are byte‑stable across runs.
