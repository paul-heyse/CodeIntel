# Analytics ↔ Core Columnar Alignment Implementation Plan

## Goal
Align the analytics Acero/DSL footprint with the unified core columnar target state:
contract-driven finalize, reader-first execution, deterministic ordering, and
first-class observability artifacts.

## Scope Items

### 1) Contract-driven finalize spec for analytics

Pattern
```python
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_table


def finalize_analytics_result(table_key: str, table: pa.Table) -> FinalizeResult:
    spec = FinalizeSpec(
        table_key=table_key,
        mode="tolerant",
        emit_artifacts=True,
        target_name=None,
    )
    # finalize_table resolves schema finalize_policy + canonical ordering.
    return finalize_table(table, spec=spec)
```

Target files
- src/codeintel/build/analytics/utilities/finalize.py
- src/codeintel/build/analytics/utilities/datasets.py
- src/codeintel/build/hamilton/post_run_quality_outputs.py
- src/codeintel/core/columnar/finalize_ops.py (policy enforcement parity)

Checklist
- [x] Remove explicit `resolve_stable_sort_keys` usage from analytics finalize helpers.
- [x] Ensure finalize emits artifacts (`emit_artifacts=True`) for analytics outputs.
- [x] Route analytics writers through the updated finalize helper.

---

### 2) Persist finalize artifacts + run manifests for analytics outputs

Pattern
```python
result = finalize_analytics_result(table_key, table)
write_dataset(..., data=result.good, ...)
write_dataset(..., data=result.errors, table_key=f"{table_key}__errors", ...)
write_dataset(..., data=result.alignment, table_key=f"{table_key}__alignment", ...)
write_dataset(..., data=result.stats, table_key=f"{table_key}__stats", ...)
```

Target files
- src/codeintel/build/analytics/utilities/datasets.py
- src/codeintel/build/hamilton/post_run_quality_outputs.py
- src/codeintel/core/columnar/run_manifest.py
- src/codeintel/core/columnar/streaming.py

Checklist
- [x] Persist `FinalizeResult` artifacts for analytics datasets.
- [x] Attach scan telemetry and runtime profile info to manifests.
- [x] Ensure post-run analytics writes include artifact datasets or manifest records.

---

### 3) Reader-first analytics pipelines (streaming finalize)

Pattern
```python
plan = snapshot_plan(table, repo=repo, commit=commit, columns=columns)
reader = plan.to_reader(use_threads=True)
result = finalize_analytics_reader(table_key, reader)
```

Target files
- src/codeintel/build/analytics/utilities/snapshot.py
- src/codeintel/build/analytics/utilities/datasets.py
- src/codeintel/build/hamilton/post_run_quality_outputs.py
- src/codeintel/core/columnar/plan_ops.py

Checklist
- [x] Replace materialization-first flows with `to_reader` + `finalize_reader`.
- [x] Keep list-decoding at the final boundary only.
- [x] Ensure ordering is applied at the boundary when required (not for determinism).

---

### 4) Canonical ordering + order-independent dedupe adoption

Pattern
```python
spec = FinalizeSpec(
    table_key=table_key,
    mode="tolerant",
    emit_artifacts=True,
)
result = finalize_table(table, spec=spec)
```

Target files
- src/codeintel/build/analytics/semantic_roles/core.py
- src/codeintel/build/analytics/functions/function_contracts.py
- src/codeintel/build/analytics/subsystems/cache.py
- src/codeintel/core/columnar/dedupe_ops.py

Checklist
- [x] Remove ad-hoc `dedupe_keep_first_after_sort` from analytics paths (semantic_roles/core.py, function_contracts.py).
- [x] Rely on schema finalize policy for dedupe + canonical ordering (semantic_roles*, function_contracts, subsystem_profile_cache).
- [x] Ensure `stable_sort_keys=()` semantics are enforced (error/downgrade).
- [x] Extend canonical ordering cleanup to remaining analytics producers that still order solely for determinism.

---

### 5) Analytics schema finalize_policy coverage

Pattern
```python
TableSchema(
    schema="analytics",
    name="py_cpg_quality_report",
    columns=[...],
    primary_key=("repo", "commit", "run_id"),
    finalize_policy=FinalizePolicy(
        required_non_null=("repo", "commit", "run_id"),
        canonical_sort_keys=("repo", "commit", "run_id"),
        dedupe=FinalizeDedupeSpec(
            keys=("repo", "commit", "run_id"),
            prefer_columns=("created_at",),
        ),
    ),
)
```

Target files
- src/codeintel/core/schemas/table_registry.py
- src/codeintel/core/schemas/contract_serde.py
- src/codeintel/core/schemas/serde.py
- src/codeintel/core/schemas/primitives.py

Checklist
- [x] Add finalize_policy for analytics tables (function_contracts, semantic_roles*, subsystem_profile_cache).
- [x] Serialize/deserialize finalize_policy consistently across schema sources (serde + contract_serde already support finalize_policy).
- [x] Ensure `canonical_sort_keys` is defined where determinism is required for the targeted tables.
- [x] Apply default finalize_policy for analytics tables with primary keys (output_registry).
- [x] Add explicit primary_key/finalize_policy for analytics tables without primary keys (analytics.function_types, analytics.hello_example).

---

### 6) Guardrail convergence with core columnar standards

Pattern
```python
if "iter_rows(" in text and "analytics" in path:
    raise SystemExit("Use Plan.aggregate or allowlist AST/graph boundaries.")
```

Target files
- tools/lint_analytics_iter_rows.py
- tools/lint_analytics_rowset_guardrails.py
- tools/lint_no_raw_pyarrow_compute_in_nodes.py
- tools/lint_no_materialize_in_nodes.py
- tools/quality_report.py

Checklist
- [ ] Keep analytics guardrails aligned with core "no raw pc" and "no materialize" rules.
- [ ] Extend guardrails to flag `to_table` outside finalize for analytics.
- [ ] Ensure quality report includes all analytics guardrails.

---

## Sequencing Recommendation
1) Contract-driven finalize spec + artifacts (Scopes 1–2). ✅
2) Reader-first execution upgrades (Scope 3). ✅ (core helpers + post-run scans)
3) Ordering/dedupe alignment and schema policies (Scopes 4–5). ✅ (canonical ordering cleanup + finalize_policy coverage).
4) Guardrail convergence (Scope 6).
