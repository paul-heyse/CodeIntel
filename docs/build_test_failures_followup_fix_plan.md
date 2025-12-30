# Build Test Failures Followup Fix Plan

## Goal
Resolve the remaining `tests/build` failures by addressing four root causes (SQL placeholder rendering, SchemaService lifecycle, Arrow min/max on list types, and compute `and_`/`or_` lookup). The harness failures should clear once these issues are fixed.

## Scope
- DuckDB SQL generation for parameterized queries.
- SchemaService lifecycle and connection safety during gateway open/close.
- Arrow schema observation stats for list/nested types.
- PyArrow compute function name normalization for `and_`/`or_`.
- Follow-on validation for build/serving harness tests.

## Non-Goals
- Refactors unrelated to the failing tests.
- Changing build/serving feature behavior beyond the described fixes.
- Running full test suites as part of this plan (tests are listed as follow-up actions).

## Plan

### 1) Fix SQL placeholder rendering for DuckDB (Critical)
**Problem**: `exp.Parameter()` renders as `$` in DuckDB SQL, which is not valid for positional bindings.  
**Approach**: Replace `exp.Parameter()` with `exp.Placeholder()` in all DuckDB-bound SQL builders that expect positional bindings.

**Implementation steps**
1. Replace `exp.Parameter()` with `exp.Placeholder()` in:
   - `src/codeintel/storage/schema/registry_provider.py`
   - `src/codeintel/storage/schema/arrow_contracts.py` (both contract schema and observation queries)
   - `src/codeintel/storage/metadata/sync.py`
   - `src/codeintel/analytics/utilities/datasets.py`
2. Keep SQL output stable except for the placeholder token (`$` → `?`).

**Acceptance**
- `test_cli_snapshot[pr78_build_validate_auto]` no longer fails with DuckDB parser error.
- Any CLI snapshot change is limited to rendered SQL if the output includes parameterized SQL.

---

### 2) Ensure SchemaService never uses a closed DuckDB connection (High)
**Problem**: `open_gateway()` calls `_schema_service_mismatches()` before rebinding the schema service to the new DuckDB connection. If the global SchemaService still points at a closed connection, mismatches trigger a connection error.  
**Approach**: Clear or rebind the SchemaService before calling `_schema_service_mismatches()`.

**Implementation steps**
1. Add a defensive guard in `src/codeintel/storage/gateway/factory.py`:
   - Before `_schema_service_mismatches()`, attempt a lightweight provider check; if it throws `duckdb.Error`, call `clear_schema_service()` and proceed.
2. Optionally, clear SchemaService on gateway close if the provider is a `RegistrySchemaProvider` bound to the gateway connection.

**Decision criteria**
- Prefer the minimal change that preserves behavior: clear on mismatch failure inside `open_gateway()` and allow `_maybe_set_schema_service_from_catalog()` to rebuild.
- Only add close-hook clearing if the first fix is insufficient.

**Acceptance**
- Gateway setup failures in `tests/build/hamilton/test_materializer.py`, `tests/build/hamilton/test_pr21_analytics_native_impl_kind.py`, and `tests/build/hamilton/test_pr28_phase4_asset_catalog.py` no longer occur.

---

### 3) Skip min/max for list/nested columns during schema observation (High)
**Problem**: Arrow `min`/`max` kernels are not defined for list/struct/map types; schema observation tries to compute them anyway and fails.  
**Approach**: Guard min/max and distinct stats for nested types; expand exception handling to include `pa.ArrowNotImplementedError`.

**Implementation steps**
1. In `src/codeintel/build/schemas/observations.py`:
   - Add a type guard in `_min_max()` to early-return `(None, None)` for list/struct/map/union types.
   - Update `_compute_scalar`, `_count_distinct`, and `_length_stats` to catch `pa.ArrowNotImplementedError` (and `pa.ArrowNotImplemented` if present).

**Acceptance**
- Materializer tests that previously failed with `Function 'min_max' has no kernel matching input types (list<...>)` now succeed.
- Observation stats remain unchanged for supported scalar types.

---

### 4) Normalize `and_`/`or_` compute lookups (Medium)
**Problem**: In some paths, compute lookup by name for `and_`/`or_` fails because the function registry uses `and`/`or`.  
**Approach**: Normalize compute function names before lookup (map `and_ → and`, `or_ → or`), and optionally fall back to `and_kleene`/`or_kleene` if needed.

**Implementation steps**
1. In `src/codeintel/build/exports/validation.py`:
   - Normalize names in `_binary_compute` before `getattr(pc, name, None)`.
2. In `src/codeintel/build/hamilton/materializers/iceberg_saver.py`:
   - Normalize in `_resolve_compute_fn` (or specifically in `_compute_and`).

**Acceptance**
- `test_internal_outputs_skip_contract_checks` no longer fails with “No function registered with name: and_”.

---

### 5) Validate cascade failures clear
**Problem**: Target harness tests fail because materialization errors propagate.  
**Approach**: Once fixes above are in place, re-run the harness subset to confirm stability.

**Suggested test subsets**
1. `uv run pytest tests/build/hamilton/test_materializer.py`
2. `uv run pytest tests/build/hamilton/test_target_harness_wrappers.py`
3. `uv run pytest tests/build/serving`
4. `uv run pytest tests/build/hamilton/test_graphs_end_to_end.py`

**Acceptance**
- Harness tests no longer fail with target_status "failed" for the previous error set.

## Notes
- The changes are safe under `pytest-xdist` because each worker has its own process, but the SchemaService guard must still tolerate in-process gateway reuse.
- If any CLI snapshots change due to SQL placeholder updates, regenerate snapshots only for affected cases.

## Rollback Plan
- All changes are localized. Revert by restoring the previous SQL placeholder usage and removing new guards in SchemaService and observation functions.
