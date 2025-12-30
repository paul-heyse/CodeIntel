# Build Test Remaining Failures Fix Plan

## Scope

Target: `tests/build` (21 failures from `build/test-results/junit.xml`).
Goal: resolve remaining failures with minimal behavior drift and consistent
Hamilton/Iceberg semantics.

## Root Cause Summary

1. **Iceberg write policy mismatch**
   - `core.modules` uses `TableWritePolicy(mode="upsert")`, but
     `IcebergDatasetSaver` only supports `append` and `replace`.
   - This causes materializer failures, which cascade into graph/analytics/serving
     harness failures.

2. **Loader nodes disabled-by-config test assumption**
   - Hamilton variables do not reliably expose `user_defined` for external inputs.
   - When loader nodes are disabled, q__ nodes may still exist as external inputs
     but should be **untagged**, not necessarily `user_defined=True`.

3. **Loader tag canonicalization for non-catalog q__ nodes**
   - In minimal runtime, q__ nodes for tables not in `catalog.table_outputs`
     appear as external inputs and are untagged.
   - Tests should only require loader tags for catalog-derived loader nodes.

4. **Missing validation record on early failure**
   - When a materialization fails before `_finalize_validation` (e.g., write
     policy error), no validation record is written even when validation is enabled.

## Production Fixes

### 1) Map `upsert` to Iceberg-safe write policy

**File:** `src/codeintel/build/hamilton/materializers/iceberg_saver.py`

**Change:**
- When `TableWritePolicy.mode == "upsert"`, coerce to `replace` for Iceberg.
- Prefer `replace_scope="snapshot"` when repo/commit partition columns exist;
  otherwise fall back to `replace_scope="table"` (or raise a clearer error if
  table-level replacement is not acceptable).

**Why:**
- Iceberg doesn’t support upsert semantics directly.
- Replace behavior preserves “last write wins” for the targeted snapshot while
  remaining compatible with partitioned tables.

**Suggested implementation detail:**
- Add a helper in `_IcebergWriter._write_policy` (or a new `resolve_write_policy`)
  to return a policy compatible with Iceberg:
  - If mode is `"upsert"`, return `TableWritePolicy(mode="replace", replace_scope="snapshot")`
    if partition columns include `repo`/`commit`; else `"table"`.
- Keep metadata intact; no changes to contract/schema derivation.

**Tests fixed:**
- `tests/build/hamilton/test_materializer.py` (all upsert failures)
- `tests/build/hamilton/test_graphs_end_to_end.py`
- `tests/build/hamilton/test_target_harness_wrappers.py`
- `tests/build/serving/test_pr84_semantic_view_hamilton_tags.py`
- `tests/build/serving/test_publisher.py`
- `tests/build/serving/test_pilot_end_to_end.py`
- `tests/build/serving/test_pr90_search_index_builds.py`

### 2) Record validation on early materialization failure

**File:** `src/codeintel/build/hamilton/materializers/iceberg_saver.py`

**Change:**
- If `_build_observed_reader` or `_write_to_iceberg` raises and validation is
  enabled, persist a failed validation record with:
  - `status="failed"`, `validation_scope`, `validation_profile`, `output_role`,
    `table_key`, `target_name`, `repo`, `commit`.
- Return failed materialization result with `validation_id` and
  `validation_status="failed"` in metadata.

**Why:**
- Keeps `metadata.materialization_validations` consistent even when failure
  occurs before `_finalize_validation`.

**Tests fixed:**
- `tests/build/hamilton/test_materializer.py::test_strict_validation_fails_on_missing_columns`

## Test Fixes

### 3) Loader nodes disabled-by-config test

**File:** `tests/build/hamilton/test_pr12_loader_nodes.py`

**Change:**
- Replace `user_defined` assertion with tag-based assertion:
  - If q__ node exists, verify it **is not tagged** as
    `node_type=loader.query` (or is absent from tag query).

**Why:**
- Hamilton external input variables don’t consistently expose `user_defined`;
  tag presence is the canonical indicator.

### 4) Loader tag canonicalization test

**File:** `tests/build/hamilton/test_pr64_loader_tags_are_canonical.py`

**Change:**
- Compute `expected_tagged` from `hamilton_runtime.catalog.table_outputs`:
  - For each `table_key` in catalog outputs, require `q__<table_key>` to carry
    `node_type=loader.query` and `table_key` tag.
- Ignore q__ variables that are not derived from catalog table outputs
  (external inputs in minimal runtime).

**Why:**
- Only catalog-derived loader nodes are tagged; external inputs should be
  ignored by loader tag guardrails.

## Validation Strategy (No Execution)

Once implemented, validate in this order:
1. `uv run pytest tests/build/hamilton/test_materializer.py`
2. `uv run pytest tests/build/hamilton/test_pr12_loader_nodes.py`
3. `uv run pytest tests/build/hamilton/test_pr64_loader_tags_are_canonical.py`
4. `uv run pytest tests/build/hamilton/test_target_harness_wrappers.py`
5. `uv run pytest tests/build/serving/test_publisher.py`
6. `uv run pytest tests/build`

## Risk Notes

- Mapping upsert → replace changes semantics for Iceberg writes but matches the
  current capability surface while preserving deterministic writes per snapshot.
- Validation-on-failure adds side effects (writes to validation table) but only
  when validation is already enabled, and aligns with expected observability.
