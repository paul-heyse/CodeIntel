# Arrow-First Cleaning Backend Implementation Plan

## Objective

Enable a true Arrow-first cleaning pipeline by introducing an `"arrow"` backend to
`VariantConfig`, implementing Arrow-native cleaning steps using `pyarrow.compute`,
and selecting those steps via `resolve_from_config` in `pipe_clean_df`. The goal is
to preserve streaming behavior, avoid Polars conversion, and keep existing pipeline
step names stable (no new step nodes beyond the current cleaning stages).

## Scope

- Add `"arrow"` to the `df_backend` variant and validation surface.
- Implement Arrow-native cleaning operations in `tabular_steps.py`.
- Update `pipe_clean_df` to select Arrow steps when `df_backend="arrow"`.
- Maintain current step naming and config-driven gating behavior.

## Non-Goals

- No new `with_columns` Arrow implementation in this pass.
- No change to default backend (`"polars_lazy"` remains default).
- No changes to non-cleaning transforms or table contract policies.

## Current State (Key References)

- `VariantConfig` only allows `"polars_lazy"`:
  `src/codeintel/core/runtime/variants.py`
- Cleaning pipeline uses `pipe_input` + `step` with `tabular_steps`:
  `src/codeintel/build/hamilton/transforms/decorators.py`
- Cleaning functions are Polars-only today:
  `src/codeintel/build/hamilton/transforms/tabular_steps.py`
- Column feature subDAG uses Polars `with_columns` only:
  `src/codeintel/build/hamilton/transforms/with_columns_backend.py`

## Design Decisions

- Keep the existing `pipe_input` step structure and names (`nulls`, `loc_clip`,
  `with_drop_bad_rows`) to preserve step gating tests and DAG stability.
- Arrow backend will operate on `pa.RecordBatchReader` without materializing to
  `pa.Table` unless required by a specific operation.
- For Arrow backend, `with_features` remains no-op unless feature sets are empty;
  invalid combinations are rejected early by config validation.

## Implementation Plan

### Phase 1: VariantConfig Arrow Backend

- [ ] Update `DataFrameBackend` Literal to include `"arrow"`.
- [ ] Add `"arrow"` to `_ALLOWED_BACKENDS`.
- [ ] Ensure `VariantConfig.from_mapping()` accepts `"arrow"`.
- [ ] Add validation to reject `feature_sets` when `df_backend="arrow"` unless
      they are empty.
- [ ] Update docstrings and comments to describe `"arrow"` backend behavior.

Files:
- `src/codeintel/core/runtime/variants.py`

### Phase 2: Arrow-Native Cleaning Steps

Implement Arrow-aware paths inside `tabular_steps.py` without changing public
function names. Each function should:

- Accept `TabularInput` and return the same logical type for Arrow inputs
  (`pa.RecordBatchReader`).
- Preserve streaming by iterating batches from the reader and returning a new
  reader constructed from the transformed batch stream.

Planned helpers (internal, private):

- `_as_arrow_reader(df: TabularInput) -> pa.RecordBatchReader | None`
- `_iter_batches(reader: pa.RecordBatchReader) -> Iterator[pa.RecordBatch]`
- `_filter_batch(batch: pa.RecordBatch, mask: pa.Array) -> pa.RecordBatch`
- `_reorder_batch(batch: pa.RecordBatch, column_order: Sequence[str]) -> pa.RecordBatch`

Planned function behavior:

- `drop_bad_rows`:
  - For Arrow inputs, compute a mask where all `required_cols` are non-null,
    using `pyarrow.compute.is_null` + boolean reductions.
  - Filter each batch with `pyarrow.compute.filter`.
- `normalize_nulls`:
  - `"preserve"`: no-op.
  - `"drop_bad_rows"`: apply a full-row non-null mask across all columns.
- `clip_numeric`:
  - For Arrow inputs, clip the target column using element-wise compute
    (`pc.if_else` or `pc.min_element_wise` with a scalar).
- `sort_columns`:
  - For Arrow inputs, reorder batch columns in the requested order.

Files:
- `src/codeintel/build/hamilton/transforms/tabular_steps.py`

### Phase 3: pipe_clean_df Backend Selection

Extend `_pipe_cleaning` in `decorators.py` to select Arrow steps when
`df_backend="arrow"`:

- Reuse the same step names and ordering:
  - `with_drop_bad_rows` (strict only)
  - `nulls`
  - `loc_clip` (if configured)
- For Arrow backend, these steps call the Arrow-capable functions from
  `tabular_steps.py` (same function names, Arrow path internally).

Files:
- `src/codeintel/build/hamilton/transforms/decorators.py`

### Phase 4: Feature-Set Guardrails (Arrow Backend)

Prevent unsupported combinations early:

- Add a check in `VariantConfig.validate()` that rejects non-empty `feature_sets`
  when `df_backend="arrow"`.
- Ensure `with_features()` returns a no-op when `feature_sets` is empty, without
  calling `select_with_columns`.

Files:
- `src/codeintel/core/runtime/variants.py`
- `src/codeintel/build/hamilton/transforms/decorators.py`

### Phase 5: Tests

Add and update tests to validate Arrow behavior:

- [ ] New unit tests for Arrow cleaning functions:
  - `drop_bad_rows` on `pa.RecordBatchReader`
  - `normalize_nulls` with `"drop_bad_rows"` on Arrow input
  - `clip_numeric` on Arrow input
  - `sort_columns` on Arrow input
- [ ] Update variant config tests to accept `"arrow"` and reject features.
- [ ] Add a small driver build test with `df_backend="arrow"` to verify the
  cleaning steps appear when `clean_mode!="off"` and do not error for Arrow
  input types.

Candidate files:
- `tests/variants/test_pipe_input_step_gating.py` (extend)
- New tests under `tests/build/hamilton/transforms/` or `tests/build/tabular/`

### Phase 6: Validation Gates

- [ ] Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- [ ] Run targeted tests for new/updated files.

## Acceptance Criteria

- `df_backend="arrow"` is accepted by `VariantConfig` and validated.
- Cleaning pipeline operates on `pa.RecordBatchReader` without converting to
  Polars.
- Existing step names remain unchanged; step gating tests still pass.
- No new polars conversions introduced for Arrow backend paths.

## Risks & Mitigations

- **Streaming semantics:** `RecordBatchReader` is single-consume.
  - Mitigation: process strictly in a single pass and return a new reader.
- **Arrow compute API availability:** some element-wise helpers may differ by
  version.
  - Mitigation: prefer broadly available `pc.if_else` and `pc.cast` patterns.
- **Feature sets with Arrow backend:** `with_columns` is Polars-only today.
  - Mitigation: explicit validation error when features are configured for Arrow.

## Rollout Plan

1. Land the backend + cleaning changes behind `df_backend="arrow"`.
2. Enable Arrow backend in a limited CLI profile or test configuration.
3. Expand to additional targets after validation stability.
