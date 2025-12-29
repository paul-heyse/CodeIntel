# Columnar Refactor Follow-on Implementation Plan

## Intent

Continue the columnar execution refactor by finishing the remaining core
workstreams:

- Row model typing + columnar validation refactor to consume the
  `ColumnTypeRegistry` and clear complexity warnings.
- Polars streaming adapter refactor to use a typed execution control plane.
- DuckDB expression builder + observation payload typing for schema metadata.

This plan aligns with `docs/best_in_class_columnar_execution_plan.md` and is
structured to remove the remaining Ruff/Pyright/Pyrefly errors while improving
maintainability.

## Scope

### In scope

- Refactor `row_models.py` and `columnar.py` to consume the registry and replace
  large conditional blocks with small dispatch tables.
- Introduce a typed Polars execution options adapter for streaming collection.
- Replace ad-hoc DuckDB expression typing with a small builder layer and typed
  schema observation payloads.

### Out of scope

- Adding new external engines or runtime backends.
- Rewriting Hamilton DAG logic or schema inference beyond the stated refactors.

## Dependencies

- `ColumnTypeRegistry` already lives in `src/codeintel/core/schemas/primitives.py`.
- Arrow SQLGlot mapping now uses safe type resolution in
  `src/codeintel/core/schemas/arrow_gen.py`.

## Workstream A: Row Model Typing + Columnar Validation Refactor

### Objectives

- Use the registry for Python type resolution in row models.
- Replace complex type-compatibility logic with a composable mapping.
- Clear Ruff complexity warnings in `row_models.py` and `columnar.py`.

### Files to touch

- `src/codeintel/core/schemas/row_models.py`
- `src/codeintel/storage/validation/columnar.py`
- `src/codeintel/core/schemas/primitives.py` (registry consumption only)
- `tests/storage/validation/*` (new tests)

### Design

- Replace `_python_type_for_column_type` with registry calls:
  `COLUMN_TYPE_REGISTRY.python_type_for(...)`.
- Extract Arrow compatibility logic into a new helper module:
  `src/codeintel/storage/validation/arrow_type_compat.py`.
- In `columnar.py`, keep `_is_compatible_type` as a thin wrapper that delegates
  to the new module.

### Implementation steps

1. Row model registry adoption
   - Import `COLUMN_TYPE_REGISTRY` in `row_models.py`.
   - Replace `_python_type_for_column_type` with a small wrapper that calls
     `COLUMN_TYPE_REGISTRY.python_type_for(...)`.
   - Ensure decimal handling remains identical to the current logic.
2. Arrow compatibility helpers
   - Add `arrow_type_compat.py` with:
     - `is_compatible_arrow_type(column: Column, actual: pa.DataType) -> bool`
     - `is_list_like(dtype: pa.DataType) -> bool`
     - `decimal_scale_zero(column_type: str) -> bool`
   - Use a `dict[str, Callable[[pa.DataType], bool]]` keyed by base type to
     replace nested `if` blocks.
3. Columnar validation integration
   - Update `columnar.py` to call the new compatibility helpers.
   - Keep the public validation API unchanged.
4. Tests
   - Add targeted tests in `tests/storage/validation/` for:
     - Decimal scale=0 int compatibility.
     - Dictionary encoded string columns.
     - LIST/MAP/STRUCT/UNION compatibility.

### Acceptance criteria

- `row_models.py` and `columnar.py` have no Ruff complexity warnings.
- Registry is the only source of Python type mapping logic.
- Validation remains backward compatible for existing types.

## Workstream B: Polars Streaming Adapter Refactor

### Objectives

- Provide a typed execution control plane for Polars streaming collection.
- Remove `dict[str, object]` kwargs and signature introspection in `stream.py`.

### Files to touch

- `src/codeintel/core/columnar/stream.py`
- `src/codeintel/core/columnar/polars_collect.py` (new)
- `tests/_helpers/columnar_streams.py`
- `tests/serving/*` (streaming regression tests if needed)

### Design

- Add a `PolarsExecutionOptions` dataclass with:
  - `streaming: bool`
  - `query_opt_flags: object | None`
  - `inspect: bool`
  - `streaming_fallback: bool`
- Add a `PolarsCollectAdapter` that:
  - Maps options to `LazyFrame.collect` and `LazyFrame.collect_batches`.
  - Encodes allowed kwargs using `TypedDict` for type safety.
  - Uses explicit feature detection for supported parameter names.

### Implementation steps

1. Adapter module
   - Implement `polars_collect.py` with the typed adapter and helpers.
   - Include a single `collect_batches(...)` and `collect(...)` entry point to
     keep `stream.py` minimal.
2. Stream integration
   - Replace `_collect_kwargs`, `_collect_batch_kwargs`, and `_signature` in
     `stream.py` with the adapter.
   - Preserve streaming fallback behavior.
3. Tests
   - Update helpers in `tests/_helpers/columnar_streams.py` to cover
     streaming/fallback paths.
   - Ensure current streaming IPC tests still pass.

### Acceptance criteria

- `stream.py` no longer has complexity warnings.
- Pyright/Pyrefly no longer report `dict[str, object]` kwargs errors.
- Streaming fallback behavior remains intact and observable.

## Workstream C: DuckDB Expression Builder + Observation Payload Typing

### Objectives

- Replace ad-hoc typing in `expressions.py` with a small builder layer.
- Formalize schema observation payload types to fix `parquet_stats` errors.

### Files to touch

- `src/codeintel/storage/duckdb_types.py`
- `src/codeintel/storage/queries/expressions.py`
- `src/codeintel/storage/warehouse.py`
- `src/codeintel/storage/tracking/schema_catalog_models.py`
- `src/codeintel/build/schemas/observations.py`
- `tests/storage/tracking/*` (schema payload tests)

### Design

- Add `ExpressionBuilder` with explicit return types:
  - `col(name: str) -> Expression`
  - `lit(value: object) -> Expression`
  - `eq(column: str, value: object) -> Expression`
  - `and_all(expressions: Iterable[Expression]) -> Expression`
- Update `duckdb_types.py` to export `ExpressionFactory` types for clarity.
- Replace `ColumnExpression` and `ConstantExpression` use in
  `expressions.py` with the builder to align with DuckDB type stubs.
- Add typed payloads:
  - `ColumnStatsPayload`
  - `DatasetStatsPayload`
  - `ParquetStatsPayload`
  - `DerivedSettingsPayload`

### Implementation steps

1. Expression builder
   - Create a small builder object (module-level functions or dataclass).
   - Update `expressions.py` to delegate to it and expand docstrings to satisfy
     DOC201/DOC501.
   - Update `warehouse.py` call sites if needed.
2. Observation payload typing
   - Add TypedDicts in `schema_catalog_models.py` and update
     `SchemaObservationRecord` fields to use them.
   - Update `observations.py` to build payloads that match the types.
3. Tests
   - Add payload shape tests under `tests/storage/tracking/`.
   - Add a minimal expression builder test if existing coverage is missing.

### Acceptance criteria

- Pyright/Pyrefly errors in `expressions.py` and `observations.py` are cleared.
- Schema observation payload fields are typed and stable.

## Sequencing

1. Workstream A (row models + validation) to reduce core complexity first.
2. Workstream B (Polars streaming adapter) to stabilize runtime options.
3. Workstream C (DuckDB builder + payload typing) to finalize storage typing.

## Quality gates

- `uv run ruff check <touched files>`
- `uv run pyright --warnings --pythonversion=3.13 <touched files>`
- `uv run pyrefly check <touched files>`
- Targeted tests:
  - `uv run pytest -q tests/storage/validation`
  - `uv run pytest -q tests/storage/tracking`
  - `uv run pytest -q tests/_helpers/columnar_streams.py`

## Risks and mitigations

- Risk: Arrow compatibility rules diverge from previous behavior.
  - Mitigation: add regression tests for existing compatibility cases.
- Risk: Polars option mapping drifts across versions.
  - Mitigation: keep feature detection in adapter and log fallback usage.
- Risk: Typed payloads block future metadata expansion.
  - Mitigation: use TypedDict `total=False` for optional keys.

## Deliverables checklist

- [ ] Registry-backed row model typing + reduced complexity in
      `src/codeintel/core/schemas/row_models.py`.
- [ ] Arrow compatibility helpers and reduced complexity in
      `src/codeintel/storage/validation/columnar.py`.
- [ ] Typed Polars collection adapter and simplified `stream.py`.
- [ ] DuckDB expression builder and typed observation payloads.
- [ ] Targeted tests added for each workstream.
