# PyArrow, Polars, Pandera, Msgspec Integration Plan

## Context and goals
- Prefer PyArrow streaming as the default execution and interchange path.
- Use Polars only when we need its execution engine or expression DSL.
- Consolidate validation and diagnostics in Pandera (including uniqueness and range checks).
- Replace orjson and hand-built JSON dicts with msgspec serialization and schemas.
- Allow permissive schema evolution when promotion is safe and easy to correct.

## Scope
- CodeIntel core and build layers, especially:
  - Arrow schema alignment and unification.
  - Dataset scanning and streaming adapters.
  - Data quality validation (Pandera).
  - Manifest and decision trace serialization (msgspec).
- Keep external APIs and storage contracts stable unless explicitly called out below.

## Non-goals
- Rework storage backends or introduce new persistence formats.
- Replace Polars with PyArrow everywhere. Polars remains for lazy graph execution.
- Rebuild CLI behavior or Hamilton orchestration logic beyond the changes below.

## Design decisions
- Schema evolution uses `pa.unify_schemas(..., promote_options="permissive")`.
- Casting stays explicit via `pyarrow.compute.CastOptions` where needed.
- Msgspec is the canonical JSON encoder/decoder; JSON schema is derived from msgspec.
- Pandera is the single source for schema validation diagnostics.
- Streaming remains via `pyarrow.RecordBatchReader` wherever feasible.

## Phase 0: Inventory and baseline
- Map orjson usage and JSON serialization entry points.
  - `src/codeintel/core/manifests.py`
  - `src/codeintel/build/hamilton/decision_trace.py`
- Map schema unification and alignment entry points.
  - `src/codeintel/core/columnar/schema.py`
  - `src/codeintel/core/columnar/schema_alignment.py`
  - `src/codeintel/core/datasets/scanning.py`
- Map validation entry points that must move to Pandera.
  - `src/codeintel/build/hamilton/data_quality.py`
  - Any validators that implement uniqueness or range checks outside Pandera.
- Map streaming adapters and collect paths.
  - `src/codeintel/core/columnar/stream.py`
  - `src/codeintel/core/columnar/polars_collect.py`
  - `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`

## Phase 1: Msgspec-first serialization
### Goals
- Eliminate orjson from manifest and trace writing paths.
- Avoid hand-built JSON dicts that duplicate schema definitions.
- Preserve deterministic output ordering and newline termination.

### Implementation steps
1. Add a shared msgspec JSON helper module.
   - New module: `src/codeintel/core/serialization/msgspec_json.py`.
   - Define `Encoder` and `Decoder` instances with:
     - `order="deterministic"`
     - strict decoding for canonical payloads.
   - Provide helpers:
     - `encode_json_bytes(obj: object) -> bytes`
     - `encode_json_text(obj: object) -> str`
     - `decode_json_bytes(buf: bytes, *, type: object) -> object`

2. Update manifest serialization.
   - `src/codeintel/core/manifests.py`:
     - Replace `orjson` usage in `write_manifest_json`, `_encode_manifest_bytes`,
       and `_encode_manifest_text`.
     - Prefer encoding msgspec Structs directly, not hand-built dicts.
   - Use `msgspec.Struct` options for schema-wire alignment:
     - `kw_only=True` (clarity for optional fields).
     - `omit_defaults=True` (reduce payload size).
     - `forbid_unknown_fields=True` for strict decode (where safe).
   - Use `msgspec.UNSET` for optional fields that should be omitted rather than
     emitted as `null`.

3. Update decision trace serialization.
   - `src/codeintel/build/hamilton/decision_trace.py`:
     - Replace `orjson.dumps` with msgspec encoder.
     - Consider `DecisionTraceRecord` to be the canonical wire object, not
       `DecisionTracePayload` dicts.

4. Backward compatibility for existing JSON files.
   - For readers that ingest persisted JSON, decode to dict and convert to
     msgspec Struct via `msgspec.convert` when needed.
   - Document compatibility expectations and provide a small migration note.

### Acceptance criteria
- No orjson usage in manifest or decision trace write/read paths.
- Msgspec JSON schema for manifest types matches the runtime serialization
  shape (no duplicated dict building).
- All serialized JSON remains deterministic and newline-terminated.

## Phase 2: Permissive schema evolution
### Goals
- Allow safe schema promotion to reduce brittle failures.
- Keep contract alignment strict unless promotion is explicitly allowed.

### Implementation steps
1. Add a promotion policy to settings.
   - Add `schema_promote_options: Literal["default", "permissive"]` to:
     - `BuildSettings`
     - `ArrowDatasetSettings` (if present) or equivalent config layer.
   - Default to `"permissive"` per user guidance.

2. Apply promotion to schema unification.
   - `src/codeintel/core/columnar/schema.py`:
     - Add `promote_options` parameter to `unify_schema_for_batches`.
     - Default to permissive.
   - `src/codeintel/core/datasets/scanning.py`:
     - When `unify_schemas=True`, pass the promotion option.

3. Align contract schema with permissive promotions.
   - `src/codeintel/core/columnar/schema_alignment.py`:
     - Replace the current `promote_options: pc.CastOptions | None` with
       a promotion policy parameter for `pa.unify_schemas`.
     - Retain explicit casting via `pc.cast` and optional `CastOptions`.

### Acceptance criteria
- Schema unification succeeds for typical type widenings (int32->int64,
  float32->float64, decimal precision increases).
- Contract metadata is preserved in unified schemas.
- Promotions are explicit and configurable via settings.

## Phase 3: Pandera consolidation for validation
### Goals
- Use Pandera as the single validation layer for:
  - Column presence and dtype enforcement.
  - Non-nullable constraints.
  - Primary key uniqueness.
  - Range checks derived from observations.
- Standardize diagnostics and failure case reporting.

### Implementation steps
1. Build a Pandera schema factory.
   - Add module: `src/codeintel/core/validation/pandera_schema.py`.
   - Inputs:
     - `TableSchema`
     - Optional observation stats (min/max, etc.)
     - Extras policy (retain/reject/drop)
   - Outputs:
     - `pandera.polars.DataFrameSchema` with:
       - `strict=True` when extras policy is reject.
       - `strict="filter"` when extras policy retains.
       - `unique` for primary keys.
       - `Check.ge`/`Check.le` or `Check.in_range` for min/max.
       - `nullable` mapped from TableSchema.
       - `coerce=False` for strict type enforcement.

2. Replace ad-hoc validators with Pandera.
   - `src/codeintel/build/hamilton/data_quality.py`:
     - Remove column presence, uniqueness, and range checks implemented outside
       Pandera.
     - Use a single Pandera-based validator for schema-level and data-level
       validation.
     - Standardize diagnostics based on `SchemaErrors.failure_cases`.

3. Preserve streaming behavior.
   - Provide two validation modes:
     - Full-table (LazyFrame) for global checks like uniqueness.
     - Streaming (RecordBatchReader) for per-batch validation where global
       checks are not required or are computed separately.
   - For streaming:
     - Convert each batch to Polars DataFrame, validate, accumulate diagnostics.
     - Provide early-exit on first failure when configured.

### Acceptance criteria
- All uniqueness and range checks run through Pandera.
- Diagnostics are consistent and include failure cases when available.
- Validation can run without full materialization when streaming is required.

## Phase 4: PyArrow-first streaming path
### Goals
- Use `pyarrow.RecordBatchReader` as the primary streaming interface.
- Minimize eager conversion to Polars except when required by Pandera.

### Implementation steps
1. Update streaming adapters.
   - `src/codeintel/core/columnar/stream.py`:
     - Keep `RecordBatchReader` as the authoritative streaming type.
     - Ensure LazyFrame streaming uses batch generators without materializing
       the full dataset.

2. Clarify Polars batch collection types.
   - `src/codeintel/core/columnar/polars_collect.py`:
     - Adjust type hints to `Iterable[DataFrame]` where Polars returns a
       generator.
     - Avoid eager list conversion.

3. Use Arrow scanners where Polars is not needed.
   - `src/codeintel/build/graphs/engine/datasets.py`:
     - Keep Arrow `Scanner` paths for read-heavy operations.
     - Use Polars only when downstream logic requires Polars expressions.

### Acceptance criteria
- Streaming flows prefer Arrow readers without full materialization.
- Polars streaming remains available but does not force full collection.

## Phase 5: Tests and quality gates
### Tests to add or update
- Msgspec serialization:
  - Round-trip encode/decode for core manifests.
  - JSON schema generation snapshots (msgspec.json.schema).
- Schema promotion:
  - Unify mismatched schemas with permissive promotion.
  - Align contract with promoted schema.
- Pandera validation:
  - Uniqueness failure cases.
  - Range checks derived from observations.
  - Streaming validation on RecordBatchReader batches.

### Quality gates
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted `pytest -q` for impacted modules, followed by per-directory runs.

## Migration and rollout
- Migrate existing JSON files by reading with msgspec and writing back with the
  new encoder as part of a maintenance task or build step.
- Maintain a temporary compatibility path that accepts old JSON formats if
  necessary (to be removed after a migration window).

## Risks and mitigations
- Pandera validation on streaming data may miss global constraints.
  - Mitigation: split validation into streaming and global phases, and surface
    which checks require global evaluation.
- Msgspec strict decoding may fail on old payloads.
  - Mitigation: use `msgspec.convert` on decoded dicts, or a fallback decoder
    for legacy payloads.
- Permissive schema promotion could mask incorrect types.
  - Mitigation: log promotions and include schema drift summaries in manifests.

## Open questions
- Which validation checks must be global (always full dataset) vs streaming?
- Do we need a formal migration step for persisted JSON or is best-effort
  backwards compatibility acceptable?

