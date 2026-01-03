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
Completed:
- Add a shared msgspec JSON helper module.
  - New module: `src/codeintel/core/serialization/msgspec_json.py`.
  - Define deterministic encoder/decoder helpers.
  - Provide helpers:
    - `encode_json_bytes(obj: object) -> bytes`
    - `encode_json_text(obj: object) -> str`
    - `decode_json_bytes(buf: bytes, *, type: object) -> object`
- Update manifest serialization.
  - `src/codeintel/core/manifests.py`:
    - Replace `orjson` usage in manifest read/write helpers.
    - Prefer encoding msgspec Structs directly.
  - Use `msgspec.Struct` options for schema-wire alignment:
    - `kw_only=True`
    - `omit_defaults=True`
    - `forbid_unknown_fields=True` with permissive fallback conversion.
- Update decision trace serialization.
  - `src/codeintel/build/hamilton/decision_trace.py` now uses msgspec helpers.
- Backward compatibility for existing JSON files.
  - Read path tolerates unknown fields via `msgspec.convert` fallback.

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
Completed:
- Add a promotion policy to settings.
  - `BuildSettings` and dataset settings include `schema_promote_options` with
    permissive default.
  - Environment overrides documented alongside existing schema settings.
- Apply promotion to schema unification.
  - `src/codeintel/core/columnar/schema.py` uses `promote_options` for unification.
  - `src/codeintel/core/datasets/scanning.py` passes the promotion option.
- Align contract schema with permissive promotions.
  - `src/codeintel/core/columnar/schema_alignment.py` unifies schemas with
    the promotion policy while retaining explicit cast options.

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
Completed:
- Build a Pandera schema factory.
  - `src/codeintel/core/validation/pandera_schema.py` added.
  - Produces `pandera.polars.DataFrameSchema` with strict mode, uniques, and
    range checks mapped from observations.
- Preserve streaming behavior in storage validation.
  - `src/codeintel/storage/validation/columnar.py` now validates
    `RecordBatchReader` with Pandera when global checks are required.

Outstanding checklist:
- Replace ad-hoc validators with Pandera in Hamilton.
  - [ ] `src/codeintel/build/hamilton/data_quality.py`:
    - [ ] Remove column presence, uniqueness, and range checks implemented
      outside Pandera.
    - [ ] Use a single Pandera-based validator for schema-level and data-level
      validation.
    - [ ] Standardize diagnostics based on `SchemaErrors.failure_cases`.
- Preserve streaming behavior for Hamilton validation.
  - [ ] Provide two validation modes:
    - [ ] Full-table (LazyFrame) for global checks like uniqueness.
    - [ ] Streaming (RecordBatchReader) for per-batch validation when possible.
  - [ ] For streaming:
    - [ ] Convert each batch to Polars DataFrame, validate, accumulate diagnostics.
    - [ ] Provide early-exit on first failure when configured.

### Acceptance criteria
- All uniqueness and range checks run through Pandera.
- Diagnostics are consistent and include failure cases when available.
- Validation can run without full materialization when streaming is required.

## Phase 4: PyArrow-first streaming path
### Goals
- Use `pyarrow.RecordBatchReader` as the primary streaming interface.
- Minimize eager conversion to Polars except when required by Pandera.

### Implementation steps
Completed:
- Update streaming adapters.
  - `src/codeintel/core/columnar/stream.py` keeps `RecordBatchReader` as the
    authoritative stream and uses Arrow datasets for LazyFrame conversion.
- Clarify Polars batch collection types.
  - `src/codeintel/core/columnar/polars_collect.py` returns iterators for
    batch collection and avoids eager list conversion.
- Use Arrow scanners where Polars is not needed.
  - `src/codeintel/build/graphs/engine/datasets.py` applies Arrow-side
    filtering before falling back to Polars expressions.

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

### Pandera rollout for build validation
Goal: roll Pandera validation across `src/codeintel/build` with clear staging,
visibility, and rollback controls.

Rollout stages:
1) Baseline (noop parity)
   - Keep validation enabled behind existing `ci_validate_outputs` gating.
   - Emit Pandera diagnostics on failures without changing existing failure
     behavior (lenient mode logs, strict mode fails).
   - Ensure the new Pandera path is used for all build datasets via
     `PanderaSchemaValidator`.
2) Stage 1 (shadow mode for strict profiles)
   - For profiles `strict` and `data-strict`, run Pandera validation and log
     diagnostics even if a legacy path would have passed.
   - Track failure rates by table key and profile to identify noisy contracts.
3) Stage 2 (enforce Pandera as source of truth)
   - Remove or disable remaining ad-hoc checks in Hamilton (already done).
   - Treat Pandera failures as authoritative for schema/uniqueness/range.
4) Stage 3 (streaming-first)
   - Use streaming validation for `RecordBatchReader` when no global checks
     are required; fall back to full-table validation when primary keys exist.
   - Document which checks require materialization (uniques).
5) Stage 4 (stabilize and monitor)
   - Add error budget dashboards / alerts (if telemetry is enabled).
   - Confirm failure cases include `failure_cases` in diagnostics for triage.

Operational checklist:
- Ensure `pandera_available()` is true in build environments (dependency present).
- Verify `resolve_extras_policy` aligns with build schema expectations.
- Confirm primary key uniqueness checks route through Pandera (global mode).
- Confirm range checks are sourced only from observation stats.
- Validate failure diagnostics include `table_key`, `error`, and `failure_cases`.

Rollback plan:
- Flip `ci_validate_outputs` or set validation profile to `schema-only` to
  bypass Pandera checks while keeping build outputs unblocked.
- Re-enable lenient mode if strict failures are too noisy.

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
