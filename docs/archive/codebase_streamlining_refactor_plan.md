# Codebase Streamlining Refactor Plan

## Goal
Consolidate shared functionality across `src/codeintel` to reduce duplication, harden behavior,
and make new functionality easier to add. The focus is on streamlining data flows (Arrow/Polars/
DuckDB), schema contracts, materialization, and export/observation pipelines without reducing
capability.

## Scope Overview
- **Tabular adapter unification** (Arrow/Polars/DuckDB/streaming inputs).
- **Schema contract translation** (TableSchema ⇄ Arrow schema ⇄ JSON schema/IPC).
- **Materialization pipeline base** (shared saver logic).
- **Observation payload codec** (typed encode/decode at one layer).
- **Manifest I/O consolidation** (dataset + schema manifests).
- **Tool-target scaffolding normalization** (Hamilton target templates).
- **Export/serialization codecs** (NDJSON and other export formats).
- **Configuration access hardening** (typed settings view).

## Sequencing Plan (Phased, Top-to-Bottom)

### Phase 0 — Inventory + Boundaries (no behavior change)
**Goal**: map current responsibilities to new module boundaries and identify all duplicate logic.

Checklist:
- Build a short inventory doc of each pipeline step (input normalization, schema alignment,
  materialization, observation, export) and its current file location.
- Add a “boundary ownership” note for each pipeline stage (one module owns each stage).
- Identify all Arrow/Polars/DuckDB conversions and list call sites.

---

### Phase 1 — Tabular Adapter Unification
**New API boundary**: `src/codeintel/core/columnar/tabular_adapter.py`

Responsibilities:
- Normalize inputs: `TabularInput` → `LazyFrame | RecordBatchReader | DuckDBRelation`.
- Provide streaming materialization utilities.
- Provide stable conversion surface: `to_lazyframe`, `to_record_batch_reader`, `to_relation`.

Exact file moves / consolidations:
- Move conversion helpers from:
  - `src/codeintel/core/columnar/stream.py`
  - `src/codeintel/core/columnar/polars_collect.py`
  - `src/codeintel/build/tabular/duckdb_relation.py`
  - `src/codeintel/storage/warehouse.py` (conversion helpers only)
- Introduce `TabularAdapter` functions in the new module and rewire call sites.

Checklist:
- Create `tabular_adapter.py` with explicit conversion functions and small helpers.
- Replace ad-hoc conversions in all call sites with adapter calls.
- Delete or deprecate duplicate helpers in the original files.

---

### Phase 2 — Schema Contract Translation Service
**New API boundary**: `src/codeintel/core/schemas/contracts.py`

Responsibilities:
- `TableSchema ↔ Arrow schema` conversion (including IPC encode/decode).
- `TableSchema ↔ JSON schema` conversion.
- Schema hashing and metadata tag generation.

Exact file moves / consolidations:
- Move/centralize logic from:
  - `src/codeintel/core/schemas/arrow_polars.py`
  - `src/codeintel/core/schemas/arrow_gen.py`
  - `src/codeintel/storage/schema/arrow_schema.py`
  - `src/codeintel/core/schemas/serde.py`

Checklist:
- Create `contracts.py` with `to_arrow_schema`, `from_arrow_schema`,
  `encode_schema_ipc`, `decode_schema_ipc`, `to_json_schema`, `from_json_schema`.
- Update all callers to use the new functions.
- Keep `arrow_polars.py` as a thin façade or delete if fully replaced.

---

### Phase 3 — Materialization Pipeline Base
**New API boundary**: `src/codeintel/build/hamilton/materializers/base_pipeline.py`

Responsibilities:
- Shared materialization flow: resolve context → validate → write → observe → result.
- Unified error handling (consistent failure result shape).

Exact file moves / consolidations:
- Extract shared logic from:
  - `src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py`
  - `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`

Checklist:
- Create `base_pipeline.py` with reusable steps and hooks.
- Update both savers to compose the shared pipeline functions.
- Ensure observation and validation are consistently invoked.

---

### Phase 4 — Observation Payload Codec Layer
**New API boundary**: `src/codeintel/storage/tracking/observation_codec.py`

Responsibilities:
- Typed encode/decode for column stats, dataset stats, derived settings.
- Single schema for observation payloads (and compatibility checks).

Exact file moves / consolidations:
- Move parsing/encoding logic from:
  - `src/codeintel/build/schemas/observations.py`
  - `src/codeintel/storage/tracking/schema_catalog.py`

Checklist:
- Add `encode_*` / `decode_*` functions and shared validators.
- Update persistence and observation emitters to use codec layer.
- Add targeted tests for codec behavior.

---

### Phase 5 — Manifest I/O Consolidation
**New API boundary**: `src/codeintel/storage/manifests/manifest_io.py`

Responsibilities:
- Canonical read/write, path resolution, hash validation for manifests.
- Dataset + schema manifest handling in one place.

Exact file moves / consolidations:
- Move logic from:
  - `src/codeintel/storage/datasets/manifests.py`
  - `src/codeintel/core/manifests.py`
  - `src/codeintel/storage/tracking/schema_catalog_compile.py` (manifest IO only)

Checklist:
- Implement `read_manifest`, `write_manifest`, `manifest_path`.
- Rewire dataset and schema workflows to new module.

---

### Phase 6 — Tool Target Scaffolding Normalization
**New API boundary**: `src/codeintel/build/hamilton/native/patterns/target_builder.py`

Responsibilities:
- Single canonical builder for tool-backed targets.
- Standardized naming, tagging, and error handling.

Exact file moves / consolidations:
- Consolidate logic from:
  - `src/codeintel/build/hamilton/native/patterns/tool_target.py`
  - `src/codeintel/build/hamilton/native/patterns/savers.py` (templating only)

Checklist:
- Extract a builder class or functional builder to construct run/ingest/materialize nodes.
- Ensure docstring summary requirements are enforced in one place.

---

### Phase 7 — Export/Serialization Codecs
**New API boundary**: `src/codeintel/core/exports/codecs.py`

Responsibilities:
- Centralized serialization: NDJSON, JSON, Parquet snapshots (as needed).
- Unified input row normalization and Arrow integration.

Exact file moves / consolidations:
- Move logic from:
  - `src/codeintel/serving/export/ndjson.py`
  - `src/codeintel/core/exports/serialization.py`

Checklist:
- Add codec registry and `encode_row`, `encode_batch`, `encode_reader`.
- Ensure NDJSON and export callers use the registry.

---

### Phase 8 — Typed Configuration Access
**New API boundary**: `src/codeintel/core/config/view.py`

Responsibilities:
- Centralized typed settings access with validation and defaults.
- Single source for computed config values.

Exact file moves / consolidations:
- Consolidate ad-hoc resolves in:
  - `src/codeintel/build/hamilton/save_to.py`
  - `src/codeintel/build/hamilton/materializers/*`
  - `src/codeintel/serving/*` and `src/codeintel/storage/*` modules

Checklist:
- Create a thin `SettingsView` with explicit accessors.
- Replace scattered `resolve_*` patterns with `SettingsView`.

---

## Execution Checklist

### Pre-Work
- Confirm the new module boundaries and names are approved.
- Capture a short baseline: lint, type checks, and any key smoke tests.
- Identify all external imports that will need transitional compatibility shims.
- Decide deprecation windows and delete gates for legacy modules.

### During Execution
- Implement boundaries first, then move call sites, then delete legacy helpers.
- Keep compatibility shims minimal and time-boxed with explicit removal points.
- Update tests alongside refactors; do not leave failing legacy tests behind.
- Track per-phase acceptance criteria in each ticket.

### Post-Phase
- Re-run lint/type checks and relevant tests for the touched domain.
- Remove transitional shims once all call sites are migrated.
- Update documentation and architecture notes for the new boundaries.
- Close the phase only when acceptance criteria are met.

---

## Per-Phase Tickets and Acceptance Criteria

### Phase 0 Tickets
- **P0-A Inventory pass**: inventory doc of pipeline stages exists; acceptance: all stages mapped to a single owning module.
- **P0-B Duplicate logic scan**: Arrow/Polars/DuckDB conversions and schema conversions enumerated; acceptance: every conversion has a call-site list.

### Phase 1 Tickets
- **P1-A Tabular adapter module**: `tabular_adapter.py` created; acceptance: new adapter API used by at least one pipeline.
- **P1-B Migration to adapter**: all conversion call sites updated; acceptance: no direct conversion helpers remain outside adapter.
- **P1-C Legacy cleanup**: delete old conversion helpers; acceptance: no duplicate conversions in `stream.py`, `polars_collect.py`, or `duckdb_relation.py`.

### Phase 2 Tickets
- **P2-A Contract translation service**: `contracts.py` created; acceptance: all schema conversions use `contracts.py` APIs.
- **P2-B Call site rewiring**: migrate schema IPC and JSON schema usage; acceptance: no direct calls into legacy schema conversion modules.
- **P2-C Decommission old modules**: delete or replace `arrow_polars.py`, `arrow_gen.py`, `arrow_schema.py`, `serde.py`; acceptance: zero imports of these modules remain.

### Phase 3 Tickets
- **P3-A Base pipeline extraction**: `base_pipeline.py` exists and is used by both savers; acceptance: shared error handling and observation logic is centralized.
- **P3-B Saver simplification**: `duckdb_relation_saver.py` and `arrow_dataset_saver.py` use shared pipeline steps; acceptance: duplicated logic removed.

### Phase 4 Tickets
- **P4-A Observation codec layer**: `observation_codec.py` added; acceptance: encode/decode logic is called by both observation emitters and persistence.
- **P4-B Test coverage**: new tests for codec layer; acceptance: both valid and invalid payloads are covered.

### Phase 5 Tickets
- **P5-A Manifest IO module**: `manifest_io.py` added; acceptance: dataset and schema manifests read/write go through it.
- **P5-B Legacy manifest removal**: remove direct manifest path helpers in old modules; acceptance: no direct manifest path logic outside `manifest_io.py`.

### Phase 6 Tickets
- **P6-A Target builder**: `target_builder.py` introduced; acceptance: new targets use it as the canonical pattern.
- **P6-B Template refactor**: existing tool targets migrated; acceptance: `tool_target.py` reduced to a thin façade or removed.

### Phase 7 Tickets
- **P7-A Codec registry**: `codecs.py` added; acceptance: NDJSON and any other export formats registered in one place.
- **P7-B Export migration**: callers use the registry; acceptance: `ndjson.py` becomes a thin wrapper or is removed.

### Phase 8 Tickets
- **P8-A Settings view**: `view.py` added with explicit accessors; acceptance: no ad-hoc resolve logic remains.
- **P8-B Call site migration**: update materializers and serving/storage paths; acceptance: all config access goes through `SettingsView`.

---

## Cross-Cutting Hardening Tasks
Checklist:
- Add “contract compliance” tests for each new boundary module.
- Ensure every new boundary has a small smoke test and a failure-mode test.
- Add a validation guardrail per module (type and schema consistency).

## Migration Strategy
- **No behavior changes in Phase 0–1**, only wiring and wrappers.
- Introduce new boundaries first, then move call sites gradually.
- Delete old helpers only after all call sites are migrated.
- Keep all public APIs stable; only internal module paths change.

## Legacy and Compatibility Decommissioning
This section lists legacy and compatibility code that will be fully deleted once each phase
is complete. The goal is to avoid long-lived wrappers and keep a single source of truth.

### Phase 1 Decommission Targets
- Delete conversion helpers in `src/codeintel/core/columnar/stream.py` after `tabular_adapter.py` is adopted.
- Delete `src/codeintel/core/columnar/polars_collect.py` after adapter covers its functionality.
- Delete conversion helpers in `src/codeintel/build/tabular/duckdb_relation.py`; keep only relation registration if still needed.
- Remove `_coerce_tabular_input` and related helpers in `src/codeintel/storage/warehouse.py` once adapter is used.

### Phase 2 Decommission Targets
- Delete `src/codeintel/core/schemas/arrow_polars.py` after `contracts.py` fully replaces it.
- Delete `src/codeintel/core/schemas/arrow_gen.py` once contract generation is centralized.
- Delete `src/codeintel/storage/schema/arrow_schema.py` after all Arrow schema IO is in `contracts.py`.
- Delete `src/codeintel/core/schemas/serde.py` once JSON schema conversion is centralized.

### Phase 3 Decommission Targets
- Remove duplicated materialization flow in `src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py`.
- Remove duplicated materialization flow in `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`.
- Delete any interim wrappers created for the migration once all savers use the shared pipeline.

### Phase 4 Decommission Targets
- Delete observation payload encode/decode helpers from `src/codeintel/build/schemas/observations.py`.
- Delete observation payload decode helpers from `src/codeintel/storage/tracking/schema_catalog.py`.

### Phase 5 Decommission Targets
- Delete manifest path helpers and IO logic in `src/codeintel/storage/datasets/manifests.py`.
- Delete manifest IO logic in `src/codeintel/core/manifests.py` after `manifest_io.py` is canonical.
- Delete manifest IO helpers in `src/codeintel/storage/tracking/schema_catalog_compile.py`.

### Phase 6 Decommission Targets
- Delete or fully replace `src/codeintel/build/hamilton/native/patterns/tool_target.py` once the new builder is canonical.
- Remove templating-only helpers from `src/codeintel/build/hamilton/native/patterns/savers.py` if unused.

### Phase 7 Decommission Targets
- Delete `src/codeintel/serving/export/ndjson.py` after export codec registry is in place.
- Delete duplicate row normalization in `src/codeintel/core/exports/serialization.py` after codec adoption.

### Phase 8 Decommission Targets
- Remove ad-hoc config resolves in `src/codeintel/build/hamilton/save_to.py`.
- Remove ad-hoc config resolves in `src/codeintel/build/hamilton/materializers/*`.
- Remove ad-hoc config resolves in `src/codeintel/serving/*` and `src/codeintel/storage/*`.

## Definition of Done
- All conversions and schema logic go through the new boundary modules.
- Materialization savers share a common pipeline implementation.
- Observation payload encoding/decoding is centralized.
- Manifests are read/written by a single I/O module.
- Export formats use a unified codec registry.
- Type checking, ruff, and tests pass with no regressions.
