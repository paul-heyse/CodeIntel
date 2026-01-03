# Core Consolidation Plan (src/codeintel/core)

## TL;DR
Consolidate `core` into a smaller, Arrow-first data plane with a single schema
authority, unified serialization boundary, and a minimal set of streaming and
query utilities. The goal is to reduce duplicate utilities, shrink module
surface area, and enforce boundary-only JSON usage.

## Goals
- Make `core` the single authority for schema, row models, and Arrow metadata.
- Enforce Arrow/Parquet streaming as the internal data flow.
- Centralize msgspec boundary decoding/encoding in one module.
- Make SQLGlot AST the canonical query representation in core.
- Remove duplicate utilities and collapse overlapping modules.

## Non-Goals
- No changes to build/storage/serving orchestration.
- No external API changes beyond internal refactors.
- No full rewrite of analytics/ingestion targets.

## Scope
Primary targets in `src/codeintel/core`:
- `schemas/**`
- `serialization/**`
- `columnar/**`
- `datasets/**`
- `queries/**`
- `helpers/**`
- `validation/**`

## Principles
- JSON is boundary-only; internal pipelines are Arrow/Parquet.
- Prefer `RecordBatchReader`/`Scanner` over `Table` materialization.
- Keep `core` free of `build`/`storage` imports.
- Consolidate to one canonical module per concern.

## Workstreams

### W1: Schema Authority Unification
Problem: schema definitions, Arrow metadata, and row models are split across
multiple modules.

Plan:
- Introduce a single `core/schemas/contract_bundle.py` that yields:
  - TableSchema
  - Arrow schema with metadata
  - msgspec row model + serializer
  - Pandera schema (if needed in core)
- Make `core/schemas/service.py` the canonical lookup interface.
- Deprecate duplicate schema/row model factories.

Checklist:
- [ ] Implement `ContractBundle` dataclass with schema + Arrow + row model
- [ ] Update `core/schemas/service.py` to return bundles by table_key
- [ ] Replace calls to `row_models` or `arrow_gen` with bundle access

Acceptance:
- One schema authority path in core.

### W2: Row Model Consolidation (msgspec-first)
Problem: TypedDicts + dataclasses + msgspec structs overlap.

Plan:
- Use msgspec Struct as the canonical row model.
- Keep TypedDicts only for external typing, not internal construction.
- Remove or freeze generated TypedDict modules.

Checklist:
- [ ] Make msgspec Struct the default in `row_models.py`
- [ ] Provide `row_struct_builder_for_table_schema`
- [ ] Deprecate `core/schemas/generated_rows/*`

Acceptance:
- Internal row construction uses msgspec Structs exclusively.

### W3: Boundary Serialization Unification
Problem: msgspec/JSON decoding is scattered and inconsistent.

Plan:
- Centralize boundary decode/encode in `core/serialization/msgspec.py`.
- Remove ad-hoc JSON normalization from other modules.
- Keep JSON helpers strictly in boundary modules.

Checklist:
- [ ] Add `decode_boundary_payload` / `encode_boundary_payload`
- [ ] Route contract/manifest decode through boundary helpers
- [ ] Trim `core/helpers/json.py` to boundary-only APIs

Acceptance:
- One boundary decoder; no JSON objects in core pipelines.

### W4: Arrow Streaming IO Consolidation
Problem: scanning utilities and stream adapters are duplicated.

Plan:
- Merge dataset scanning helpers into a single `core/columnar/streaming.py`.
- Enforce `RecordBatchReader`/`Scanner` in all internal IO paths.
- Remove eager `Table.from_batches(list(reader))` in core.

Checklist:
- [ ] Consolidate scanner helpers
- [ ] Replace eager conversions with streaming readers
- [ ] Add guardrails for `to_pylist`, `to_table`, `read_all` in core

Acceptance:
- Core data flow is streaming-first.

### W5: SQLGlot AST Canonicalization
Problem: SQL string handling still leaks into core utilities.

Plan:
- Create `core/queries/ast.py` with AST-first parsing/transform helpers.
- Ensure all core query utilities accept/return SQLGlot expressions.
- Restrict SQL string rendering to storage/export boundaries only.

Checklist:
- [ ] Add AST-first query API
- [ ] Replace string-based flows in core queries
- [ ] Add AST diff utilities in core

Acceptance:
- AST is the canonical query representation in core.

### W6: Validation Consolidation
Problem: Pandera and schema validation logic is duplicated.

Plan:
- Add `core/validation/engine.py` as the single entrypoint for:
  - schema-only vs data-light vs data-strict validation
  - Pandera schema creation + execution
- Replace direct Pandera calls across core with the engine.

Checklist:
- [ ] Implement validation engine API
- [ ] Update core call sites
- [ ] Add unit tests for validation profiles

Acceptance:
- One validation entrypoint in core.

### W7: Helper Consolidation
Problem: helper modules overlap (payload/json/normalization).

Plan:
- Collapse helper logic into two modules:
  - `core/serialization/*` for msgspec/JSON boundary
  - `core/columnar/*` for Arrow/Parquet normalization
- Remove duplicate conversion utilities.

Checklist:
- [ ] Merge payload + json normalization into serialization
- [ ] Remove duplicated helpers and update imports

Acceptance:
- Minimal helper surface area.

## Phased Implementation Plan

### Phase 0: Inventory + Deprecation Map (1-2 days)
- Catalog all core utilities and identify duplicates.
- Define old->new mapping and deprecation plan.

### Phase 1: Schema + Row Models (3-5 days)
- Implement W1 + W2.
- Add tests for bundle outputs and row struct builders.

### Phase 2: Serialization + Streaming (3-5 days)
- Implement W3 + W4.
- Replace eager conversions in core.

### Phase 3: Queries + Validation (2-4 days)
- Implement W5 + W6.
- Add AST diff and validation profile tests.

### Phase 4: Cleanup (1-2 days)
- Implement W7 and remove deprecated helpers.
- Add guardrails for legacy imports/usage.

## Acceptance Criteria
- Core has one schema authority path and one boundary decoder.
- No JSON objects flow through internal core pipelines.
- All core IO paths are streaming-first.
- SQLGlot AST is canonical in core.
- Validation logic is centralized and tested.

## Testing Strategy
- Contract bundle tests (schema + Arrow + row model outputs).
- Boundary decode tests for msgspec payloads.
- Streaming IO tests using RecordBatchReader/Scanner.
- AST canonicalization tests.
- Validation profile tests.

## Risks and Mitigations
- Risk: large refactor surface in core modules.
  - Mitigation: phased rollout + compatibility wrappers.
- Risk: streaming regressions.
  - Mitigation: targeted performance tests and batch-size controls.

## Deliverables Checklist
- [ ] Contract bundle implementation
- [ ] msgspec-first row model path
- [ ] Boundary serialization consolidation
- [ ] Unified streaming IO helpers
- [ ] AST-first query API
- [ ] Core validation engine
- [ ] Helper cleanup + guardrails
