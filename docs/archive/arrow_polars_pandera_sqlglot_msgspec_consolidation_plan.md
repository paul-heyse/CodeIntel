# Arrow/Polars/Pandera/SQLGlot/msgspec Consolidation Implementation Plan

## TL;DR
Consolidate schema, serialization, streaming IO, and query handling across
`core`, `build`, `config`, and `cli` into a single Arrow-first data plane. The
end state is fewer sources of truth, smaller module surface area, and strict
boundary-only JSON usage, with Polars lazy + Pandera validation and SQLGlot AST
as the canonical interfaces.

## Goals
- Establish a single schema authority pipeline that yields: TableSchema, Arrow
  schema, Pandera schema, and msgspec models from one core contract bundle.
- Enforce boundary-only JSON and msgspec ingress decoding in one place.
- Make Arrow RecordBatchReader/Scanner the default internal data flow.
- Unify Polars lazy execution and streaming sinks behind core utilities.
- Make SQLGlot AST the canonical query representation (SQL rendering only at
  storage/export boundaries).
- Reduce duplication across `core`, `build`, `config`, and `cli` by removing
  parallel utilities and registries.

## Non-Goals
- No change to external API semantics beyond refactors needed for consolidation.
- No full rewrite of storage/serving orchestration.
- No immediate removal of legacy exports beyond boundary adapters.

## Scope
Primary targets:
- `src/codeintel/core/schemas/**`
- `src/codeintel/core/serialization/**`
- `src/codeintel/core/columnar/**`
- `src/codeintel/core/datasets/**`
- `src/codeintel/core/queries/**`
- `src/codeintel/build/schemas/**`
- `src/codeintel/build/tabular/**`
- `src/codeintel/build/exports/**`
- `src/codeintel/config/datasets/**`
- `src/codeintel/cli/handlers/**`

Secondary targets:
- `src/codeintel/storage/queries/**` (AST boundary only)
- `src/codeintel/storage/validation/**` (Pandera entrypoints)

## Constraints
- `core` must not import `build` or `storage`.
- JSON is boundary-only; internal pipelines are Arrow/Parquet.
- Arrow schema metadata must include `codeintel.schema_hash`,
  `codeintel.schema_digest`, and `codeintel.provenance`.
- All schema validation gates must use `schema_metadata_errors` before
  materialization.

## Target Architecture

Canonical flow:
```
config -> core (contract bundle + Arrow + Pandera + msgspec)
            |
            v
build/cli (read-only consumption of core contracts)
            |
            v
storage (AST->SQL boundary, materialization only)
```

Core owns:
- Schema authority (TableSchema + Arrow schema + Pandera schema).
- Row models (msgspec Struct) and boundary serializers.
- Streaming IO helpers (RecordBatchReader, Scanner, Arrow IPC).
- Query AST handling (SQLGlot expressions).

Build/CLI own:
- Orchestration, runtime assembly, and entrypoints.
- No schema creation or JSON normalization logic.

Storage owns:
- SQL rendering, database IO, and schema persistence.

## Consolidation Map (Old -> New Canonical)
- Schema authority
  - `config/datasets/*` + `build/schemas/*` -> `core/schemas/contract_bundle.py`
  - `core/schemas/service.py` becomes the only public read interface
- JSON/msgspec boundary
  - Ad-hoc JSON decode helpers -> `core/serialization/msgspec.py`
- Row models
  - `core/schemas/generated_rows/*` + `core/schemas/row_models.py` ->
    `core/schemas/row_models.py` (msgspec Struct primary, TypedDict optional)
- Arrow streaming
  - `core/datasets/scanning.py` + `core/columnar/dataset_scanner.py` ->
    `core/columnar/streaming.py`
- Polars lazy execution
  - Scattered `pl.scan_*`/`collect` logic -> `core/columnar/polars_collect.py`
- SQLGlot
  - `core/sqlglot_tools.py` + `core/queries/filter_compiler.py` ->
    `core/queries/ast.py` (AST-first API)
- Validation
  - `storage/validation/columnar.py` + `core/validation/pandera_schema.py` ->
    `core/validation/engine.py` (single entrypoint)

## Workstreams

### W1: Schema Authority Consolidation
Problem: Schema primitives, contract assembly, and registry logic are spread
across `config`, `core`, and `build` with overlapping responsibilities.

Plan:
- Create `core/schemas/contract_bundle.py` that builds:
  - TableSchema
  - Arrow schema + metadata
  - Pandera schema
  - msgspec row model + serializer
- Make `core/schemas/service.py` the only public read interface.
- Downgrade `build/schemas/*` to thin adapters and remove schema creation.
- Move any remaining schema registry logic from `config/datasets/*` into core.

Checklist:
- [ ] Implement `ContractBundle` dataclass with schema, arrow_schema,
      pandera_schema, row_struct, row_serializer
- [ ] Update `core/schemas/service.py` to expose bundles and cache by table_key
- [ ] Make `build/schemas/*` call `core` service only
- [ ] Replace `config/datasets/contracts.py` usage with core contract bundle

Acceptance:
- One authoritative schema path in core; build/cli read-only.
- No schema creation logic outside core.

### W2: Boundary Serialization Consolidation
Problem: msgspec/JSON normalization exists in multiple modules with inconsistent
fallbacks.

Plan:
- Centralize boundary decode/encode in `core/serialization/msgspec.py`.
- Replace ad-hoc JSON decoding across build/cli/core with shared helpers.
- Keep JSON only for ingress/egress; no internal JSON objects in core pipelines.

Checklist:
- [ ] Add `decode_boundary_payload` / `encode_boundary_payload` helpers
- [ ] Route all contract/manifest decoding through the boundary helpers
- [ ] Remove duplicate JSON normalization helpers

Acceptance:
- All boundary decoding uses strict msgspec decoders + legacy stripping.

### W3: Row Model Consolidation (msgspec-first)
Problem: TypedDicts, dataclasses, and row model helpers overlap and diverge.

Plan:
- Treat msgspec Struct as the canonical row model.
- Keep TypedDict only for external API typing or CLI IO if required.
- Replace manual row tuple assembly with schema-driven builders.

Checklist:
- [ ] Make msgspec Struct the default row model in `row_models.py`
- [ ] Provide one `row_struct_builder_for_table_schema`
- [ ] Delete or freeze `generated_rows/*` to legacy/compat only

Acceptance:
- Internal row construction uses msgspec Structs and schema-driven builders.

### W4: Arrow Streaming IO Unification
Problem: Dataset scanning and Arrow IO utilities are duplicated across core
modules, with mixed Table/Reader usage.

Plan:
- Merge dataset scanning helpers into a single streaming module.
- Enforce RecordBatchReader/Scanner as internal interfaces.
- Replace eager `Table.from_batches(list(reader))` usage except in exports/tests.

Checklist:
- [ ] Create `core/columnar/streaming.py` and move scanner helpers there
- [ ] Update dataset read/write to accept Reader/Scanner only
- [ ] Add guardrails against `to_pylist`, `to_table`, `read_all` in core

Acceptance:
- Core data flow is streaming-first (Reader/Scanner).

### W5: Polars Lazy Pipeline Consolidation
Problem: Polars execution logic is spread across build/core, leading to
inconsistent streaming behaviors.

Plan:
- Centralize LazyFrame execution options in core.
- Require `scan_*` + `sink_*` in build/cli pipelines where possible.

Checklist:
- [ ] Single LazyFrame execution adapter in core
- [ ] Replace eager `pl.read_*` in build/cli with `scan_*`
- [ ] Add tests to ensure streaming execution paths

Acceptance:
- All Polars pipelines flow through core lazy execution helper.

### W6: Pandera Validation Consolidation
Problem: Pandera schema creation and validation happens in multiple modules
with inconsistent profiles.

Plan:
- Create `core/validation/engine.py` with a single entrypoint to produce
  Pandera schema + validate (schema-only vs data-light vs data-strict).
- Remove validation logic from storage/build modules; they call core only.

Checklist:
- [ ] Centralize Pandera schema creation and validation entrypoints
- [ ] Replace direct Pandera calls in storage/build with core engine
- [ ] Add CI test to compile Pandera schemas for all tables

Acceptance:
- One Pandera entrypoint; validation profiles enforced consistently.

### W7: SQLGlot AST Canonicalization
Problem: SQL string parsing/rendering and AST manipulation are scattered.

Plan:
- Create a core AST-first query API that returns `sqlglot.exp.Expression`.
- Render SQL strings only at storage/export boundaries.

Checklist:
- [ ] Add `core/queries/ast.py` with parse/transform/canonicalize helpers
- [ ] Replace string-based query flows with AST in build/cli
- [ ] Restrict SQL rendering to storage/exports

Acceptance:
- AST is the canonical query representation in core/build/cli.

### W8: CLI/Config Simplification
Problem: CLI and config layers implement schema/serialization logic that
belongs in core.

Plan:
- Make CLI handlers thin orchestrators that call core services.
- Consolidate config parsing to msgspec structs in core.

Checklist:
- [ ] Route CLI dataset/contract operations through core bundles
- [ ] Remove config-level schema helpers and use core exclusively
- [ ] Replace CLI JSON normalization with core boundary helpers

Acceptance:
- CLI and config do not construct schemas or decode JSON directly.

### W9: Deprecation + Cleanup
Problem: Legacy utilities remain after consolidation.

Plan:
- Add temporary compatibility wrappers with deprecation warnings.
- Remove old modules after downstream call sites migrate.

Checklist:
- [ ] Deprecation map doc (old -> new)
- [ ] Remove wrappers after two cutovers
- [ ] Update guardrails to fail on old imports

Acceptance:
- Old modules removed or isolated behind compatibility layer.

## Phased Implementation Plan

### Phase 0: Inventory + Design (1-2 days)
- Audit all schema/serialization/IO/query utilities.
- Build a consolidation map and deprecation plan.

### Phase 1: Core Foundations (3-6 days)
- Implement W1, W2, W3 in core.
- Add unit tests for msgspec boundary decoding + row struct builders.

### Phase 2: Streaming + Polars (3-6 days)
- Implement W4 and W5.
- Update build/cli pipelines to use streaming primitives.

### Phase 3: Validation + SQLGlot (2-4 days)
- Implement W6 and W7.
- Replace string-based SQL usage in build/cli.

### Phase 4: CLI/Config Cutover (2-4 days)
- Implement W8 (thin CLI + config)
- Remove any remaining schema creation outside core.

### Phase 5: Cleanup + Guardrails (1-3 days)
- Implement W9 and enforce guardrails.
- Remove dead modules and update docs.

## Acceptance Criteria
- `core` is the single schema authority and streaming interface.
- No JSON objects flow through internal core pipelines.
- All Arrow schema metadata validated prior to materialization.
- Polars pipelines use lazy scan + streaming sinks.
- SQL strings only rendered in storage/export layers.
- CLI/config code does not own schema or serialization logic.

## Testing Strategy
- Unit tests for contract bundle outputs (TableSchema/Arrow/Pandera/msgspec).
- Boundary decode tests for msgspec payloads and legacy JSON.
- Streaming IO tests using RecordBatchReader + Scanner.
- Pandera schema compilation for all tables.
- SQLGlot AST diff tests for canonicalization.

## Risks and Mitigations
- Risk: Large refactor surface area.
  - Mitigation: staged phases + compatibility wrappers.
- Risk: Streaming performance regressions.
  - Mitigation: add perf checkpoints with batch sizes and scan metrics.
- Risk: Schema drift across layers.
  - Mitigation: guardrails + schema metadata validation gates.

## Deliverables Checklist
- [ ] Contract bundle implementation in core
- [ ] Boundary serialization consolidation
- [ ] msgspec row model primary path
- [ ] Unified Arrow streaming IO
- [ ] Polars lazy pipeline adapter
- [ ] Pandera validation engine
- [ ] SQLGlot AST-first query API
- [ ] CLI/config cutover to core services
- [ ] Guardrails + deprecation cleanup
