# Codebase Streamlining Parallel Implementation Plan

## Purpose
Deliver a parallel, non-disruptive set of refactors that consolidate shared functionality,
reduce duplication, and improve maintainability while the primary streamlining refactor
plan executes independently.

## Principles
- Do not block or reorder the primary streamlining phases.
- Avoid long-lived compatibility shims; delete legacy code immediately after migration.
- Keep behavior stable; if behavior must change, update callers and remove old paths immediately.
- Ship with tests and quality gates for every touched module.

## Parallel Workstreams

### Workstream A: Path Normalization Consolidation
**Goal**: Make `src/codeintel/core/paths/normalize.py` the single canonical path normalizer.

**Current duplication**
- `src/codeintel/core/catalog/span_index.py` has a local `normalize_path`.
- `src/codeintel/ingestion/infrastructure/__init__.py` has `normalize_rel_path`.
- `src/codeintel/serving/config.py` has `normalize_optional_path`.
- `src/codeintel/config/models.py` has path normalization helpers.

**Plan**
1. Inventory all path normalization entrypoints and their behavior differences.
2. Define canonical helpers in `src/codeintel/core/paths/normalize.py`:
   - `normalize_path`
   - `normalize_rel_path` (or equivalent using `Path.as_posix()` + `normalize_path`)
   - `normalize_optional_path`
3. Replace local helpers with imports from `codeintel.core.paths`.
4. Remove any local helpers immediately after call sites are migrated.
5. Add tests that lock down canonical behavior for:
   - Windows separators
   - `./` and `../` normalization
   - Empty path handling
   - Optional path normalization

**Acceptance criteria**
- No local normalize helpers remain outside `codeintel.core.paths`.
- All call sites use core helpers.
- Tests cover path normalization behavior across call sites.

**Validation**
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted path tests.

---

### Workstream B: Deterministic Serialization for Hashing and Cache Keys
**Goal**: Centralize and version deterministic serialization used for hashes and cache keys.

**Current duplication**
- `src/codeintel/core/cache/keying.py` `_serialize_value`
- `src/codeintel/core/hashing/fingerprint.py` `_serialize_value`
- `src/codeintel/cli/core/results.py` `_serialize_value`
- `src/codeintel/core/serialization/converters.py` `serialize_value`

**Plan**
1. Create a canonical serializer module, e.g. `src/codeintel/core/serialization/stable.py`.
2. Define:
   - `stable_serialize_value(value: object) -> str`
   - `stable_serialize_json(value: object) -> JsonValue`
   - `stable_hash(*args: object) -> str` (if needed)
3. Ensure deterministic ordering for dicts and predictable handling for:
   - `Path`, `datetime`, `date`, `Enum`, `bytes`, dataclasses.
4. Update all internal call sites to use the new serializer.
5. Accept cache invalidation as part of migration in design phase.
6. Remove duplicate serializers after migration.

**Acceptance criteria**
- One canonical serializer used for hashing and cache keys.
- Versioned serialization clearly separated and documented.
- Tests prove deterministic output for complex inputs.

**Validation**
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Dedicated serializer tests for deterministic ordering and type handling.

---

### Workstream C: AST and CST Parsing Consolidation
**Goal**: Move shared AST parsing and literal extraction helpers into `src/codeintel/core/parsing`.

**Current duplication**
- `src/codeintel/ingestion/infrastructure/ast_utils.py` parsing helpers.
- `src/codeintel/analytics/utilities/ast.py` literal extraction and unparse helpers.

**Plan**
1. Add `src/codeintel/core/parsing/ast_utils.py`:
   - `parse_python_module`
   - `timed_parse`
   - `literal_value`, `literal_str`, `literal_int`, `literal_bool`
   - `safe_unparse`
2. Move or re-export helpers and update imports.
3. Remove duplicate logic immediately after call sites are migrated.

**Acceptance criteria**
- All AST parsing/literal helpers sourced from `codeintel.core.parsing`.
- No behavior change in parsing or literal extraction.
- Facade modules are thin and deprecated.

**Validation**
- Parser unit tests for valid/invalid modules.
- Literal extraction tests for int, bool, negative numbers, lists, dicts.

---

### Workstream D: Shared Serving Export Delivery Adapter
**Goal**: Deduplicate export dispatch logic between HTTP and MCP.

**Current duplication**
- `src/codeintel/serving/http/export_dispatch.py`
- `src/codeintel/serving/mcp/export_dispatch.py`

**Plan**
1. Introduce `src/codeintel/serving/export/dispatch.py` with shared logic:
   - Build export plan
   - Row iteration and streaming helpers
   - File export helper
2. Refactor HTTP and MCP dispatch to thin wrappers:
   - HTTP handles response construction + metrics
   - MCP handles resource store specifics
3. Keep behavior identical for supported formats and error cases.

**Acceptance criteria**
- One shared dispatch module for plan + row delivery.
- HTTP and MCP keep their unique response handling without duplicated control flow.

**Validation**
- Unit tests for export plan delivery mapping.
- Integration tests for JSONL stream and binary exports.

---

### Workstream E: Transport-Level Error Mapping Helper
**Goal**: Centralize ProblemDetail mapping across HTTP and MCP.

**Current duplication**
- `src/codeintel/serving/http/errors.py`
- `src/codeintel/serving/mcp/middleware_errors.py`

**Plan**
1. Add `src/codeintel/serving/errors/transport.py`:
   - `problem_detail_from_error_response_with_context(...)`
   - shared cleaning of extensions and correlation IDs
2. Update HTTP and MCP to call the shared helper.
3. Keep `src/codeintel/serving/errors/problem_adapter.py` as the single conversion source.

**Acceptance criteria**
- One transport helper used in HTTP and MCP.
- ProblemDetail payloads are identical across transports for the same ErrorResponse.

**Validation**
- Snapshot tests for ProblemDetail payload fields.
- Integration tests covering error mapping paths.

---

### Workstream F: Table-Key Splitting Standardization
**Goal**: Remove local `_split_table_key` helpers and use
`src/codeintel/storage/helpers/table_key.py`.

**Current duplication**
- `src/codeintel/core/schemas/contract_policy.py`
- `src/codeintel/core/schemas/contract_factory.py`
- `src/codeintel/build/schemas/observations.py`

**Plan**
1. Replace local helpers with:
   - `parse_table_key`, `split_table_key`, or `try_parse_table_key`
2. Preserve existing "None on invalid" behavior using `try_parse_table_key`.
3. Ensure all table-key parsing errors are consistent across modules.
4. Remove local `_split_table_key` definitions after migration.

**Acceptance criteria**
- No local `_split_table_key` helpers remain.
- Error behavior consistent across modules.

**Validation**
- Table key parsing tests for valid and invalid inputs.

---

### Workstream G: Row Serialization Alignment
**Goal**: Consolidate row serialization through `src/codeintel/core/schemas/row_serialization.py`.

**Current duplication**
- `src/codeintel/config/datasets/columns.py` `serialize_row`

**Plan**
1. Update `serialize_row` to delegate to `row_to_tuple` when a table key is available.
2. Introduce a small adapter for cases that only have column sequences.
3. Ensure ordering is based on schema registry when possible.
4. Deprecate or remove redundant serialization helpers.

**Acceptance criteria**
- One canonical row serialization path for table-key driven serialization.
- Clear separation for schema-backed vs ad-hoc serialization.

**Validation**
- Tests verifying column ordering and stable tuple output.

---

### Workstream H: Observability Registry Consolidation
**Goal**: Clarify and consolidate observability registries.

**Current duplication**
- `src/codeintel/observability/instrument_registry.py`
- `src/codeintel/observability/instrumentation_registry.py`

**Plan**
1. Define a unified public surface in `src/codeintel/observability/registry.py`:
   - `get_instrument_registry`
   - `get_instrumentation_registry`
2. Clarify naming:
   - "InstrumentRegistry" for meter-scoped caches.
   - "InstrumentationRegistry" for status tracking and telemetry.
3. Update imports to use the unified entrypoint.
4. Remove legacy modules immediately after imports are migrated.

**Acceptance criteria**
- Single canonical registry module for imports.
- No ambiguity in naming or responsibility.

**Validation**
- Unit tests for registry caching and instrumentation status reporting.

---

## Workstream Checklists

### Workstream A Checklist
- [x] Add canonical helpers in `src/codeintel/core/paths/normalize.py`.
- [x] Update imports in `src/codeintel/core/catalog/span_index.py`.
- [x] Update imports in `src/codeintel/ingestion/infrastructure/__init__.py`.
- [x] Update imports in `src/codeintel/serving/config.py` and `src/codeintel/config/models.py`.
- [x] Add tests for windows separators, dot segments, and optional paths.
- [x] Remove duplicate helpers in the same PR as migration.
- [x] Add test coverage for empty path handling if not already covered.

### Workstream B Checklist
- [x] Add `src/codeintel/core/serialization/stable.py` with stable serialization APIs.
- [x] Add deterministic serialization tests for dict ordering and complex types.
- [x] Migrate internal cache key call sites to the new serializer.
- [x] Remove duplicate serializers in the same PR as migration.
- [x] Decide whether to deprecate/align `serialize_value` in `src/codeintel/core/serialization/converters.py` and update exports if needed.

### Workstream C Checklist
- [x] Add `src/codeintel/core/parsing/ast_utils.py` with parsing and literal helpers.
- [x] Re-export or update imports in ingestion and analytics modules.
- [x] Remove duplicate logic in the same PR as migration.
- [x] Add tests for parse success/failure and literal extraction behavior.

### Workstream D Checklist
- [x] Add `src/codeintel/serving/export/dispatch.py` shared adapter.
- [x] Refactor HTTP dispatch to use shared adapter and keep metrics handling intact.
- [x] Refactor MCP dispatch to use shared adapter and keep resource store handling intact.
- [x] Add tests for JSONL streaming and binary export paths.
- [x] Verify error handling parity for all formats.

### Workstream E Checklist
- [x] Add `src/codeintel/serving/errors/transport.py` helper.
- [x] Update HTTP error mapping to use the shared helper.
- [x] Update MCP middleware mapping to use the shared helper.
- [x] Add tests for ProblemDetail payload parity across transports.

### Workstream F Checklist
- [x] Replace local `_split_table_key` usage with `table_key` helpers.
- [x] Preserve None-on-invalid semantics using `try_parse_table_key`.
- [x] Remove local helpers in the same PR as migration.
- [x] Add tests for invalid table keys and unqualified keys.

### Workstream G Checklist
- [x] Add adapter for column-only serialization where no table key exists.
- [x] Update `src/codeintel/config/datasets/columns.py` to use canonical serializer.
- [x] Delegate schema-backed serialization to `row_to_tuple` when table keys are available.
- [x] Add tests for tuple ordering and schema-aligned output.

### Workstream H Checklist
- [x] Add `src/codeintel/observability/registry.py` as canonical entrypoint.
- [x] Update imports across the codebase to use the new entrypoint.
- [x] Remove legacy modules in the same PR as import migration.
- [x] Add tests for registry caching and instrumentation summary behavior.

---

## Execution Enhancements

### Entry and Exit Criteria
- Workstream A: Entry = path behavior inventory done; Exit = all call sites migrated and
  duplicate helpers removed with tests passing.
- Workstream B: Entry = stable serializer added; Exit = all hash/key call sites migrated,
  legacy serializers removed, cache invalidation accepted.
- Workstream C: Entry = core AST helpers added; Exit = ingestion/analytics imports migrated and
  duplicate helpers removed.
- Workstream D: Entry = shared dispatch adapter added; Exit = HTTP/MCP dispatches refactored and
  tests covering streaming/binary exports pass.
- Workstream E: Entry = transport helper added; Exit = HTTP/MCP error mapping unified and
  ProblemDetail parity tests pass.
- Workstream F: Entry = table key helper availability confirmed; Exit = all local helpers removed
  and table-key parsing tests pass.
- Workstream G: Entry = row serialization adapter added; Exit = column serialization uses
  canonical row serializer and tests pass.
- Workstream H: Entry = registry entrypoint added; Exit = imports migrated and legacy modules
  deleted.

### Testing Matrix
| Workstream | Targeted pytest subsets | Smoke tests |
| --- | --- | --- |
| A | `tests/core/paths/test_paths.py`, `tests/config/test_serving_models_paths.py`, `tests/serving/test_settings.py` | `tests/test_pipeline_smoke.py` |
| B | `tests/build/hamilton/test_pr56_schema_hashing.py`, `tests/build/hamilton/test_pr58_fingerprinting_schema_hash.py`, `tests/build/hamilton/test_pr62_row_model_cache_keys_on_schema_hash.py`, `tests/plugins/test_pack_fingerprint_invalidation.py` | `tests/test_pipeline_smoke.py` |
| C | `tests/ingestion/test_ast_utils.py`, `tests/analytics/test_ast_utils.py`, `tests/analytics/test_ast_metrics.py` | `tests/test_pipeline_smoke.py` |
| D | `tests/serving/http/test_export_dispatch.py`, `tests/serving/test_streaming_ndjson.py`, `tests/serving/http/test_export.py`, `tests/serving/mcp/test_resources.py` | `tests/serving/test_http_mcp_integration.py` |
| E | `tests/serving/test_problem_detail_adapter.py`, `tests/serving/mcp/test_error_catalog.py`, `tests/serving/mcp/test_middleware_logging_smoke.py` | `tests/serving/test_http_mcp_integration.py` |
| F | `tests/storage/test_schema_helpers.py`, `tests/storage/test_table_goldens.py`, `tests/storage/test_schema_roundtrip.py` | `tests/test_pipeline_smoke.py` |
| G | `tests/build/hamilton/test_row_serialization_registry.py`, `tests/ingestion/test_row_serialization.py`, `tests/storage/test_bulk_insert_normalization.py` | `tests/test_pipeline_smoke.py` |
| H | `tests/observability/test_observability_smoke.py`, `tests/observability/test_metrics_views.py`, `tests/observability/test_attribute_taxonomy.py` | `tests/observability/test_observability_smoke.py` |

### Immediate Migration and Legacy Deletion Policy
- Migrations must remove legacy helpers and compatibility shims in the same PR.
- If sequencing requires two PRs, the cleanup PR must land immediately after the migration PR.
- No long-lived deprecation windows or runtime shims; design phase permits direct replacement.

### Rollback Notes
- Revert a workstream by restoring the previous helper module and call sites together.
- Do not leave mixed old/new call paths; rollback should be atomic per workstream.
- If cache keys change (Workstream B), expect cache invalidation on rollback or reapply.

### Documentation Touchpoints
- Update `docs/architecture.md` if a new canonical module boundary is introduced.
- Update `docs/observability_shared_components_plan.md` for Workstream H.
- Update `docs/codebase_streamlining_refactor_plan.md` if any boundary names change.

## Initial PR Sequence and Dependencies

Migration PRs in each workstream depend on their foundation PR, and cleanup PRs depend on
the migration PR. Across workstreams, these PRs can run in parallel unless noted below.

### Foundation PRs (start in parallel)
| PR | Scope | Depends On | Notes |
| --- | --- | --- | --- |
| PR-A1 | Add canonical path helpers and tests | none | No call site changes. |
| PR-B1 | Add stable serialization module and tests | none | No cache migration yet. |
| PR-C1 | Add core AST helpers and tests | none | No call site changes. |
| PR-D1 | Add shared export dispatch adapter and tests | none | No HTTP/MCP refactors yet. |
| PR-E1 | Add transport error helper and tests | none | No HTTP/MCP refactors yet. |
| PR-F1 | Add table key helper tests and migration plan | none | Optional if tests already exist. |
| PR-G1 | Add row serialization adapter and tests | none | No call site changes. |
| PR-H1 | Add observability registry entrypoint and tests | none | No import migration yet. |

### Migration PRs (run in parallel after foundations)
| PR | Scope | Depends On | Notes |
| --- | --- | --- | --- |
| PR-A2 | Migrate path normalization call sites | PR-A1 | Touches path-related call sites. |
| PR-B2 | Migrate cache key call sites to new serializer | PR-B1 | Removes legacy serializers. |
| PR-C2 | Update ingestion and analytics imports | PR-C1 | Removes legacy helpers. |
| PR-D2 | Refactor HTTP and MCP export dispatch | PR-D1 | Touches serving HTTP and MCP. |
| PR-E2 | Refactor HTTP and MCP error mapping | PR-E1 | Touches serving HTTP and MCP. |
| PR-F2 | Replace local table key helpers | PR-F1 | Touches schemas and observations. |
| PR-G2 | Update dataset column serialization | PR-G1 | Touches config datasets helpers. |
| PR-H2 | Update imports to registry entrypoint | PR-H1 | Touches observability imports. |

### Cleanup PRs (only if sequencing requires)
If a migration PR cannot include deletions due to coordination, a cleanup PR must follow
immediately after and remove legacy helpers before any new workstream PRs stack on top.

### Parallel Execution Notes
- Avoid overlapping modifications to the same files in concurrent PRs.
- Schedule PR-D2 and PR-E2 with awareness that both touch serving layers, but separate files.
- Keep main refactor plan phases isolated from these PRs unless a compatibility shim is needed.

### Conflict Matrix (Likely File Overlaps)
- PR-A2 and PR-C2: potential overlap in `src/codeintel/ingestion/infrastructure/__init__.py`.
- PR-A2 and PR-C2: potential overlap in `src/codeintel/analytics/utilities/ast.py` if it adopts core path helpers.
- PR-D2 and PR-E2: both touch serving layers; keep changes scoped to
  `src/codeintel/serving/http/export_dispatch.py`, `src/codeintel/serving/mcp/export_dispatch.py`,
  `src/codeintel/serving/http/errors.py`, and `src/codeintel/serving/mcp/middleware_errors.py`.
- PR-B2 and PR-G2: potential overlap if row serialization feeds into cache key generation.
- PR-F2 and PR-G2: possible overlap in table key handling if row serialization uses table keys.

## Cross-Workstream Execution Plan

### Phase P-A: Baselines and Inventory
- Capture baselines for each workstream (existing behavior and call sites).
- Establish test files and utilities for each workstream.
- Confirm compatibility shims and deprecation windows.

### Phase P-B: Implementation in Parallel
- Execute Workstreams A-H concurrently with separate PRs or tickets.
- Restrict each workstream to its file boundaries to avoid conflicts with the main plan.

### Phase P-C: Cleanup and Convergence
- Remove shims after all call sites are migrated.
- Update documentation and architectural notes for canonical helpers.

## Shared Quality Gates
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted `uv run pytest -q` for each touched domain, followed by segmented test runs.

## Remaining Execution Checklist
- [ ] Run targeted pytest subsets for updated modules.
- [ ] Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json` at the end.

## Risks and Mitigations
- **Cache key drift**: Cache invalidation is expected in design phase; test deterministic output.
- **Behavior regressions**: No compatibility windows; rely on targeted tests and fast rollback.
- **Parallel conflicts**: Keep workstreams focused and avoid touching main-plan files
  unless needed for compatibility shims.

## Deliverables
- One PR per workstream with:
  - Updated code boundaries
  - Tests
  - Compatibility shims and removal gates
- A final cleanup PR after all migrations complete.
