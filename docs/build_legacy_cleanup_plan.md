# Build Legacy Cleanup Plan (Native-Only + Contract-First)

## Context and Decisions
- No external consumers depend on `codeintel.build.hamilton.templates.*` or `get_support_module`.
- `src/codeintel/build/hamilton/native/target_override_tables.py` is not authoritative long-term.
- We must identify compute results that emit `errors` and migrate to `warnings` before removing the
  compatibility fallback.

## Goals
- Make OutputContracts the sole source of truth for output table schemas and artifacts.
- Remove legacy template fallbacks, stub targets, and placeholder schemas.
- Eliminate compatibility behavior that hides schema or execution issues.
- Preserve a clean, native-only build DAG surface with strict validation.

## Non-goals
- No new features beyond contract/schema correctness and legacy removal.
- No backward compatibility shims for external consumers.

## Success Criteria
- `TARGET_SPEC_OVERRIDES` and `target_override_tables.py` are removed.
- `compile_output_targets_from_driver()` builds contracts from real TableSchema data, not placeholders.
- `codeintel.build.hamilton.templates` is fully removed (code + tests).
- No compute results expose an `errors` attribute; `ExecutionResultLike` no longer relies on it.
- Quality gates pass with targeted test subsets and the full build test segments.

## Workstreams

## Status Update (Prep Work Performed)
- W1 schema migration completed: overrides removed, contracts now resolve real TableSchema data.
- Canonical output schemas migrated into the core registry (see `src/codeintel/core/schemas/output_registry.py`).
- Placeholder schema helpers removed from contracts and schema index; tests updated to use helpers.
- Quality report passes (ruff/pyright/pyrefly/guardrails/contract checks).

### W0: Baseline Inventory and Impact Map
**Objective**: Enumerate all legacy touchpoints and confirm removal scope.

**Steps**
1. Inventory template usage and stub generation references.
   - `rg -n "codeintel.build.hamilton.templates|get_template_module|get_support_module" src tests`
2. Inventory OutputContract schema sources and overrides.
   - `rg -n "TARGET_SPEC_OVERRIDES|target_override_tables|placeholder_table_schema" src`
3. Inventory compute results with `errors` or non-standard result shapes.
   - `rg -n "errors\s*:\s*|\.errors\b" src/codeintel/build`
   - `rg -n "ExecutionResultLike|to_execution_result" src`

**Outputs**
- A checklist of files to edit/delete, grouped by workstream.
- A list of compute result types that require migration.

**Acceptance**
- Inventory list is complete and cross-checked against `tests/` references.

---
**Inventory Results (Concrete Task List)**  
_(captured from W0 scans; use as the definitive edit/delete checklist)_

**Template + support module touchpoints**
- Replace template imports in native modules (move to new helper module in W2):
  - `src/codeintel/build/hamilton/native/graphs/call_graph.py`
  - `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
  - `src/codeintel/build/hamilton/native/graphs/graph_targets.py`
  - `src/codeintel/build/hamilton/native/graphs/import_graph.py`
- Template package to delete (after extraction):
  - `src/codeintel/build/hamilton/templates/__init__.py`
  - `src/codeintel/build/hamilton/templates/all_targets.py`
  - `src/codeintel/build/hamilton/templates/materialize_template.py`
  - `src/codeintel/build/hamilton/templates/multi_table_pipeline.py`
  - `src/codeintel/build/hamilton/templates/rows_helpers.py`
  - `src/codeintel/build/hamilton/templates/tool_pipeline.py`
- Support module exports to update/remove:
  - `src/codeintel/build/hamilton/nodes/__init__.py` (exports `get_support_module`)
  - `src/codeintel/build/hamilton/nodes/support_factory.py` (defines `get_support_module`)
- Tests to remove or rewrite once templates are removed:
  - `tests/build/hamilton/test_executor_pipeline_template.py`
  - `tests/build/hamilton/test_phase2_ibis_pipeline_template.py`
  - `tests/build/hamilton/test_multi_table_pipeline_template.py`
  - `tests/build/hamilton/test_rows_pipeline_helpers.py`
  - `tests/build/test_hamilton_phase1.py` (uses `get_support_module`)

**Override + placeholder schema touchpoints**
- Override registry and usage to remove:
  - `src/codeintel/build/hamilton/native/target_override_tables.py`
  - `src/codeintel/build/hamilton/native/target_overrides.py`
  - `src/codeintel/build/hamilton/driver_factory.py` (uses `TARGET_SPEC_OVERRIDES`)
  - `src/codeintel/build/hamilton/nodes/support_factory.py` (uses `TARGET_SPEC_OVERRIDES`)
- Placeholder schema helpers to delete after migration:
  - `src/codeintel/build/contracts.py` (`OutputContract.simple`, `placeholder_table_schema`,
    `is_placeholder_schema`)
  - `src/codeintel/build/hamilton/target_spec_compiler.py` (uses `placeholder_table_schema`)
  - `src/codeintel/build/schemas/schema_index.py` (uses `is_placeholder_schema`)
- Tests and helpers using placeholders/`OutputContract.simple` to update:
  - `tests/build/test_state.py`
  - `tests/build/test_readiness_registry_resources_resolver.py`
  - `tests/build/test_contract_resolution_seams.py`
  - `tests/build/test_contracts_parameters_state.py`
  - `tests/build/hamilton/test_pr09_planner.py`
  - `tests/build/hamilton/test_coverage_targets.py`
  - `tests/build/hamilton/test_phase2_ibis_pipeline_template.py`
  - `tests/build/test_targets.py`
  - `tests/build/hamilton/test_target_spec_tags.py`
  - `tests/build/test_hashing_plan_targets.py`
  - `tests/build/hamilton/test_materializer.py`
  - `tests/build/hamilton/test_schema_index_overrides.py`
  - `tests/build/hamilton/test_executor_pipeline_template.py`
  - `tests/build/hamilton/test_metrics_targets.py`
  - `tests/build/hamilton/test_graph_targets.py`
  - `tests/build/hamilton/test_multi_table_pipeline_template.py`
  - `tests/build/hamilton/test_pr80_schema_compile_uses_batch_inference.py`
  - `tests/_helpers/build.py`

**ExecutionResult compatibility audit**
- Compatibility fallback to remove:
  - `src/codeintel/build/hamilton/execution_result.py` (`_extract_warnings` uses `errors`)
- Compute result types passed to `to_execution_result` (verify warning/error fields and migrate if
  needed):
  - `src/codeintel/build/hamilton/native/graphs/call_graph.py` (`CallGraphExtractResult`)
  - `src/codeintel/build/hamilton/native/graphs/import_graph.py` (`ImportGraphExtractResult`)
  - `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py` (`CFGExtractResult`, `DFGExtractResult`)
  - `src/codeintel/build/hamilton/native/graphs/graph_targets.py`
    (`GoidExtractResult`, `SymbolUsesExtractResult`)
- Non-executor compute result with `errors` field (decide whether to standardize to warnings):
  - `src/codeintel/build/hamilton/native/graphs/graph_targets.py` (`GraphValidationResult`)

---

### W1: Migrate Override Schemas into OutputContracts
**Objective**: Eliminate `target_override_tables.py` by resolving TableSchema directly into
OutputContracts.

**Design Decisions**
- OutputContracts must contain non-placeholder TableSchema definitions for all contract outputs.
- TableSchema source is the canonical schema registry/provider (not a target-specific override map).

**Implementation Steps**
1. Add a `TableSchemaResolver` in the target spec compilation path. **Done**
   - Candidate location: `src/codeintel/build/hamilton/target_spec_compiler.py`.
   - Resolve schema via the canonical provider (`codeintel.build.schemas.registry`).
   - Error if schema is missing for any contract output.
2. Update `compile_output_targets_from_driver()` to use the resolver instead of
   `placeholder_table_schema()` and override tables. **Done**
   - Removed override-table logic in `_resolve_table_schemas()`.
3. Move TableSchema definitions from `target_override_tables.py` into the canonical registry. **Done**
   - Implemented via `src/codeintel/core/schemas/output_registry.py` and
     `src/codeintel/core/schemas/table_registry.py`.
   - Ensured table keys are registered once and only once.
4. Delete `src/codeintel/build/hamilton/native/target_override_tables.py` and
   `src/codeintel/build/hamilton/native/target_overrides.py`. **Done**
5. Remove placeholder helpers from `src/codeintel/build/contracts.py` if they are not used by tests
   after migration (`OutputContract.simple`, `placeholder_table_schema`, `is_placeholder_schema`).
   **Done**
6. Update tests that referenced overrides or placeholders to use explicit contracts or registry
   schemas. **Done**

**Acceptance**
- No references to `TARGET_SPEC_OVERRIDES` or `placeholder_table_schema` remain in `src/`.
- `OutputContract.tables` is fully populated with real TableSchema objects for all targets.
- Build-time schema validation fails fast for missing table schemas.

---

### W2: Remove Template Package and Test-Only Utilities
**Objective**: Remove legacy template fallback infrastructure and keep only native helpers used by
production code.

**Implementation Steps**
1. Extract `executor_materialize` into a non-template module.
   - Candidate new location: `src/codeintel/build/hamilton/materialization_helpers.py`.
   - Update native graph modules importing `executor_materialize`.
2. Remove unused template utilities and tests:
   - `src/codeintel/build/hamilton/templates/all_targets.py`
   - `src/codeintel/build/hamilton/templates/tool_pipeline.py`
   - `src/codeintel/build/hamilton/templates/multi_table_pipeline.py`
   - `src/codeintel/build/hamilton/templates/rows_helpers.py`
   - `src/codeintel/build/hamilton/templates/materialize_template.py` (after extraction)
   - `src/codeintel/build/hamilton/templates/__init__.py`
3. Remove template-only tests:
   - `tests/build/hamilton/test_executor_pipeline_template.py`
   - `tests/build/hamilton/test_phase2_ibis_pipeline_template.py`
   - `tests/build/hamilton/test_multi_table_pipeline_template.py`
   - `tests/build/hamilton/test_rows_pipeline_helpers.py`
4. Update any native modules that were using template-only helpers to import from
   new native helper locations.

**Acceptance**
- No `codeintel.build.hamilton.templates` imports exist in `src/` or `tests/`.
- Native DAG still materializes via the extracted helper(s).

---

### W3: Remove Stub Target Generation and Template Fallback Docs
**Objective**: Eliminate unused stub target nodes and fallback narratives.

**Implementation Steps**
1. Remove stub-generation logic in `src/codeintel/build/hamilton/nodes/support_factory.py`:
   - Delete `_create_stub_target_node_function()` and any call paths.
   - Remove `include_target_stubs` from `SupportGenerationOptions` if no longer used.
2. Remove `get_template_module()` in `src/codeintel/build/hamilton/templates/all_targets.py` (or
   remove entire module as part of W2).
3. Update docstrings and module comments referencing Phase 1 templates or fallback overrides.
   - Focus on `src/codeintel/build/hamilton/native/analytics/*.py` headers and
     `src/codeintel/build/hamilton/templates/*` (as part of removal).

**Acceptance**
- No stub target nodes are generated or referenced.
- Module headers accurately reflect native-only execution.

---

### W4: Remove ExecutionResult Compatibility Fallback (`errors` -> `warnings`)
**Objective**: Standardize compute results and remove compatibility behavior in
`ExecutionResult` conversion.

**Implementation Steps**
1. Identify all compute result dataclasses that expose `errors` or omit `warnings`.
   - Use the W0 inventory results to build a migration list.
2. For each compute result type:
   - Rename `errors` to `warnings` (or map to `warnings` with a deprecation period in code).
   - Ensure `warnings` is typed as `tuple[str, ...]` (or compatible sequence type).
3. Update `ExecutionResultLike` to remove the `errors` fallback in `_extract_warnings()` and
   drop any compatibility branches.
4. Update tests and fixtures to use `warnings` consistently.

**Acceptance**
- `ExecutionResultLike` does not reference `errors`.
- No compute result object defines an `errors` attribute.
- All tests pass with warnings-only semantics.

---

### W5: Cleanup, Docs, and Validation
**Objective**: Finalize documentation, remove dead references, and validate quality gates.

**Implementation Steps**
1. Update relevant documentation to reflect native-only, contract-first design.
   - Candidate docs: `docs/architecture.md`, `docs/centralization_big_move_*.md` (only if needed).
2. Remove dead references in error messages or guidance that mention overrides/templates.
   - Example: `src/codeintel/build/errors.py` guidance text.
3. Run quality gates and segmented tests per AGENTS.md:
   - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
   - `uv run pytest -q tests/build`
   - `uv run pytest -q tests/ingestion` (if touched)
   - `uv run pytest -q tests/graphs` (if touched)

**Acceptance**
- Quality report is clean (ruff, pyright, pyrefly all green).
- Targeted test segments pass.

## Rollout Plan
1. Land W1 (schema/contract migration) first to lock in contract correctness.
2. Land W2 + W3 to remove templates and stubs.
3. Land W4 once all compute results are migrated.
4. Conclude with W5 cleanup and validation.

## Risks and Mitigations
- Risk: Missing schema registrations break contract compilation.
  - Mitigation: add a preflight check that enumerates contract table keys and validates registry
    coverage before removing placeholders.
- Risk: Removing template helpers breaks native modules that relied on template imports.
  - Mitigation: extract required helpers into native modules before deletion; update imports in
    a single changeset.
- Risk: `errors` -> `warnings` migration misses a compute result type.
  - Mitigation: add a temporary test that fails if any compute result exposes `errors`.

## Definition of Done
- Legacy files removed, no `templates` package remains.
- OutputContracts carry full TableSchema data without overrides or placeholders.
- No stub target nodes; native-only DAG paths confirmed.
- Compatibility fallbacks removed; warnings-only compute results enforced.
- Quality gates and targeted tests pass.
