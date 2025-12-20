# Serving Consolidation + Extensibility - Implementation Plan

## Executive Summary

This plan consolidates serving functionality across HTTP and FastMCP adapters to:

- Reduce duplicated logic and stringly-typed contracts.
- Harden behavior via typed models and single-source-of-truth registries.
- Improve extensibility by centralizing query planning, export orchestration, and metrics.

Scope covers both high-leverage consolidations and secondary cleanups identified in the
serving review. The refactor preserves existing behavior while tightening contracts.

## Constraints and Assumptions

- Design phase: immediate migration allowed (no deprecation pathway).
- FastMCP is guaranteed >= 2.14 everywhere.
- Serving must remain transport-agnostic at core; adapters remain thin.
- All changes must pass Ruff, Pyright, Pyrefly, and serving tests.

## Goals (Definition of Done)

1. **Single source of truth** for:
   - metrics definitions
   - export execution logic
   - query planning/compilation
   - HTTP route contracts
   - MCP resource URI templates
2. **Typed responses** flow end-to-end:
   - operations return typed models instead of raw dicts
   - adapters only validate/serialize, not rebuild payloads
3. **Reduced adapter coupling**:
   - MCP does not import HTTP modules
   - HTTP does not rely on MCP-specific helpers
4. **Maintain or improve functionality** with no behavior regressions.

## Scope

High leverage
- Transport-agnostic metrics module.
- Shared export execution engine for HTTP + MCP.
- Typed operations layer outputs.
- Shared export snapshot/meta assembly.
- Consolidated SQL fingerprinting.
- Centralized query planning pipeline.

Secondary cleanups
- Shared error context assembly.
- Canonical HTTP route registry.
- Canonical resource URI registry/builders.
- Unified feature gating (ServingFeatureSet).

## Phased Implementation Plan

Each phase includes tasks and acceptance gates.

### Phase 0 - Baseline and Safety Nets

Objective: ensure serving-only gates run cleanly before refactors.

Tasks
1. Confirm serving-only checks:
   - `uv run ruff check --fix src/codeintel/serving tests/serving`
   - `uv run pyright --warnings --pythonversion=3.13 src/codeintel/serving tests/serving`
   - `uv run pyrefly check src/codeintel/serving tests/serving`
2. Confirm serving operation contract check:
   - `uv run python -m codeintel.serving.contracts.check_operation_contracts`

Acceptance gates
- All commands complete without errors.

---

### Phase 1 - Transport-Agnostic Metrics

Objective: remove MCP dependency on HTTP modules.

Tasks
1. Create `src/codeintel/serving/metrics.py`:
   - Move `QueryMetrics` and `log_query_metrics` from HTTP.
2. Update imports in:
   - HTTP route modules
   - MCP tools
   - any other serving modules
3. Remove or replace `src/codeintel/serving/http/metrics.py`:
   - preferred: delete and update imports everywhere.

Acceptance gates
- Serving-only gates from Phase 0.
- Tests in `tests/serving/http/test_metrics.py` updated if module path changes.

---

### Phase 2 - Shared Export Execution Engine

Objective: one export orchestration path for HTTP and MCP.

Tasks
1. Create `src/codeintel/serving/export/engine.py`:
   - Define `ExportExecutionPlan`:
     - `format`, `content_type`, `filename`, `row_iter`, `write_file_fn`, `row_count`
   - Implement `build_export_plan(ops, request, *, query_hash, schema_hash)`:
     - NDJSON row stream
     - JSON list for HTTP
     - Parquet/Arrow file writers
2. Refactor HTTP export dispatch:
   - `src/codeintel/serving/http/export_dispatch.py` uses `build_export_plan`.
3. Refactor MCP export dispatch:
   - `src/codeintel/serving/mcp/export_dispatch.py` uses same plan and selects store writer.
4. Ensure metrics row_count and headers remain stable.

Acceptance gates
- Export tests pass (`tests/serving/http/test_export.py`).
- MCP resource tests remain stable where export data is read.

---

### Phase 3 - Typed Operations Outputs

Objective: eliminate dict-returning operations for catalog/describe/meta.

Tasks
1. Update `ServingKernelProtocol` to return typed models:
   - `SemanticCatalogResponse`
   - `SemanticViewDescriptionResponse`
   - `ServingMeta` (new typed model in `serving/meta/models.py`)
2. Update `ServingOperations` to return typed models directly.
3. Update HTTP routes to return models without `model_validate`.
4. Update MCP tools/resources to use typed responses directly.

Acceptance gates
- Pyright and Pyrefly in serving pass with no Any leaks.
- JSON outputs match prior schemas.

---

### Phase 4 - Export Snapshot/Metadata Assembly

Objective: ensure export metadata is built via a single canonical helper.

Tasks
1. Introduce `src/codeintel/serving/export/meta.py`:
   - `build_export_snapshot(pointer, meta_payload)` -> `ServingExportSnapshot`
   - `build_export_spec(...)` helper for `ExportArtifactSpec`
2. Update MCP export tool to use these helpers.
3. Update HTTP export responses to use the same snapshot info where relevant.

Acceptance gates
- MCP export response schemas unchanged.
- HTTP export response schemas unchanged.

---

### Phase 5 - SQL Fingerprint Consolidation

Objective: unify SQL canonicalization and hashing.

Tasks
1. Move `sqlglot_canonical_sha256` into `src/codeintel/serving/semantic/fingerprints.py`
   (or a new `serving/sql.py`) and update imports.
2. Delete `src/codeintel/serving/mcp/sql_fingerprint.py`.

Acceptance gates
- No references to deleted module.
- Hashes remain stable for identical SQL input.

---

### Phase 6 - Query Planning Pipeline

Objective: centralize resolve -> plan -> compile logic.

Tasks
1. Create `src/codeintel/serving/semantic/planner.py`:
   - `SemanticQueryPlanner` with:
     - `resolve_view_context`
     - `plan_query`
     - `compile_query`
     - `compile_export`
2. Replace duplicated logic in `src/codeintel/serving/semantic/kernel.py`.
3. Ensure query/explain/export/fingerprint paths all use shared planner.

Acceptance gates
- Semantic query tests pass.
- Export SQL/plan behavior unchanged.

---

### Phase 7 - Shared Error Context Assembly

Objective: consistent error metadata for HTTP and MCP.

Tasks
1. Add `build_error_context_*` helpers in `src/codeintel/serving/errors/mapping.py`:
   - `from_http_request`
   - `from_mcp_context`
2. Replace custom context extraction in:
   - `src/codeintel/serving/http/errors.py`
   - `src/codeintel/serving/mcp/middleware_errors.py`

Acceptance gates
- HTTP ProblemDetails and MCP ErrorResponse remain schema-identical.

---

### Phase 8 - Canonical HTTP Route Registry

Objective: a single definition for serving HTTP routes.

Tasks
1. Create `src/codeintel/serving/contracts/http_routes.py`:
   - `RouteSpec(method, path, name)`
   - `SERVING_HTTP_ROUTES` list
2. Update `check_operation_contracts` to reference this registry.
3. Optionally use registry for docs generation or tests.

Acceptance gates
- Contract check passes with no drift.

---

### Phase 9 - Canonical Resource URI Registry

Objective: eliminate duplicated URI string literals.

Tasks
1. Create `src/codeintel/serving/uris.py`:
   - constants for all canonical URIs and templates.
   - small builder helpers (e.g., `export_uri(export_id)`).
2. Replace URI string literals across:
   - MCP tools
   - MCP resources
   - MCP models where examples are defined
3. Keep docstrings aligned with new constants.

Acceptance gates
- MCP tests using resource URIs remain stable.

---

### Phase 10 - Unified Feature Gating

Objective: consistent feature toggles across adapters.

Tasks
1. Introduce `ServingFeatureSet` in `src/codeintel/serving/settings.py` or
   `src/codeintel/serving/features.py`.
2. Compute derived flags from `ServingSettings`:
   - enable_export_http, enable_export_mcp, enable_search, enable_meta, etc.
3. Update `mcp/app.py` and HTTP routes to use `ServingFeatureSet`.

Acceptance gates
- Feature toggles behave identically to current behavior.

---

### Phase 11 - Cleanup and Documentation

Objective: consolidate docs and remove stale references.

Tasks
1. Update docs in `docs/serving_refinement` to reflect:
   - new module locations
   - new single-source registries
2. Remove legacy references to old modules if any remain.

Acceptance gates
- `rg` confirms no references to deleted modules.

## Verification Checklist

- `uv run ruff check --fix src/codeintel/serving tests/serving`
- `uv run pyright --warnings --pythonversion=3.13 src/codeintel/serving tests/serving`
- `uv run pyrefly check src/codeintel/serving tests/serving`
- `uv run python -m codeintel.serving.contracts.check_operation_contracts`
- `uv run pytest -q tests/serving`

## Risk Mitigation

- Keep each phase small and validated by serving-only gates.
- Avoid broad refactors in a single change; commit each phase separately.
- Maintain JSON output stability by adding tests for schema parity.

## Rollback Strategy

Each phase is scoped for clean reversion. If any gate fails, revert the phase
or fix forward with targeted adjustments.
