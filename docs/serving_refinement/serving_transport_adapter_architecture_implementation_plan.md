# Serving Consolidation + Transport-Adapter Architecture — Implementation Plan

## Goals

1. **Single source of truth** for:
   - Export formats + MIME types + file suffixes
   - Tooling/runtime version metadata + mismatch warnings
   - Filter operator semantics
   - Error catalog and HTTP/MCP parity mapping
2. **Transport-adapter architecture**:
   - Move “what we do” (semantic/search/meta/export operations) into a transport-agnostic layer.
   - Keep “how we expose it” (FastAPI routes, FastMCP tools/resources) as thin adapters.
3. **Best-in-class maintainability**:
   - Minimize duplication across `src/codeintel/serving/http` and `src/codeintel/serving/mcp`.
   - Make contracts explicit and testable.
   - Keep concurrency model aligned with **`uvicorn_workers=1`** and the FastMCP design.

## Constraints / Non-Goals

- **FastMCP-specific**: MCP surfaces must be implemented using `fastmcp` (gofastmcp 2.x), not the `mcp` library.
- **Single-process serving**: Align design with `uvicorn_workers=1` (in-process memory caches and state are acceptable).
- **Immediate migration acceptable**: This is a design-stage refactor; we can remove legacy code directly.
- **Do not touch `src/codeintel/build`** as part of this work (parallel effort may be ongoing there).

## Current Consolidation Opportunities (Scope to Implement)

This plan implements the consolidation recommendations captured during the serving review:

1. Remove dead/legacy MCP response envelope modules:
   - `src/codeintel/serving/mcp/response.py`
   - `src/codeintel/serving/mcp/models.py`
2. Remove unused helpers/constants:
   - `src/codeintel/serving/mcp/app.py` (`_fastmcp_type_globals`, `_ERR_*`, `_view_not_found_msg`)
   - `src/codeintel/serving/mcp/response_models.py` (`_pydantic_runtime_types`)
3. Consolidate duplicated “tooling meta” logic:
   - `_runtime_versions*` + `_tooling_mismatch_warnings*` duplicated between
     `src/codeintel/serving/mcp/app.py` and `src/codeintel/serving/mcp/resources.py`
4. Consolidate export registry + formats:
   - MIME constants duplicated between `src/codeintel/serving/mcp/resources.py` and
     `src/codeintel/serving/mcp/resource_store.py`
   - HTTP hard-coded export MIME strings in `src/codeintel/serving/http/routes/v1/export.py`
     and `src/codeintel/serving/http/streaming.py`
   - `ExportFormat` type alias duplicated between `src/codeintel/serving/semantic/models.py` and
     `src/codeintel/serving/mcp/response_models.py`
5. Centralize Protocols (reduce “mini-protocol drift”):
   - Local Protocols in `src/codeintel/serving/mcp/resources.py` and similar shapes in prompts/app
6. Consolidate HTTP boilerplate:
   - Repeated `isinstance(...)` dependency checks (semantic/search/export routes)
   - Repeated query metrics assembly and logging patterns
7. Consolidate `SemanticQueryKernel` internals:
   - Factor a shared “resolve → normalize → compile” pipeline reused by query/explain/export/fingerprint/sql
8. Consolidate filter operator semantics:
   - Remove duplicate “allowed ops by dtype” logic in prompts; use the authoritative query-builder semantics
9. Consolidate error modeling across transports:
   - Unify MCP error catalog (`src/codeintel/serving/mcp/errors.py`) and HTTP Problem Details
     (`src/codeintel/serving/http/errors.py`) via a canonical serving error catalog and adapters.
10. Consolidate runtime/app factories:
    - De-duplicate DB manager + kernel construction between `src/codeintel/serving/http/app.py` and
      `src/codeintel/serving/mcp/server.py`
11. Implement the high-leverage **transport-adapter architecture**:
    - Create a transport-agnostic operations layer for semantic/search/meta/export
    - Keep FastAPI and FastMCP as thin adapters around it

## Guiding Architecture

### Target Layering

1. **Transport-agnostic operations layer** (`codeintel.serving.operations.*`)
   - Pure(ish) orchestration over the kernel + persistence/registry.
   - Returns domain models (Pydantic models are fine if they are transport-neutral).
   - Raises domain errors (typed exceptions with stable error codes).
2. **Transport adapters**
   - HTTP adapter (`codeintel.serving.http.*`): request parsing/validation, response shaping,
     Problem Details mapping, streaming/file responses, dependency wiring.
   - FastMCP adapter (`codeintel.serving.mcp.*`): tool/resource registration, FastMCP middleware,
     structured MCP error mapping, ResourceStore integration, progress reporting.

### Canonical “Single Source of Truth” Modules (Proposed)

- `codeintel.serving.export.formats`
  - `ExportFormat` (canonical alias)
  - `ExportFormatSpec` (format → `mime_type`, `suffix`)
  - `EXPORT_FORMATS` mapping and helpers
- `codeintel.serving.meta.tooling`
  - `runtime_versions()` and `tooling_mismatch_warnings(...)`
- `codeintel.serving.semantic.filter_ops`
  - `allowed_ops_for_column_type(...)`
  - optional: `parse_filter_value(...)` helpers (shared with prompts)
- `codeintel.serving.serving_errors` (or `codeintel.serving.errors.catalog`)
  - Canonical error code catalog
  - Domain exception types
  - Adapters: MCP `ErrorResponse` mapping + HTTP `ProblemDetail` mapping
- `codeintel.serving.runtime`
  - `build_db_manager(cfg)`
  - `build_kernel(db, cfg)`
  - `build_serving_state(cfg, db, kernel)`

## Phased Implementation Plan

Each phase has (1) concrete refactor tasks, (2) acceptance gates, and (3) rollback notes.
Phases are ordered to deliver quick wins early while setting up architecture for the larger changes.

### Phase 0 — Baseline + Guardrails

**Objective**: lock in confidence that serving changes are safe and measurable.

Tasks
- Ensure serving-only checks are stable and fast to run:
  - `uv run ruff check --fix src/codeintel/serving tests/serving`
  - `uv run pyright --warnings --pythonversion=3.13 src/codeintel/serving tests/serving`
  - `uv run pyrefly check src/codeintel/serving tests/serving`
  - `uv run pytest -q -o addopts= --confcutdir=tests/serving tests/serving`
- Verify `src/codeintel/serving/contracts/check_operation_contracts.py` executes as part of
  local validation (or add a targeted test that runs it).

Acceptance gates
- All serving-only tests pass.
- Ruff/Pyright/Pyrefly clean for `src/codeintel/serving` + `tests/serving`.

Rollback notes
- None; this phase is diagnostic only.

---

### Phase 1 — Delete Dead/Legacy Code (Zero-Risk Consolidation)

**Objective**: remove unused modules and helpers to reduce confusion and reduce refactor surface area.

Tasks
1. Delete:
   - `src/codeintel/serving/mcp/response.py`
   - `src/codeintel/serving/mcp/models.py`
2. Remove unused items:
   - In `src/codeintel/serving/mcp/app.py` remove `_fastmcp_type_globals`, `_ERR_*`,
     `_view_not_found_msg` (and any now-unused imports).
   - In `src/codeintel/serving/mcp/response_models.py` remove `_pydantic_runtime_types`
     (and any now-unused imports).
3. Run a strict reference check to ensure nothing imports these names.

Acceptance gates
- Phase 0 gates.

Rollback notes
- If any downstream dependency unexpectedly referenced these modules, re-add in a single revert.
  (In practice, this should be a clean deletion.)

---

### Phase 2 — Export Registry Consolidation (Formats, MIME Types, Suffixes)

**Objective**: one canonical registry for export serialization formats across HTTP + FastMCP.

Tasks
1. Create `src/codeintel/serving/export/formats.py` with:
   - `ExportFormat = Literal["json", "ndjson", "parquet", "arrow"]` (canonical)
   - `ExportFormatSpec` (dataclass: `format`, `mime_type`, `suffix`)
   - `EXPORT_FORMATS: dict[ExportFormat, ExportFormatSpec]`
   - helpers:
     - `mime_type_for_export_format(fmt)`
     - `suffix_for_export_format(fmt)`
2. Update callers to use the registry:
   - MCP:
     - `src/codeintel/serving/mcp/resource_store.py` (artifact path selection + MIME selection)
     - `src/codeintel/serving/mcp/resources.py` (resource `mime_type` checks and declarations)
   - HTTP:
     - `src/codeintel/serving/http/streaming.py` (NDJSON media type)
     - `src/codeintel/serving/http/routes/v1/export.py` (Parquet/Arrow/JSON media types)
3. Canonicalize `ExportFormat` typing:
   - Replace duplicated aliases in:
     - `src/codeintel/serving/semantic/models.py`
     - `src/codeintel/serving/mcp/response_models.py`
   - Both should import `ExportFormat` from `codeintel.serving.export.formats`.
4. Ensure export format validation uses a single validator:
   - Either (a) Pydantic validation on the canonical alias or (b) a shared normalization function
     in `formats.py` used by both tool inputs and HTTP payload models.

Acceptance gates
- Phase 0 gates.
- Add/extend tests asserting:
  - All formats have stable mime/suffix mapping.
  - HTTP export endpoints emit correct `Content-Type` for each format.
  - MCP store/resource both accept/emit the same MIME types.

Rollback notes
- Keep registry small and dependency-free to avoid import cycles; if cycles appear, move registry to
  `codeintel.serving.semantic` or `codeintel.serving.common`.

---

### Phase 3 — Consolidate Tooling Meta (Runtime Versions + Mismatch Warnings)

**Objective**: eliminate duplicated tooling-meta logic and guarantee consistent warnings/fields.

Tasks
1. Add `src/codeintel/serving/mcp/tooling_meta.py` with:
   - `runtime_versions() -> dict[str, str]`
   - `tooling_mismatch_warnings(snapshot_tools: dict[str, object], runtime: dict[str, str])`
     returning a canonical list/tuple of warning strings
2. Refactor:
   - `src/codeintel/serving/mcp/app.py` to call `tooling_meta.runtime_versions()` and
     `tooling_meta.tooling_mismatch_warnings(...)`
   - `src/codeintel/serving/mcp/resources.py` similarly
3. Ensure both surfaces produce the same schema shape for `runtime_versions` + warnings.

Acceptance gates
- Phase 0 gates.
- Add tests for mismatch warnings determinism:
  - stable ordering
  - stable string format

Rollback notes
- Ensure tooling meta remains in MCP package only if it is MCP-only; otherwise move to a shared
  `codeintel.serving.meta.runtime_versions` module.

---

### Phase 4 — Centralize Protocols for MCP (Reduce “Mini-Protocol Drift”)

**Objective**: have exactly one place that defines the minimal interfaces needed by the MCP surface.

Tasks
1. Create `src/codeintel/serving/mcp/protocols.py`:
   - `ServingDBManagerProtocol`
   - `ServingSnapshotPointerProtocol`
   - `SemanticKernelProtocol` (only the methods needed by MCP tools/resources/prompts)
2. Update:
   - `src/codeintel/serving/mcp/resources.py` to import protocols rather than defining them inline
   - `src/codeintel/serving/mcp/prompts.py` to use the shared kernel protocol
   - `src/codeintel/serving/mcp/app.py` (if it defines additional protocols) to reuse shared protocols

Acceptance gates
- Phase 0 gates.
- Ensure no circular imports are introduced (prefer `Protocol` + minimal method sets).

Rollback notes
- If import cycles appear, move protocols to `codeintel.serving.protocols` (top-level) and keep them
  extremely lightweight.

---

### Phase 5 — HTTP Route Consolidation (Type Checks + Metrics)

**Objective**: remove duplicated boilerplate while preserving (or improving) runtime safety.

Tasks
1. Choose a single runtime-hardening strategy:
   - **Option A (preferred)**: rely on typed dependencies and keep runtime checks only in
     dependency providers (`get_kernel`, `_get_state`), not per-route.
   - **Option B**: keep route-level runtime checks, but implement one shared helper that all routes call.
2. Implement shared primitives:
   - `src/codeintel/serving/http/route_utils.py` (proposed)
     - `ensure_fastapi_injected_types(...) -> None` (Option B)
     - `metrics_context(...)` builder helpers for `QueryMetrics`
3. Refactor routes:
   - `src/codeintel/serving/http/routes/v1/semantic.py`
   - `src/codeintel/serving/http/routes/v1/search.py`
   - `src/codeintel/serving/http/routes/v1/export.py`
   to use the shared helpers consistently.
4. Consider aligning route signatures around `Kernel` dependency:
   - `from codeintel.serving.http.dependencies import Kernel`
   - Replace `kernel: SemanticQueryKernel = Depends(get_kernel)` with `kernel: Kernel`

Acceptance gates
- Phase 0 gates.
- Ensure OpenAPI remains unchanged (except for docstring improvements).

Rollback notes
- Keep refactor mechanical; if anything becomes less clear, revert and do a smaller consolidation.

---

### Phase 6 — SemanticQueryKernel Consolidation (Resolve + Compile Context)

**Objective**: reduce duplication and prevent drift across query/explain/export/fingerprint/sql generation.

Tasks
1. Introduce internal “compiled query context” dataclasses (private to kernel module):
   - `ResolvedViewContext`:
     - view spec, allowed columns, column types, snapshot identity, schema hash
   - `CompiledSemanticQuery`:
     - normalized request, ibis expression (or SQL + params), query hash, sql fingerprint
2. Implement a single internal helper:
   - `_compile_semantic_query(request: SemanticQueryRequest | SemanticExportRequest) -> CompiledSemanticQuery`
   - This helper must be the sole place that:
     - resolves view ID → registry/view spec
     - normalizes select/order/filters
     - builds query plan (QueryBuilder)
     - compiles SQL and computes fingerprints/hashes
3. Update public entrypoints to reuse the same context:
   - `query`, `explain`, `compile_query_sql`, `export_fingerprint`, `export_rows`, `export_sql`,
     `export_to_parquet`, `export_to_arrow_ipc`
4. Verify that both MCP and HTTP surfaces now inherit consistent normalization behavior.

Acceptance gates
- Phase 0 gates.
- Add/extend tests for normalization parity:
  - same request produces identical `query_hash`, `schema_hash`, compiled SQL, and fingerprints across
    query/explain/export_sql where applicable.

Rollback notes
- Keep compilation helpers private; do not prematurely freeze them as public API.

---

### Phase 7 — Consolidate Filter Operator Semantics (Prompts + Query Builder)

**Objective**: eliminate duplicated “ops-by-dtype” logic and guarantee prompts only suggest valid filters.

Tasks
1. Create `src/codeintel/serving/semantic/filter_ops.py`:
   - `allowed_ops_for_column_type(column_type: str | None) -> tuple[Op, ...]`
     - must reflect what the query builder supports
   - `parse_filter_value(column_type: str | None, op: Op, raw: str) -> object` (optional)
2. Update `src/codeintel/serving/semantic/query_builder.py` to import and use the shared op catalog
   (or build the shared catalog directly from query_builder internals and export it).
3. Update `src/codeintel/serving/mcp/prompts.py`:
   - Replace `_allowed_ops_for_dtype` with `semantic.filter_ops.allowed_ops_for_column_type`
   - Optionally share parsing logic with `parse_filter_value` to reduce drift.

Acceptance gates
- Phase 0 gates.
- Add tests ensuring:
  - every op prompts can emit is accepted by query builder
  - invalid ops are rejected deterministically

Rollback notes
- If query builder needs different behavior by engine, keep the catalog engine-aware but still shared.

---

### Phase 8 — Canonical Error Catalog + HTTP/MCP Parity

**Objective**: unify error semantics across transports and eliminate drift in status/retryability/guidance.

Tasks
1. Choose the canonical catalog location:
   - **Recommended**: promote `src/codeintel/serving/mcp/errors.py` catalog into a serving-wide catalog
     (e.g., `src/codeintel/serving/errors/catalog.py`) because it already contains stable error codes and
     `http_status` parity fields.
2. Define domain exceptions:
   - e.g., `ServingDomainError(code: str, *, params: dict[str, object] | None = None)`
   - Specialized subclasses for common errors (`ViewNotFoundError`, `InvalidQueryError`, etc.)
3. Create adapter mappers:
   - HTTP mapper: `domain_error → ProblemDetail` (`application/problem+json`)
   - MCP mapper: `domain_error → ErrorResponse` (existing canonical MCP error model)
4. Refactor HTTP errors:
   - Replace or wrap `ServingError` with the domain error mechanism
   - Keep HTTP-only concerns (headers, instance URL, correlation_id) inside the adapter
5. Refactor MCP error mapping middleware to map domain errors to MCP errors consistently.
6. Update contracts/tests:
   - Assert that “equivalent” operations in HTTP and MCP return consistent error semantics
     (status code, retryable guidance, stable code).

Acceptance gates
- Phase 0 gates.
- Add tests for parity:
  - For each canonical error, verify HTTP status matches `http_status`
  - Verify MCP error `code` matches the canonical catalog key

Rollback notes
- If the refactor is large, do it in two steps:
  - Step 1: introduce domain errors + adapters without removing existing `ServingError`
  - Step 2: migrate callers; delete old path

---

### Phase 9 — Runtime Builder Consolidation (HTTP + MCP)

**Objective**: ensure identical configuration and lifecycle for DB manager + kernel across transports.

Tasks
1. Create `src/codeintel/serving/runtime.py`:
   - `build_db_manager(cfg: ServingSettings) -> ServingDBManager`
   - `build_kernel(db: ServingDBManager, cfg: ServingSettings) -> SemanticQueryKernel`
   - `build_state(cfg, db, kernel) -> ServingState`
2. Refactor:
   - `src/codeintel/serving/http/app.py` to use `runtime.py` builders
   - `src/codeintel/serving/mcp/server.py` to use `runtime.py` builders
3. Ensure auth hardening is applied consistently:
   - `cfg.validate_auth_for_host()`
   - `cfg.validate_mcp_single_worker(...)` when mounting MCP

Acceptance gates
- Phase 0 gates.
- Ensure there is no behavior drift between HTTP and MCP instantiation.

Rollback notes
- Keep builder functions thin; avoid moving “transport-only” options into `runtime.py`.

---

### Phase 10 — Transport-Adapter Architecture (High-Leverage Strategic Refactor)

**Objective**: centralize serving business logic in an operations layer, leaving HTTP/FastMCP as thin adapters.

Deliverable overview
- New package: `src/codeintel/serving/operations/`
  - `operations/errors.py` — domain error types + helpers
  - `operations/models.py` — transport-agnostic response models (or reuse existing semantic/search models)
  - `operations/semantic.py` — semantic ops (catalog/describe/query/explain/compile/export_sql/export_rows)
  - `operations/search.py` — search ops
  - `operations/meta.py` — meta ops
  - `operations/export.py` — export orchestration shared across transports when possible
  - `operations/protocols.py` — kernel/db abstractions if needed

Step-by-step migration plan
1. Introduce an `Operations` facade (composition over inheritance):
   - `ServingOperations(kernel: SemanticQueryKernel, *, settings: ServingSettings)`
   - Methods mirror the public contract:
     - `catalog()`, `describe(view_id)`, `query(req)`, `explain(req)`, `compile_sql(req)`
     - `search(req)`
     - `export_rows(req)`, `export_sql(req)`, `export_fingerprint(req)`
2. Keep the kernel as a dependency:
   - Operations should call kernel methods initially (thin wrapper), then gradually move shared orchestration
     and normalization into operations if it results in a cleaner separation.
3. Move cross-transport concerns into operations:
   - Query hashing and schema hashing (if currently scattered)
   - SQL fingerprinting (canonical location for `sqlglot` hashing)
   - Stable “resolved view context” normalization (if not already consolidated in Phase 6)
4. Refactor HTTP adapter:
   - Replace direct kernel usage with `ServingOperations`.
   - HTTP remains responsible for:
     - dependency injection + auth (`require_api_key`)
     - Problem Details mapping
     - streaming/file response mechanics
     - metrics emission (or call a shared metrics hook)
5. Refactor FastMCP adapter:
   - Replace direct kernel usage with `ServingOperations`.
   - MCP remains responsible for:
     - tool/resource registration
     - ResourceStore lifecycle and file artifacts
     - progress reporting (`ctx.report_progress`) and sampling
     - middleware stack configuration
6. Unify export logic carefully:
   - Operations should expose:
     - canonical `export_rows` iterator
     - canonical `export_sql` + safety checks (`assert_single_select_statement`)
     - canonical `export_fingerprint`
   - Adapters decide representation:
     - HTTP adapter writes Parquet/Arrow temporary files and returns `FileResponse`
     - MCP adapter writes artifacts into `ResourceStore` and exposes resources
7. Update contracts:
   - Extend `src/codeintel/serving/contracts/check_operation_contracts.py` to validate that:
     - HTTP and MCP operations still exist
     - schemas remain coherent
     - error parity is preserved for a minimal matrix of known failures

Acceptance gates
- Phase 0 gates.
- New operations layer has direct unit tests (no transport) validating:
  - query normalization
  - error raising semantics
  - export iterator behavior (including limit/offset)
- HTTP + MCP adapter tests remain green with no contract regressions.

Rollback notes
- Build `ServingOperations` as a wrapper first; only move logic out of kernel once adapter refactor is stable.

## Implementation Ordering Summary (Recommended)

1. Phase 1 (delete dead code)
2. Phase 2 (export registry)
3. Phase 3 (tooling meta)
4. Phase 4 (protocols)
5. Phase 6 (kernel resolve/compile consolidation)
6. Phase 7 (filter ops consolidation)
7. Phase 8 (canonical error catalog)
8. Phase 9 (runtime builder)
9. Phase 5 (HTTP boilerplate cleanup; can be earlier if desired)
10. Phase 10 (transport-adapter architecture)

## Quality Gates (Run After Each Phase)

Serving-only validation (recommended during parallel build work)
- `uv run ruff check --fix src/codeintel/serving tests/serving`
- `uv run pyright --warnings --pythonversion=3.13 src/codeintel/serving tests/serving`
- `uv run pyrefly check src/codeintel/serving tests/serving`
- `uv run pytest -q -o addopts= --confcutdir=tests/serving tests/serving`

Repo-level validation (recommended once parallel build work stabilizes)
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- `uv run pytest -q`

## Expected Outcomes

- **Lower cognitive load**: fewer duplicated constants/types and fewer “two implementations of the same idea”.
- **Higher hardness**: canonical error semantics, canonical query compilation normalization, fewer drift bugs.
- **Better extensibility**: new endpoints/tools/resources can be added by implementing operations once and
  exposing through thin HTTP/FastMCP adapters.
- **Maintainability**: significantly smaller per-transport modules and clearer contracts between layers.
