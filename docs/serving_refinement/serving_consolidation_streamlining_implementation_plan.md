# Serving Streamlining + Consolidation — Implementation Plan

## Executive Summary

This plan is a **serving-only** roadmap to consolidate duplicated functionality, tighten boundaries, and reduce
maintenance cost across:

- `src/codeintel/serving/http` (FastAPI adapter)
- `src/codeintel/serving/mcp` (FastMCP adapter)
- `src/codeintel/serving/operations` + `src/codeintel/serving/semantic` (transport-agnostic core)

It is designed to preserve or improve functionality while materially improving:

- **Hardness** (clear invariants, fewer implicit contracts, safer boundaries)
- **Extensibility** (adding endpoints/tools/exports without multi-file drift)
- **Maintainability** (smaller modules, less duplication, clearer layering)

This document is an add-on to (and should remain consistent with):

- `docs/serving_refinement/serving_transport_adapter_architecture_implementation_plan.md`

## Constraints / Assumptions

- **FastMCP-only** for MCP surfaces (no `mcp` library server implementations).
- **`uvicorn_workers=1`** when MCP is mounted (sessionful MCP contract).
- **Design-stage refactor allowed**: we can remove legacy surfaces directly where safe.
- Follow repo quality gates (Ruff + Pyright + Pyrefly, then pytest) as described in `AGENTS.md`.

## Goals (What “Done” Looks Like)

1. **Transport-agnostic core is truly transport-agnostic**
   - HTTP does not depend on MCP packages for “domain” concepts.
   - MCP does not own the canonical error catalog or canonical “meta/introspection” logic.
2. **Single source of truth** modules exist and are actually used everywhere:
   - Snapshot identity model + serialization
   - Serving meta/introspection assembly
   - Error catalog + mapping helpers
   - Export formats + capability classification
3. **Adapters are thin**
   - HTTP route modules are mostly: parse → call ops → shape response.
   - FastMCP modules are mostly: register tool/resource → call ops → shape response.
4. **Module sizes and responsibilities are sane**
   - No “god modules” (`mcp/app.py`, `mcp/response_models.py`, `mcp/resources.py`) that mix unrelated concerns.
5. **Contracts remain checked**
   - `src/codeintel/serving/contracts/check_operation_contracts.py` stays green and is expanded where useful.

## Scope (Items To Implement)

This plan actions the consolidation opportunities identified in the serving review:

1. Make `codeintel.serving.errors` **truly transport-agnostic** (remove HTTP → MCP dependency chain).
2. Consolidate “serving meta / introspection” into a **single service** used by HTTP + MCP tools + MCP resources.
3. Move transport-agnostic helpers out of MCP (e.g., `tooling_meta`).
4. Unify snapshot identity representation across the serving layer (typed model + conversion helpers).
5. Collapse repeated metrics + threadpool scaffolding in HTTP routes into a shared runner/helper.
6. Standardize export route dispatch + metadata handling via a single export responder/dispatcher.
7. Make row-count semantics consistent for binary exports (Parquet/Arrow) and export metadata.
8. Centralize export-format capability classification in `codeintel.serving.export.formats`.
9. Trim unused `ResourceStore` API surface (remove dead helpers, keep minimal public API).
10. Split oversized MCP modules by responsibility (tools/resources/models) while keeping stable entrypoints.
11. Replace implicit exception attribute probing in `ServingOperations` with explicit domain exceptions.
12. Consider splitting `SemanticQueryKernel` responsibilities (meta/search/export sub-services).
13. Unify auth policy semantics across transports (bearer/api-key parity and consistent error mapping).
14. Audit and simplify re-export/compat modules (remove or clearly mark “compat-only”).

## Guiding Architecture (Target Layering)

### Target Module Boundaries

1. **Domain / Core (transport-agnostic)**
   - `codeintel.serving.errors.*` — canonical error catalog, domain exceptions, mapping helpers
   - `codeintel.serving.snapshot.*` — typed snapshot identity model + conversions
   - `codeintel.serving.meta.*` — meta/introspection assembler service
   - `codeintel.serving.export.*` — export formats and export capability classification
   - `codeintel.serving.operations.*` — facade used by adapters
   - `codeintel.serving.semantic.*` — kernel + query builder + registry/inventory, etc.
2. **Transport Adapters**
   - `codeintel.serving.http.*` — FastAPI routes/middleware, RFC 9457 Problem Details mapping
   - `codeintel.serving.mcp.*` — FastMCP app/tools/resources, middleware stack, MCP error mapping

### “Single Source of Truth” Modules (End State)

| Topic | Single source | Notes |
|---|---|---|
| Error codes + domain exceptions | `codeintel.serving.errors.*` | MCP and HTTP import from here |
| Snapshot identity model | `codeintel.serving.snapshot.models` | Adapters serialize from this |
| Serving meta/introspection | `codeintel.serving.meta.service` | Used by HTTP `/meta`, MCP tool, MCP resources |
| Export formats + capabilities | `codeintel.serving.export.formats` | `is_text_format`, `supports_preview`, suffix/MIME, etc. |
| Runtime/tool versions | `codeintel.serving.meta.tooling` (or `meta.service`) | Remove from `mcp/` namespace |
| Filter operator semantics | `codeintel.serving.semantic.filter_ops` | Prompts and query builder share it |

## Phased Implementation Plan

Each phase includes:

- **Tasks**: concrete refactor steps
- **Acceptance gates**: how we verify it’s correct
- **Notes/Risks**: typical failure modes and mitigations

### Phase 0 — Baseline, Safety Nets, and Fast Feedback Loops

**Objective**: ensure we can validate serving changes quickly and isolate from unrelated failures.

Tasks
- Ensure we have a stable “serving-only” validation command set:
  - `uv run ruff check --fix src/codeintel/serving tests/serving`
  - `uv run pyright --warnings --pythonversion=3.13 src/codeintel/serving tests/serving`
  - `uv run pyrefly check src/codeintel/serving tests/serving`
  - `uv run pytest -q -o addopts= --confcutdir=tests/serving tests/serving`
- Add (or adjust) a dedicated test that executes:
  - `codeintel.serving.contracts.check_operation_contracts` end-to-end.

Acceptance gates
- Commands above complete without errors.

Notes/Risks
- If `tests/conftest.py` causes cross-package import failures, rely on `--confcutdir=tests/serving`.

---

### Phase 1 — Make Errors Transport-Agnostic (Remove HTTP → MCP Dependency)

**Objective**: domain errors and error catalog must not live under `codeintel.serving.mcp.*`.

Current issue (example)
- `src/codeintel/serving/errors/catalog.py` re-exports from `src/codeintel/serving/mcp/errors.py`,
  which makes “domain errors” transitively MCP-owned.

Tasks
1. Create a new canonical error package:
   - `src/codeintel/serving/errors/models.py`:
     - `ErrorKind`, `ErrorInfo`, `ErrorResponse`, `ErrorContext`
   - `src/codeintel/serving/errors/catalog.py`:
     - `ERROR_CODE_CATALOG` + templates
   - `src/codeintel/serving/errors/exceptions.py`:
     - `CodeIntelDomainError` + specific domain exceptions (Semantic/Export/Auth/etc.)
   - `src/codeintel/serving/errors/mapping.py`:
     - `error_from_code(...)`
     - `exception_to_error_response(...)`
2. Update HTTP error mapping:
   - `src/codeintel/serving/http/errors.py` imports from `codeintel.serving.errors.*` only.
3. Update FastMCP error mapping middleware:
   - `src/codeintel/serving/mcp/middleware_errors.py` imports from `codeintel.serving.errors.*`.
4. Keep compatibility re-exports temporarily (optional but often useful for incremental refactors):
   - `src/codeintel/serving/mcp/errors.py` can become a thin re-export shim (or be deleted if we do a clean cut).
5. Expand/strengthen `src/codeintel/serving/contracts/check_operation_contracts.py`:
   - Explicit parity checks between MCP error payloads and HTTP Problem Details for a representative sample set.

Acceptance gates
- Phase 0 gates.
- Contract check passes and verifies:
  - stable code strings
  - stable http_status mapping
  - stable retryable/kind semantics.

Notes/Risks
- Avoid import cycles: `codeintel.serving.errors.*` must depend only on stdlib + pydantic (and not on HTTP/MCP).

---

### Phase 2 — Unify Snapshot Identity (Typed Model + Conversions)

**Objective**: eliminate multiple divergent “snapshot dict” representations.

Tasks
1. Introduce canonical snapshot model:
   - `src/codeintel/serving/snapshot/models.py`
     - `ServingSnapshotRef` (repo/commit/run_id/published_at/semantic_layer_version, etc.)
     - `from_pointer(pointer)` and `to_dict()` helpers
2. Update core responses to use this model consistently:
   - Replace raw `dict[str, str]` snapshot shapes in:
     - `src/codeintel/serving/semantic/models.py`
     - `src/codeintel/serving/search/models.py`
3. Update kernel + operations to return the canonical snapshot model (or canonical dict derived from it):
   - Remove duplicated `_snapshot_dict` builders where feasible.
4. Update MCP models:
   - Either re-use the canonical snapshot model directly, or make `SnapshotRef` a thin wrapper around it.

Acceptance gates
- Phase 0 gates.
- Contract check ensures MCP/HTTP still emit the expected snapshot fields (schema stability).

Notes/Risks
- If public schemas must remain unchanged, keep the serialized JSON stable and only change internal typing.

---

### Phase 3 — Build a Single Introspection Service (Meta + Environment + Templates)

**Objective**: one assembly path for serving meta/introspection information.

Tasks
1. Create `src/codeintel/serving/meta/service.py`:
   - Inputs:
     - `ServingOperations`
     - `ServingSettings`
   - Outputs:
     - typed “serving meta” response (for MCP tool)
     - JSON dict (for HTTP `/meta`)
     - resource template listing (for MCP resources)
     - environment/runtime mismatch warnings
2. Relocate tooling version reporting out of MCP:
   - Move `src/codeintel/serving/mcp/tooling_meta.py` to `src/codeintel/serving/meta/tooling.py`
     (or fold into `meta/service.py` if you want fewer modules).
3. Refactor callsites:
   - HTTP `/meta` uses the introspection service instead of calling `ops.meta()` directly.
   - MCP `serving_meta` tool uses the introspection service.
   - MCP `codeintel://meta/environment` and `codeintel://meta/resources` resources use the introspection service.

Acceptance gates
- Phase 0 gates.
- Add targeted tests asserting:
  - HTTP `/meta` and MCP `serving_meta` share consistent fields/values for a mocked pointer/kernel.

Notes/Risks
- Keep the introspection service “pure-ish”: avoid performing heavy DB work; prefer reuse of existing cached
  pointer/context where possible.

---

### Phase 4 — Export Format Capabilities as a Single Registry (No More Ad-Hoc Sets)

**Objective**: all format behavior derives from `codeintel.serving.export.formats`.

Tasks
1. Extend `src/codeintel/serving/export/formats.py` with typed helpers:
   - `is_text_export_format(fmt: ExportFormat) -> bool`
   - `supports_preview(fmt: ExportFormat) -> bool`
   - `supports_line_chunks(fmt: ExportFormat) -> bool`
   - `supports_byte_chunks(fmt: ExportFormat) -> bool`
   - `default_export_format() -> ExportFormat` (optional, if useful)
2. Refactor callsites to use these helpers:
   - HTTP export route dispatch (`src/codeintel/serving/http/routes/v1/export.py`)
   - MCP export tool (`src/codeintel/serving/mcp/app.py`)
   - MCP resources (`src/codeintel/serving/mcp/resources.py`)
   - MCP prompts (`src/codeintel/serving/mcp/prompts.py`) should derive format choices from `EXPORT_FORMATS`.

Acceptance gates
- Phase 0 gates.
- Add tests that enumerate all formats and assert consistent capability classification.

---

### Phase 5 — HTTP Route Consolidation (Metrics + Threadpool Runner)

**Objective**: reduce repeated boilerplate in HTTP handlers and standardize metrics.

Tasks
1. Add a shared runner in `src/codeintel/serving/http/route_utils.py` (or a new module):
   - Encapsulate:
     - correlation id retrieval
     - timing measurement
     - `run_in_threadpool(...)`
     - metrics emission on both success and failure
2. Refactor:
   - `src/codeintel/serving/http/routes/v1/semantic.py`
   - `src/codeintel/serving/http/routes/v1/search.py`
   - `src/codeintel/serving/http/routes/v1/export.py` (as much as makes sense)
3. Standardize metric naming:
   - Ensure endpoint names are consistent and stable (e.g., always `/v1/...` vs unversioned alias).

Acceptance gates
- Phase 0 gates.
- Add/adjust tests to assert metrics helper is called with correct values (can be unit tests with a mocked logger).

---

### Phase 6 — Standardize Export Dispatch and Row-Count Semantics

**Objective**: export logic should not drift between MCP and HTTP; row-count should have coherent meaning.

Tasks
1. Decide row-count policy for binary exports:
   - Option A (preferred): make `SemanticQueryKernel.export_to_parquet(...) -> int` and return row_count.
   - Option B: allow “unknown row_count” in metadata (make it optional) and handle that consistently in all
     responses and metrics.
2. Implement the chosen policy end-to-end:
   - Kernel (`src/codeintel/serving/semantic/kernel.py`)
   - Operations (`src/codeintel/serving/operations/ops.py`)
   - HTTP export route metrics (`src/codeintel/serving/http/routes/v1/export.py`)
   - MCP `ResourceStore` metadata (`src/codeintel/serving/mcp/resource_store.py`)
   - MCP `semantic_export` tool response/meta (`src/codeintel/serving/mcp/app.py`)
3. Introduce an “export responder/dispatcher”:
   - One place in HTTP that maps `ExportFormat -> Response builder`.
   - One place in MCP that maps `ExportFormat -> ResourceStore writer`.

Acceptance gates
- Phase 0 gates.
- New tests:
  - Binary export row_count is correct (or explicitly None) in:
    - HTTP metrics
    - MCP export metadata sidecar
    - MCP export meta resource response.

---

### Phase 7 — Prune `ResourceStore` Public API Surface

**Objective**: shrink the API surface area to only what adapters use.

Tasks
- Remove unused methods (`put_json`, `put_ndjson`) if they are not called anywhere.
- Keep the minimal API:
  - `put_with_metadata`
  - `put_with_metadata_stream`
  - `put_generated_file_with_metadata`
  - `get`, `get_meta`, `get_preview`
  - `delete`, `cleanup_expired`, `mark_cancelled`
- Ensure docstrings stay accurate and sidecars remain canonical.

Acceptance gates
- Phase 0 gates.
- No unused code paths remain; all callers updated.

---

### Phase 8 — Decompose MCP Modules (Tools/Resources/Models)

**Objective**: `mcp/` becomes maintainable by splitting responsibilities into smaller modules.

Tasks
1. Split `src/codeintel/serving/mcp/app.py` into:
   - `src/codeintel/serving/mcp/tools/catalog.py`
   - `src/codeintel/serving/mcp/tools/describe.py`
   - `src/codeintel/serving/mcp/tools/query.py`
   - `src/codeintel/serving/mcp/tools/explain.py`
   - `src/codeintel/serving/mcp/tools/search.py`
   - `src/codeintel/serving/mcp/tools/export.py`
   - `src/codeintel/serving/mcp/tools/meta.py`
   - Keep `build_mcp_app(...)` as the stable entrypoint that wires everything together.
2. Split `src/codeintel/serving/mcp/resources.py` similarly:
   - `mcp/resources/meta.py` (environment/templates/views_sql)
   - `mcp/resources/exports.py` (exports payload/meta/preview/chunks)
3. Split `src/codeintel/serving/mcp/response_models.py` into themed files:
   - `mcp/models/primitives.py`, `mcp/models/meta.py`, `mcp/models/export.py`, etc.
4. Update `src/codeintel/serving/contracts/check_operation_contracts.py` if it imports MCP tool lists/schemas.

Acceptance gates
- Phase 0 gates.
- Contract check explicitly verifies tool names, route sets, and schema stability.

Notes/Risks
- Use compatibility re-exports to avoid massive “one PR touches everything” if desired, but direct migration is OK.

---

### Phase 9 — Replace Implicit Exception Contracts with Explicit Domain Exceptions

**Objective**: remove patterns like `getattr(exc, "unknown", None)` from operations layers.

Tasks
1. Define explicit exceptions in `codeintel.serving.errors.exceptions`:
   - e.g., `SemanticUnknownColumnsError(unknown: tuple[str, ...], allowed: tuple[str, ...])`
2. Update `semantic/query_builder.py` and/or kernel validation to raise these exceptions.
3. Update `ServingOperations` to map these exceptions directly to error codes without attribute probing.

Acceptance gates
- Phase 0 gates.
- Add tests that assert:
  - raising the exception produces stable error response code + details in both HTTP and MCP.

---

### Phase 10 — Kernel Decomposition (Meta/Search/Export Subservices)

**Objective**: reduce “god-kernel” responsibilities and clarify ownership.

Tasks
1. Extract search engine logic:
   - Move search query templates and execution into `src/codeintel/serving/search/engine.py`.
   - Kernel delegates to this service.
2. Extract meta assembly:
   - Kernel should not build the “meta dict”; the introspection service should.
3. (Optional) Extract export compilation/execution into `serving/export/engine.py`:
   - unify export SQL compilation, row streaming, parquet/arrow writing.

Acceptance gates
- Phase 0 gates.
- Performance sanity check: no new large allocations in query paths (especially export).

---

### Phase 11 — Unify Auth Policy Across Transports

**Objective**: one coherent auth story, consistent errors, consistent hints.

Tasks
1. Create `src/codeintel/serving/auth/policy.py`:
   - defines accepted credential modes (bearer token, api key, none)
   - defines how to validate per transport
2. HTTP:
   - `require_api_key` becomes a thin adapter around the canonical policy.
3. MCP:
   - `create_bearer_auth` remains, but policy controls whether auth is required and what errors to emit.
4. Ensure parity in error code mapping:
   - always `CODEINTEL_AUTH_FORBIDDEN` for auth failures, with consistent hint structure.

Acceptance gates
- Phase 0 gates.
- Add tests for:
  - missing/invalid credentials in HTTP and MCP map to the same canonical error code and category.

---

### Phase 12 — Remove or Clarify Re-export/Compat Modules

**Objective**: reduce indirection and eliminate dependency direction mistakes.

Targets (examples)
- Legacy HTTP compat re-exports (`serving/http/routes/semantic.py`, `.../search.py`) removed in
  the serving legacy cleanup.
- `src/codeintel/serving/mcp/protocols.py` re-exports operations protocols.
- `src/codeintel/serving/errors/catalog.py` currently acts as a shim (should become real canonical home after Phase 1).

Tasks
- For each compat module, choose:
  - Delete and update imports everywhere, OR
  - Keep but add an explicit “compat-only” docstring + stable re-export policy.
- Ensure import graphs flow core → adapters, not adapters → core.

Acceptance gates
- Phase 0 gates.
- `rg` audit shows no unexpected adapter-to-adapter imports.

## Validation Checklist (Run For Each Phase)

- `uv run ruff check --fix src/codeintel/serving tests/serving`
- `uv run pyright --warnings --pythonversion=3.13 src/codeintel/serving tests/serving`
- `uv run pyrefly check src/codeintel/serving tests/serving`
- `uv run pytest -q -o addopts= --confcutdir=tests/serving tests/serving`
- `uv run python -m codeintel.serving.contracts.check_operation_contracts` (or an equivalent pytest wrapper)

## Rollout Strategy

Because we’re in a design-stage refactor, the preferred approach is:

- **Direct migration** of imports and module moves, with
- **Short-lived compatibility shims** only when they materially reduce change risk (e.g., temporary re-export modules).

## Suggested Prioritization (If We Need To Timebox)

1. Phase 1 (errors) + Phase 3 (introspection) — removes the biggest architectural coupling and drift risk.
2. Phase 5 (HTTP route runner) — fastest payoff in LOC reduction.
3. Phase 8 (MCP module decomposition) — improves maintainability without changing behavior.
4. Phase 6 + Phase 9 (export semantics + explicit exceptions) — improves correctness/hardness.
5. Phase 10 + Phase 11 (kernel split + auth unification) — larger refactors; do when the above is stable.
