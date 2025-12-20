# Serving Legacy Compatibility Cleanup - Implementation Plan

## Executive Summary

This plan removes all legacy compatibility surfaces in `src/codeintel/serving` now that we
are in design phase and can migrate immediately. The work focuses on:

- Removing unversioned HTTP endpoints and the root alias.
- Deleting compatibility re-export modules.
- Simplifying the FastMCP compatibility shim to assume fastmcp 2.14+.
- Updating serving contract checks and documentation to match the new, strict surface.

No deprecation path is required. The goal is a single, versioned HTTP surface and a
clean FastMCP integration without fallbacks.

## Constraints and Assumptions

- Design phase: remove legacy surfaces immediately.
- FastMCP is guaranteed to be >= 2.14.0 everywhere.
- No production deprecation pathways are needed.
- All quality gates from `AGENTS.md` must pass for touched files.

## Scope

In scope
- Remove HTTP root alias for v1 and unversioned endpoints.
- Remove compatibility re-export modules in HTTP and MCP.
- Simplify `codeintel.serving.mcp._compat` to remove fallback behavior and unused exports.
- Update contract checks to only expect versioned endpoints.
- Update docs that reference legacy import paths or unversioned routes.

Out of scope
- Semantic behavior changes beyond the removal of legacy paths.
- New endpoints or new FastMCP features.
- Schema changes to response payloads.

## Implementation Plan

### Phase 0 - Inventory and Alignment

Objective: confirm all current references to legacy paths and shims.

Tasks
1. Locate all imports of legacy modules and paths:
   - `rg -n "serving.http.routes.search|serving.http.routes.semantic" src docs`
   - `rg -n "serving.mcp.tooling_meta" src docs`
2. Locate all references to unversioned endpoints in docs:
   - `rg -n "/semantic|/search|/export" docs`
3. Snapshot the current contract expectations:
   - `src/codeintel/serving/contracts/check_operation_contracts.py`

Acceptance gates
- All legacy references identified and ready for update/removal.

### Phase 1 - Remove Unversioned HTTP Routes

Objective: HTTP surface is versioned only (`/v1/...`).

Tasks
1. Remove the root alias from the HTTP router:
   - Edit `src/codeintel/serving/http/routes/__init__.py` to remove the
     `include_router(..., include_in_schema=False)` call and update the docstring.
2. Update serving contract expectations:
   - Edit `src/codeintel/serving/contracts/check_operation_contracts.py`:
     - Remove unversioned routes from `EXPECTED_HTTP_ROUTES`.
     - Update the route prefix filter to only include `/v1/...` endpoints.
3. Update docs that mention unversioned endpoints to `/v1/...`.

Acceptance gates
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- `uv run pytest -q` (or a serving-only subset if that is the current norm).
- `codeintel.serving.contracts.check_operation_contracts` passes.

### Phase 2 - Delete Compatibility Re-exports

Objective: remove dead compat modules now that imports should be canonical.

Tasks
1. Delete HTTP compat re-export modules:
   - `src/codeintel/serving/http/routes/search.py`
   - `src/codeintel/serving/http/routes/semantic.py`
2. Delete MCP compat re-export module:
   - `src/codeintel/serving/mcp/tooling_meta.py`
3. Update any remaining imports to canonical paths:
   - `codeintel.serving.http.routes.v1.search`
   - `codeintel.serving.http.routes.v1.semantic`
   - `codeintel.serving.meta.tooling`
4. Update docs referencing these modules:
   - `docs/serving_refinement/serving_transport_adapter_architecture_implementation_plan.md`
   - `docs/serving_refinement/serving_consolidation_streamlining_implementation_plan.md`

Acceptance gates
- `rg` shows zero references to removed modules.
- Quality gates and contract check pass.

### Phase 3 - Simplify FastMCP Shim for 2.14+

Objective: remove compatibility logic that is only needed for older FastMCP versions.

Tasks
1. Simplify `src/codeintel/serving/mcp/_compat.py`:
   - Remove EventStore feature detection and `HAS_EVENT_STORE`.
   - Remove unused `ToolError` import and `create_bearer_auth`.
   - Import `EventStore` directly from `fastmcp.server.event_store`.
   - Update `__all__` to only export used symbols.
2. Update EventStore usage:
   - Edit `src/codeintel/serving/http/app.py` to remove the `EventStore is None` branch.
   - Always instantiate `EventStore()` when `mcp_enable_event_store` is True.
3. Re-run `rg` to ensure no dangling references to removed exports.

Acceptance gates
- Quality gates and contract check pass.
- MCP app still mounts cleanly with EventStore enabled.

### Phase 4 - Documentation and Examples Cleanup

Objective: docs reflect the new strict, versioned surface and canonical imports.

Tasks
1. Update any mention of unversioned endpoints to `/v1/...`.
2. Replace any mention of `codeintel.serving.mcp.tooling_meta` with
   `codeintel.serving.meta.tooling`.
3. Remove references to the root alias and "backwards compatibility" notes.

Acceptance gates
- `rg` in `docs/` shows no legacy references.

## Verification Checklist

- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- `uv run pytest -q`
- `rg -n "tooling_meta|http.routes.search|http.routes.semantic" src docs` returns no results
- `rg -n "/semantic|/search|/export" docs` shows only `/v1/...` forms

## Rollback Plan (If Needed)

Rollback is a single revert set if tests or contract checks fail. Since this is
design phase with no production dependencies, the preferred approach is to fix
forward rather than re-introduce compatibility shims.
