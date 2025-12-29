# Guardrails Inference-First Alignment Plan

## Intent

Align guardrails with the inference-first, graph-authoritative architecture so they
protect the streaming Hamilton pipeline and tag taxonomy, while treating schema
drift as an observable signal (not a hard gate). This plan replaces schema drift
gating with checks that enforce inference invariants, module discovery boundaries,
and streaming-only data flow.

## Goals

- Make guardrails enforce inference-first invariants and streaming-only behavior.
- Ensure Hamilton graph + tag taxonomy is authoritative and consistent across build/serve.
- Eliminate guardrail failures caused by drift in contract catalogs or meta stores.
- Add static guardrails that prevent regression to row-based or eager materialization paths.
- Provide explicit guardrails for module discovery and plugin boundaries.

## Non-goals

- Rewriting schema inference or registry persistence logic.
- Introducing new schema pinning or gating on drift.
- Changing runtime behavior beyond guardrail checks and logging.

## Architecture Alignment

- Hamilton graph inference is authoritative; guardrails should validate the DAG and
  tag taxonomy, not block on storage catalog drift.
- Streaming-first: guardrails should forbid eager table materialization in build
  or serving layers and enforce reader/batch usage.
- Target discovery is module-based; guardrails should catch legacy registry or
  static target list usage outside the resolver.

## Implementation Plan

### Phase 0: Decisions and configuration hooks

- Decide where to centralize guardrail options (suggested: dedicated helper in
  `tools/guardrails.py` or a `StorageConfig.for_guardrails()` factory).
- Define allowed paths for streaming exceptions (tests/helpers, fixtures, or
  explicitly named dev utilities).
- Enumerate the canonical tag taxonomy source (e.g., reuse existing DAG tag
  validation utilities in `codeintel.build.hamilton.graph_validation`).

Acceptance:
- Documented allowlist and runtime config for guardrails.

### Phase 1: Guardrails should not gate on storage catalog drift

- Update `tools/guardrails.py` to open gateways with:
  - `validate_schema=False` or `validation_mode=OFF`.
  - Avoid writing validation summaries when invoked via guardrails.
- Replace reliance on `_apply_contract_validation()` and Arrow drift logs with
  a new inference-centric check:
  - Ensure each materialized table has a recent schema observation or renderer
    cache entry (Arrow schema IPC bytes) in the registry.

File targets:
- `tools/guardrails.py`
- `src/codeintel/storage/gateway/config.py` (optional helper for guardrails)
- `src/codeintel/storage/gateway/factory.py` (optional hook for guardrails-only mode)

Acceptance:
- Guardrails do not fail due to schema drift warnings.
- Guardrails still validate DAG consistency and report inference issues.

### Phase 2: Add streaming-first static guardrails

Add new static guardrail patterns in `tools/guardrails.py` to disallow:

- `to_table()` and `read_all()` in build/serving/inference modules.
- `.arrow()` on DuckDB relations or `.fetchall()` usage in build/serving code.
- `to_pandas()` or `.values` (except in tests or explicit debug utilities).

Allowlist paths:
- `tests/**`
- `tools/**` (explicit debug scripts only)
- `docs/_scripts/**` (if required)

File targets:
- `tools/guardrails.py`

Acceptance:
- Guardrail scan fails on eager materialization usage in source paths.
- Tests remain allowed to materialize tables when necessary.

### Phase 3: Enforce graph-authoritative tag taxonomy

- Extend guardrail runtime checks after runtime bundle composition to validate:
  - required tags on `t__*` anchors (domain, target, kind, schema_ref)
  - consistent use of `@schema.output` and `@check_output` wrappers from the SDK
    (no direct `hamilton.function_modifiers.schema` usage outside SDK or tagging helpers)
- Add static guardrails to forbid direct imports of:
  - `hamilton.function_modifiers.schema`
  - `hamilton.function_modifiers.check_output`
  - `hamilton.function_modifiers.tag` (already banned)

Allowed usage:
- `src/codeintel/build/hamilton/tagging.py`
- `src/codeintel/sdk/**`

File targets:
- `tools/guardrails.py`
- `src/codeintel/build/hamilton/graph_validation.py` (if adding an explicit
  tag taxonomy validator)

Acceptance:
- Guardrails fail when tags are missing or a direct modifier import is used.
- Tag taxonomy validation errors report provenance (origin + module path).

### Phase 4: Enforce module discovery boundaries

Add static guardrails to forbid:

- Hard-coded target registries or static target lists outside the resolver.
- Direct imports of `codeintel_targets` modules from non-resolver contexts.

Suggested patterns to ban:
- `TARGETS = [` in non-test modules under `src/codeintel/`.
- `codeintel_targets.` imports in `src/codeintel/` except `runtime/module_resolver.py`.

File targets:
- `tools/guardrails.py`

Acceptance:
- Guardrails fail when module discovery is bypassed or targets are manually listed.

### Phase 5: Inference observation guardrails

- Add a runtime guardrail check to ensure the registry contains inferred schema
  observations (IPC bytes + stats) for all output tables discovered in the DAG.
- Treat missing observation entries as guardrail failures, but do not block on
  drift or differences between observation and catalog schemas.

File targets:
- `tools/guardrails.py`
- `src/codeintel/build/schemas/inference_service.py` or a new helper that can
  verify observation presence for a table key.

Acceptance:
- Guardrails fail when inference outputs are missing observation records.
- Drift is logged but not considered a guardrail failure.

## Acceptance Criteria

- Guardrails enforce streaming-only behavior in build/serving code.
- Guardrails validate the Hamilton DAG + tag taxonomy with provenance-rich errors.
- Guardrails no longer gate on DuckDB meta catalog drift or contract mismatches.
- Module discovery boundaries are enforced (targets are modules only).
- Inference observations are required for each DAG output table.

## Test Plan

- Add unit tests for guardrail pattern matching in `tools/guardrails.py`.
- Add a DAG validation test that fails when tags are missing or invalid.
- Add a guardrail runtime test that fails when observation records are missing.
- Add a static scan test that ensures `to_table()` usage is banned in
  `src/codeintel/build/**` and `src/codeintel/serving/**`.

Suggested test files:
- `tests/tools/test_guardrails_patterns.py`
- `tests/build/hamilton/test_tag_taxonomy_guardrails.py`
- `tests/build/schemas/test_inference_observation_guardrails.py`

## Rollout Strategy

1) Update guardrails runtime behavior (Phase 1) to remove drift gating.
2) Add streaming and modifier static guardrails (Phase 2-3).
3) Add module discovery and observation checks (Phase 4-5).
4) Harden tests and ensure guardrails pass in CI.

## Open Questions

- Where should guardrail configuration live (in `tools/guardrails.py` vs a new
  runtime settings block)?
- Do we want a single allowlist for streaming exceptions, or separate allowlists
  for tests, tools, and docs?
- Should missing observation entries be fatal in all environments, or only in CI?
