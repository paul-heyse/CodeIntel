# Pytest Failure Remediation Plan

## Purpose

Provide an execution-ready plan to resolve the current pytest failures and harden the
test/production design for inventory consistency, schema service initialization, and
observability runtime behavior.

## Snapshot of Current Failures

Primary failure clusters (from `build/test-results/junit.xml` and `build/test-results/pytest-report.json`):

1. **core.repo_map / core.modules inconsistency** (83 setup errors + 7 direct failures)
2. **SchemaService not configured** (2 setup errors + 1 failure)
3. **OpenTelemetry ProxyTracerProvider missing `add_span_processor`** (6 failures)
4. **Docs export parquet mapping mismatch** (1 failure)
5. **Callgraph alias resolution mismatch** (1 failure)
6. **Graph validation path expectation mismatch** (1 failure)
7. **Span consistency mismatch in CFG GOIDs** (1 failure)
8. **Ingestion docstrings target failed** (1 failure)
9. **Coverage ingest TargetRunRecord missing** (1 failure)
10. **SCIP tables empty with harness/stubs** (2 failures)
11. **Module repository file summary missing** (1 failure)

## Root-Cause Analysis Summary

- **Inventory consistency failures** point to CorePack seeding `core.repo_map` with a
  larger module map than the subset of `core.modules` actually inserted. The new
  `ModulesAssertions.inventory_consistent()` check is correctly flagging a mismatch.
- **SchemaService not configured** indicates docs-view gateways or tests are
  invoking schema-driven utilities without a global `SchemaService` set.
- **ProxyTracerProvider missing `add_span_processor`** indicates tests or runtime
  are using a non-SDK provider when span processors are added.
- **Docs export mapping mismatch** indicates export mapping is being compared against
  a list that no longer matches the expanded registry of datasets.
- **Callgraph alias failure** suggests the callgraph builder behavior or fixtures
  do not guarantee a resolved alias edge; tests may need a stable resolution path.
- **Graph validation path mismatch** suggests `seed_graph_validation_gaps` produces
  a different rel_path than expected or run context resolves catalogs differently.
- **CFG GOID mismatch** indicates span/coverage seeding yields different GOID types
  or values; likely a mismatch between expected GOID seed and data produced.
- **Docstrings target failure** indicates `docstrings` target depends on repo_map
  content or filesystem scan; failure suggests inventory inconsistency or missing
  expected repo_map/module rows post-manipulation.
- **Coverage ingest TargetRunRecord missing** indicates the target result is not
  produced, likely due to name mapping or runner contract changes.
- **SCIP empty tables** suggests the stub artifacts and/or tools path do not match
  the ingestion path expected by the target.
- **Module repository file summary missing** suggests minimal docs export seed
  does not populate required analytics for module summary views.

## Design Goals

- **Inventory invariants**: enforce repo_map/modules consistency at the seed layer.
- **Single source of truth**: module inventory derived from a unified helper and
  passed through packs and seeders.
- **Explicit schema initialization**: schema service is configured for gateways
  that depend on schema services, without relying on test order.
- **Observability resilience**: observability setup should handle missing SDK
  providers gracefully and be testable with a stub provider.
- **Stable test contracts**: tests assert stable interfaces instead of internal
  transient behavior (for example, export mapping comparison is schema-driven).

## Implementation Plan

### Phase 1: Fix inventory consistency (highest impact)

**Targets**
- CorePack and any seed pack inserting `core.repo_map`/`core.modules`.
- Fixtures and harnesses that call `ModulesAssertions.inventory_consistent()`.

**Actions**
1. **Align CorePack seeding with module_count filtering**
   - Update CorePack to insert `core.repo_map` using the same subset inserted into
     `core.modules` (respecting `module_count` and `include_util`).
   - Ensure the repo_map payload is based on the actual inserted modules.
2. **Add invariant helper**
   - Introduce a small helper to compute `module_map` and `modules_rows` from a
     canonical map with optional filtering, so this remains consistent across packs.
3. **Audit seed packs**
   - Verify all seed packs that insert repo_map also insert corresponding modules.
   - For packs with custom maps (for example, docs export, span/pipeline) ensure
     map construction excludes any module not inserted.
4. **Add/adjust test assertions**
   - Where `ModulesAssertions.inventory_consistent()` is invoked, ensure
     `repo_map` payload matches modules content after any test-specific edits.

**Expected result**
- The large cluster of setup errors with repo_map/modules inconsistency should clear.

### Phase 2: SchemaService initialization hardening

**Targets**
- docs views fixtures and any tests that use `bootstrap_metadata_datasets`.
- Storage gateway setup paths that do not already set SchemaService.

**Actions**
1. **Set SchemaService during test gateway provisioning**
   - Ensure `docs_views_ready_gateway` and `provision_docs_export_ready` call a
     helper that sets SchemaService when missing (using storage schema provider).
2. **Add a `tests/_helpers/schemas.py` helper**
   - Provide a centralized `ensure_schema_service()` used by fixtures and seeders.
3. **Document expectations**
   - Update docs/view tests to explicitly assert schema service configured (if needed).

**Expected result**
- `RuntimeError: SchemaService has not been configured` failures eliminated.

### Phase 3: Observability test resilience

**Targets**
- `src/codeintel/observability/otel.py`
- Observability tests that use `ProxyTracerProvider`

**Actions**
1. **Add guard for missing `add_span_processor`**
   - In `_build_tracer_provider`, verify the provider implements
     `add_span_processor`; if not, fall back to a no-op or disable span processor
     addition and log a debug message.
2. **Test harness provider injection**
   - Provide a test helper to install an SDK provider in tests or use an
     in-memory provider that implements `add_span_processor`.
3. **Explicit capability check**
   - Add a lightweight interface check to avoid attribute errors.

**Expected result**
- 6 observability failures resolved and more robust tracing setup.

### Phase 4: Docs export mapping parity update

**Targets**
- `tests/docs_export/test_export_parity.py`
- Dataset registry/contract registry expectations

**Actions**
1. **Base expectations on dataset contracts**
   - Replace hard-coded `required_tables` with a computed subset derived from
     the contract registry (core/graph/analytics tables).
2. **Categorize exports**
   - If a smaller set is intended, define an explicit allowlist for export mapping
     instead of assuming a fixed table count.
3. **Add documentation**
   - Document the export scope and the rule for the mapping set.

**Expected result**
- Parity test aligns with actual dataset registry and passes.

### Phase 5: Graph/callgraph correctness tests

**Targets**
- `tests/graphs/test_callgraph_builder.py`
- `tests/graphs/test_graph_validation_catalog.py`
- `tests/graphs/test_span_consistency_integration.py`

**Actions**
1. **Callgraph alias resolution**
   - Ensure fixture repo includes alias patterns resolvable by the callgraph
     builder; if logic is nondeterministic, update test to accept
     multiple valid resolution kinds or validate edge existence by evidence.
2. **Graph validation path**
   - Ensure `seed_graph_validation_gaps` uses rel_path consistent with test
     expectations; adjust to fixed canonical value or update test to read
     from seeded value.
3. **Span GOID consistency**
   - Ensure seeding for span coverage uses consistent GOID types and values;
     cast/normalize GOIDs to integers before comparisons.

**Expected result**
- Graph-related functional tests match the updated seeding behavior.

### Phase 6: Ingestion/harness contract alignment

**Targets**
- `tests/ingestion/test_docstrings_inventory.py`
- `tests/ingestion/test_runner_plumbing.py`
- `tests/ingestion/test_scip_ingest.py`
- `tests/_helpers/harnesses/hamilton_build.py`

**Actions**
1. **Docstrings target run**
   - Validate repo_map updates after module deletion and ensure downstream
     docstrings run uses repo_map correctly. If target depends on filesystem,
     add a test helper to disable filesystem scan or force repo_map usage.
2. **Coverage ingest target record**
   - Ensure `run_targets(["coverage_ingest"])` returns a TargetRunRecord in
     `HamiltonBuildResult.outputs`; if target names are remapped, add a lookup
     helper in harness to resolve the correct node name.
3. **SCIP ingestion expectations**
   - Confirm stub artifacts match ingestion path; if ingestion expects a specific
     file layout, update `write_dummy_scip_artifacts()` to match that layout.

**Expected result**
- Ingestion harness tests align with production target runtime behavior.

### Phase 7: Docs export seed completeness

**Targets**
- `tests/_helpers/orchestration/seeding_docs.py`
- `tests/storage/repositories/test_repositories.py`

**Actions**
1. **Extend docs export seed**
   - Ensure `seed_docs_export_minimal` includes the minimal rows required for
     module summary/hints views (file summary).
2. **Add validations**
   - Add test helper checks that seed produces module summary data when used
     by ModuleRepository.

**Expected result**
- Module repository reads succeed in docs export tests.

## Execution Checklist

1. Fix CorePack repo_map/modules mismatch and add shared helper.
2. Audit and align all seed packs that touch repo_map/modules.
3. Add `ensure_schema_service()` and wire into docs view gateways.
4. Guard observability tracer provider for missing `add_span_processor`.
5. Update export parity tests to be contract-driven.
6. Adjust graph tests and seeders for deterministic behavior.
7. Align ingestion harness target recording and SCIP artifact paths.
8. Extend docs export seed for module summaries.
9. Run targeted pytest subsets:
   - `tests/_helpers`
   - `tests/analytics`
   - `tests/graphs`
   - `tests/ingestion`
   - `tests/docs_export`
   - `tests/observability`
   - `tests/storage`
10. Run full `uv run python -m tools.quality_report` and a segmented pytest suite.

## Notes on Design Principles

- Centralize seed logic; avoid duplicated module inventory derivations.
- Promote explicit, structured seed options objects instead of long parameter lists.
- Treat schema service as part of gateway lifecycle, not test state.
- Ensure harnesses surface useful diagnostics when targets are missing.
- Keep tests resilient by asserting invariants rather than internal implementation details.
