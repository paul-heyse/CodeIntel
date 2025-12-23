# Change: Expand test helper usage across ingestion, graphs, analytics, and fixtures

## Status
Archived with validation deferred per user confirmation (full pytest recently run; minor failures).

## Why
The modules-first and harness helpers exist, but several tests still construct module inventories
by hand, bypass repo_map consistency checks, or duplicate BuildEnv wiring. This results in drift
from production scanning behavior, weaker diagnostics when module maps diverge, and duplicated
runtime setup in tests that could share a common Hamilton runtime. Expanding helper usage in a
targeted way improves production parity and makes failures higher signal without altering runtime
behavior.

## What Changes
### SCIP ingestion harness coverage
- Add a harness-based SCIP test path that runs `scip` through `HamiltonBuildHarness` when the
  tooling binaries exist on PATH, validating `TargetRunRecord` row counts and artifact paths.
- Keep the deterministic artifact-based SCIP path for tool-free execution using
  `HarnessArtifacts.write_dummy_scip_artifacts(...)`, ensuring parity with the production
  ingestion path but without subprocess requirements.
- Preserve the existing integration test that directly checks tool binaries and row persistence.

### Modules-first seeding parity
- Replace helper-level module path derivation that currently uses `repo_root.rglob("*.py")` with
  `modules_expected_from_repo_tree(...)` / `module_paths_expected_from_repo_tree(...)` so the
  same ignore and filtering logic as production scanning is applied.
- Apply this specifically to helper utilities that seed `core.modules` / `core.repo_map`
  inventories and helper-derived module lists used for downstream assertions.

### Graph inventory consistency checks
- Introduce `ModulesAssertions.inventory_consistent()` in graph tests that rely on module
  catalogs (for example graph loaders in `tests/graphs/test_engine_nx.py`) so graph behavior
  failures surface module inventory mismatch earlier and with a consistent message.
- Expand the same consistency checks to other graph loader tests that seed core.modules
  directly (e.g., other `tests/graphs/*` cases and orchestration seed packs).

### Golden diffs for analytics module maps
- When analytics tests compare module/path mappings (e.g., provider factories or dependency
  helpers), replace inline equality checks with `format_missing_extra(...)` or
  `format_module_map_diff(...)` to provide actionable diffs.
- Use `module_map_from_path_map(...)` when comparing `path -> module` maps from core tables.
- Extend golden diffs to other module/path comparisons outside analytics (docs export tests,
  storage repository tests) so module map mismatches are consistently high-signal.

### Shared runtime fixture consolidation
- Align helper fixtures (e.g., `tests/_helpers/hamilton_fixtures.py`) with the shared runtime
  or harness path so Hamilton graph construction is reused and skip/manifest behavior remains
  consistent across tests.
- Route additional BuildEnv fixtures in analytics/graphs conftests through the shared runtime
  or harness path where direct BuildEnv construction still exists.

### Broader harness and manifest usage
- Expand `HamiltonBuildHarness` usage to ingestion targets beyond modules/scip
  (docstrings, typing, tests_ingest, coverage_ingest) where tests currently bypass the harness.
- Use `ManifestPriming` and `HarnessArtifacts` for tests that seed manifests or artifacts
  manually (schema manifest, export targets, serving_artifacts).
- Use `tool_sandbox` for integration tests that currently require real binaries to
  make tool behavior deterministic while still exercising subprocess paths.

## Scope Inventory
### Ingestion
- `tests/ingestion/test_scip_ingest.py` (harness-based real tools path + deterministic artifacts).
 - Ingestion target tests for docstrings, typing, tests_ingest, and coverage ingestion that
   currently execute without the harness.

### Helpers and seeds
- `tests/_helpers/ingestion.py` (modules-first path derivation).
- Helper seed packs under `tests/_helpers/seeds/*.py` where module inventories are derived
  from repo contents rather than hard-coded lists.
 - Orchestration helpers under `tests/_helpers/orchestration/*.py` with manual module path
   derivations where repo scanning parity is desired.

### Graphs
- `tests/graphs/test_engine_nx.py` (module inventory assumptions and graph loaders).
 - Other `tests/graphs/*` cases that insert module rows directly and rely on module catalogs.

### Analytics
- `tests/analytics/resources/test_provider_factory.py` (module map comparisons, provider outputs).
- `tests/analytics/test_dependencies.py` (module/path mapping in alias map helpers).
 - Analytics tests that compare module/path lists or maps outside ingestion/storage.

### Docs export / storage repositories
- Docs export tests that compare module/path lists or repo map inventories.
- Storage repository tests that compare module/path mappings.

### Fixtures
- `tests/_helpers/hamilton_fixtures.py`
- `tests/_helpers/hamilton_execution.py`
 - BuildEnv fixtures in analytics/graphs conftests and other helper modules.

## Impact
- Affected specs: test-helpers
- Affected code:
  - `tests/ingestion/test_scip_ingest.py`
  - `tests/_helpers/ingestion.py`
  - `tests/_helpers/seeds/*.py`
  - `tests/graphs/test_engine_nx.py`
  - `tests/analytics/resources/test_provider_factory.py`
  - `tests/analytics/test_dependencies.py`
  - `tests/_helpers/hamilton_fixtures.py`
  - `tests/_helpers/hamilton_execution.py`
