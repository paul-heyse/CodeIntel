## Context
Modules-first expectations, golden diffs, and the Hamilton build harness are implemented, but
several test areas still compute module paths via raw filesystem scans or duplicate BuildEnv
setup. This change expands helper usage in targeted areas (SCIP ingestion, graph loaders,
analytics module maps, and fixtures) to align tests with production scanning and execution
behavior while keeping the runtime logic unchanged.

## Goals / Non-Goals
- Goals:
  - Use harness pathways for SCIP tests that need real tools.
  - Normalize module inventory seeding via modules-first helpers.
  - Apply ModulesAssertions where graph tests depend on module catalogs.
  - Use golden diffs for module/path comparisons outside ingestion/storage.
  - Reduce duplicated Hamilton runtime wiring in fixtures.
- Non-Goals:
  - No production target changes.
  - No new tool plugins or external dependencies.

## Decisions
- Decision: Keep the existing integration SCIP test and add a harness-based real-tools variant.
  Alternatives considered: Replacing the integration test (rejected; want both paths).
- Decision: Use modules-first helpers for path inventories in helper utilities and seeds.
  Alternatives considered: Keep rglob-based lists (rejected; drift with filters).
- Decision: Add ModulesAssertions checks in graph tests that rely on module catalogs.
  Alternatives considered: Leave implicit assumptions (rejected; unclear failures).
- Decision: Use golden diff helpers for analytics module map mismatches to standardize failure
  messages.
  Alternatives considered: Keep plain equality asserts (rejected; low-signal diffs).
- Decision: Align Hamilton fixtures with shared runtime/harness paths.
  Alternatives considered: Keep BuildEnv-only fixtures (rejected; redundant runtime builds).
- Decision: Extend helper usage into docs export and storage repository tests where module/path
  comparisons exist, standardizing golden diff diagnostics.
  Alternatives considered: Leave localized asserts (rejected; low-signal diffs).
- Decision: Use tool_sandbox for integration tests that require real binaries to achieve a
  deterministic subprocess path.
  Alternatives considered: Skip tools or rely on PATH binaries (rejected; non-deterministic).

## Risks / Trade-offs
- Risk: Additional harness usage may increase test setup time.
  Mitigation: Reuse shared runtime fixtures and keep deterministic artifact paths.

## Migration Plan
- Add harness-based SCIP test and keep the existing integration variant.
- Swap helper inventory collection to modules-first expectations where appropriate.
- Introduce ModulesAssertions in graph inventory-adjacent tests.
- Update analytics module/path comparisons to use golden diffs.
- Align Hamilton fixtures with shared runtime/harness utilities.
- Apply golden diffs to docs export and storage repository module/path comparisons.
- Expand harness usage to additional ingestion targets and use manifest/artifact helpers where
  tests currently seed outputs manually.
- Route integration tests through tool_sandbox where practical to stabilize tool behavior.

## Implementation Notes
- SCIP harness test should validate `TargetRunRecord` row counts and verify the artifact paths
  returned by the harness artifacts helper.
- Module inventory helper changes should prefer `modules_expected_from_repo_tree(...)` for
  repo-based seeding and `module_paths_expected_from_repo_tree(...)` when only paths are needed.
- Graph tests should call `ModulesAssertions.inventory_consistent()` after module insertions
  and before graph loader execution to surface mismatches early.
- Analytics module map comparisons should route through `format_missing_extra(...)` and
  `format_module_map_diff(...)` (using `module_map_from_path_map(...)` where applicable).
- Fixture alignment should prefer reusing the shared `hamilton_runtime` fixture and/or
  `HamiltonBuildHarness` for environment construction.
- Docs export and storage repository comparisons should use golden diffs whenever lists/maps
  of module paths are compared.
- For ingestion targets beyond SCIP, prefer harness-based execution to ensure skip/manifest
  behavior is exercised consistently.
- Use tool_sandbox to provide stub binaries for integration tests that require subprocesses
  but should remain deterministic in CI.

## Open Questions
- Should analytics module map comparisons be centralized in a shared assertion helper?
- Are there additional graph loader tests beyond `test_engine_nx.py` that assume module catalogs
  and should adopt ModulesAssertions in this change?
- Which docs export and storage repository tests currently compare module/path lists and should
  be prioritized for golden diffs?
