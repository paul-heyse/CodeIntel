# Phase 2 Remaining Plan: Tool Realism Adoption

## Goal
Complete Phase 2 by standardizing tool payload fixtures and ToolSandbox usage
across the remaining tests that still use ad-hoc JSON payloads or rely on real
tool binaries.

## Completed So Far
- `tests/ingestion/test_runner_plumbing.py` now uses shared payload builders.
- `tests/ingestion/test_scip_ingest.py` uses `ToolSandbox` stubs for the harness path.
- `tests/_helpers/tool_sandbox.py` default stubs now use shared payload builders.
- `tests/_helpers/orchestration/tooling.py` uses payload builders for stub outputs.
- `tests/_helpers/hamilton_harness_artifacts.py` and `tests/_helpers/ingestion.py`
  generate pytest/scip payloads via shared builders.

## Remaining Scope (Phase 2)
### 1) Standardize tooling tests in `tests/ingestion/test_tools.py`
**Why**
This file still mixes inline payloads and ad-hoc JSON for tool outputs. It should
rely on payload builders for consistency with the new tool realism layer.

**Targets**
- `tests/ingestion/test_tools.py`

**Planned changes**
- Replace any inline pytest JSON payloads with `pytest_report_payload(...)`.
- Replace inline coverage JSON payloads with `coverage_json_payload(...)`.
- Replace inline scip JSON payloads with `scip_json_payload(...)`.
- Where tests use `write_pytest_report(...)`, ensure the payload semantics
  align with the shared builder (summary keys and root metadata if needed).
- Keep existing expectations; do not change behavior or assertions.

**Acceptance criteria**
- All tool JSON payloads in the file are produced by the shared builders.
- No functional change in test assertions beyond payload construction.

### 2) Harmonize ToolSandbox integration tests
**Why**
The sandbox integration tests should assert behavior using the canonical payload
builders to ensure consistent parsing paths.

**Targets**
- `tests/tools/test_tool_sandbox_integration.py`

**Planned changes**
- Use `pytest_report_payload(...)` when constructing any report JSON file content.
- If the test relies on stubbed tool JSON output, ensure the stub payload matches
  the shared builder output (especially for `pytest`).
- Keep existing test behavior and assertions.

**Acceptance criteria**
- Stub payloads match the shared payload builders.
- No change in assertions.

### 3) Align helper provisioning paths that generate tooling artifacts
**Why**
Some helpers still create payloads directly when provisioning artifacts. They
should use the shared builders to avoid drift.

**Targets**
- `tests/_helpers/orchestration/provisioning.py` (only if any inline payloads exist)
- `tests/_helpers/cli_stubs.py` (only if any inline payloads exist)

**Planned changes**
- Replace any inline pytest/coverage/scip payloads with shared builders.
- Keep output paths and filenames unchanged.

**Acceptance criteria**
- All generated payloads come from `tool_payloads.py`.

## Execution Steps (Order)
1. Audit `tests/ingestion/test_tools.py` for inline payloads and replace them with
   `pytest_report_payload`, `coverage_json_payload`, `scip_json_payload`.
2. Update `tests/tools/test_tool_sandbox_integration.py` to use the shared builders.
3. Scan `tests/_helpers/orchestration/provisioning.py` and `tests/_helpers/cli_stubs.py`
   for inline payloads and replace as needed.
4. Run targeted tests (below) to ensure behavior is unchanged.

## Targeted Tests
- `pytest -q tests/ingestion/test_tools.py`
- `pytest -q tests/tools/test_tool_sandbox_integration.py`
- If provisioning/cli stubs updated: run the smallest affected subset.

## Rollback Strategy
All changes are localized to test payload construction. If a test fails due to
unexpected schema differences, revert the single test’s payload to the previous
inline structure and add a follow-up note to extend the shared builders.
