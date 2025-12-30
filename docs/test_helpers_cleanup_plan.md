# Test Helpers Production Alignment Cleanup Plan

## Scope

This plan focuses on the helpers under `tests/_helpers` to align them with
production behavior and remove drift. The changes are limited to test helpers
and tests that consume them (no production runtime behavior changes).

## Goals

- Make test helpers use the same entry points, configuration paths, and data
  contracts as production code.
- Remove helpers that are unused or encode deprecated production behavior.
- Reduce duplicated helper APIs and consolidate on a small, explicit surface.
- Keep test intent clear while preventing silent schema or environment drift.

## Non-Goals

- Rewriting unrelated tests or changing production code semantics.
- Introducing new test frameworks or changing the test runner.
- Large-scale reorganizations of `tests/` beyond the specific helper cleanups.

## Workstreams And Detailed Steps

### 1) Consolidate Environment Helpers

**Motivation**
Multiple environment mutation helpers exist (`tests/_helpers/env.py`,
`tests/_helpers/env_vars.py`, and local `_temporary_env` utilities). This creates
inconsistent behavior and encourages ad-hoc implementations.

**Plan**
1. Pick a single canonical helper (recommend `tests/_helpers/env.py` because it
   already supports mapping + kwargs and None to unset).
2. Add small convenience wrappers there to cover the single key/value pattern
   currently used in `tests/_helpers/env_vars.py`.
3. Update call sites to import only from the canonical helper.
4. Replace local `_temporary_env` helpers in
   `tests/build/hamilton/hooks/test_lifecycle.py` and
   `tests/build/hamilton/adapters/test_parallel.py` with the canonical helper.
5. Remove `tests/_helpers/env_vars.py` once all call sites are migrated.

**Files**
- Canonicalize: `tests/_helpers/env.py`
- Remove: `tests/_helpers/env_vars.py`
- Update imports in:
  - `tests/storage/test_extension_policy.py`
  - `tests/storage/test_duckdb_session_init_sql.py`
  - `tests/observability/test_otel_config.py`
  - `tests/observability/test_observability_smoke.py`
  - `tests/cli/test_cli_telemetry.py`
  - `tests/build/hamilton/hooks/test_lifecycle.py`
  - `tests/build/hamilton/adapters/test_parallel.py`

**Acceptance Criteria**
- No `temporary_env` helper definitions outside `tests/_helpers/env.py`.
- No remaining imports from `tests._helpers.env_vars`.

---

### 2) Remove Deprecated Macro Helpers

**Motivation**
`GatewayFactory.with_macros`, `memory_con_with_macros`, and related helpers
encode a production behavior that no longer exists. Keeping them as no-ops
invites false confidence and drift.

**Plan**
1. Remove `GatewayFactory.with_macros` and the deprecated macro helpers in
   `tests/_helpers/gateway.py`.
2. Replace `.with_macros()` call sites with the canonical `.open()` or
   `.with_schema()` flows.
3. Update ingestion helpers to drop the `with_macros` variant and remove any
   macro-specific variants in `tests/_helpers/ingestion.py`.
4. Update orchestration helpers that expose `memory_con_with_macros` or
   `gateway_with_macros` to point to the non-macro variants.

**Files**
- Remove helpers from: `tests/_helpers/gateway.py`
- Update usage in:
  - `tests/_helpers/ingestion.py`
  - `tests/_helpers/orchestration/gateway.py`
  - `tests/_helpers/orchestration/__init__.py`

**Acceptance Criteria**
- `rg -n "with_macros|memory_con_with_macros|gateway_with_macros" tests/_helpers`
  returns no results.
- Ingestion helpers still build gateways using the canonical factory path.

---

### 3) Align Row Factories With Production Schemas

**Motivation**
`tests/_helpers/fixtures/rows.py` currently uses name-based JSON defaults
(`_JSON_LIST_COLUMNS`, `_JSON_DICT_COLUMNS`). This can silently drift as schemas
change and allows incorrect shapes to pass.

**Plan**
1. Inventory JSON columns from production schemas and generated row models.
2. Replace the name-based JSON defaults with schema-driven defaults:
   - Prefer defaults derived from generated row models where available.
   - Where no default exists, require explicit values and raise if missing.
3. Update row builders and seed packs to provide explicit JSON values where
   required (likely in `tests/_helpers/builders/`, `tests/_helpers/seeds/`, and
   `tests/_helpers/fixtures/rows.py`).
4. Add a small utility for JSON defaults that is keyed by `TableSchema` and
   column metadata (not column names).

**Files**
- Primary: `tests/_helpers/fixtures/rows.py`
- Likely updates in:
  - `tests/_helpers/builders/*`
  - `tests/_helpers/seeds/*`
  - `tests/_helpers/fixtures/rows.py`

**Acceptance Criteria**
- No name-based JSON default sets (`_JSON_LIST_COLUMNS`, `_JSON_DICT_COLUMNS`).
- Missing JSON fields in row helpers raise clear errors unless explicitly
  defaulted via schema/model metadata.

---

### 4) Align CLI Helpers With Production Entry Points

**Motivation**
`tests/_helpers/cli.py` manually manipulates `os.environ`, `sys.argv`, and
catches exceptions, which can diverge from the production entrypoint behavior.
A production-aligned CLI harness already exists under `tests/cli/_harness`.

**Plan**
1. Re-implement `tests/_helpers/cli.run_cli` on top of `CliTestHarness` so
   invocation matches production wiring.
2. Replace direct `app(...)` calls and custom exception routing in
   `tests/_helpers/cli.py` with harness usage.
3. Ensure `temp_repo_context` returns inputs compatible with
   `CliTestHarness` and `CLIProjectHarness`.
4. Update tests that rely on `CliResult` semantics to match the harness result
   object or provide compatibility shims.

**Files**
- Update: `tests/_helpers/cli.py`
- Reference harness: `tests/cli/_harness/__init__.py`

**Acceptance Criteria**
- CLI tests execute through `CliTestHarness` (no direct `app(...)` calls).
- Environment and working directory behavior matches the harness path.

---

### 5) Restrict Fakes To Unit Tests (Production-Parity By Default)

**Motivation**
Several fakes return simplified or empty data and can mask production behavior
in integration tests. Fakes should be limited to unit tests or explicit
behavioral tests of port-level logic.

**Plan**
1. Inventory fakes under `tests/_helpers/fakes/*` and identify current usage.
2. Mark fakes as unit-test-only (e.g., with docstrings and module-level
   warnings), and update integration tests to use real gateways/harnesses.
3. For fakes that must remain, ensure they implement the same protocol surface
   as production ports (e.g., `IngestStoragePort`).
4. Introduce a small policy: integration tests must use real gateways and
   harnesses unless explicitly justified.

**Files**
- Review and update:
  - `tests/_helpers/fakes/storage.py`
  - `tests/_helpers/fakes/fake_providers.py`
  - `tests/_helpers/fakes/*`
- Update tests that should move to real gateways/harnesses.

**Acceptance Criteria**
- Integration tests no longer depend on fakes for storage or tool execution.
- Fakes are clearly annotated and used only in unit-level tests.

---

### 6) Remove Unused Helper Modules

**Motivation**
`tests/_helpers/ports/*` and `tests/_helpers/cli_stubs.py` are unused and add
maintenance burden without benefit.

**Plan**
1. Delete:
   - `tests/_helpers/ports/gateway.py`
   - `tests/_helpers/ports/repo.py`
   - `tests/_helpers/ports/__init__.py`
   - `tests/_helpers/cli_stubs.py`
2. Verify no imports remain (already absent in search).

**Acceptance Criteria**
- `rg -n "tests._helpers.ports|cli_stubs" tests` returns no results.

---

### 7) Consolidate Hamilton Build Helper Paths

**Motivation**
`tests/_helpers/hamilton_fixtures.py` and
`tests/_helpers/harnesses/hamilton_build.py` overlap and can drift in defaults
and runtime configuration.

**Plan**
1. Choose `tests/_helpers/harnesses/hamilton_build.py` as the canonical path.
2. Convert `tests/_helpers/hamilton_fixtures.py` into thin wrappers or remove
   duplicates.
3. Normalize default `BuildEnv` construction to a single source of truth.

**Acceptance Criteria**
- Only one place defines the default `BuildEnv`/harness creation.
- Tests import a single canonical helper for Hamilton build setup.

---

## Execution Order

1. Consolidate environment helpers (Workstream 1).
2. Remove macro helpers (Workstream 2).
3. Remove unused helper modules (Workstream 6).
4. Align CLI helpers (Workstream 4).
5. Consolidate Hamilton build helpers (Workstream 7).
6. Align row factories (Workstream 3).
7. Restrict fakes to unit tests (Workstream 5).

This order minimizes broad ripple effects early (env + macros), then cleans
unused surface area, then targets higher-touch refactors.

## Validation

- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- Run targeted tests for affected areas:
  - CLI: `uv run pytest -q tests/cli`
  - Hamilton: `uv run pytest -q tests/build/hamilton`
  - Storage/test helpers: `uv run pytest -q tests/storage tests/_helpers`
- Final segmented pass per major test area as described in AGENTS.md.

## Open Questions

- Do we want to enforce explicit JSON defaults (failing when omitted), or
  should we derive defaults from generated row models when available?
- Should `tests/_helpers/cli.py` become a small wrapper around the harness or
  be removed in favor of direct harness usage?
- Is there any historical test dependency on macro registration that must be
  preserved (likely none)?
