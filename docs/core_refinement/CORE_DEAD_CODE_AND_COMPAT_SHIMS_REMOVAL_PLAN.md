# Core Refinement: Dead Code Removal + Compatibility Shim Deletion Plan

## Scope

This plan covers **only** the items previously identified under:

1. **Dead / unused** (code exists but has no runtime importers in `src/` and/or is test/doc-only)
2. **Compatibility / legacy shims (still used somewhere)** (intentionally kept for transitional APIs)

It does **not** cover other legacy/compat topics (e.g., parsing “graph compatibility fields”).

## Goals

- Remove unused core packages/modules that provide no production value and add maintenance surface.
- Remove compatibility shims in favor of a single canonical API per concept.
- Preserve correctness and keep Ruff/Pyright/Pyrefly clean at every step.

## Non-goals

- Large architectural redesigns.
- Behavior changes unrelated to shim removal / dead code deletion.
- Public API stability outside this repository (assume this repo is the source of truth).

## Acceptance Gates (run per milestone)

- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- `uv run pytest -q`
- “No dangling imports” checks (exact commands listed in each milestone).

## Inventory (current evidence)

### Dead / unused candidates

| Item | Evidence | Notes |
|---|---|---|
| `src/codeintel/core/context/` | No imports found in `src/` or `tests/` (only referenced in archived docs) | Safe deletion after confirming no dynamic imports |
| `src/codeintel/core/data/` | No imports found in `src/` or `tests/` | Appears superseded by `codeintel.core.cache` |
| `src/codeintel/core/models/` | No imports found in `src/` or `tests/` | Not used anywhere (only self-export) |
| `src/codeintel/core/observability/` | No imports found in `src/` or `tests/` (only referenced in archived docs) | Safe deletion |
| `src/codeintel/core/results/` | No imports found in `src/` or `tests/` (only referenced in docstrings) | Duplicates other result types in build |
| `src/codeintel/core/runtime/` | No imports found in `src/` or `tests/` (only referenced in docstrings) | Separate from ingestion step tracking |
| `src/codeintel/core/validation/rules.py` | No imports found in `src/` or `tests/` | Candidate single-file deletion |
| `src/codeintel/core/constants/` | Imported only by `tests/build/hamilton/test_pr73_json_schema_generation.py:27` | “Test-only” usage; requires a test update before deletion |

### Compatibility / legacy shims (in active use)

| Shim | Location | Current usage |
|---|---|---|
| `BatchResult.rows_written`, `BatchResult.table_key`, `BatchResult.from_write` | `src/codeintel/core/ports/storage.py` | Used by ingestion code and test fakes via `IngestStoragePort` |
| `STATUS_CODES` mapping | `src/codeintel/core/errors/taxonomy.py` | Used only by CLI (`src/codeintel/cli/errors/taxonomy.py`) |
| Pydantic v1 fallback | `src/codeintel/core/plugins/execution/options.py` | Appears unused (repo uses Pydantic v2); must confirm before deleting |
| Provider compatibility layer | `src/codeintel/core/providers/` | Only used by `src/codeintel/build/schemas/contract_provider.py` for module-lazy loading |

## Recommended Execution Strategy

Do this in **small, verifiable milestones**, each ending with the acceptance gates.
The dependency order below avoids large “all-at-once” breakage:

1. Remove `STATUS_CODES` (low blast radius, few call sites).
2. Remove Pydantic v1 branch (low blast radius, but requires confidence checks).
3. Remove storage `BatchResult` compatibility aliases (moderate blast radius, localized to ingestion + test fakes).
4. Remove `core.providers` after migrating the single consumer.
5. Delete dead/unused packages (safe once no imports remain).
6. Delete `core.constants` only after test migration.

---

## Milestone 0 — Baseline & Safeguards

1. Record current green status:
   - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
   - `uv run pytest -q`
2. Confirm there are no dynamic imports of “dead” modules:
   - `rg -n "importlib\\.import_module\\(|__import__\\(" -S src`
   - `rg -n "codeintel\\.core\\.(context|data|models|observability|results|runtime)\\b" -S src tests`

Deliverable: baseline logs and a confirmed list of real importers.

---

## Milestone 1 — Delete `STATUS_CODES` Compatibility Mapping

### Why

`ErrorCode` already carries `status`; `STATUS_CODES` is redundant and currently used only by the CLI.

### Files expected to change

- `src/codeintel/core/errors/taxonomy.py`
- `src/codeintel/core/errors/__init__.py`
- `src/codeintel/cli/errors/taxonomy.py`
- `src/codeintel/cli/handlers/docs.py` (only if function signatures change)

### Plan

1. Replace CLI functions (`validation_error`, `operation_error`, etc.) to use `ErrorCode` objects directly:
   - Change parameters from `ValidationErrorCode`/`OperationErrorCode` enums to the corresponding `ErrorCode` constants
     (e.g., accept `MISSING_REQUIRED` rather than `ValidationErrorCode.MISSING_REQUIRED`).
   - Compute:
     - `type`: use `.code` to build the CLI URN (`urn:codeintel:cli:{category}:{code}`).
     - `status`: use `.status` from the `ErrorCode` instance.
2. Update the few call sites:
   - `src/codeintel/cli/handlers/docs.py` currently passes enum values (e.g., `ValidationErrorCode.INVALID_FORMAT`).
   - Replace with the `ErrorCode` constant equivalent (e.g., `INVALID_FORMAT`).
3. Remove `STATUS_CODES`:
   - Delete the dict from `src/codeintel/core/errors/taxonomy.py`.
   - Remove it from `__all__` and from `src/codeintel/core/errors/__init__.py`.
4. Verification:
   - `rg -n "\\bSTATUS_CODES\\b" -S src tests` returns no hits.
   - Run acceptance gates.

### Acceptance criteria

- CLI still emits the same statuses as before (400/404/500/etc), now sourced from `ErrorCode.status`.
- No remaining imports of `STATUS_CODES`.

---

## Milestone 2 — Remove Pydantic v1 Fallback in `PluginOptionsResolver`

### Why

The codebase uses Pydantic v2 (`uv run python -c "import pydantic; print(pydantic.__version__)"` reports v2).
The v1 fallback adds complexity and introduces a structural-typing hazard (objects with `.copy()` but incompatible signature).

### Files expected to change

- `src/codeintel/core/plugins/execution/options.py`
- Possibly add/update tests near plugin option resolution (if tests already exist; do not introduce a brand-new testing framework).

### Pre-checks (must be true before editing)

- No imports of `pydantic.v1` anywhere:
  - `rg -n "pydantic\\.v1" -S src tests`
- No option model depends on `.copy(update=...)` semantics:
  - `rg -n "\\.copy\\(\\s*update\\s*=" -S src tests`

### Plan

1. Remove `_PydanticV1Model` protocol and the `isinstance(base, _PydanticV1Model)` branch.
2. Keep support for:
   - Frozen dataclasses: `dataclasses.replace`
   - Pydantic v2 models: `model_copy(update=...)`
   - Plain Python objects: attribute assignment fallback (existing behavior)
3. Add a minimal regression test if there is an existing test location for this functionality; otherwise, rely on acceptance gates.
4. Verification:
   - `rg -n "_PydanticV1Model" -S src` returns no hits.
   - Run acceptance gates.

### Acceptance criteria

- No behavior change for Pydantic v2 option models.
- No new type-checking or lint errors.

---

## Milestone 3 — Remove `BatchResult` Backward-Compatibility Aliases

### Why

The canonical storage result fields are:

- `BatchResult.table`
- `BatchResult.rows_affected`

Ingestion currently uses “legacy” naming (`table_key`, `rows_written`) and uses `BatchResult.from_write`.
This milestone completes the unification by moving ingestion to the canonical fields and removing aliases.

### Files expected to change (known call sites)

- Canonical type:
  - `src/codeintel/core/ports/storage.py`
- Ingestion (rows_written access):
  - `src/codeintel/ingestion/compute/ast_extract.py`
  - `src/codeintel/ingestion/compute/tests_ingest.py`
  - `src/codeintel/ingestion/compute/scip_ingest.py`
  - `src/codeintel/ingestion/compute/coverage_ingest.py`
  - `src/codeintel/ingestion/compute/typing_ingest.py`
  - `src/codeintel/ingestion/compute/config_ingest.py`
  - `src/codeintel/ingestion/compute/repo_scan.py`
  - `src/codeintel/ingestion/compute/base.py`
  - `src/codeintel/ingestion/compute/__init__.py` (docstring/print)
- Storage adapters / fakes (from_write):
  - `src/codeintel/ingestion/adapters/duckdb_storage.py`
  - `tests/_helpers/fakes/storage.py`
- Ingestion ports documentation:
  - `src/codeintel/ingestion/ports/storage.py`

### Plan

1. Update ingestion call sites to canonical names:
   - Replace `.rows_written` with `.rows_affected` everywhere in ingestion compute code.
   - If any `.table_key` property is used on a `BatchResult`, replace with `.table`.
2. Replace `BatchResult.from_write(...)` uses:
   - In `DuckDBStorageAdapter.write_batch`, return `BatchResult.ok(table_key, inserted, duration_s=...)`.
   - In `FakeIngestStorage.write_batch`, return `BatchResult.ok(table_key, len(rows), duration_s=...)`.
3. Update ingestion docs to describe canonical names:
   - In `src/codeintel/ingestion/ports/storage.py`, change prose referencing “rows_written/table_key” on results to “rows_affected/table”.
4. Remove the compatibility API surface from `BatchResult`:
   - Remove `.table_key` and `.rows_written` properties.
   - Remove `BatchResult.from_write`.
   - Ensure docstrings/examples use canonical names.
5. Verification:
   - `rg -n "\\.rows_written\\b" -S src tests` returns no hits.
   - `rg -n "BatchResult\\.from_write\\(" -S src tests` returns no hits.
   - Run acceptance gates.

### Acceptance criteria

- Ingestion compute still produces correct table_counts and totals.
- `BatchResult` has only one naming scheme (no aliases).

---

## Milestone 4 — Remove `codeintel.core.providers` (migrate the single consumer)

### Why

There are two “provider” systems in core:

- `codeintel.core.resources` (actively used)
- `codeintel.core.providers` (appears to be a parallel/legacy abstraction)

Only `src/codeintel/build/schemas/contract_provider.py` uses `LazyProvider` from `codeintel.core.providers`.
Remove this duplication by migrating that file to use `codeintel.core.resources` (or a local minimal cache).

### Files expected to change

- `src/codeintel/build/schemas/contract_provider.py`
- `src/codeintel/core/providers/` (delete entire package after migration)

### Plan

1. Replace `LazyProvider` usage:
   - Introduce a tiny private `ResourceProviderBase[ModuleType]` subclass inside `contract_provider.py`
     that loads via `importlib.import_module(...)` and caches the module.
   - Or use `SingletonHolder` for one-time lazy initialization (if it keeps type-checkers happy).
2. Ensure the replacement preserves:
   - Lazy import behavior (avoid import-time side effects/cycles).
   - No function-scoped imports are introduced.
3. Delete `src/codeintel/core/providers/` package:
   - Remove `base.py`, `lazy.py`, `protocol.py`, and `__init__.py`.
4. Verification:
   - `rg -n "codeintel\\.core\\.providers\\b" -S src tests` returns no hits.
   - Run acceptance gates.

### Acceptance criteria

- Build schema contract provider still works (imports succeed; lazy behavior preserved).
- No remaining references to `codeintel.core.providers`.

---

## Milestone 5 — Delete Dead / Unused Core Packages

### Why

These packages are not imported by `src/` or `tests/` and appear to be remnants of earlier consolidation work.

### Targets and plan per package

For each target:
1. Confirm no importers:
   - `rg -n "<module_path>" -S src tests`
2. Delete the directory/file.
3. Run acceptance gates.

#### 5.1 `src/codeintel/core/context/`

- Delete: `src/codeintel/core/context/__init__.py`, `base.py`, `builder.py`, `protocol.py`
- Post-check: `rg -n "codeintel\\.core\\.context\\b" -S src tests` has no hits

#### 5.2 `src/codeintel/core/data/`

- Delete: `src/codeintel/core/data/__init__.py`, `loader.py`, `protocol.py`
- Post-check: `rg -n "codeintel\\.core\\.data\\b" -S src tests` has no hits

#### 5.3 `src/codeintel/core/models/`

- Delete: `src/codeintel/core/models/__init__.py`, `rows.py`
- Post-check: `rg -n "codeintel\\.core\\.models\\b" -S src tests` has no hits

#### 5.4 `src/codeintel/core/observability/`

- Delete: `src/codeintel/core/observability/__init__.py`, `metrics.py`, `protocol.py`, `tracing.py`
- Post-check: `rg -n "codeintel\\.core\\.observability\\b" -S src tests` has no hits

#### 5.5 `src/codeintel/core/results/`

- Delete: `src/codeintel/core/results/__init__.py`, `base.py`, `execution.py`, `protocol.py`
- Post-check: `rg -n "codeintel\\.core\\.results\\b" -S src tests` has no hits

#### 5.6 `src/codeintel/core/runtime/`

- Delete: `src/codeintel/core/runtime/__init__.py`, `protocol.py`, `tracking.py`
- Post-check: `rg -n "codeintel\\.core\\.runtime\\b" -S src tests` has no hits

#### 5.7 `src/codeintel/core/validation/rules.py`

- Delete: `src/codeintel/core/validation/rules.py`
- Post-check: `rg -n "core\\.validation\\.rules|validation\\.rules" -S src tests` has no hits

---

## Milestone 6 — Delete `core.constants` (test-only usage cleanup)

### Why

`src/codeintel/core/constants/crypto.py` is currently used only by a single test.
If the project does not want a dedicated “constants” package, this should be removed and tests should inline
the value or use an existing canonical constant location.

### Files expected to change

- `tests/build/hamilton/test_pr73_json_schema_generation.py`
- Delete:
  - `src/codeintel/core/constants/__init__.py`
  - `src/codeintel/core/constants/crypto.py`

### Plan

1. Replace the import in `tests/build/hamilton/test_pr73_json_schema_generation.py` with a local constant:
   - Use `SHA256_HEX_DIGEST_LENGTH = 64` in the test module, or assert directly against `64`.
2. Delete `src/codeintel/core/constants/`.
3. Verification:
   - `rg -n "codeintel\\.core\\.constants\\b" -S src tests` returns no hits.
   - Run acceptance gates (the test suite is the important gate here).

### Acceptance criteria

- Tests remain green.
- No constants package remains in `src/codeintel/core/`.

---

## Final Verification Checklist

- No references remain:
  - `rg -n "codeintel\\.core\\.(context|data|models|observability|results|runtime|providers|constants)\\b" -S src tests`
  - `rg -n "\\bSTATUS_CODES\\b|BatchResult\\.from_write\\(|\\.rows_written\\b" -S src tests`
- Quality gates are green:
  - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
  - `uv run pytest -q`

## Rollback Strategy

Each milestone is designed to be revertible by restoring the deleted module(s) and undoing the local call-site edits.
Avoid mixing multiple milestones in a single change if you want straightforward rollback.

