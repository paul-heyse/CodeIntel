# Build + Storage Warehouse Alignment & Guardrails Implementation Plan

**Status**: Implementation plan (followup alignment)  
**Last updated**: 2025-12-18  
**Primary scope**: `src/codeintel/build/**`, `src/codeintel/storage/**`  
**Secondary scope**: `tools/guardrails.py`, `tests/**`  

## 0) Why This Plan Exists

We now have a canonical shared Ibis typing seam (`src/codeintel/core/ibis_typing.py`) and a canonical table
acquisition seam (`src/codeintel/storage/gateway/ibis_facade.py`) that remove repeated type work and
call-site friction.

This plan takes the next alignment step: **make the storage `Warehouse` the single durable I/O boundary**
for snapshot-scoped writes, and **enforce** the new architecture so build/storage cannot drift back into
ad-hoc write paths or table-access patterns.

## 1) Goals and Non‑Goals

### Goals (must)

1) **Single write boundary**
   - All snapshot-scoped table writes originating in build flow through `codeintel.storage.warehouse.Warehouse`.
2) **Build ergonomics**
   - Build DAG nodes and materializers get a small, consistent API surface via `BuildEnv` so they don’t
     re-implement “construct warehouse, scope reads, write + record assets”.
3) **Protocol simplification**
   - Reduce union typing and “gateway variants” by tightening the gateway protocol layering.
4) **Table key hardening**
   - Standardize `TableKey` semantics (validation, parsing) so build and storage share the same contract.
5) **Guardrails + tests**
   - Enforce invariants via `tools/guardrails.py` and a fast unit test that scans `src/codeintel/build/**`
     and `src/codeintel/storage/**`.

### Non‑Goals (for this tranche)

- Rewriting analytics/ingestion call sites beyond what is required for build/storage alignment.
- Changing dataset contracts, schema providers, or the build target catalog semantics.
- Adding new suppressions (`# type: ignore` is not a strategy here).

## 2) Target Architecture (End State)

### 2.1 Read/compute/write layering

- **Core typing seam**
  - `src/codeintel/core/ibis_typing.py`: the only allowed place for minimal `cast("Any", ...)` to smooth Ibis
    operator typing friction.
- **Storage read seam**
  - `src/codeintel/storage/gateway/ibis_facade.py`: the standard way to acquire an Ibis table expression from a
    gateway (`table(gateway, table_key) -> ir.Table`).
- **Storage write seam**
  - `src/codeintel/storage/warehouse.py`: the standard way to materialize data (rows, mappings, dataframes, or
    Ibis tables) into DuckDB with snapshot semantics and asset tracking.
- **Build orchestration seam**
  - `src/codeintel/build/hamilton/env.py`: build nodes receive only `env: BuildEnv`; nodes should not directly
    perform low-level persistence patterns.
  - Build persistence is performed by:
    - Hamilton materializers/savers (build-owned), which delegate to `Warehouse` for actual writes.

### 2.2 Allowed write paths (canonical)

- `Warehouse.materialize_table(...)` for `ir.Table` outputs.
- `Warehouse.materialize_rows(...)` / `Warehouse.materialize_dataframe(...)` / `Warehouse.materialize_mappings(...)`
  for row-shaped outputs and typed mapping rows.
- Storage internal DDL/mutation remains behind `gateway.policy` and `gateway.ibis.write/delete`, but build code should
  not use those low-level APIs directly except inside the `Warehouse` and the dedicated build materializers.

## 3) Acceptance Gates (keep green)

Run after each workstream (or at least at the end):

```bash
scripts/bootstrap_codex.sh
uv sync

uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

For tight iteration:

```bash
uv run ruff check --fix src/codeintel/build src/codeintel/storage tools tests
uv run pyright --warnings --pythonversion=3.13 src/codeintel/build src/codeintel/storage tools tests
uv run pyrefly check src/codeintel/build src/codeintel/storage tools tests
uv run pytest -q tests/build tests/storage
```

## 4) Workstreams and Sequencing

### W0 — Inventory and classification (1 pass, must be explicit)

**Objective**: identify and classify all remaining “write boundary” and “table acquisition” drift.

Tasks:

1) Find direct build writes via policy backend:
   - `rg -n "\\.policy\\.(ensure_table|delete_for_snapshot|bulk_insert|bulk_insert_mappings|delete\\()\" src/codeintel/build`
2) Find direct build writes via ibis adapter:
   - `rg -n \"\\.ibis\\.(write|delete)\\(\" src/codeintel/build`
3) Find remaining direct `gateway.ibis.table(...)` call sites:
   - `rg -n \"\\.ibis\\.table\\(\" src/codeintel/build src/codeintel/storage`
4) Produce a short checklist grouped by module:
   - “convert to Warehouse” vs “keep (allowed storage internals)”

Deliverable:

- A small table in this doc (or a scratch note) enumerating each offender:
  - file, symbol/function, write/read pattern, target table(s), and recommended replacement.

### W1 — Tighten gateway protocol layering (simplify typing and seams)

**Objective**: remove `MinimalGateway | StorageGateway` unions and make gateway composition more uniform.

Target change:

- Make `StorageGateway` extend `MinimalGateway` in `src/codeintel/storage/gateway/protocol.py` so “anything that is a
  StorageGateway is also a MinimalGateway”.

Tasks:

1) Update `src/codeintel/storage/gateway/protocol.py`:
   - `class StorageGateway(MinimalGateway, Protocol): ...`
   - Ensure no attribute conflicts and the protocol remains structural.
2) Update `src/codeintel/storage/gateway/ibis_facade.py`:
   - Prefer `table(gateway: MinimalGateway, table_key: TableKey) -> ir.Table` once the protocol layering is correct.
3) Sweep call sites (if needed):
   - Remove union typing that becomes unnecessary.

Acceptance criteria:

- `ibis_facade.table(...)` no longer needs to accept union gateway types.
- Call sites do not need extra casts or local Protocol definitions.

### W2 — Canonical `TableKey` contract (shared semantics across build + storage)

**Objective**: unify “table key” handling and reduce implicit stringly-typed drift.

Design direction:

- Storage owns canonical parsing/validation; build re-exports only the alias/type and uses storage validation.

Tasks:

1) Extend `src/codeintel/storage/helpers/table_key.py`:
   - Add `type TableKey = str` (or a small `NewType` if it remains ergonomic).
   - Add `parse_table_key(table_key: TableKey) -> ParsedTableKey` (schema + name).
   - Add `validate_table_key(table_key: TableKey) -> None` with explicit error messages.
   - Keep `split_table_key` as a compatibility wrapper only if still widely used; otherwise migrate and delete.
2) Build-side re-export:
   - Update `src/codeintel/build/hamilton/boundary_types.py` to import/re-export `TableKey` (or define it as an alias
     pointing to the storage type) so build boundary typing references the storage canonical contract.
3) Update storage and build call sites:
   - Replace ad-hoc `str` table key parameters with `TableKey` where it improves clarity at module boundaries.

Acceptance criteria:

- Single canonical implementation for parsing/validation of table keys exists in storage.
- Build code does not implement its own table-key parsing logic.

### W3 — BuildEnv convenience seam (make the happy path easy)

**Objective**: provide a small, consistent set of “build runtime primitives” that DAG nodes use.

Recommended API:

- Add `BuildEnv.warehouse` (property) returning `Warehouse(self.gateway)`.
- Optionally add:
  - `BuildEnv.read_table(table_key: TableKey, *, scope_to_snapshot: bool = True) -> ir.Table`
  - `BuildEnv.materialize_*` thin wrappers if it meaningfully reduces boilerplate in target modules.

Tasks:

1) Update `src/codeintel/build/hamilton/env.py`:
   - Add a small property that constructs `Warehouse` (no caching required; it’s a thin wrapper).
2) Adopt in a small number of modules first:
   - One “graphs” target module
   - One “analytics” target module
   - One materializer (e.g., rows saver)
3) Ensure API design remains minimal:
   - Prefer a single `env.warehouse` entry point over many helper methods unless the helpers are clearly reused.

Acceptance criteria:

- Build code constructs `Warehouse` via `env.warehouse` (or equivalent) in the common case.
- Nodes read tables via the canonical seam (`Warehouse.read` and/or `ibis_facade.table`).

### W4 — Migrate build writes to `Warehouse` (decommission legacy patterns)

**Objective**: eliminate copy/pasted “ensure/delete/bulk_insert” patterns and centralize snapshot semantics.

Migration rules:

- In build code, “ensure table + delete snapshot + bulk insert” must be replaced with:
  - `Warehouse.materialize_rows(..., options=MaterializeOptions(snapshot=env.snapshot, mode="replace", ...))`
  - or `Warehouse.materialize_mappings(...)` when call sites already have mapping rows.
- Use `MaterializeOptions.owner_target` and `MaterializeOptions.input_hash` where available to improve traceability.

Tasks:

1) Replace helper-level write utilities:
   - Refactor or delete `persist_rows(...)` in `src/codeintel/build/hamilton/helpers.py` once all call sites migrate.
2) Update build native targets that write via `gateway.policy.*`:
   - Common offenders are graph modules under `src/codeintel/build/hamilton/native/graphs/**`.
3) Update any build modules writing via `gateway.ibis.write/delete` directly:
   - Replace with Warehouse materialization methods.
4) Remove deprecated code:
   - Delete legacy helper functions once the last call site is migrated.
   - Ensure no compatibility shims remain in build for the deprecated write path.

Acceptance criteria:

- No build module performs snapshot-scoped persistence by calling `gateway.policy.*` directly (outside the Warehouse
  implementation and the dedicated build materializers).
- All migrated call sites preserve behavior (same tables populated, same snapshot scope).

### W5 — Guardrails + fast tests (prevent regression)

**Objective**: make it impossible to drift back into forbidden patterns.

Guardrails to add/tighten in `tools/guardrails.py`:

1) Ban `.ibis.table(` outside allowed files:
   - Allow in:
     - `src/codeintel/storage/gateway/ibis_facade.py`
     - `src/codeintel/storage/ibis_adapter.py`
   - Disallow elsewhere (prefer `ibis_facade.table`).
2) Ban build-side `.policy.*` write calls outside allowed locations:
   - Allow in:
     - `src/codeintel/storage/warehouse.py`
     - build materializers that are explicitly the “write boundary” (and even there, prefer delegating to Warehouse).
3) Error messages must point to the canonical replacements:
   - “Use `Warehouse.materialize_*`”
   - “Use `ibis_facade.table(...)`”

Tests:

- Add `tests/test_build_storage_architecture_invariants.py` (or similar) that:
  - Scans `src/codeintel/build/**` for forbidden `.policy.(ensure_table|delete_for_snapshot|bulk_insert*)`
  - Scans `src/codeintel/build/**` and `src/codeintel/storage/**` for forbidden `.ibis.table(` outside allowlist.
  - Uses stable relative paths and prints clear offender lists.

Acceptance criteria:

- Guardrails fail fast on architectural regressions.
- The test suite contains at least one fast invariant test enforcing the new alignment.

### W6 — Verification + cleanup (no legacy residue)

Tasks:

1) Run acceptance gates (see Section 3).
2) Re-run inventory commands from W0; ensure they return empty (or only allowlisted hits).
3) Confirm deprecated functions/files are deleted after migration:
   - No old helper functions remain if unused.
4) Update any docs that describe the old write path:
   - Ensure docs consistently describe:
     - `Warehouse` for writes
     - `ibis_facade` for reads
     - `core.ibis_typing` for typing friction

Deliverable:

- A short “Completed” section appended to this document listing:
  - migrated modules
  - deleted legacy helpers
  - guardrails/tests added

