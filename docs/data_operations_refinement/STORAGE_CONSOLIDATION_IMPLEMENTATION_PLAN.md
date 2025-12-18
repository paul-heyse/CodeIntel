# Storage Layer Consolidation — Best‑in‑Class Implementation Plan

**Status**: Implementation plan (storage-only)  
**Last updated**: 2025-12-18  
**Primary scope**: `src/codeintel/storage/**`  
**Secondary scope (deferred)**: `src/codeintel/serving/**`, `src/codeintel/build/**` call-site updates only when needed.

**Related references**
- `docs/data_operations_refinement/ADVANCED_DUCKDB_IBIS_SQLGLOT_BEST_IN_CLASS_IMPLEMENTATION_PLAN.md`

---

## 0) Purpose and Scope

This document translates the storage-only consolidation opportunities into a concrete implementation roadmap that:

- **Streamlines the storage design** (fewer parallel abstractions; fewer “special paths”).
- **Hardens behavior** (consistent session policy, extension policy, temp lifecycle hygiene).
- **Improves maintainability** (shared AST utilities; less duplicated write orchestration).
- **Preserves (or improves) functionality** while deleting legacy/compat paths created during the refactor.

### In scope (must)

- Unify the **DuckDB open path** so connection/session policy is applied consistently.
- Consolidate **SQLGlot utilities** into one canonical “toolkit”.
- Convert internal metadata DDL to be **contract/AST driven** (no raw-string DDL islands).
- Reduce duplication in `Warehouse` by refactoring into a single **materialization pipeline**.
- Unify snapshot scoping semantics (`repo`/`commit`) across `Warehouse` and repositories.
- Consolidate extension loading and feature gates (FTS) into the same connection policy layer.
- Standardize temp relation/table lifecycle management across storage.

### Out of scope (for this tranche)

- Serving/build feature work not required to keep storage contracts consistent.
- Product feature expansion unrelated to storage hardening/consolidation.

---

## 1) Current Baseline (What Exists Today)

The storage tree already contains strong primitives that we should consolidate around rather than proliferate new variants:

- **Connection wiring** (env → DuckDB connect config, attach history, schema apply):
  - `codeintel.storage.gateway.connection.connect` (`src/codeintel/storage/gateway/connection.py:33`)
- **Session lifecycle wrapper** (secrets, init SQL, fsspec registration, read-only defaults):
  - `codeintel.storage.backend.DuckDBSession` (`src/codeintel/storage/backend/duckdb_session.py:44`)
- **AST-first compilation seam**:
  - `IbisGateway.to_sqlglot` (`src/codeintel/storage/ibis_adapter.py:246`)
- **DDL generation from contracts**:
  - `create_table_ast` (`src/codeintel/storage/schema_roundtrip.py:70`)
  - Policy backend DDL entrypoints (`src/codeintel/storage/duckdb_policy_backend.py:774`)
- **Lifecycle-safe staging helper**:
  - `registered_temp_relation` (`src/codeintel/storage/staging.py:19`)
- **SQL diffs + dependency extraction**:
  - `views/diff.py` (`src/codeintel/storage/views/diff.py:98`)
  - `views/dependencies.py` (`src/codeintel/storage/views/dependencies.py:29`)
- **SQL ingress perimeter validator (additive)**:
  - `SqlIngressPolicy` + `assert_select_perimeter` (`src/codeintel/storage/queries/safe.py:119`)

### Key inconsistency to fix first

Session policy is not applied uniformly because some “opening paths” bypass `DuckDBSession`:

- `open_gateway()` calls `connect()` directly (`src/codeintel/storage/gateway/factory.py:31`).
- `ReadPoolWarehouse` opens connections via `connect()` directly (`src/codeintel/storage/gateway/pool.py:61`).

This means secrets/init SQL/fsspec hooks are “best effort” rather than guaranteed.

---

## 2) Guiding Principles (Design Contracts)

1. **One open path**: there must be exactly one canonical way to open DuckDB connections in production code.
2. **AST-first**: compile to SQL strings only at the final execution boundary; do not “SQL string → parse back to AST” except at explicit ingress perimeters.
3. **Storage owns policy**: extension/secrets/init SQL/filesystems belong to storage session policy, not feature modules.
4. **Deterministic lifecycle hygiene**: temp relations/tables must be cleaned up even on exceptions/cancellation.
5. **Decommission by default**: new work must delete legacy/compat code produced by earlier steps (no long-lived shims).

---

## 3) Quality Gates (Acceptance Gates Per Phase)

Run after each phase (and after any wide refactor):

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q tests/storage
```

When serving/build call sites are updated later, use full suite:

```bash
uv run pytest -q
```

---

## 4) Workstreams and Sequencing (Minimize Rework)

1. **W1 — Canonical Session Open Path** (highest ROI; fixes inconsistent policy)
2. **W2 — SQLGlot Toolkit Consolidation** (prevents semantic drift across modules)
3. **W3 — Metadata DDL Contractization** (remove raw-string DDL islands)
4. **W4 — Warehouse Materialization Pipeline Refactor** (reduce duplication; enable telemetry later)
5. **W5 — Snapshot Scoping Unification** (avoid diverging semantics)
6. **W6 — Extension/Feature Gate Unification (FTS)** (no ad-hoc LOAD/INSTALL)
7. **W7 — Temp/Staging Standardization** (no leaked temp objects)
8. **W8 — Storage Read Surface Convergence** (accessors vs repos vs warehouse)

---

## 5) Workstream Details

### W1 — Canonical Session Open Path

**Objective**: Ensure every production connection open applies the same hooks (env config, extension policy, secrets, fsspec, init SQL, attach history, schema/view bootstrap).

**Primary design decision**: pick one and delete/retire the other as a public entrypoint.

- **Option A (recommended)**: Make `DuckDBSession` the canonical open layer and treat `connect()` as low-level.
- **Option B**: Fold all session hooks into `connect()` and delete the session layer.

**Implementation tasks**

1) **Inventory and categorize open paths**
   - Enumerate all `connect()` and `duckdb.connect(...)` uses under `src/codeintel/storage/**`.
   - Define which are “production opens” vs “ephemeral/test-only”.

2) **Choose canonical API surface**
   - If Option A: introduce a new helper (e.g., `open_gateway_with_session(...)`) or update `open_gateway(...)` to use `DuckDBSession.open()/open_reader()` (`src/codeintel/storage/backend/duckdb_session.py:58`).
   - If Option B: move the `_bootstrap_*` hooks from `DuckDBSession` into `connect()` and delete `DuckDBSession` (and update imports).

3) **Update serving pool connection creation**
   - Ensure `ReadPoolWarehouse` uses the canonical open path so reader defaults and hooks are applied (`src/codeintel/storage/gateway/pool.py:61`).
   - Remove duplicated config typing in pool by importing `DuckDBConnectConfig` from `src/codeintel/storage/gateway/connection.py:23`.

4) **Clarify read-only semantics**
   - Reconcile `StorageConfig.for_readonly(...)` defaults (`src/codeintel/storage/gateway/config.py:61`) with the actual behavior we want in serving:
     - should read-only ever attempt `INSTALL`? (likely no)
     - should it `LOAD` explicitly requested extensions? (likely yes)
     - should it run init SQL? (decide; usually yes for PRAGMA/policy)

**Acceptance criteria**

- A “storage session policy” is applied consistently no matter how the gateway/pool is opened.
- There is a single documented and enforced public open path (others are internal/private).
- `ReadPoolWarehouse` and `open_gateway()` agree on read-only behavior and apply the same hooks.

**Tests to add**

- Reader connection applies `DuckDBSession` policy knobs (init SQL executed; secrets registration attempted safely; fsspec requested protocols registered).
- Pool uses read-only defaults (no INSTALL on read-only; LOAD behavior as designed).

---

### W2 — SQLGlot Toolkit Consolidation

**Objective**: Prevent semantic drift by centralizing parsing/canonicalization/table-ref extraction/fingerprinting/diff in a single module.

**Current scattered implementations**

- Canonicalization and hashing: `src/codeintel/storage/views/diff.py:28`
- CTE-safe dependency extraction: `src/codeintel/storage/views/dependencies.py:29`
- Ingress perimeter validation: `src/codeintel/storage/queries/safe.py:130`

**Implementation tasks**

1) Create a single toolkit module/package (recommended: `src/codeintel/storage/sqlglot_tools.py` or `src/codeintel/storage/sql/…`)
   - `parse_one_duckdb(sql: str) -> exp.Expression`
   - `canonical_sql_duckdb(sql: str) -> str`
   - `extract_physical_table_keys_duckdb(...) -> frozenset[str]` using `traverse_scope` (CTE-safe)
   - `sql_fingerprint_duckdb(sql: str) -> str` (stable hash of canonical SQL)
   - `diff_sql_structural_duckdb(before: str, after: str) -> SqlStructuralDiffSummary` (one canonical path)

2) Refactor call sites to depend on toolkit
   - `views/diff.py` delegates canonicalization/fingerprint/diff to toolkit.
   - `views/dependencies.py` delegates table extraction to toolkit (or share the same extraction primitive).
   - `queries/safe.py` reuses toolkit parsing primitives (while keeping policy semantics local).

**Acceptance criteria**

- Only one implementation exists for “canonical SQL” and “table refs”.
- Structural diff and fingerprinting are stable across call sites (same SQL → same hash).

**Tests to add**

- Table ref extraction correctness for:
  - CTE shadowing
  - nested queries
  - unqualified references
- Fingerprint stability for semantically identical SQL that differs only by whitespace/casing.

---

### W3 — Metadata DDL Contractization

**Objective**: Remove raw-string DDL islands by expressing metadata tables as `TableSchema` and generating DDL via SQLGlot/policy backend.

**Current raw-string DDL**

- `METADATA_SCHEMA_DDL` tuple of SQL strings (`src/codeintel/storage/metadata/bootstrap.py:72`)

**Implementation tasks**

1) Define metadata tables as `TableSchema` (and indexes) in a single place:
   - recommended new module: `src/codeintel/storage/metadata/schema.py`
2) Update `apply_metadata_ddl(...)` to use:
   - `MinimalStorageGateway(con).policy.create_schema_if_not_exists(...)` and
   - `create_table_from_schema(..., if_not_exists=True)` (`src/codeintel/storage/duckdb_policy_backend.py:774`)
3) Ensure table/index creation is idempotent and preserves existing data (match current semantics).

**Acceptance criteria**

- No raw SQL DDL constants remain for metadata schema creation.
- Metadata schema evolution becomes diffable/testable using the same contract mechanisms as the rest of storage.

**Tests to add**

- Idempotent bootstrap: calling metadata bootstrap twice yields no errors and schema matches expected.
- Index creation idempotence.

---

### W4 — Warehouse Materialization Pipeline Refactor

**Objective**: Remove duplicated orchestration across `materialize_*` methods while preserving semantics (transaction safety, profiling artifacts, asset recording).

**Duplication hotspots**

- `materialize_table`, `materialize_dataframe`, `materialize_rows`, `materialize_mappings` all repeat:
  - table ensure
  - snapshot delete-for-replace
  - transaction
  - optional profiling enable/disable
  - asset record creation and write
  - repeated result assembly
  - `Warehouse` (`src/codeintel/storage/warehouse.py:189`)

**Implementation tasks**

1) Introduce a single private helper (e.g., `_materialize(...)`) that owns:
   - validation of `MaterializeOptions`
   - `ensure_table(...)`
   - replace delete semantics
   - transaction + error handling
   - profiling enable/disable
   - asset record writing
2) Keep public API stable; keep `MaterializationResult` stable.
3) Ensure “write path selection” remains explicit and testable (Ibis insert-select vs bulk insert).

**Acceptance criteria**

- All four `materialize_*` methods share one orchestration path.
- Profiling artifacts are still produced and disabled correctly (`src/codeintel/storage/warehouse.py:798`).
- Transaction rollback behavior remains correct (no partial writes).

**Tests to add**

- Write-path equivalence tests across `materialize_*` variants (same input yields same persisted rows).
- Profiling enable/disable correctness (profiling is disabled after operation even when exceptions occur).

---

### W5 — Snapshot Scoping Unification

**Objective**: Ensure “snapshot filtering when `repo`/`commit` columns exist” behaves identically everywhere.

**Current duplication**

- Warehouse read scoping (`src/codeintel/storage/warehouse.py:124`)
- BaseRepository `_ibis_table` scoping (`src/codeintel/storage/repositories/base.py:73`)

**Implementation tasks**

1) Add a single helper function (suggested module: `src/codeintel/storage/snapshot_scoping.py`)
   - `maybe_scope_table(expr: ir.Table, *, repo: str, commit: str) -> ir.Table`
   - `maybe_scope_by_snapshot(expr: ir.Table, snapshot: SnapshotRef) -> ir.Table`
2) Update `Warehouse.read(...)` and `BaseRepository._ibis_table(...)` to use it.
3) Optional: converge repository reads onto `Warehouse.read(...)` as a follow-on (still storage-only, but may require careful call-site review).

**Acceptance criteria**

- No duplicated logic for snapshot filtering remains.
- Snapshot scoping is consistent and test-covered.

**Tests to add**

- Tables with/without repo+commit columns behave correctly (filtered vs unfiltered).
- Repository and warehouse read results are identical for the same snapshot.

---

### W6 — Extension/Feature Gate Unification (FTS)

**Objective**: Ensure “extension availability” is handled once by connection/session policy; feature modules do not ad-hoc `LOAD` or `INSTALL`.

**Current split**

- Env-driven extension load at connect time (`src/codeintel/storage/gateway/connection.py:135`)
- Ad-hoc FTS loading (`src/codeintel/storage/serving/search_index.py:170`)

**Implementation tasks**

1) Create a storage-owned extension helper (e.g., `src/codeintel/storage/gateway/extensions.py`)
   - validates extension names
   - applies policy: `INSTALL` only when allowed, but `LOAD` when requested
2) Update `connect()` (or session policy) to call extension helper.
3) Update `ensure_fts_index(...)` to rely on “extension is already loaded” (or to call the same helper explicitly).

**Acceptance criteria**

- There is one implementation of extension name validation and loading rules.
- Read-only never triggers `INSTALL` (by policy).

**Tests to add**

- Invalid extension name is rejected consistently.
- FTS “not available” error is stable and actionable.

---

### W7 — Temp/Staging Standardization

**Objective**: Ensure all in-memory registrations and temp objects have deterministic cleanup.

**Current primitive**

- `registered_temp_relation(...)` (`src/codeintel/storage/staging.py:19`)

**Implementation tasks**

1) Ensure all places that call `con.register(...)` or `con.unregister(...)` use `registered_temp_relation`.
2) If storage uses temp *tables* (not just registered relations), add:
   - `temporary_table(...)` context manager that creates/drops a temp table deterministically.
3) Add “temp object hygiene” rules to storage docs and enforce via code review (no inline register/unregister).

**Acceptance criteria**

- No leaked temp relations/tables across exceptions.
- Long-lived processes (serving) do not accumulate temp artifacts.

**Tests to add**

- Temp relation is unregistered on exception.
- Temp table is dropped on exception (if introduced).

---

### W8 — Storage Read Surface Convergence (Accessors vs Repositories vs Warehouse)

**Objective**: Pick and enforce a single “read surface” pattern so future work does not create parallel variants.

**Current situation**

- Typed accessors exist (`src/codeintel/storage/gateway/accessors.py:1`).
- Repositories provide a read interface and implement their own scoping (`src/codeintel/storage/repositories/base.py:73`).
- Warehouse provides the intended I/O boundary (`src/codeintel/storage/warehouse.py:107`).

**Decision to make**

- Preferred: **Warehouse is the single I/O boundary** and repositories use it for reads/writes.
- Accessors remain as ergonomic “relation getters” only (no policy/semantics).

**Implementation tasks**

1) Document the chosen pattern in storage module docs.
2) Refactor repositories to depend on Warehouse (where feasible) and remove duplicated helpers.
3) Delete unused/dead APIs created by the convergence (decommission fully).

**Acceptance criteria**

- New code has an obvious “one way” to access data.
- Reduced duplication and fewer places to apply policy changes.

---

## 6) Decommissioning and Legacy Code Purge Policy

For each workstream:

1) Identify any new compatibility layer introduced to ease migration.
2) Track it in a “compat removal checklist” section in the PR/branch notes.
3) Remove it before finishing the overall consolidation tranche.

The end state must not contain:

- duplicate open paths
- duplicated SQL parsing/canonicalization logic
- ad-hoc extension loading in feature modules
- raw-string metadata DDL islands

---

## 7) Deferred Follow‑Ons (After Serving/Build Changes Land)

These are explicitly deferred but should be enabled by this consolidation work:

- Serving can consume stable `query_hash`/fingerprints derived from the toolkit without re-implementing hashing.
- Serving pool can rely on storage policy to ensure extensions/secrets/init SQL are consistent.
- Build can rely on a single gateway/session open path to avoid environment drift between build and serving.

