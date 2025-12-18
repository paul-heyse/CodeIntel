# Storage Refinement — Best-in-Class Consolidation Implementation Plan

**Status**: Implementation plan  
**Last updated**: 2025-12-18  
**Primary scope**: `src/codeintel/storage/**`  
**Secondary scope** (required call-site migrations): `src/codeintel/core/**`, `src/codeintel/serving/**`,
`src/codeintel/cli/**`, `src/codeintel/config/**`, and `tests/**`.

## 0) Goals (Best-in-Class Target Shape)

By the end of this plan:

1) **One canonical DuckDB runtime bootstrap** (consistent behavior across build/serving/tests)
   - A single owner for env parsing, connection config, extension/secrets/init SQL policy, schema/bootstrap.
   - Serving pool connections behave identically to non-pooled read-only connections.

2) **One canonical DDL/Schema builder surface**
   - No duplicated SQLGlot DDL builders for schemas/indexes.
   - Table DDL remains contract-driven via `TableSchema` → `ibis.Schema` → SQLGlot.

3) **Clear JSON Schema story (no “three truths”)**
   - Explicit separation (or explicit unification) of:
     - contract schema (DDL + drift + contract validation), and
     - export schema (API/export row-shape guarantees).
   - No remaining coupling to `src/codeintel/config/schemas/export/*.json` as a source-of-truth.

4) **Views are definition-first, orchestration-owned**
   - View definitions remain in `storage/views/*`.
   - View discovery/compilation/materialization/toposort/lineage-sync live in a dedicated views orchestrator
     module (not inside `DuckDBPolicyBackend`).
   - All “legacy create_* view functions” are removed after call-site migration.

5) **Repository layer is consistent and minimal**
   - All read surfaces use the same conversion/normalization helpers (Ibis→dicts, scalar coercion, JSON decode).
   - No ad-hoc DataFrame conversion/`to_dict` patterns outside a small set of canonical helpers.

6) **One canonical storage error surface**
   - No overlapping `QueryError`/`SchemaError` definitions across `codeintel.core` and `codeintel.storage`.
   - Clear layering: serving and storage share a single structured error taxonomy when appropriate.

7) **No dead/compatibility code left behind**
   - Any temporary wrappers created during migration are removed by the closeout phase.
   - “Empty/placeholder” packages and unused modules are deleted.

## 1) Current State (Why We’re Changing)

### A) Connection bootstrapping is split across multiple modules

- `src/codeintel/storage/gateway/connection.py` parses env config + opens connection + loads extensions + applies
  schema.
- `src/codeintel/storage/backend/duckdb_session.py` adds secrets/init SQL/fsspec registration and has its own
  read-only defaults.
- `src/codeintel/storage/gateway/factory.py` adds metadata bootstrap + contract/schema validation + views.
- `src/codeintel/storage/gateway/pool.py` rebuilds DuckDB connect config again for serving pools.

Result: drift risk (especially for extension policy, read-only invariants, and env-driven config).

### B) DDL builders are duplicated

`CREATE SCHEMA` and `CREATE INDEX` SQLGlot builders exist in both:
- `src/codeintel/storage/metadata/bootstrap.py`
- `src/codeintel/storage/duckdb_policy_backend.py`

### C) JSON Schema currently has multiple competing “sources”

- TypedDict-driven export schemas: `src/codeintel/storage/schema/json_schema.py`
- TableSchema-driven JSON schemas: `src/codeintel/storage/contracts/json_schema.py`
- Catalog hashing reads file artifacts under `src/codeintel/config/schemas/export/*`:
  `src/codeintel/storage/datasets/catalog.py`

### D) View orchestration is embedded in the policy backend

`DuckDBPolicyBackend.ensure_all_views(...)` performs orchestration (discovery, compile, toposort, materialize,
lineage sync) even though dedicated modules already exist in `src/codeintel/storage/views/*`.

### E) Scalar/value coercion + JSON normalization are repeated

- `src/codeintel/storage/query_results.py` provides canonical coercion helpers, but `Warehouse` still has its own
  `_coerce_int` and repositories contain additional ad-hoc coercers.
- JSON encoding/decoding rules exist in multiple places (`helpers/json.py`, policy backend JSON coercion).

### F) Error types overlap between core and storage

- `src/codeintel/storage/exceptions.py` defines `QueryError`, but `src/codeintel/storage/queries/safe.py` uses
  `codeintel.core.errors.storage.QueryError`.

## 2) Design Decisions Required Up Front (Avoid Rework)

### 2.1 JSON Schema strategy (must choose)

Pick one of the following and implement it consistently:

- **Option A (recommended): two explicit schema products**
  - **Contract schema** (TableSchema): governs DDL, drift detection, contract validation.
  - **Export schema** (RowBinding TypedDict): governs export/API payload validation.
  - Both are generated from Python sources and emitted as build artifacts (not treated as repo-tracked truth).

- **Option B: one schema product**
  - TableSchema is the only basis for JSON Schema.
  - RowBinding TypedDict exists only for runtime decoding/typing, not schema generation.

This choice determines what gets renamed vs deleted under `storage/schema/*` and `storage/contracts/*`, and how the
catalog computes schema digests.

### 2.2 Canonical error surface (must choose)

Choose whether storage errors live canonically in:
- `codeintel.core.errors.storage` (recommended: shared across serving/build/storage), or
- `codeintel.storage.exceptions` (less ideal: pushes serving to depend on storage).

If choosing core as canonical, `codeintel.storage.exceptions` must become a re-export module (or be removed).

### 2.3 “Single bootstrapping owner” (recommended decision)

Adopt `DuckDBSession` as the single owner for connection bootstrapping and env policy. Everything else delegates.

## 3) Workstreams & Phases (Sequenced to Minimize Rework)

### Phase 0 — Inventory + invariants (no behavior change)

**Goal**: lock in decisions and create an executable checklist baseline.

**Deliverables**

1) Confirm the decisions in Section 2 (JSON Schema strategy + error canonicalization).
2) Record the current env vars used for DuckDB bootstrapping and their semantics:
   - `CODEINTEL_DUCKDB_*` config vars
   - `CODEINTEL_DUCKDB_EXTENSIONS`
   - `CODEINTEL_DUCKDB_SECRETS`
   - `CODEINTEL_DUCKDB_INIT_SQL`
3) Record current call sites of:
   - `open_gateway(...)`, `open_memory_gateway(...)`
   - `DuckDBSession.open/open_reader`
   - `ReadPoolWarehouse`
   - `ensure_all_views`
   - `src/codeintel/config/schemas/export` consumers

**Acceptance gates**

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

---

### Phase 1 — Unify DuckDB connection lifecycle + env wiring

**Goal**: one canonical “bootstrap a DuckDB connection” path, shared by all modes (build/serving/tests).

**Primary changes**

1) Make `src/codeintel/storage/backend/duckdb_session.py` the single owner for:
   - env → DuckDB connect config parsing/merge (currently in `gateway/connection.py`)
   - extension loading policy (install vs load, read-only restrictions)
   - attach history policy
   - schema apply policy
   - secrets/init SQL/fsspec registration
2) Demote `src/codeintel/storage/gateway/connection.py` into a lower-level helper (or delete it and inline).
3) Update:
   - `src/codeintel/storage/gateway/factory.py` to delegate connection creation to `DuckDBSession`.
   - `src/codeintel/storage/gateway/pool.py` to reuse the session bootstrap path (no bespoke config assembly).
   - `src/codeintel/storage/gateway/ephemeral.py` to reuse session bootstrapping invariants where appropriate.

**Migration notes**

- Keep public API stable during this phase:
  - `open_gateway` stays as the public entrypoint, but delegates.
  - `DuckDBSession.open/open_reader` become the “true” underlying source.

**Acceptance gates**

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

Add/extend tests under `tests/storage/` to assert:
- read-only connections never attempt `INSTALL` extensions
- pool connections are configured identically (init SQL / secrets / extensions / schema behavior)

---

### Phase 2 — Extension requirements as a first-class policy

**Goal**: feature modules declare extension requirements; they do not implement extension handling or messaging.

**Primary changes**

1) Extend `src/codeintel/storage/gateway/extensions.py` with a small “require extension” API, e.g.:
   - `require_extension(con, name, *, allow_install: bool) -> None`
   - optional `has_extension(...) -> bool` if needed
2) Update feature modules (at minimum):
   - `src/codeintel/storage/serving/search_index.py` to call the extension requirement API instead of catching
     arbitrary `duckdb.Error` and embedding env var guidance.

**Acceptance gates**

- Unit tests for extension requirement behavior, including deterministic error message content.
- Full repo checks (quality report + pytest).

---

### Phase 3 — DDL/Schema builder consolidation

**Goal**: one canonical location for SQLGlot DDL AST builders used across storage.

**Primary changes**

1) Create a shared module (name illustrative):
   - `src/codeintel/storage/schema/sqlglot_ddl.py`
   - owning:
     - `create_schema_if_not_exists_ast(schema: str) -> exp.Create`
     - `create_index_if_not_exists_ast(...) -> exp.Create`
2) Replace duplicated DDL builders in:
   - `src/codeintel/storage/metadata/bootstrap.py`
   - `src/codeintel/storage/duckdb_policy_backend.py`
3) Keep table DDL routed through `create_table_ast` / schema round-trip.

**Acceptance gates**

- Add tests that render DDL and assert stable semantics (IF NOT EXISTS, UNIQUE, column order).
- Full repo checks (quality report + pytest).

---

### Phase 4 — Canonical scalar/value coercion + JSON normalization primitives

**Goal**: remove repeated edge-case logic and consolidate type-normalization.

**Primary changes**

1) Make `src/codeintel/storage/query_results.py` the canonical scalar coercion module:
   - Replace `Warehouse._coerce_int` (`src/codeintel/storage/warehouse.py`) with `coerce_int`/`execute_int`.
   - Migrate repository coercers (e.g., GOID coercion) to use shared helpers.
2) Centralize JSON write normalization:
   - Align policy backend JSON insert coercion with `src/codeintel/storage/helpers/json.py`.
   - Introduce a single helper for “DuckDB JSON column value normalization” so both `bulk_insert_mappings` and
     any other JSON writes follow the same rules.

**Acceptance gates**

- Add focused unit tests around edge cases currently handled in multiple places:
  - bool vs int coercion
  - Decimal coercion
  - JSON column insert of dict/list/set values
- Full repo checks.

---

### Phase 5 — Views: separate orchestration from policy backend + delete legacy view APIs

**Goal**: `DuckDBPolicyBackend` stops owning orchestration; views become a cohesive subsystem.

**Primary changes**

1) Introduce a view orchestration module (name illustrative):
   - `src/codeintel/storage/views/materialization.py`
   - owning:
     - discovery (`discover_view_builders`)
     - compilation (`ibis.compile` / `IbisGateway` compile)
     - dependency graph (`views/dependencies.py`)
     - materialization (create views in correct order)
     - derived lineage sync (when repo/commit identity is present)
2) Make `DuckDBPolicyBackend.ensure_all_views` a thin wrapper that calls the orchestrator.
3) Split `src/codeintel/storage/views/ibis_views.py` into smaller modules by domain (docs/analytics/core/etc).
4) Remove “legacy create_*” functions after migrating call sites.

**Acceptance gates**

- Add/extend tests to assert:
  - deterministic view discovery order
  - dependency ordering/toposort behavior
  - strict cycle detection behavior (when enabled)
- Full repo checks.

---

### Phase 6 — Validation package ownership and naming consolidation

**Goal**: one coherent `storage/validation/*` package with clear boundaries.

**Primary changes**

1) Move Pandera DataFrame validation out of `src/codeintel/storage/contracts/validation.py` into
   `src/codeintel/storage/validation/pandera_df.py` (or similar).
2) Ensure there is one canonical “contract integrity validation” entrypoint and that schema drift checks are not
   duplicated across:
   - `storage/schema/ddl.py`
   - `storage/validation/contract.py`
3) Update imports and delete old modules once all call sites migrate.

**Acceptance gates**

- Full repo checks.
- Add targeted tests validating “warn/strict/skip” semantics for DataFrame validation.

---

### Phase 7 — JSON Schema consolidation (execute the decision from Phase 0)

**Goal**: eliminate competing JSON schema paths and remove coupling to repo-tracked export schema files.

**Primary changes (Option A: two schema products)**

1) Rename modules to reflect intent:
   - Contract schema generation lives under `storage/contracts/` (or `storage/schema/contracts.py`).
   - Export schema generation lives under `storage/schema/export_schema.py` (or similar).
2) Replace `src/codeintel/storage/datasets/catalog.py` schema digest logic so it:
   - computes digests from generated schema content, or
   - reads from build artifact outputs (not `src/codeintel/config/schemas/export`).
3) If `src/codeintel/config/schemas/export/*.json` is still needed, make it an emitted artifact directory that is
   regenerated deterministically, not hand-edited.

**Acceptance gates**

- Add tests that:
  - generate schemas deterministically (stable JSON output)
  - verify digest computation matches emitted artifacts when applicable
- Full repo checks.

---

### Phase 8 — Error surface unification (delete overlapping names)

**Goal**: a single canonical error taxonomy for storage/query failures.

**Primary changes**

1) Choose the canonical module (recommended: `codeintel.core.errors.storage`).
2) Remove duplicate error definitions in `src/codeintel/storage/exceptions.py`:
   - convert it to a re-export module, or delete it entirely.
3) Standardize storage call sites to raise the canonical errors:
   - connection errors (`open_gateway`)
   - query failures (`queries/safe.py`, repositories)
4) Ensure serving uses a single mapping from storage errors → problem details / API responses.

**Acceptance gates**

- `rg -n "class QueryError\\b" src/codeintel/storage src/codeintel/core/errors/storage.py` shows one canonical
  definition.
- Full repo checks.

---

### Phase 9 — Metadata + repository consolidation follow-through

**Goal**: reduce one-off modules and align repository patterns.

**Primary changes**

1) Split metadata into:
   - `metadata/ddl.py` (DDL only)
   - `metadata/sync.py` (populate/refresh from contracts: datasets, schema hashes, dataflow graph)
2) Make `src/codeintel/storage/repositories/data_models.py` a normal `BaseRepository` implementation (or provide a
   `DataModelsRepository`) so it shares:
   - Ibis acquisition
   - conversion/normalization
   - snapshot scoping semantics
3) Delete redundant conversion/coercion utilities once migrated.

**Acceptance gates**

- Full repo checks.
- Add repository-focused tests for data model reads (ensuring the same row shapes as before).

## 4) Closeout Gates (No Dead/Compat Code Left)

These are required at the end of the final phase:

1) **Quality gates**

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

2) **No remaining legacy/compat scaffolding in storage**

Run and review:

```bash
rg -n "backward compatibility|compat(ibility)?|legacy create_" src/codeintel/storage
```

Any remaining hits must be:
- truly intentional and justified (e.g., dataset “deprecated” column), or
- removed as part of the closeout.

3) **No empty placeholder packages**

Delete any empty packages (e.g., a `storage/sql/` directory that contains no source) once confirmed unused.

## 5) Notes / Risk Management

- **Risk: cross-package ripple**. Moving errors, schema generation, or view orchestration may touch serving/CLI.
  Mitigation: keep “thin wrapper modules” only temporarily, with explicit deletion tasks in Phase 8/closeout.
- **Risk: circular imports**. Consolidation can surface hidden cycles.
  Mitigation: keep orchestration modules depend on protocols/facades (`MinimalGateway`, `ibis_facade`) and avoid
  importing higher-level owners (serving/build).
- **Risk: behavior drift in read-only mode**. Centralizing bootstrapping must preserve invariants.
  Mitigation: add targeted tests early (Phase 1) for read-only connection semantics and extension behavior.

