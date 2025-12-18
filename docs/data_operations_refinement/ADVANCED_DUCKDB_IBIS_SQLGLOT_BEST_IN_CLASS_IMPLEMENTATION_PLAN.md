# Advanced DuckDB + Ibis + SQLGlot “Best‑in‑Class” Deployment — Implementation Plan

**Status**: Implementation plan (storage + serving)  
**Related references**:
- `docs/python_library_reference/duckdb_advanced.md`
- `docs/python_library_reference/DuckDB_advanced_connection_and_relational_api.md`
- `docs/python_library_reference/DuckDB_advanced_types.md`
- `docs/python_library_reference/ibis_advanced.md`
- `docs/python_library_reference/SQLGlot_advanced.md`

---

## 0) Purpose and Scope

This document turns the advanced DuckDB/Ibis/SQLGlot design recommendations into a concrete, end‑to‑end implementation plan that improves:

- **Functionality**: richer, safer query and data-movement capabilities.
- **Extensibility**: centralized “power knobs” for extensions, secrets, filesystems, attach patterns.
- **Hardening**: stronger SQL perimeter validation; fewer SQL-string “escape hatches”; better lifecycle cleanup.
- **Maintainability**: AST‑first compilation pipeline; fewer string↔parse round-trips; consistent parameterization.

### Primary scope (must cover)

- `src/codeintel/storage/**`
- `src/codeintel/serving/**`

### Secondary scope (allowed only as needed)

- Build-side interfaces that touch storage/serving contracts or serving artifacts.

### Non-goals (explicitly out of scope)

- Introducing brand new product features unrelated to data operations hardening.
- Full adoption of DuckDB complex/nested types across contracts (kept as an explicit deferred pilot).

---

## Status (Up-To-Date)

**Last updated**: 2025-12-18  
**Status scope**: Storage-only tranche completed; serving/build follow-ons deferred until FastMCP and build work lands.

### Phase Status Summary

- **Phase 1 (DuckDB session policy)**: **Partially complete (storage-only)** — core hooks implemented; remaining work is mostly serving/build integration and policy tightening.
- **Phase 2 (AST-first compilation)**: **Partially complete (storage-only)** — key storage round-trips removed; remaining work includes additional call sites and governance transforms.
- **Phase 3 (Dual-mode templating + named DB-API params)**: **Not started** (primarily serving-facing).
- **Phase 4 (SQLGlot governance toolkit)**: **Partially complete (storage-only)** — scope-aware deps + structural diff primitive added.
- **Phase 5 (Staging primitives + relation adoption)**: **Partially complete (storage-only)** — lifecycle-safe staging helper added and used in storage fast lane.
- **Phase 6 (Perimeter v2)**: **Partially complete (storage-only)** — additive policy validator exists; endpoint wiring deferred.
- **Phase 7 (UDFs + complex types)**: **Not started / deferred**.

### Completed Work (Storage-Only)

- **DuckDB connect-time configuration knobs** (env → DuckDB connect config):
  - `CODEINTEL_DUCKDB_AUTOINSTALL_KNOWN_EXTENSIONS`
  - `CODEINTEL_DUCKDB_AUTOLOAD_KNOWN_EXTENSIONS`
  - `CODEINTEL_DUCKDB_ENABLE_EXTERNAL_FILE_CACHE`
  - `CODEINTEL_DUCKDB_PARQUET_METADATA_CACHE`
  - Implementation: `src/codeintel/storage/gateway/connection.py:67`
- **Read-only “serving profile” defaults** (disable extension autoinstall/autoload in reader connections):
  - Implementation: `src/codeintel/storage/backend/duckdb_session.py:41` and `src/codeintel/storage/backend/duckdb_session.py:60`
- **Secrets bootstrap hook (opt-in)** via `CODEINTEL_DUCKDB_SECRETS` (JSON array):
  - Implementation: `src/codeintel/storage/backend/duckdb_session.py:219`
- **fsspec filesystem registration hook (opt-in)** via `CODEINTEL_DUCKDB_FSSPEC_FILESYSTEMS`:
  - Implementation: `src/codeintel/storage/backend/duckdb_session.py:334`
- **AST-first storage write path** (avoid `ibis.to_sql(...)` → parse round-trips):
  - Ibis expression → SQLGlot AST via `IbisGateway.to_sqlglot(...)`
  - Policy backend accepts `select_sql` as `str | sqlglot.Expression`
  - Implementations: `src/codeintel/storage/ibis_adapter.py:453`, `src/codeintel/storage/duckdb_policy_backend.py:628`
- **AST-first storage delete predicate derivation** (avoid SQL string parse for WHERE extraction):
  - Implementation: `src/codeintel/storage/ibis_adapter.py:640` and `src/codeintel/storage/ibis_adapter.py:681`
- **Scope-aware view dependency extraction** (CTE-safe / avoids false deps):
  - Implementation: `src/codeintel/storage/views/dependencies.py:29`
- **Structural diff primitive (additive)** via SQLGlot diff (kept alongside legacy diff summary):
  - Implementation: `src/codeintel/storage/views/diff.py:98`
- **Lifecycle-safe staging helper (storage-only)**:
  - `registered_temp_relation(...)` context manager
  - Storage DF fast lane uses it
  - Implementations: `src/codeintel/storage/staging.py:19`, `src/codeintel/storage/ibis_adapter.py:544`
- **Perimeter v2 foundation (additive)**:
  - `SqlIngressPolicy` + `assert_select_perimeter(...)`
  - Implementation: `src/codeintel/storage/queries/safe.py:119`

### Validation Notes

- Storage quality gates were run successfully for the storage tree:
  - `uv run ruff check src/codeintel/storage --fix`
  - `uv run pyright --warnings --pythonversion=3.13 src/codeintel/storage`
  - `uv run pyrefly check src/codeintel/storage`
  - `uv run pytest -q tests/storage`
- Full `uv run pytest -q` currently fails due to serving/FastMCP changes in progress (out of scope for this tranche).

## 1) Current Baseline (What Already Exists)

This plan assumes the repo already includes the following foundational capabilities (or very close equivalents):

- **DuckDB session/connection seam**:
  - `src/codeintel/storage/gateway/connection.py`
  - `src/codeintel/storage/backend/duckdb_session.py`
- **Ibis → SQLGlot AST hook**: `IbisGateway.to_sqlglot(...)` in `src/codeintel/storage/ibis_adapter.py`
- **Contract-driven DDL via schema round-trip**: `src/codeintel/storage/schema_roundtrip.py`
- **Serving export streaming** using Arrow record batches: `src/codeintel/serving/semantic/kernel.py`
- **SQL perimeter validation** (single statement, SELECT-only): `src/codeintel/storage/queries/safe.py`
- **SQL diffs + dependency extraction utilities**:
  - `src/codeintel/storage/views/diff.py`
  - `src/codeintel/storage/views/dependencies.py`
- **Dual-mode query templating scaffold**:
  - Ibis-first: `QueryTemplate` / `BoundQuery`
  - DB‑API: `DbApiTemplate` / `DbApiQuery`
  - `src/codeintel/serving/semantic/templates.py`

If any of the above differs in your current branch, adjust the plan sequencing to first re-establish these seams.

---

## 2) Guiding Principles (Design Contracts)

1. **AST-first pipeline**:
   - Ibis expression → SQLGlot AST → (optional transforms/analysis) → SQL string at the last moment.
   - Avoid “compile to SQL string then parse back to AST” except at explicit perimeters.

2. **Session owns power knobs**:
   - Connection configuration, init SQL, extension policy, secrets, filesystem registration, attach/export/import helpers.
   - Feature modules must not silently `INSTALL` extensions or mutate session-global settings.

3. **Two execution modes; both safe**:
   - **Ibis mode** for most query building (typed, composable).
   - **DB‑API mode** for a small set of vetted hot paths that truly benefit from native placeholders/prepared caching.

4. **Reproducibility > convenience**:
   - Serving/read-only must not auto-install extensions or “fetch stuff from the network”.
   - Artifacts are deterministic (stable ordering, canonical SQL, stable hashing).

5. **Lifecycle hygiene is mandatory**:
   - Any temporary registration (memtables, Arrow/DF staging) must have deterministic cleanup.
   - No “best effort” cleanup that leaks temp objects on cancellation/errors.

---

## 3) Quality Gates (Acceptance Gates for Every Phase)

Run these at the end of each phase (and after any wide refactor):

```bash
scripts/bootstrap_codex.sh
uv sync

uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

Additionally:

- Add/extend targeted tests for each new contract (per-phase below).
- Do not leave compatibility shims by the end of the final phase; decommission legacy paths fully.

---

## 4) Phase Plan Overview (Sequencing and Dependencies)

**Recommended order** (minimize rework):

1. Phase 1 — DuckDBSession “single source of truth” (extensions/secrets/fs/config)
2. Phase 2 — AST‑first compilation refactors (remove string↔parse ping‑pong)
3. Phase 3 — Parameterization + dual-mode templating (named params + coherent public API)
4. Phase 4 — SQLGlot governance toolkit (metadata, scope-aware deps, true semantic diff, optimizer canonicalization)
5. Phase 5 — Staging primitives + Relation API adoption (lifecycle-safe temp tables, faster write paths)
6. Phase 6 — Stronger SQL perimeters (metadata allowlists, function restrictions, cross-db checks)
7. Phase 7 — UDF strategy + deferred complex types pilot (optional, explicitly gated)

---

## 5) Phase 1 — DuckDBSession as the Single Policy Layer

### Objective

Make `DuckDBSession` the canonical place to configure and constrain DuckDB behaviors:

- connect-time config knobs
- init SQL
- extension policy (INSTALL vs LOAD)
- secrets creation
- filesystem registration (fsspec) when needed
- attach/export/import helpers

### 1.1 Unify and formalize “DuckDB runtime config”

**Work items**

- Introduce a typed config model owned by storage (not serving) that describes:
  - connect config (threads/memory_limit/temp_directory/…)
  - init SQL statements (structured; deterministic ordering)
  - extension policy (requested extensions; install allowed or not)
  - secrets policy (what secrets exist; temp vs persistent)
  - filesystem registrations (protocol → factory)
  - external file caching knobs

**Acceptance criteria**

- Serving can open a read-only connection that applies the *same* config shape every time.
- Build/write paths can enable additional capabilities without requiring per-feature ad-hoc wiring.

**Tests**

- Unit tests for env/config parsing and ordering determinism.

### 1.2 Extension policy: eliminate hidden auto-install in serving

DuckDB can auto-install/load known extensions. The plan is to explicitly control this:

- Writer/build/admin paths may `INSTALL` (policy controlled).
- Serving/read-only paths must be `LOAD-only` and fail fast if extension is missing.

**Work items**

- Add session-level switches that enforce:
  - `autoinstall_known_extensions=false` for serving/read-only
  - `autoload_known_extensions=false` for serving/read-only (optional; decide based on behavior desired)
- Replace ad-hoc feature-local extension loads with session-managed “ensure extension loaded” calls.

**Acceptance criteria**

- Serving never attempts an `INSTALL` (network/mutation).
- If an extension is required but missing, errors are deterministic and actionable.

**Tests**

- Regression test: serving path uses read-only session and cannot install extensions.

### 1.3 Secrets: add first-class support at the session seam

DuckDB supports a secrets manager (`CREATE SECRET`). Even if you only use local files today, wiring this now prevents future invasive refactors.

**Work items**

- Define a minimal secrets interface:
  - ephemeral (session-only) secrets for serving
  - optional persistent secrets (likely only for admin/build; policy-gated)
- Implement “load secrets at session start”:
  - from env/config to SQL statements
  - strict validation (no accidental logging)

**Acceptance criteria**

- No secrets appear in logs or error messages by default.
- Secrets are created before any query touches remote resources.

**Tests**

- Unit tests: secret SQL is generated without leaking secret values via repr/logging.

**Status**

- Implemented (storage-only): `CODEINTEL_DUCKDB_SECRETS` bootstrap hook in `src/codeintel/storage/backend/duckdb_session.py:219`.
- Deferred: integration/usage patterns in serving/build, plus end-to-end tests.

### 1.4 Filesystem registration (fsspec) as an explicit capability

DuckDB’s Python client can register fsspec filesystems. Add this in a policy-controlled way.

**Work items**

- Add optional filesystem registration hooks in session open.
- Keep it explicitly off unless configured (avoid unexpected third-party imports).

**Acceptance criteria**

- When enabled, DuckDB can read from non-httpfs protocols in a consistent way.

**Status**

- Implemented (storage-only): `CODEINTEL_DUCKDB_FSSPEC_FILESYSTEMS` registration hook in `src/codeintel/storage/backend/duckdb_session.py:334`.
- Deferred: production policy (which protocols are allowed) and end-to-end integration tests.

---

## 6) Phase 2 — AST‑First Compilation Pipeline (Delete Parse Round-Trips)

### Objective

Remove “SQL string → parse_one → AST” patterns when the AST already exists upstream.

### 2.1 Define the canonical compilation interface

**Work items**

- Introduce a single “compile shape” utility that returns:
  - SQLGlot AST (always)
  - SQL string (rendered only on demand)
  - optional metadata (tables/columns) when requested

**Acceptance criteria**

- All internal transforms/analysis run on AST, not strings.

### 2.2 Remove string↔parse round-trips from mutation helpers

Priority targets (common sources of drift/fragility):

- Deriving DELETE WHERE clauses by compiling Ibis to SQL then parsing back.
- Any “generate SQL then re-parse” patterns used for policy operations.

**Work items**

- Replace those with AST extraction from `to_sqlglot(...)`.
- Ensure dialect is explicit (`duckdb`) when rendering.

**Acceptance criteria**

- Mutation helpers no longer depend on `parse_one` of generated SQL for correctness.
- There is exactly one parsing perimeter for untrusted SQL strings (Phase 6 expands this).

**Tests**

- Unit tests around delete/where derivation for representative predicates.

**Status**

- Implemented (storage-only):
  - `IbisGateway.delete()` no longer compiles to SQL + parses back; it extracts WHERE from the compiled SQLGlot AST (`src/codeintel/storage/ibis_adapter.py:640`).
  - Storage writes from Ibis expressions now use SQLGlot AST rather than SQL strings (`src/codeintel/storage/ibis_adapter.py:453`).
  - Policy backend `insert_select` / `upsert_select` accept `str | sqlglot.Expression` and only parse SQL when given strings (`src/codeintel/storage/duckdb_policy_backend.py:628`).
- Remaining:
  - Inventory any remaining “SQL string → parse_one → AST” patterns outside `IbisGateway` and eliminate them where the AST is already available.

---

## 7) Phase 3 — Parameterization + Dual-Mode Templating (Cohesive Public API)

### Objective

Standardize parameterization across:

- Ibis-mode execution (`ibis.param` bindings)
- DB‑API-mode execution (DuckDB placeholders, prefer `$name` + dict)

### 3.1 Make DB‑API templates support named parameters

**Work items**

- Extend the DB‑API template layer to support:
  - `$name` placeholders
  - `params: Mapping[str, object]` binding
  - strict validation of required parameter names (no silent missing keys)

**Acceptance criteria**

- No DB‑API hot path relies on positional ordering for optional filters.

**Tests**

- Unit tests: missing params raise clear errors; extra params are either rejected or ignored deterministically.

### 3.2 Expand DB‑API mode coherently (only where it wins)

DB‑API mode should not become the default; it should be the “fast lane” for a small set of cases:

- search
- a small number of high-volume select endpoints (if any)

**Work items**

- Identify which endpoints remain DB‑API for performance/plan caching reasons.
- Ensure those endpoints:
  - use named params
  - pass through the SQL perimeter validator
  - report query hashes/telemetry uniformly

**Acceptance criteria**

- DB‑API mode is a coherent public API, not an ad-hoc one-off.

### 3.3 Prepared-statement style caching (optional, but designed in now)

DuckDB may cache prepared statements under the hood for repeated parameterized SQL.

**Work items**

- Ensure DB‑API SQL text is stable (named params support this).
- Optionally add an explicit prepare cache if measurement shows it matters:
  - keep it entirely behind session/policy layer
  - keyed by stable template id

**Acceptance criteria**

- No correctness regression; any caching is opt-in and observable.

**Tests**

- Behavioral tests (not microbenchmarks): repeated executions yield identical results.

---

## 8) Phase 4 — SQLGlot Governance Toolkit (Metadata, Diffs, Deps, Canonicalization)

### Objective

Treat SQLGlot as the repo’s “query governance toolkit”:

- structural diffing
- scope-aware dependency extraction
- metadata extraction (tables/columns/functions)
- canonicalization via optimizer for stable hashing and cache keys

### 4.1 Upgrade semantic SQL diffs to structural diffs

Current diffing should evolve from “canonical string + referenced table set” to “AST-level diff actions”.

**Work items**

- Implement a structural diff summary based on `sqlglot.diff(...)`.
- Add a stable, JSON-friendly categorization layer:
  - projection changes
  - filter changes
  - join shape changes
  - grouping/order/limit changes

**Acceptance criteria**

- Diffs are robust to formatting/alias changes and highlight meaningful semantic shifts.

**Tests**

- Unit tests with pairs of queries that differ only by formatting vs by semantics.

**Status**

- Implemented (storage-only): additive structural diff helper `diff_sql_structural(...)` in `src/codeintel/storage/views/diff.py:98`.
- Remaining:
  - Add categorization (projection/filter/join/etc.) on top of raw SQLGlot diff actions.
  - Decide how/where to persist structural diff artifacts (build-serving artifact pipeline).

### 4.2 Make dependency extraction scope-aware (CTE-safe)

Naively collecting `exp.Table` can misclassify CTEs or shadowed names as real tables.

**Work items**

- Replace view dependency extraction with a scope-aware approach (CTE-aware).
- Provide deterministic fallback behavior when scope resolution fails.

**Acceptance criteria**

- View materialization ordering is correct even with CTE-heavy view SQL.

**Tests**

- Add tests with a view SQL containing a CTE named like a schema table.

**Status**

- Implemented (storage-only): dependency extraction now uses SQLGlot scope traversal (`src/codeintel/storage/views/dependencies.py:29`).
- Remaining:
  - Add deterministic fallback behavior when scope resolution fails (and tests for that case).

### 4.3 Canonicalization policy: optimizer-backed AST hashes

**Work items**

- Define an internal canonicalization function that:
  - parses SQL to AST (duckdb dialect)
  - optionally runs SQLGlot optimizer with schema when available
  - renders canonical SQL (or produces a stable AST hash)
- Use this canonicalization for:
  - internal caching keys
  - diff baselines
  - upgrade gates

**Acceptance criteria**

- Hashes remain stable across harmless formatting differences.
- Hash changes when semantics change.

---

## 9) Phase 5 — Staging Primitives + Relation API Adoption (Lifecycle-Safe Data Movement)

### Objective

Replace fragile temporary-object patterns (especially memtable naming assumptions) with lifecycle-safe staging primitives, and use DuckDB Relation API where it eliminates SQL glue.

### 5.1 Introduce a shared staging abstraction

**Work items**

- Create a context-managed `StagedTable` (or similar) that:
  - registers Arrow/DF data under a stable unique name
  - returns the name for use in Ibis/SQLGlot/DuckDB relation building
  - guarantees cleanup via `unregister` or DROP

**Acceptance criteria**

- No staging path depends on Ibis memtable naming behavior.
- Cleanup occurs on normal completion and on cancellation/errors.

**Tests**

- Integration test: staged object does not remain registered after request cancellation.

**Status**

- Implemented (storage-only):
  - `registered_temp_relation(...)` context manager (`src/codeintel/storage/staging.py:19`).
  - Storage DataFrame fast lane uses lifecycle-safe staging (`src/codeintel/storage/ibis_adapter.py:544`).
- Remaining:
  - Apply the same staging primitive to serving-side IN-list staging and streaming cancellation paths (serving deferred).
  - Add cancellation/leak tests once FastMCP/serving changes land.

### 5.2 Rework IN-list strategies into a reusable primitive

Target a 3-tier strategy:

1. small list → literal `.isin(...)`
2. medium list → DuckDB `= ANY($param)` when viable (DB‑API mode)
3. huge list → staged table + semi-join

**Work items**

- Implement a `ListParamStrategy` helper used by serving query builder.
- Ensure staged-table cleanup is deterministic in long-lived serving processes.

**Acceptance criteria**

- Large filter lists do not create pathological SQL or leak temp objects.

### 5.3 Expand “write-path fast lanes” using relations

You already have a DataFrame fast lane that registers a temp and does `INSERT ... SELECT`.

**Work items**

- Standardize on relation-based ingestion where it improves performance:
  - DF/Arrow registration → relation → insert/select/upsert
- Avoid Python tuple normalization loops for large writes whenever possible.

**Acceptance criteria**

- Large writes are Arrow/Relation-driven, not Python-loop-driven.

**Tests**

- Equivalence tests: DF writes and tuple writes produce identical table contents for sample fixtures.

---

## 10) Phase 6 — Stronger Raw SQL Perimeter Validation (Beyond SELECT‑Only)

### Objective

Keep the “SELECT-only” rule, but add higher-order policy enforcement using SQLGlot metadata extraction:

- allowlisted tables
- allowlisted columns
- forbid cross-db references unless explicitly allowed
- optional function allowlist/denylist for volatile or unsafe functions

### 6.1 Metadata extraction helpers

**Work items**

- Add utility that extracts from AST:
  - referenced tables
  - referenced columns
  - function calls
  - presence of CTEs / subqueries

**Acceptance criteria**

- The perimeter validator can produce actionable error messages (“table X is not allowed”).

### 6.2 Enforce per-endpoint ingress policies

**Work items**

- Define ingress policies per endpoint (e.g., search vs semantic export vs admin endpoints).
- Apply them consistently anywhere SQL strings can enter the system.

**Acceptance criteria**

- No endpoint accepts SQL strings without passing through the perimeter layer.

**Tests**

- Negative tests: multi-statement, DDL, injected comments, cross-db access.
- Positive tests: legitimate SELECT queries still pass.

**Status**

- Implemented (storage-only): `SqlIngressPolicy` + `assert_select_perimeter(...)` as an additive perimeter validator (`src/codeintel/storage/queries/safe.py:119`).
- Remaining:
  - Column-level allowlisting and richer metadata extraction (beyond tables/functions).
  - Enforce ingress policies at actual SQL-string entry points (serving/build deferred).

---

## 11) Phase 7 — UDF Strategy + Deferred Complex Types (Explicitly Gated)

### Objective

Clarify and standardize:

- when to use Ibis builtin wrappers (preferred)
- when to use DuckDB Python UDFs (rare; session-managed)
- how to pilot complex/nested types safely without destabilizing contracts

### 7.1 UDF governance

**Work items**

- Define a registry and policy:
  - UDF definitions live in one module
  - session registers/unregisters deterministically
  - serving read-only paths default to “no custom UDFs” unless explicitly enabled

**Acceptance criteria**

- UDFs do not silently appear in ad-hoc modules.

### 7.2 Complex types pilot (explicitly deferred / experimental)

**Work items**

- Extend the schema/contract type system only when ready:
  - contracts → Ibis schema → SQLGlot DDL mapping
  - validators for nested data
- Pilot with 1–2 tables only.

**Acceptance criteria**

- Pilot is behind a feature gate; rollback is trivial.

---

## 12) Decommissioning Checklist (Legacy/Compatibility Purge)

By the end of the final implementation:

- Remove any “string replace placeholder” hacks introduced to bridge named params.
- Remove any memtable-name-dependent cleanup paths.
- Remove any feature-local extension `INSTALL`/`LOAD` calls that bypass session policy.
- Remove any duplicated SQL perimeter checks and route all ingress through the canonical validator.
- Remove dead/legacy helper functions that are made obsolete by:
  - AST-first compilation utilities
  - staging primitives
  - unified templates

Each PR (or phase) should include a “what got deleted” section to ensure the codebase converges toward the go-forward architecture.

---

## 13) Suggested Milestones (Practical Roadmap)

These are “integration-friendly” checkpoints that keep the repo shippable:

1. **Session policy**: extensions + init SQL + no auto-install in serving
2. **AST-first refactor**: delete parse round-trips; stabilize compilation utilities
3. **Named DB‑API params**: search and any other DB‑API hot paths migrated
4. **Governance**: scope-aware deps + structural diffs + optimizer canonicalization for hashing
5. **Staging primitive**: IN-list staging + cleanup reliability
6. **Perimeter v2**: metadata allowlists (tables/columns/functions) where SQL strings are accepted
7. **Optional**: prepared caching validation; UDF registry; complex types pilot (gated)
