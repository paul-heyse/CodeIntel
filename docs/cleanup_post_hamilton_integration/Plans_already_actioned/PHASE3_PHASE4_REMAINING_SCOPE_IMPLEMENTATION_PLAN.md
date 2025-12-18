# Phase 3–4 Remaining Scope — Implementation Plan

**Source plan**: `docs/data_operations_refinement/holistic_data_operations_enhancement_plan.md`  
**Scope covered here**: Phase 3 + Phase 4 items listed under “remaining scope” (Serving + Storage + Build artifacts)  
**Non-goal**: This document does *not* re-plan Phase 1/2 (already implemented); it plans only Phase 3/4 closeout.

---

## Goals (What “Done” Means)

By the end of this plan:

1. **Serving emits stable query fingerprints** (`query_hash`) for semantic query + export + search requests.
2. **Serving provides lightweight telemetry** (duration, row counts, query_hash) that is consistent across HTTP and MCP.
3. **Build produces stable serving artifacts** per run:
   - `environment.json` (versions + bootstrap settings)
   - `views_sql.json` (compiled SQL per view)
   - `views_sql_diff.json` (semantic diffs between consecutive runs)
4. **Derived lineage edges are persisted deterministically** per run and are queryable for governance/debugging.
5. **Raw SQL ingress is perimeter-validated** anywhere user-adjacent SQL strings can enter the system.
6. **Compiler upgrade gates exist** (golden SQL + execution validation) to make SQLGlot/Ibis upgrades predictable.
7. **DuckDB complex types remain explicitly deferred** behind contract/validator readiness.

---

## Guiding Principles

- **One-way data plane**: Serving reads immutable snapshots; artifacts and metadata are written in build or admin paths only.
- **Deterministic artifacts**: Prefer stable ordering, stable JSON canonicalization, and explicit schemas.
- **No “clever” caching**: Start with fingerprints and observability; add caching only after correctness gates exist.
- **Perimeter validation** is *scoped*: enforce “SELECT-only” at user ingress boundaries, not globally across build/storage.

---

## Phase 3 — Caching, Telemetry, and Build Artifacts

### 3.1 Serving Query Fingerprinting (`query_hash`)

**Objective**: Compute a stable fingerprint for semantic requests suitable for observability, future caching keys, and diffing.

**Key design decision**: Maintain two related hashes:

- `query_hash` (value-aware): incorporates **validated inputs**, including scalar values and lists.
- `query_shape_hash` (shape-only, optional): excludes scalar values and focuses on query *structure* to maximize cache reuse.

If only one hash is shipped initially, ship `query_hash` first.

**Proposed canonical inputs**

- Identity:
  - `snapshot.repo`, `snapshot.commit`, `snapshot.run_id`
  - `view_id` (semantic queries/exports)
  - `endpoint_kind` (`semantic_query`, `semantic_export`, `search`)
- Request (post-validation):
  - `select` (resolved to concrete list)
  - `filters` (normalized; stable ordering; stable JSON representation)
  - `order_by` (resolved to concrete list)
  - `limit`, `offset`
- Optional: `schema_hash` (from schema manifest for the resolved view/table)
- Optional: `semantic_registry_version` (to avoid hash collisions across registry revisions)

**Implementation sketch**

1. Add a small utility module, e.g. `src/codeintel/serving/semantic/fingerprints.py`:
   - `canonicalize_request_dict(...) -> dict[str, object]`
   - `stable_json_dumps(...) -> str` (sorted keys, compact separators)
   - `sha256_16(...) -> str` (or 32 chars; pick and standardize)
2. Update serving response models (HTTP and MCP) to optionally return:
   - `query_hash: str | None`
   - `schema_hash: str | None` (where known)
3. Ensure hash computation occurs *after*:
   - view resolution
   - allowed column resolution
   - filter validation
   - IN-list staging decision (so shape hash can reflect staging mode if desired)

**Acceptance criteria**

- Hashes are stable across repeated calls for identical inputs.
- Hash changes when any validated input changes.
- Hash changes when the schema hash changes (if included).
- Golden tests cover stability and key sensitivity.

**Tests**

- Unit tests: canonicalization stability, ordering invariance (e.g., filters ordering).
- Integration tests: kernel `query()` returns `query_hash` and it matches expected pattern.

---

### 3.2 Lightweight Serving Telemetry (Kernel-level)

**Objective**: Emit consistent, low-cost telemetry for:

- `duration_ms`
- `row_count` (and `truncated` where applicable)
- `query_hash` (when available)
- `view_id` / `query` (for search)
- snapshot metadata (repo/commit/run_id)

**Implementation strategy**

1. Add a small internal telemetry “hook” interface:
   - function or protocol: `emit(event: QueryEvent) -> None`
2. Wire kernel operations to record start/end timestamps with `try/finally`.
3. For streaming exports:
   - emit a “start” event at iterator creation
   - emit a “finish” event in generator `finally`, including `row_count`
4. Ensure HTTP and MCP share the same event payload schema where possible.

**Integration points**

- `src/codeintel/serving/semantic/kernel.py` (core)
- existing telemetry path (if present): `src/codeintel/serving/http/metrics.py` and MCP metrics usage

**Acceptance criteria**

- Telemetry exists for `query`, `search`, `export_rows`, `export_to_parquet`, `export_to_arrow_ipc`.
- Export telemetry still fires on cancellation (`generator.close()` and client disconnect scenarios).

**Tests**

- Integration: export generator close triggers telemetry finalization (row_count <= emitted count).
- Unit: event payload has required fields.

---

### 3.3 Environment Stamping End-to-End (`environment.json`)

**Objective**: Make environment/tooling metadata a first-class artifact of each build run and visible in serving meta.

**Artifact schema (minimum)**

- `generated_at` (UTC ISO string)
- `tools`:
  - python version
  - duckdb version
  - ibis version
  - sqlglot version
  - pyarrow version
- `duckdb`:
  - `config` (threads/memory_limit/temp_directory if configured)
  - `extensions_loaded` and/or `extensions_requested`
- `codeintel`:
  - `git_sha` (optional)
  - `settings` relevant to bootstrap (optional)

**Implementation strategy**

1. In build, write `environment.json` into the serving artifacts directory for the run.
2. In serving, load it via the pointer snapshot context and expose it through:
   - `/meta` (HTTP)
   - `serving_meta` (MCP)

**Acceptance criteria**

- `environment.json` exists for each run and is valid JSON.
- Serving `meta()` includes `environment` when available.
- Missing artifact is handled gracefully (no crash; `environment=None`).

**Tests**

- Build-side unit test: environment writer produces required keys.
- Serving integration test: meta includes environment contents when pointer references it.

---

### 3.4 Semantic SQL Diffs as Build Artifacts (`views_sql.json`, `views_sql_diff.json`)

**Objective**: Persist compiled view SQL per run and generate semantic diffs between consecutive runs.

**Artifacts**

- `views_sql.json`: mapping of `{view_id|table_key -> compiled_sql}` plus snapshot metadata.
- `views_sql_diff.json`: computed diff summary between “previous run” and “current run”.

**Key design decision**: Define “previous run” deterministically.

Options:

1. Previous pointer for the same repo/commit (if you keep history).
2. Previous successful run_id for the same repo/commit.
3. Previous run in the same build output directory (if running locally).

Pick one and encode it in the artifact metadata.

**Implementation strategy**

1. During build export/materialization, collect compiled SQL per semantic view.
2. Write `views_sql.json` with stable ordering.
3. Compute diffs using SQLGlot parse+diff utilities (already centralized in `src/codeintel/storage/views/diff.py`).
4. Write `views_sql_diff.json` including:
   - added/removed views
   - changed SQL (and summarized change categories)

**Acceptance criteria**

- Artifacts are stable (ordering and formatting deterministic).
- Diffs explain “what changed” with human-readable summaries.

**Tests**

- Build unit test: produces `views_sql.json` with expected shape.
- Build unit test: diff output stable for a controlled change.

---

### 3.5 Derived Lineage Edges Persisted Per Run

**Objective**: Persist derived lineage edges (view → upstream tables/views) deterministically and refresh per run.

**Implementation strategy**

1. Derive dependencies from compiled SQL (preferred) or SQLGlot AST:
   - Use existing dependency extraction (`src/codeintel/storage/views/dependencies.py`).
2. Persist via a single sync function (insert/delete per repo/commit/run):
   - Prefer a dedicated table (e.g., `metadata.derived_lineage_edges`) and a single “sync” function.
3. Optional: compare derived lineage against contract lineage:
   - Emit a warning artifact or add a “drift report” section into `views_sql_diff.json`.

**Acceptance criteria**

- Derived edges are:
  - stable ordering (deterministic)
  - refreshed per run for the given repo/commit
- Serving and debugging tools can query derived edges without scanning full view definitions.

**Tests**

- Storage unit/integration test: known view SQL produces expected upstream edges.
- Integration: running the sync twice yields identical persisted edges (idempotent).

---

### 3.6 Expand DB-API Param Execution Beyond Search (Optional)

**Objective**: Extend DB-API mode beyond search *only if there are additional hot SQL-string paths* where:

- plan reuse is valuable, and
- the query is strictly SELECT-only, and
- parameterization is stable.

**Decision gate**

Before implementing, produce a short audit:

- list all serving SQL-string call sites
- classify as:
  - generated SQL (safe)
  - user-adjacent input (needs perimeter validation)
  - internal-only (no need)

**Acceptance criteria**

- No net increase in SQL-string surface area.
- Any new DB-API execution mode reuses the same perimeter validation and telemetry.

---

### 3.7 Compiler Upgrade Gates (Golden SQL + Execution Validation)

**Objective**: Treat SQLGlot/Ibis bumps like compiler upgrades with predictable diffs.

**Test categories**

1. **Golden SQL snapshots** for representative expressions:
   - semantic query builder examples (filters/order/limit/offset)
   - export query compilation
   - DDL generation for a representative schema
2. **Execution validation** on DuckDB:
   - compile then execute against a small fixture dataset
   - validate row counts and key invariants

**Golden file policy**

- Use existing golden harness patterns if available (UPDATE_GOLDEN workflow).
- Keep golden fixtures minimal, but representative.

**Acceptance criteria**

- A planned SQLGlot/Ibis bump produces explicit test diffs explaining changes.
- Execution tests catch semantic regressions (not just string diffs).

---

## Phase 4 — Governance + Deferred Experiments

### 4.1 Repo/Commit Scoping Uniform by Default (Audit + Finish)

**Objective**: Ensure snapshot scoping is applied consistently for repository-style accessors and warehouse reads.

**Implementation strategy**

1. Audit all repository query builders:
   - ensure scoping is handled in a single base layer
   - remove duplicated `repo/commit` filters at call sites where safe
2. Ensure warehouse access patterns follow the same conventions:
   - `Warehouse.read(..., snapshot=...)` is the default for snapshot-aware reads
3. Add tests for:
   - tables with and without repo/commit columns
   - multi-table joins where only some tables are scoped

**Acceptance criteria**

- No repeated ad-hoc scoping logic in leaf repository modules.
- Snapshot scoping is applied exactly where schema supports it (no false assumptions).

---

### 4.2 Extend Raw SQL Perimeter Validation Beyond Kernel

**Objective**: Any path that accepts or executes SQL strings in a user-adjacent way must validate:

- single statement only
- select-only perimeter
- no mutation/DDL

**Implementation strategy**

1. Define explicit “safe” execution entrypoints, e.g.:
   - `execute_select_sql(sql, params)` (validates then executes)
2. Replace direct `execute_sql(sql, ...)` calls in serving boundaries with the safe variant.
3. Keep build/storage internal DDL/mutation paths unrestricted (they generate SQL programmatically).

**Acceptance criteria**

- All user-adjacent SQL-string execution sites in serving use perimeter validation.
- Tests cover:
  - multiple statements rejection
  - disallowed operations rejection (INSERT/UPDATE/COPY/etc.)

---

### 4.3 DuckDB Complex Types Pilot (Deferred / Experimental)

**Status**: Deferred. Do not implement until prerequisites are met.

**Prerequisites**

- Contract language supports nested types end-to-end:
  - schema hashing
  - validators (Pandera or equivalent)
  - DDL generation
  - serialization in exports
- Ibis + SQLGlot round-trips proven for STRUCT/LIST/MAP in this repo’s dialect and versions.

**Pilot approach (when ready)**

- Add 1–2 “v2” tables side-by-side (do not migrate critical tables first).
- Provide read-only views that project nested columns into flattened compatibility views.
- Add explicit benchmarks and export validation for Arrow/Parquet outputs.

---

## Execution Order (Recommended)

1. Serving query_hash utilities + response surface (3.1)
2. Kernel telemetry hooks (3.2)
3. Environment stamping artifact schema + serving exposure (3.3)
4. Views SQL artifacts + semantic diffs (3.4)
5. Derived lineage sync + optional drift report (3.5)
6. Compiler upgrade gates (3.7)
7. Governance audits: scoping + perimeter hardening (4.1, 4.2)
8. Complex types pilot remains deferred (4.3)

---

## Quality Gates (Mandatory)

Run locally before considering Phase 3/4 complete:

```bash
scripts/bootstrap.sh
uv sync
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

---

## Deliverables Checklist

- `query_hash` present in semantic query/export responses (HTTP + MCP as applicable)
- Kernel emits telemetry for query/search/export
- Build writes `environment.json`, `views_sql.json`, `views_sql_diff.json`
- Derived lineage edges persisted and queryable per run
- SQL ingress perimeter validated for all user-adjacent SQL-string execution
- Compiler upgrade gates tests added and documented
- Complex types explicitly deferred with prerequisites documented

