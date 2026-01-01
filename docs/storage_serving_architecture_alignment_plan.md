# Storage and Serving Architecture Alignment Implementation Plan

## Context and decisions

This plan incorporates the decisions you provided and replaces prior
assumptions from earlier reviews:

- Polars is not a peer execution engine. It should be fed from DuckDB
  relations, used as a complementary columnar tool when needed.
- Hamilton is the canonical view source. The Hamilton DAG creates datasets
  and joins; storage only maintains relations among datasets and metadata.
- DuckDB is authoritative for serving contracts. Serving should not read
  Arrow schemas directly; Arrow is the zero-copy interchange layer.
- Avoid raw SQL templates and manual parameter interpolation wherever
  possible; prefer SQLGlot AST, DuckDB Expression API, and programmatic
  relation operations so correctness is intrinsic.

## Goals

- One canonical query IR (SQLGlot AST) for all query semantics.
- A single operator catalog and filter compiler that can emit SQLGlot AST,
  DuckDB Expression API, PyArrow dataset expressions, and Polars expressions.
- DuckDB relation plans are the single execution backbone for serving.
- Polars is used via DuckDB relations (`relation.pl(lazy=True)`) only.
- Hamilton DAGs are the only source of view definitions and dataset lineage.
- Serving contracts are resolved from DuckDB, not Arrow schema metadata.
- Raw SQL removed from query paths; SQL strings limited to DDL/bootstrapping.
- Correctness is intrinsic via contract-typed projections and AST capability
  envelopes, not post-hoc validation.
- Legacy code is decommissioned and deleted with a documented teardown plan.

## Non-goals

- Rewriting Hamilton DAG logic in storage or serving.
- Building a second execution engine that bypasses DuckDB.
- Supporting ad hoc raw SQL ingress for serving queries.
- Retaining Arrow schema resolution as a contract source in serving.

## Target architecture (ASCII sketch)

```
Hamilton DAG
  -> PyArrow outputs (datasets, stats, metadata)
  -> DuckDB zero-copy relation inputs

Serving request
  -> SQLGlot AST (canonical IR + capability envelope)
  -> Unified filter compiler
  -> DuckDB Expression API + relation plan
  -> DuckDB relation execution
  -> optional Polars LazyFrame (relation.pl(lazy=True))
  -> Arrow IPC / Parquet / NDJSON outputs

Contracts
  -> DuckDB catalog and metadata tables
  -> Arrow schema derived only for zero-copy interchange
```

## Status summary

Completed
- DuckDB contract resolver is now authoritative in serving
  (`src/codeintel/serving/semantic/duckdb_contracts.py`);
  `src/codeintel/serving/semantic/schema_contracts.py` removed.
- Serving schema inventory now loads from DuckDB first and no longer falls
  back to Arrow schema metadata
  (`src/codeintel/serving/semantic/inventory.py`,
  `src/codeintel/serving/db/manager.py`).
- Typed DuckDB scan adapter introduced and used by relation builder
  (`src/codeintel/serving/semantic/duckdb_scan_adapter.py`,
  `src/codeintel/serving/semantic/duckdb_relation_builder.py`).
- Export fingerprints now include canonical AST + SQL fingerprints and are
  propagated across HTTP + MCP export metadata.
- Routing now prefers DuckDB only
  (`src/codeintel/serving/semantic/routing.py`).
- `src/codeintel/serving/semantic/polars_query_builder.py` removed.
- Serving AST capability checks are enforced during query build
  (`src/codeintel/serving/semantic/query_ast.py`).
- DuckDB relation builder expanded to cover non-equi join predicates,
  qualified column references, and date/time/interval expressions
  (`src/codeintel/serving/semantic/duckdb_relation_builder.py`).
- Raw SQL removed from catalog persistence and safe query helpers:
  `src/codeintel/storage/metadata/catalogs.py`,
  `src/codeintel/storage/queries/safe.py`.
- Raw SQL removed from metadata sync/views/validation/bootstrap and storage
  helpers (`search_index`, `snapshot_service`, `dataflow`, `module_index`,
  `datasets/registry`, `schema/ddl`, `validation/contract`).

Remaining from original plan
- Complete Phase 0 inventory and finalize the deprecation map for raw SQL
  and view sources (initial pass captured below).
- Hamilton registry replacement for `view_ast_map.json` and build-time view
  map usage in `src/codeintel/build/hamilton/native/views/view_outputs.py`.
- Remove remaining raw SQL templates and string-based metadata queries across storage.
- Unified filter compiler + capability envelope reporting/expansion.
- Storage/warehouse contract alignment to DuckDB (remove Arrow-based contract
  resolution in `src/codeintel/storage/warehouse.py`).
- Scan tuning and Polars optimization alignment using advanced APIs.
- Observability and guardrail enhancements beyond fingerprints.
- Final legacy cleanup and doc/test references updates.

Phase status snapshot
- Phase 0: in progress (inventory pass 2)
- Phase 1: partial
- Phase 2: partial
- Phase 3: not started
- Phase 4: partial (serving done, storage pending)
- Phase 5: in progress
- Phase 6: partial
- Phase 7: partial
- Phase 8: partial

## Sequenced execution plan (remaining work)

The sequence below minimizes rework by locking contract/type sources and the
query IR early, expanding execution coverage next, and migrating raw SQL last.

1. Finish Phase 0 inventory and finalize deprecation targets.
2. Phase 4 (storage contract authority) plus shared type mapping for nested
   types (DuckDB/Arrow/Polars) so downstream compiler work is stable.
3. Phase 3 (Hamilton registry migration) to stabilize view metadata and tags.
4. Phase 1 (unified filter compiler + AST capability envelope and
   unsupported-level enforcement).
5. Phase 2 (expand SQLGlot -> relation coverage, add contract-typed projections).
6. Phase 6 (unified scan/tuning pipeline and advanced PyArrow/Polars tuning).
7. Phase 5 (raw SQL migrations to AST/relations using the new compiler).
8. Phase 7 (observability, lineage extraction, deterministic explain outputs).
9. Phase 8 (cleanup and delete deprecated artifacts).

## Phase 0: Inventory and alignment gates (Status: in progress)

Deliverables
- System-level inventory of raw SQL, view sources, schema sources, and
  execution entrypoints.
- Deprecation list for SQL templates, param interpolation, and legacy
  view registries.
- Concrete boundaries for Hamilton vs storage responsibilities.

Work items
- Inventory raw SQL usage in `src/codeintel/storage/**` and
  `src/codeintel/serving/**`, including:
  - `src/codeintel/storage/metadata/sync.py`
  - `src/codeintel/storage/metadata/views.py`
  - `src/codeintel/storage/serving/search_index.py`
  - `src/codeintel/storage/serving/snapshot_service.py`
  - `src/codeintel/storage/helpers/module_index.py`
  - `src/codeintel/storage/schema/ddl.py`
  - `src/codeintel/storage/queries/safe.py` (migrated to relations)
  - `src/codeintel/storage/metadata/catalogs.py` (migrated to SQLGlot AST)
- Inventory view sources:
  - Hamilton tags and DAG outputs
  - `src/codeintel/storage/views/view_ast_map.json`
  - `src/codeintel/build/hamilton/native/views/view_outputs.py`
  - `src/codeintel/serving/semantic/registry.py`
- Inventory schema sources:
  - `src/codeintel/serving/semantic/duckdb_contracts.py`
  - `src/codeintel/storage/schema/arrow_schema.py`
  - `src/codeintel/storage/warehouse.py`

Initial inventory findings (pass 2)
- Raw SQL usage in serving query paths: none found; serving query execution
  is AST + DuckDB Expression API driven:
  - `src/codeintel/serving/semantic/query_ast.py`
  - `src/codeintel/serving/semantic/sqlglot_query_builder.py`
  - `src/codeintel/serving/semantic/duckdb_relation_builder.py`
  - `src/codeintel/serving/search/engine.py`
- Raw SQL usage in storage query paths (deprecate/migrate to AST/relations):
  - Metadata and validation:
    `src/codeintel/storage/metadata/sync.py` (migrated),
    `src/codeintel/storage/metadata/views.py` (migrated),
    `src/codeintel/storage/metadata/validation.py` (migrated),
    `src/codeintel/storage/metadata/bootstrap.py` (migrated)
  - Tracking and registry:
    `src/codeintel/storage/tracking/build_tracking.py`,
    `src/codeintel/storage/tracking/run_tracking.py`,
    `src/codeintel/storage/tracking/asset_tracking.py`,
    `src/codeintel/storage/tracking/schema_catalog.py`,
    `src/codeintel/storage/datasets/registry.py` (migrated)
  - Serving helpers in storage:
    `src/codeintel/storage/serving/search_index.py` (migrated),
    `src/codeintel/storage/serving/snapshot_service.py` (migrated)
  - Repositories and helpers:
    `src/codeintel/storage/repositories/dataflow.py` (migrated),
    `src/codeintel/storage/helpers/module_index.py` (migrated)
  - Schema and query safety:
    `src/codeintel/storage/schema/arrow_schema.py`,
    `src/codeintel/storage/schema/ddl.py` (migrated),
    `src/codeintel/storage/validation/contract.py` (migrated)
  - Policy backend metadata probes (evaluate conversion feasibility):
    `src/codeintel/storage/duckdb_policy_backend.py`
- Raw SQL usage in DDL/bootstrapping (allowed but tracked):
  `src/codeintel/storage/backend/duckdb_session.py`,
  `src/codeintel/storage/gateway/extensions.py`,
  `src/codeintel/storage/metadata/ddl.py`,
  `src/codeintel/storage/metadata/meta_catalog.py`,
  `src/codeintel/storage/warehouse.py`
- Migrated to AST/relations since pass 1:
  `src/codeintel/storage/metadata/catalogs.py`,
  `src/codeintel/storage/queries/safe.py`,
  `src/codeintel/storage/metadata/sync.py`,
  `src/codeintel/storage/metadata/views.py`,
  `src/codeintel/storage/metadata/validation.py`,
  `src/codeintel/storage/metadata/bootstrap.py`,
  `src/codeintel/storage/serving/search_index.py`,
  `src/codeintel/storage/serving/snapshot_service.py`,
  `src/codeintel/storage/repositories/dataflow.py`,
  `src/codeintel/storage/helpers/module_index.py`,
  `src/codeintel/storage/datasets/registry.py`,
  `src/codeintel/storage/schema/ddl.py`,
  `src/codeintel/storage/validation/contract.py`
- View sources:
  - Hamilton tag discovery and compilation:
    `src/codeintel/storage/views/discovery.py`,
    `src/codeintel/serving/semantic/registry_compiler.py`
  - Static view map (deprecate):
    `src/codeintel/storage/views/view_ast_map.json` referenced by
    `src/codeintel/build/hamilton/native/views/view_outputs.py`
  - Serving registry loader:
    `src/codeintel/serving/semantic/registry.py`
- Schema sources:
  - Serving contracts (DuckDB authoritative):
    `src/codeintel/serving/semantic/duckdb_contracts.py`
  - Storage contract alignment (Arrow-based today, to migrate):
    `src/codeintel/storage/schema/arrow_schema.py`,
    `src/codeintel/storage/warehouse.py`
  - Dataset manifest schema metadata (interchange-only):
    `src/codeintel/storage/datasets/contracts.py`

Deprecation checklist (initial pass)
- [x] Replace raw SQL in `src/codeintel/storage/metadata/catalogs.py` with SQLGlot AST.
- [x] Replace raw SQL in metadata and validation modules with SQLGlot AST or
  DuckDB relation APIs:
  `src/codeintel/storage/metadata/sync.py`,
  `src/codeintel/storage/metadata/views.py`,
  `src/codeintel/storage/metadata/validation.py`,
  `src/codeintel/storage/metadata/bootstrap.py`
- [ ] Replace raw SQL in tracking modules with SQLGlot AST or DuckDB relations:
  `src/codeintel/storage/tracking/build_tracking.py`,
  `src/codeintel/storage/tracking/run_tracking.py`,
  `src/codeintel/storage/tracking/asset_tracking.py`,
  `src/codeintel/storage/tracking/schema_catalog.py`
- [x] Replace raw SQL in storage serving helpers with AST/relations:
  `src/codeintel/storage/serving/search_index.py`,
  `src/codeintel/storage/serving/snapshot_service.py`
- [x] Replace repository/helper SQL reads with relation helpers:
  `src/codeintel/storage/repositories/dataflow.py`,
  `src/codeintel/storage/helpers/module_index.py`,
  `src/codeintel/storage/datasets/registry.py`
- [ ] Replace Arrow-based contract resolution in storage with DuckDB-backed
  contracts:
  `src/codeintel/storage/schema/arrow_schema.py`,
  `src/codeintel/storage/warehouse.py`
- [ ] Remove static view map registry and migrate build outputs to Hamilton:
  `src/codeintel/storage/views/view_ast_map.json`,
  `src/codeintel/build/hamilton/native/views/view_outputs.py`
- [ ] Delete retired view materialization after callers removed:
  `src/codeintel/storage/views/materialization.py`
- [ ] Update docs/tests that reference removed legacy modules:
  `docs/storage_serving_best_in_class_plan.md`,
  `docs/storage_serving_phase1_phase2_ticket_backlog.md`,
  `docs/duckdb_arrow_polars_alignment_rollout_plan.md`
- [x] Replace raw SQL in `src/codeintel/storage/queries/safe.py` with relations.

Acceptance criteria
- A list of raw SQL call sites and their replacement strategy.
- A clear decision record for view registry ownership (Hamilton only).
- A clear decision record for schema contract source (DuckDB only).

## Phase 1: Canonical SQLGlot AST pipeline (Status: partial)

Deliverables
- Single canonical AST for all semantic queries.
- AST normalization, fingerprinting, and lineage extraction centralized
  in `src/codeintel/storage/sqlglot_tools.py`.
- Capability envelope enforcement using SQLGlot unsupported-level checks.

Completed
- Serving queries are built and canonicalized via SQLGlot
  (`src/codeintel/serving/semantic/query_ast.py`).
- Canonical AST/SQL fingerprints are propagated to export metadata.
- AST capability checks are enforced during serving AST build using
  `ensure_ast_capability`.

Remaining work
- Expand capability envelope reporting to include deterministic feature logs
  and centralize enforcement outside serving query paths.
- Implement a unified filter compiler that emits:
  - SQLGlot AST fragments (`src/codeintel/serving/semantic/sqlglot_query_builder.py`)
  - DuckDB Expression API (`src/codeintel/serving/semantic/duckdb_relation_builder.py`)
  - PyArrow dataset expressions (`src/codeintel/serving/semantic/datasets.py`)
  - Polars expressions (`src/codeintel/serving/semantic/engines/polars_engine.py`)
- Centralize operator validation to eliminate per-engine drift
  (`src/codeintel/serving/semantic/filter_ops.py`).

Acceptance criteria
- Every serving query produces a canonical AST and stable fingerprint.
- Unsupported AST nodes are rejected consistently before execution.
- Filters behave identically across DuckDB/Arrow/Polars execution paths.

## Phase 2: DuckDB relation plan as the execution backbone (Status: partial)

Deliverables
- AST -> DuckDB Expression API compiler (no SQL strings).
- Relation plan executed by DuckDB, streaming Arrow readers.
- Polars integration only via DuckDB relations.

Completed
- Typed DuckDB scan adapter introduced and used for Parquet/Arrow scans.
- Routing prefers DuckDB only (`src/codeintel/serving/semantic/routing.py`).
- Polars query builder removed; Polars remains downstream of relations.
- Non-equi join predicates now supported in relation plans.
- Date/time functions and interval expressions supported in relation plans.
- Qualified column references preserved in joins, projections, and ordering.

Remaining work
- Finish remaining AST coverage in `src/codeintel/serving/semantic/duckdb_relation_builder.py`
  (edge-case function aliases, JSON/list/struct access gaps, remaining join cases).
- Introduce contract-typed projections (explicit casts to contract types) so
  outputs cannot drift from contract expectations.
- Use DuckDB replacement scans consistently for Arrow/Parquet inputs with
  explicit aliasing and catalog qualification where needed.

Acceptance criteria
- Serving queries run entirely via DuckDB relations without raw SQL.
- Polars usage is always downstream of a DuckDB relation.
- Projection and filter semantics are identical between AST and relation plans.

## Phase 3: Hamilton canonical view registry (Status: not started)

Deliverables
- A single Hamilton-sourced view registry used by storage and serving.
- Deprecation of SQLGlot view maps and static JSON registries.

Work items
- Replace `src/codeintel/storage/views/view_ast_map.json` with Hamilton DAG
  outputs (or a compiler step over Hamilton metadata).
- Update `src/codeintel/build/hamilton/native/views/view_outputs.py` to consume
  Hamilton outputs instead of static view maps.
- Generate `semantic_registry.json` directly from Hamilton tags via
  `src/codeintel/serving/semantic/registry_compiler.py`.
- Remove any remaining references to view maps in tests and docs.

Acceptance criteria
- Serving registry is derived from Hamilton outputs only.
- No view map artifacts remain in storage or build outputs.

## Phase 4: DuckDB authoritative contracts and zero-copy alignment (Status: partial)

Deliverables
- Serving contracts resolved from DuckDB metadata, not Arrow schema metadata.
- Zero-copy alignment: Arrow inputs are registered or scanned into DuckDB
  without materialization.

Completed
- DuckDB-backed contract resolver for serving
  (`src/codeintel/serving/semantic/duckdb_contracts.py`).
- Arrow schema fallback removed from serving inventory.

Remaining work
- Update `src/codeintel/storage/warehouse.py` to resolve contract schemas from
  DuckDB metadata instead of `src/codeintel/storage/schema/arrow_schema.py`.
- Consolidate contract metadata (schema hash/version) into DuckDB catalog
  tables produced by Hamilton outputs.
- Ensure Arrow schemas used in storage are derived from DuckDB metadata
  (not the other way around).
- Standardize complex/nested type mapping across DuckDB, Arrow, and Polars
  with a single mapping table and schema normalizer.

Acceptance criteria
- Serving and storage resolve contract schemas from DuckDB only.
- Arrow schemas are used strictly for zero-copy interchange.

## Phase 5: Remove raw SQL templates and parameter interpolation (Status: in progress)

Deliverables
- Programmatic query construction across serving and storage.
- SQL templates limited to DDL/bootstrapping only.

Completed
- `src/codeintel/storage/metadata/catalogs.py` migrated to SQLGlot AST.
- `src/codeintel/storage/queries/safe.py` migrated to DuckDB relations.
- `src/codeintel/storage/metadata/sync.py` migrated to SQLGlot AST.
- `src/codeintel/storage/metadata/views.py` migrated to SQLGlot AST.
- `src/codeintel/storage/metadata/validation.py` migrated to SQLGlot AST.
- `src/codeintel/storage/metadata/bootstrap.py` migrated to SQLGlot AST.
- `src/codeintel/storage/serving/search_index.py` migrated to SQLGlot AST.
- `src/codeintel/storage/serving/snapshot_service.py` migrated to SQLGlot AST.
- `src/codeintel/storage/repositories/dataflow.py` migrated to SQLGlot AST.
- `src/codeintel/storage/helpers/module_index.py` migrated to DuckDB relations.
- `src/codeintel/storage/datasets/registry.py` migrated to SQLGlot AST.
- `src/codeintel/storage/schema/ddl.py` migrated to SQLGlot AST.
- `src/codeintel/storage/validation/contract.py` migrated to SQLGlot AST.

Work items
- Replace remaining raw SELECT/INSERT/UPDATE templates with SQLGlot AST builders or
  DuckDB Expression/Relation API:
  - `src/codeintel/storage/tracking/build_tracking.py`
  - `src/codeintel/storage/tracking/run_tracking.py`
  - `src/codeintel/storage/tracking/asset_tracking.py`
  - `src/codeintel/storage/tracking/schema_catalog.py`
- Use DuckDB relation APIs (`create_view`, `create`, `to_parquet`) for
  view/materialization paths in:
  - `src/codeintel/storage/duckdb_policy_backend.py`
  - `src/codeintel/storage/warehouse.py`
- Ensure ingress policy enforcement uses AST capability checks rather than
  string validation alone.

Acceptance criteria
- No raw SQL templates in serving query paths.
- Parameter interpolation removed outside of DDL and bootstrap paths.

## Phase 6: Scan pushdown and tuning (Status: partial)

Deliverables
- Column and predicate pushdown derived from the AST and unified filter compiler.
- Dataset tuning metadata drives scan behavior across Arrow, DuckDB, and Polars.

Completed
- Projection columns derived from AST when possible and applied to scans.
- Dataset manifest tuning influences batch sizing.

Remaining work
- Unify dataset scanning options across storage + serving
  (`src/codeintel/storage/datasets/arrow_store.py`,
  `src/codeintel/serving/semantic/datasets.py`).
- Apply advanced PyArrow dataset knobs: `Scanner.from_fragments`, fragment
  pruning, `use_threads`, row-group sizing, dictionary encoding, and
  `unify_schemas` for schema evolution.
- Feed tuning metadata into Polars `QueryOptFlags` and streaming controls
  (`collect_batches`, `sink_batches`, `collect_all`, `profile`) where appropriate.
- Standardize fragment readahead, batch sizing, and memory pool usage.

Acceptance criteria
- Scan path honors projection and filter pushdown end-to-end.
- Tuning metadata consistently influences DuckDB, Arrow, and Polars scans.

## Phase 7: Observability and guardrails (Status: partial)

Deliverables
- End-to-end query fingerprinting, provenance, and explain outputs.
- Guardrails based on AST feature envelopes, not ad hoc checks.
- Plan introspection hooks for DuckDB and Polars.

Completed
- AST/SQL fingerprints now included in export metadata.

Remaining work
- Add deterministic feature logging and broaden capability envelope enforcement
  beyond serving query builds.
- Integrate SQLGlot lineage extraction into metadata for derived columns.
- Normalize explain outputs across DuckDB relation plans.
- Add Polars profiling hooks (`profile`, `collect_schema`) and record
  plan metrics for observability.

Acceptance criteria
- Every serving response includes canonical fingerprints and a stable
  capability envelope report.
- Explain and plan outputs are deterministic and traceable to the AST.

## Phase 8: Cleanup and deprecation (Status: partial)

Deliverables
- Removal of legacy registries, obsolete engines, and unused helpers.
- Updated documentation and migration notes.

Completed
- `src/codeintel/serving/semantic/schema_contracts.py` removed.
- `src/codeintel/serving/semantic/polars_query_builder.py` removed.

Remaining work
- Remove view map artifacts and retired materialization stubs:
  - `src/codeintel/storage/views/view_ast_map.json`
  - `src/codeintel/build/hamilton/native/views/view_outputs.py` map loader
  - `src/codeintel/storage/views/materialization.py`
- Remove remaining raw SQL helper pathways after migration to AST/relations.
- Update docs and tests that reference removed legacy modules.

Acceptance criteria
- No legacy view registries or peer engines remain.
- Documentation aligns with the new architecture.

## Ticket checklists (outstanding work)

### Phase 0 tickets

#### T0.1 Finalize raw SQL inventory and classification
- [ ] Enumerate all `execute(...)` call sites in `src/codeintel/storage/**` and
  `src/codeintel/serving/**`.
- [ ] Classify each site as DDL/bootstrapping vs query path.
- [ ] Map each query-path site to its replacement strategy (AST, relation API).
- [ ] Update Phase 0 inventory findings with the final list and counts.

#### T0.2 Finalize view source inventory and decision record
- [ ] Confirm all Hamilton tag sources and registry compilation entrypoints.
- [ ] Trace all `view_ast_map.json` references and note removal dependencies.
- [ ] Record the canonical view source and deprecation timeline in Phase 0.

#### T0.3 Finalize schema source inventory and decision record
- [ ] Identify all contract schema sources (DuckDB, Arrow, manifest metadata).
- [ ] Record the authoritative source (DuckDB) and derived sources (Arrow).
- [ ] Capture required DuckDB metadata tables produced by Hamilton outputs.

#### T0.4 Freeze deprecation map and removal criteria
- [ ] Convert the deprecation checklist into explicit removal milestones.
- [ ] Define removal criteria (tests updated, no call sites) per legacy target.
- [ ] Record the sequence in the Phase 8 cleanup plan.

### Phase 1 tickets

#### T1.1 AST capability envelope enforcement
- [x] Enforce capability checks in `src/codeintel/serving/semantic/query_ast.py`.
- [ ] Define the supported AST node set and unsupported-level behavior.
- [ ] Centralize capability reporting beyond serving query builds.

#### T1.2 Unified filter compiler
- [ ] Introduce a single compiler that emits SQLGlot AST, DuckDB Expression API,
  PyArrow dataset expressions, and Polars expressions.
- [ ] Migrate filter construction sites in:
  `src/codeintel/serving/semantic/sqlglot_query_builder.py`,
  `src/codeintel/serving/semantic/duckdb_relation_builder.py`,
  `src/codeintel/serving/semantic/datasets.py`,
  `src/codeintel/serving/semantic/engines/polars_engine.py`.

#### T1.3 Centralize operator validation
- [ ] Consolidate operator allowlists in
  `src/codeintel/serving/semantic/filter_ops.py`.
- [ ] Remove per-engine operator validation branches.

### Phase 2 tickets

#### T2.1 Expand SQLGlot to relation coverage
- [x] Add non-equi join predicates (AND/OR/NOT + comparison ops) in
  `src/codeintel/serving/semantic/duckdb_relation_builder.py`.
- [x] Add date/time operations (date_add/date_diff/date_trunc/extract) and
  interval literals in `src/codeintel/serving/semantic/duckdb_relation_builder.py`.
- [x] Preserve qualified column references for joins/projections/order-by.
- [ ] Fill remaining expression coverage gaps (JSON/list/struct edge cases,
  function alias coverage, and any remaining join shapes).

#### T2.2 Contract-typed projections
- [ ] Emit contract-typed projections in
  `src/codeintel/serving/semantic/sqlglot_query_builder.py`.
- [ ] Source the type mapping from DuckDB contracts in
  `src/codeintel/serving/semantic/duckdb_contracts.py`.

#### T2.3 Replacement scans and aliasing
- [ ] Standardize Arrow/Parquet replacement scans with explicit aliasing in
  `src/codeintel/serving/semantic/duckdb_relation_builder.py`.
- [ ] Ensure catalog qualification is deterministic in relation plans.

### Phase 3 tickets

#### T3.1 Replace view map with Hamilton outputs
- [ ] Replace `src/codeintel/storage/views/view_ast_map.json` usage with
  Hamilton outputs or a compiler step over Hamilton metadata.

#### T3.2 Update build-time view output generation
- [ ] Update `src/codeintel/build/hamilton/native/views/view_outputs.py` to
  consume Hamilton outputs instead of the static view map.

#### T3.3 Generate serving registry from Hamilton tags
- [ ] Ensure `semantic_registry.json` is generated directly from Hamilton tags
  via `src/codeintel/serving/semantic/registry_compiler.py`.

#### T3.4 Remove view map references in docs/tests
- [ ] Remove view map references across tests and docs after migration.

### Phase 4 tickets

#### T4.1 Storage contract resolution from DuckDB
- [ ] Replace Arrow-based contract lookup in
  `src/codeintel/storage/warehouse.py` with DuckDB metadata lookups.

#### T4.2 Persist contract metadata in DuckDB catalog
- [ ] Persist schema hash/version metadata in DuckDB catalog tables as part of
  Hamilton outputs.

#### T4.3 Arrow schema derivation from DuckDB metadata
- [ ] Update `src/codeintel/storage/schema/arrow_schema.py` to derive Arrow
  schemas from DuckDB metadata only.

#### T4.4 Cross-engine complex type mapping
- [ ] Introduce a single complex/nested type mapping table used by DuckDB,
  Arrow, and Polars.
- [ ] Apply the mapping in `src/codeintel/storage/duckdb_types.py`,
  `src/codeintel/storage/schema/arrow_schema.py`,
  `src/codeintel/serving/semantic/duckdb_relation_builder.py`.

### Phase 5 tickets

#### T5.1 Metadata SQL migration
- [x] Replace raw SQL in `src/codeintel/storage/metadata/catalogs.py`.
- [x] Replace raw SQL in:
  `src/codeintel/storage/metadata/sync.py`,
  `src/codeintel/storage/metadata/views.py`,
  `src/codeintel/storage/metadata/validation.py`,
  `src/codeintel/storage/metadata/bootstrap.py`.

#### T5.2 Tracking and registry SQL migration
- [ ] Replace raw SQL in:
  `src/codeintel/storage/tracking/build_tracking.py`,
  `src/codeintel/storage/tracking/run_tracking.py`,
  `src/codeintel/storage/tracking/asset_tracking.py`,
  `src/codeintel/storage/tracking/schema_catalog.py`,
  `src/codeintel/storage/datasets/registry.py`.

#### T5.3 Storage serving helper SQL migration
- [x] Replace raw SQL in:
  `src/codeintel/storage/serving/search_index.py`,
  `src/codeintel/storage/serving/snapshot_service.py`.

#### T5.4 Repository/helper SQL migration
- [x] Replace raw SQL in:
  `src/codeintel/storage/repositories/dataflow.py`,
  `src/codeintel/storage/helpers/module_index.py`,
  `src/codeintel/storage/datasets/registry.py`.

#### T5.5 Schema and query safety SQL migration
- [x] Replace raw SQL in `src/codeintel/storage/queries/safe.py`.
- [x] Replace raw SQL in:
  `src/codeintel/storage/schema/ddl.py`,
  `src/codeintel/storage/validation/contract.py`.

#### T5.6 Relation-based view/materialization APIs
- [ ] Use DuckDB relation APIs (`create_view`, `create`, `to_parquet`) in:
  `src/codeintel/storage/duckdb_policy_backend.py`,
  `src/codeintel/storage/warehouse.py`.

#### T5.7 Ingress policy enforcement via AST envelope
- [ ] Replace string-based ingress validation with AST capability checks in
  `src/codeintel/storage/queries/safe.py`.

### Phase 6 tickets

#### T6.1 Unified dataset scanning options
- [ ] Consolidate scan options between
  `src/codeintel/storage/datasets/arrow_store.py` and
  `src/codeintel/serving/semantic/datasets.py`.

#### T6.2 Advanced PyArrow dataset tuning
- [ ] Implement `Scanner.from_fragments`, fragment pruning, `use_threads`,
  row-group sizing, dictionary encoding, and `unify_schemas`.

#### T6.3 Polars tuning integration
- [ ] Feed tuning metadata into Polars `QueryOptFlags`.
- [ ] Wire `collect_batches`, `sink_batches`, `collect_all`, and `profile`
  controls in `src/codeintel/serving/semantic/engines/polars_engine.py`.

#### T6.4 Standardize scan performance knobs
- [ ] Standardize fragment readahead, batch sizing, and memory pool usage
  across DuckDB, Arrow, and Polars.

### Phase 7 tickets

#### T7.1 Capability envelope reporting
- [ ] Emit deterministic feature logs for AST capability envelope checks.

#### T7.2 Lineage extraction automation
- [ ] Use SQLGlot lineage extraction to populate derived column lineage in
  `src/codeintel/storage/sqlglot_tools.py` and
  `src/codeintel/storage/schema/arrow_schema.py`.

#### T7.3 Explain output normalization
- [ ] Normalize explain outputs across DuckDB relation plans for stable tracing.

#### T7.4 Polars observability hooks
- [ ] Add Polars profiling and schema collection metrics in
  `src/codeintel/serving/semantic/engines/polars_engine.py`.

### Phase 8 tickets

#### T8.1 Remove view map artifacts
- [ ] Delete `src/codeintel/storage/views/view_ast_map.json` and the loader in
  `src/codeintel/build/hamilton/native/views/view_outputs.py` after migration.

#### T8.2 Remove retired view materialization
- [ ] Delete `src/codeintel/storage/views/materialization.py` once unused.

#### T8.3 Remove legacy raw SQL helpers
- [ ] Remove any remaining raw SQL helper pathways after Phase 5 migrations.

#### T8.4 Update docs/tests for legacy removals
- [ ] Update legacy references in:
  `docs/storage_serving_best_in_class_plan.md`,
  `docs/storage_serving_phase1_phase2_ticket_backlog.md`,
  `docs/duckdb_arrow_polars_alignment_rollout_plan.md`.

## Legacy decommissioning and deletion plan

Decommissioning workflow (apply to each legacy component)
1. Add a replacement path and route all call sites to it.
2. Backfill tests to lock new behavior and remove reliance on legacy APIs.
3. Remove legacy code, exports, and docs references in the same change set.
4. Verify quality gates and remove any temporary compatibility shims.

Legacy targets (explicit deletions)
- View map registry: remove `src/codeintel/storage/views/view_ast_map.json` and
  its loader in `src/codeintel/build/hamilton/native/views/view_outputs.py`
  after Hamilton registry output is canonical.
- Retired view materialization: delete
  `src/codeintel/storage/views/materialization.py` once no callers remain.
- Arrow-based contract enforcement in storage: migrate
  `src/codeintel/storage/warehouse.py` away from
  `src/codeintel/storage/schema/arrow_schema.py` for contract resolution,
  then remove or narrow Arrow schema helpers to interchange-only use.
- Raw SQL metadata paths: delete legacy string-based queries after SQLGlot/Relation
  replacements land in `src/codeintel/storage/metadata/*.py` and related modules.
- Docs/test references: remove references to legacy modules in:
  - `docs/storage_serving_best_in_class_plan.md`
  - `docs/storage_serving_phase1_phase2_ticket_backlog.md`
  - `docs/duckdb_arrow_polars_alignment_rollout_plan.md`

## Validation and quality gates

- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- Validate targeted tests by subsystem (serving, storage, build).
- Add focused tests for:
  - Unified filter compiler (SQLGlot/DuckDB/Arrow/Polars parity)
  - DuckDB contract resolution in storage + serving
  - Hamilton registry output replacing view map usage

## Risks and mitigations

- Risk: Removing raw SQL templates reduces flexibility for ad hoc queries.
  Mitigation: Provide a comprehensive SQLGlot AST builder API with clear
  coverage of supported features and extension points.
- Risk: DuckDB contract resolution may diverge from Hamilton schemas.
  Mitigation: Persist schema and lineage metadata in DuckDB as part of
  Hamilton outputs, and validate in CI.
- Risk: Polars feature gaps compared to DuckDB.
  Mitigation: Treat Polars as optional tooling for post-relation
  transformations only.

## Immediate next steps

- Complete Phase 0 inventory validation and confirm the deprecation targets.
- Implement the unified filter compiler skeleton and capability envelope.
- Plan the Hamilton registry migration to eliminate `view_ast_map.json`.
