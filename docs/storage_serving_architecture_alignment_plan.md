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
- DuckDB relation plans are the single execution backbone for serving.
- Polars is used via DuckDB relations (`relation.pl(lazy=True)`) only.
- Hamilton DAGs are the only source of view definitions and dataset lineage.
- Serving contracts are resolved from DuckDB, not Arrow schema metadata.
- Raw SQL removed from query paths; SQL strings limited to DDL only.
- Query correctness and safety are enforced by construction, not by
  post-hoc validation.

## Non-goals

- Rewriting Hamilton DAG logic in storage or serving.
- Building a second execution engine that bypasses DuckDB.
- Keeping legacy SQLGlot view maps or SQL template registries.
- Retaining Arrow schema resolution in serving (beyond zero-copy transport).

## Target architecture (ASCII sketch)

```
Hamilton DAG
  -> PyArrow outputs (datasets, stats, metadata)
  -> DuckDB zero-copy relation inputs

Serving request
  -> SQLGlot AST (canonical IR)
  -> DuckDB Expression API + relation plan
  -> DuckDB relation execution
  -> optional Polars LazyFrame (relation.pl(lazy=True))
  -> Arrow IPC / Parquet / NDJSON outputs

Contracts
  -> DuckDB catalog and metadata tables
  -> Arrow schema used only for zero-copy interchange
```

## Phase 0: Inventory and alignment gates

Deliverables
- System-level inventory of raw SQL, view sources, schema sources, and
  execution entrypoints.
- Deprecation list for SQL templates, param interpolation, and legacy
  view registries.
- Concrete boundaries for Hamilton vs storage responsibilities.

Work items
- Inventory raw SQL usage in `src/codeintel/storage/**` and
  `src/codeintel/serving/**`, including:
  - `src/codeintel/serving/search/engine.py`
  - `src/codeintel/storage/exports/service.py`
  - `src/codeintel/storage/helpers/sql_params.py`
- Inventory view sources:
  - Hamilton tags and DAG outputs
  - `src/codeintel/storage/views/sqlglot_views.py`
  - `src/codeintel/serving/semantic/view_registry.py`
- Inventory schema sources:
  - `src/codeintel/storage/schema/arrow_schema.py`
  - `src/codeintel/serving/semantic/schema_contracts.py`

Acceptance criteria
- A list of raw SQL call sites and their replacement strategy.
- A clear decision record for view registry ownership (Hamilton only).
- A clear decision record for schema contract source (DuckDB only).

## Phase 1: Canonical SQLGlot AST pipeline

Deliverables
- Single canonical AST for all semantic queries.
- AST normalization, fingerprinting, and lineage extraction centralized
  in `src/codeintel/storage/sqlglot_tools.py`.

Work items
- Treat `ServingQuery` AST as the only query specification for
  serving and storage query paths:
  - `src/codeintel/serving/semantic/query_ast.py`
  - `src/codeintel/serving/semantic/sqlglot_query_builder.py`
- Add a reusable AST normalization pass (qualify, optimize, canonical
  rendering) for deterministic fingerprints:
  - `src/codeintel/storage/sqlglot_tools.py`
  - `src/codeintel/serving/semantic/fingerprints.py`
- Add AST-level capability checks to drive allowed operations, not
  ad hoc validation.

Acceptance criteria
- Every serving query produces a canonical AST and stable fingerprint.
- AST normalization output is the single source of truth for later
  compilation.

## Phase 2: DuckDB relation plan as the execution backbone

Deliverables
- AST -> DuckDB Expression API compiler (no SQL strings).
- Relation plan executed by DuckDB, streaming Arrow readers.
- Polars integration only via DuckDB relations.

Work items
- Extend `src/codeintel/serving/semantic/duckdb_relation_builder.py`
  to cover all supported AST constructs and generate Expression API
  predicates and projections.
- Ensure dataset scans are routed through DuckDB relations, not directly
  through Polars or Arrow as primary engines.
- Replace `src/codeintel/serving/semantic/engines/polars_engine.py` with a
  DuckDB-backed adapter that uses:
  - `DuckDBPyRelation.pl(lazy=True)` for Polars LazyFrame
  - Polars operations only after relation creation
- Simplify engine routing so DuckDB is the primary engine and Polars is
  optional tooling rather than a peer engine:
  - `src/codeintel/serving/semantic/routing.py`
  - `src/codeintel/serving/semantic/engines/registry.py`

Acceptance criteria
- No direct dataset scans in Polars engine paths.
- Polars usage is always downstream of DuckDB relations.
- All serving queries can run with DuckDB only.

## Phase 3: Hamilton canonical view registry

Deliverables
- A single Hamilton-sourced view registry used by storage and serving.
- Deprecation of SQLGlot view map registry.

Work items
- Build view registry artifacts from Hamilton DAG discovery:
  - Use `src/codeintel/storage/views/discovery.py` as the discovery
    foundation, but drive it from Hamilton outputs.
  - Generate the serving registry JSON from Hamilton outputs rather than
    SQL plan maps:
    - `src/codeintel/serving/semantic/registry.py`
- Deprecate and remove:
  - `src/codeintel/storage/views/sqlglot_views.py`
  - `src/codeintel/storage/views/view_plan_map.json`
- Ensure serving view specs and table keys are sourced from Hamilton
  tags and DAG outputs.

Acceptance criteria
- Serving registry is derived from Hamilton outputs only.
- No SQL plan map registry remains in storage or serving.

## Phase 4: DuckDB authoritative contracts and zero-copy alignment

Deliverables
- Serving contracts resolved from DuckDB metadata, not Arrow schema
  metadata.
- Zero-copy alignment: Arrow inputs are registered or scanned into
  DuckDB without materialization.

Work items
- Replace `contract_schema_for_table_key` usage with a DuckDB-backed
  contract resolver:
  - Deprecate `src/codeintel/serving/semantic/schema_contracts.py`.
  - Add a DuckDB contract service that uses catalog metadata plus
    relation schemas for column types and ordering.
- Ensure Arrow is only used for interchange:
  - Use DuckDB replacement scans or `con.from_arrow` for Arrow tables
    and scanners.
  - Avoid direct Arrow schema use in serving paths.
- Align dataset metadata with DuckDB catalog entries (table, column,
  and lineage metadata stored in DuckDB).

Acceptance criteria
- Serving contract resolution never consults Arrow schema metadata.
- Arrow inputs are registered into DuckDB without materialization.

## Phase 5: Remove raw SQL templates and parameter interpolation

Deliverables
- Programmatic query construction across serving and storage.
- SQL templates limited to DDL only.

Work items
- Replace `src/codeintel/storage/helpers/sql_params.py` usage with
  Expression API or SQLGlot AST builders.
- Replace query templates in serving and search with SQLGlot AST
  generation + relation operations:
  - `src/codeintel/serving/search/engine.py`
  - `src/codeintel/storage/exports/service.py`
- Audit remaining SQL string usage and migrate to:
  - SQLGlot AST builder
  - DuckDB Expression API
  - Relation operations

Acceptance criteria
- No raw SQL templates in serving query paths.
- Parameter interpolation removed outside of DDL and bootstrap paths.

## Phase 6: Scan pushdown and tuning

Deliverables
- Column and predicate pushdown derived from the AST.
- Dataset tuning metadata drives scan behavior.

Work items
- Derive projection columns from AST and pass to DuckDB scans and
  Arrow scanners:
  - `src/codeintel/serving/semantic/datasets.py`
  - `src/codeintel/serving/semantic/duckdb_relation_builder.py`
- Apply dataset tuning metadata to scanner options and Polars
  optimization flags:
  - `DatasetManifestEntry.inferred_settings`
  - `DatasetManifestEntry.write_settings`
- Align fragment readahead and batch sizing across DuckDB and Arrow
  scanners.

Acceptance criteria
- Scan path honors projection and filter pushdown.
- Tuning metadata is used to set scanner options and optimization flags.

## Phase 7: Observability and guardrails

Deliverables
- End-to-end query fingerprinting, provenance, and explain outputs.
- Guardrails based on AST feature envelopes, not ad hoc checks.

Work items
- Emit canonical AST fingerprints and SQL fingerprints for all
  serving responses:
  - `src/codeintel/serving/semantic/fingerprints.py`
- Standardize explain outputs through DuckDB relation plans and
  capture normalized SQL for tracing.
- Add scan metrics and query plan metadata to response payloads
  without increasing cardinality.

Acceptance criteria
- Every serving response includes query fingerprint metadata.
- Explain and plan outputs are deterministic and derived from the
  canonical AST.

## Phase 8: Cleanup and deprecation

Deliverables
- Removal of legacy registries, obsolete engines, and unused helpers.
- Updated documentation and migration notes.

Work items
- Remove deprecated files and update call sites.
- Update architecture docs and alignment plans to reflect the
  DuckDB-first execution model.
- Add a migration guide for any downstream users of legacy SQL
  templates or view registries.

Acceptance criteria
- No legacy view registries or peer engines remain.
- Documentation aligns with the new architecture.

## Validation and quality gates

- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- Validate targeted tests by subsystem (serving, storage, build).
- Add focused tests for:
  - AST -> DuckDB Expression compilation
  - DuckDB relation -> Polars LazyFrame conversion
  - Contract resolution from DuckDB

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

- Approve this plan and identify the first subsystem to migrate
  (recommended: contract resolution + engine routing).
- Create migration tasks for each phase with owners and estimated scope.
- Publish the plan in the docs index and link it from existing
  architecture and Hamilton migration documents.
