# Meta Schema Implementation Plan

## Overview

This plan delivers a unified, DAG-driven schema system where `meta.duckdb` is the
sole source of truth for schema recipes, versions, and provenance. It is designed
for fast end-to-end deployment with explicit decommissioning of legacy schema
registries and strict boundary-only enforcement.

## Goals

- Centralize all schema state in `meta.duckdb` (authoritative registry).
- Make schema recipes first-class DAG outputs (Hamilton tags and manifests).
- Render all external schema formats from one canonical IR (`TableSchema`).
- Enforce contracts only at I/O boundaries (ingress/egress).
- Delete all legacy, duplicate, or transitional schema code.
- Enable schema versioning, lineage, and semantic metadata.

## Non-Goals

- Preserve compatibility with legacy schema registries.
- Maintain dual-write or dual-read transition paths.
- Keep obsolete registry code for fallback.

## Guiding Principles

- One canonical schema IR: `TableSchema` + provenance in meta.
- DAG outputs drive schema recipes; metadata is persisted per run.
- Meta registry drives inference and planning; inference is fallback.
- Schema renderers are pure functions from `TableSchema`.
- Every legacy registry is removed once replaced.

## Target Architecture (End State)

- `meta.duckdb` attached to every write-capable connection.
- `metadata.schema_versions` stores content-addressed schema versions.
- `metadata.table_schema_registry` points table_key -> schema_digest.
- `metadata.schema_manifest_runs` links runs to manifest catalogs.
- `metadata.canonical_catalogs` stores dataset contracts + schema manifests.
- Hamilton targets emit schema manifests with provenance and derivation.
- Schema enforcement at boundaries (savers/loaders, not internal nodes).

## Part D Alignment (Make_Hamilton_graph_authoritative_partD.md)

- Relation-first: metadata access uses DuckDB relations/SQL with parameter binding; no new Ibis
  dependencies in the metadata path.
- Catalog qualification: metadata tables are referenced via a shared meta catalog helper
  (e.g., `meta.metadata.*`) rather than `USE`-based context switching.
- Tag discovery: manifest compilation relies on Hamilton driver tag_filter; no bespoke TagIndex.
- Boundary discipline: metadata sync avoids dataframe materialization; I/O boundaries remain
  the only materialization points.

## Phased Plan (Critical Path Sequencing)

### Phase 0: Preflight Alignment (Fast, Low Risk)

Objective: lock invariants and define the canonical schema/metadata contract.

Deliverables:
- Finalized schema IR contract (`TableSchema`) and renderers to support.
- Finalized meta catalog schema (table list + columns + indexes).
- Decision record: boundary-only contracts, no transition support.

Acceptance:
- Spec checklist signed off (IR fields, version hash semantics, lineage fields).

Phase 0 Outputs (Completed):
- Canonical schema IR: `TableSchema` with `Column`, `Index`, `TableWritePolicy` as the only source of truth.
- Renderer targets in scope: DuckDB DDL, JSON Schema 2020-12, Pandera, pyarrow Schema, and Ibis schema.
- Meta catalog tables finalized: `metadata.schema_versions`, `metadata.table_schema_registry`,
  `metadata.schema_manifest_runs`, plus `metadata.canonical_catalogs` entries for
  `dataset_contracts` and `schema_manifest_v2`.
- Hashing semantics locked: `schema_digest` = fingerprint of `TableSchema` JSON; `schema_hash` =
  legacy shape hash for compatibility.
- Lineage fields locked: `table_key`, `derivation_kind`, `derivation_source`,
  `inference_status`, `inference_error`, `catalog_hash`, `updated_at`.
- Contract boundary rule locked: enforce contracts only at ingress/egress, never on internal nodes.
- Deployment invariant locked: `meta.duckdb` attached for all write-capable gateways; read-only
  attachment in serving environments.
- Transition policy locked: no dual-write, no fallback; legacy registries are deleted once replaced.

---

### Phase 1: Meta Storage Foundation (Core Plumbing)

Objective: establish the canonical registry in `meta.duckdb`.

Tasks:
- Add `metadata.schema_versions`, `metadata.table_schema_registry`,
  `metadata.schema_manifest_runs` tables.
- Implement schema catalog accessors:
  - `SchemaCatalogTracking.upsert_schema_manifest(...)`
  - `SchemaCatalogTracking.load_table_schema(...)`
  - `SchemaCatalogTracking.prefill_schema_index(...)`
- Ensure all gateways attach `meta.duckdb` with consistent catalog name.
- Introduce shared metadata table qualification helpers and route metadata DDL/queries through them.
- Use parameterized SQL and batch upserts for registry writes (no Ibis in metadata paths).

Deliverables:
- New metadata tables and DDL applied.
- Accessor module wired into `StorageGateway.schemas`.
- CLI command `codeintel meta.sync` to compile + persist manifest.
- Metadata reads/writes are catalog-qualified and do not depend on Ibis adapters.

Acceptance:
- `meta.sync` writes schema manifest and registry rows deterministically.
- Reads from registry return canonical `TableSchema`.
- Metadata registry operations succeed via meta catalog without Ibis dependencies.

---

### Phase 2: DAG-Driven Schema Recipes (Authoritative Outputs)

Objective: schema recipes come from DAG outputs, not code registries.

Tasks:
- Add schema recipe emitter nodes:
  - Hamilton tags on view/table builders (schema output kind + table_key).
  - Manifest compilation from DAG outputs.
- Replace bespoke tag indexes with Hamilton driver tag_filter queries for schema discovery.
- Persist schema manifests on every build:
  - Hook `serving_artifacts` and/or build executor to call
    `SchemaCatalogTracking.upsert_schema_manifest(...)`.
- Capture derivation metadata:
  - derivation_kind, inference_status, derivation_source.

Deliverables:
- Schema manifests emitted from builds and persisted to meta.
- Provenance included for each table_key.
- Manifest compilation uses `Driver.list_available_variables(tag_filter=...)` as the only
  tag discovery mechanism.

Acceptance:
- Manifest catalog hash is stable across identical DAG inputs.
- `table_schema_registry` reflects latest DAG-derived schemas.
- No runtime path depends on a bespoke TagIndex for schema discovery.

---

### Phase 3: Renderer and Validation Layer (Semantics + Contracts)

Objective: render all schema formats and validate at boundaries.

Tasks:
- Add renderers:
  - `TableSchema -> pyarrow.Schema` with field metadata.
  - `TableSchema -> JSON Schema 2020-12`.
  - `TableSchema -> Pandera` (if used for validation).
- Enrich semantic metadata using Arrow field metadata:
  - PII class, key role, provenance, version, lineage edges.
- Enforce contracts only in I/O boundaries:
  - savers/loaders validate schema (no internal node contracts).
- Add validation summary views and persist results in meta.

Deliverables:
- Renderer functions with deterministic outputs.
- Boundary-only validation pipeline with metadata artifacts.

Acceptance:
- Validation results recorded per run with clear pass/fail semantics.
- No internal nodes require schema declarations.

---

### Phase 4: Inference and Planning Refactor (Remove Early Registry Dependence)

Objective: decouple compilation from static schema registries.

Tasks:
- Replace target compilation logic to use `TableOutputDescriptor` (table_key only).
- Prefill schema inference cache from meta prior to inference.
- Limit overrides to declared sources + non-inferable outputs only.

Deliverables:
- Target compilation no longer imports table registry.
- Schema inference uses meta cache as first choice.

Acceptance:
- Removing a registry schema for inferable outputs does not break compile.
- Inference is skipped when meta provides schema.

---

### Phase 5: Decommission Legacy Registries (Delete, Not Migrate)

Objective: remove duplicate, legacy, or transitional schema code.

Decommission Steps:
- Remove inferable schemas from:
  - `core.schemas.table_registry.TABLE_SCHEMAS`
  - `core.schemas.output_registry.OUTPUT_TABLE_SCHEMAS`
- Delete bootstrap paths that import build-time contracts:
  - Remove storage bootstrapping of contract catalogs.
- Remove unused helpers that enforce schema at compile time.
- Remove legacy schema views and catalogs not backed by meta.

Deliverables:
- All inferable schemas removed from python registries.
- Legacy bootstrap code deleted.
- Dead helper functions removed.

Acceptance:
- No code path references legacy registries.
- All schema resolution occurs via meta catalog or declared sources.

---

### Phase 6: Deployment, Observability, and Hardening

Objective: reliable production rollout with monitoring and guardrails.

Tasks:
- Add operational dashboards:
  - schema drift rate, validation failures, latest good version.
- Add registry health checks:
  - missing manifest, stale registry detection.
- Configure read-only `meta.duckdb` for serving deployments.
- Add regression tests for schema round-trip + registry correctness.

Deliverables:
- Production health checks and alerts.
- CI tests for manifest round-trip and registry alignment.

Acceptance:
- Stable deployment with live registry updates and zero legacy code usage.

## Decommissioning Checklist (Explicit Deletes)

The following legacy components are removed after Phase 4/5:

- `core.schemas.table_registry.TABLE_SCHEMAS` entries for inferable outputs.
- `core.schemas.output_registry.OUTPUT_TABLE_SCHEMAS` (or replaced by a
  minimal non-inferable overrides module).
- `build/hamilton/target_spec_compiler.py` lookup of table schemas.
- Storage bootstrap imports from build schema services.
- Any duplicate schema derivation utilities that are superseded by meta catalog.

Deletion Gates:
- All replacements in place and used in production.
- `meta.duckdb` contains authoritative schema manifest and registry.
- No tests or runtime paths import deleted modules.

## Sequencing Summary (Fastest End-to-End Deployment)

1) Phase 1 (meta tables + accessors)  
2) Phase 2 (DAG emits schema manifests + persistence)  
3) Phase 4 (remove compile-time schema dependency)  
4) Phase 5 (delete legacy registries and bootstrap code)  
5) Phase 3 (renderer/validation enrichment)  
6) Phase 6 (observability + hardening)

Rationale: phases 1/2/4/5 complete the core authoritative loop quickly and
eliminate legacy drag. Phase 3 can proceed in parallel once the registry is
stable, and Phase 6 finalizes operational readiness.

## Acceptance Gates (End-to-End)

- `meta.duckdb` fully populated and attached in all write connections.
- Schema manifests stored and referenced by catalog hash.
- All schema resolution uses meta registry or declared source overrides.
- Legacy schema registries deleted.
- Boundary validations pass and report to meta.
- Metadata table access is catalog-qualified and independent of Ibis adapters.

## Test Strategy (Must-Have)

- Manifest round-trip: compile -> persist -> load -> schema equality.
- Schema index prefill: inferable tables do not invoke inference when meta exists.
- Registry alignment: every table_key has a current schema digest in meta.
- Contract enforcement: only I/O boundaries enforce schemas.
- Meta catalog queries work via catalog-qualified SQL without Ibis usage.

## Deployment Notes

- Use DuckDB prepared statements and batch upserts for registry writes.
- Attach `meta.duckdb` read-only for serving, read-write for build/ingest.
- Track schema registry version and publish a "latest good" pointer.
- Keep metadata DDL and queries catalog-qualified to avoid `USE` reliance.
