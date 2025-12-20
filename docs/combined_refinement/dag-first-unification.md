---
name: dag-first-unification
description: DAG-first consolidation of schema, contracts, validation, errors, exports, and write paths
---

# Plan

Unify build, storage, serving, and core around a DAG-first, artifact-first architecture where the
Hamilton global graph is the single source of truth for schemas, contracts, and exports. This plan
introduces a canonical artifact boundary, consolidates validation/error/export/write paths, and
accepts breaking changes to achieve maximum determinism, extensibility, and maintainability.

## Requirements
- Hamilton DAG SchemaIndex/SchemaService is the sole schema authority; runtime layers must not
  import build modules.
- Build outputs include deterministic artifacts: SchemaManifest, DatasetCatalog, SemanticRegistry,
  and ExportCatalog (if separated), all derived from the global DAG.
- Storage and serving load schemas/contracts exclusively from artifact-backed providers.
- JSON Schema generation is centralized in core (one generator, one registry, one hash canonical).
- Export format registry is canonicalized to ndjson (no jsonl as a canonical format).
- Validation is executed by core ValidationRunner with profile-driven, DAG-derived check sets.
- Error envelope is canonicalized to RFC 9457 ProblemDetail with a single error catalog.
- All write/materialization semantics are centralized in a single storage writer (Warehouse or new
  writer facade), with consistent schema validation and hashing.
- Import-time safety is preserved (no DAG evaluation or contract resolution at import).
- Deterministic ordering and hashing for all manifests and catalogs.

## Scope
- In:
  - Artifact-first boundary and loaders (SchemaManifest, DatasetCatalog, SemanticRegistry).
  - DAG-first schema provider and manifest-backed SchemaService for runtime layers.
  - Dataset contract/catalog consolidation and derived metadata tables.
  - Validation unification using core ValidationRunner.
  - Error envelope unification to ProblemDetail.
  - Export format registry consolidation and ndjson rename.
  - Write path consolidation (DataSavers/Ibis IO/Warehouse).
  - Doc updates and new tests for determinism/import-time safety.
- Out:
  - Changes to analytics algorithms or graph semantics.
  - Performance tuning beyond removing duplication.
  - New feature surfaces unrelated to schema/contract/export/write consolidation.

## Files and entry points
- `docs/build_refinement/dag-first-schema-derivation.md`
- `src/codeintel/build/schemas/schema_index.py`
- `src/codeintel/build/schemas/provider_unified.py`
- `src/codeintel/build/schemas/service.py`
- `src/codeintel/build/schemas/json_schema_registry.py`
- `src/codeintel/build/schemas/manifest.py`
- `src/codeintel/build/hamilton/contracts/schemas/registry.py`
- `src/codeintel/core/schemas/service.py`
- `src/codeintel/core/schemas/json_schema_gen.py`
- `src/codeintel/core/validation/runner.py`
- `src/codeintel/core/errors/problem_details.py`
- `src/codeintel/storage/contracts/provider.py`
- `src/codeintel/storage/datasets/registry.py`
- `src/codeintel/storage/metadata/bootstrap.py`
- `src/codeintel/storage/metadata/sync.py`
- `src/codeintel/serving/semantic/inventory.py`
- `src/codeintel/serving/semantic/registry.py`
- `src/codeintel/serving/export/formats.py`
- `src/codeintel/serving/export/engine.py`
- `src/codeintel/serving/http/export_dispatch.py`
- `src/codeintel/serving/mcp/export_dispatch.py`
- `src/codeintel/serving/errors/models.py`
- `src/codeintel/serving/errors/mapping.py`
- `src/codeintel/build/exports/engine.py`
- `src/codeintel/build/hamilton/materializers/duckdb_saver.py`
- `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py`
- `src/codeintel/build/hamilton/io/ibis_adapter.py`
- `src/codeintel/storage/warehouse.py`
- `tests/build/hamilton/test_import_time_schema_safety.py`

## Data model / API changes
- Add an ArtifactProvider API in core (or a dedicated `codeintel.core.artifacts` module) that
  loads SchemaManifest, DatasetCatalog, and SemanticRegistry from build outputs and exposes
  a manifest-backed SchemaProvider and DatasetRegistry.
- Expand SchemaManifest (v2 additive) to include provenance for tables/views and optional
  artifact lineage; ensure a stable schema hash per table and derivation metadata.
- Introduce DatasetCatalog JSON schema (canonical contract payload), containing:
  - table_key, dataset name, schema hash/version, export specs, row binding metadata,
    dependencies, validation profile, ownership/tags, view/table flags, and deprecation info.
- Replace `metadata.datasets` as a source of truth; derive it from DatasetCatalog at bootstrap.
  Split runtime-only stats into a new `metadata.dataset_stats` table if needed.
- Canonicalize export format to ndjson across build/serving/CLI; remove jsonl from schema and
  filename defaults; retain only a temporary alias if migration is required.
- Replace serving ErrorResponse as canonical with ProblemDetail (or wrap ProblemDetail inside it)
  using a single error catalog and mapping policy.

## Action items
[ ] Define canonical artifact outputs and file layout (paths, naming, versioning) and document
    them in the DAG-first design doc.
[ ] Implement the ArtifactProvider API and manifest-backed SchemaProvider/SchemaService factory
    in core, including deterministic loading and cache invalidation.
[ ] Build-phase: emit SchemaManifest, DatasetCatalog, and SemanticRegistry from the global DAG;
    enforce determinism, stable ordering, and consistent hashing.
[ ] Replace fragmented schema providers with the manifest-backed provider in storage/serving;
    remove direct build imports from runtime layers.
[ ] Centralize JSON Schema generation via core SchemaService and remove or route around
    `storage/schema/json_schema.py` and other duplicate generators.
[ ] Rewire storage contracts/datasets to use DatasetCatalog artifact; update metadata bootstrap
    and sync to treat metadata tables as derived.
[ ] Standardize validation on core ValidationRunner; define build/storage/serving check sets
    and make validation profile selection DAG-driven.
[ ] Unify error handling by adopting ProblemDetail as the canonical envelope; update serving
    error models/mapping and build error wrappers to the shared catalog.
[ ] Consolidate export format registry and planning into one module; rename jsonl -> ndjson and
    update all export artifact specs and validation paths accordingly.
[ ] Consolidate write paths by routing Hamilton DataSavers and Ibis IO through a single writer
    (Warehouse or new facade) with shared validation/hash logic.
[ ] Remove deprecated providers/shims and update docs, tests, and sample configs to reflect
    new artifacts and ndjson naming.
[ ] Add or update tests for import-time safety, manifest determinism, catalog loading, error
    mapping, and export format consistency; run local quality gates.

## Testing and validation
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- `uv run pytest -q`
- Targeted tests:
  - Import-time safety for schema/catalog loading.
  - Deterministic manifest/catalog hashing.
  - Artifact-backed SchemaProvider resolution.
  - Error envelope mapping (ProblemDetail output for HTTP/MCP).
  - ndjson export paths and filenames.

## Risks and edge cases
- Large breaking changes may ripple into tests, tooling, and any external users.
- Artifact versioning drift can break runtime if manifests are missing or mismatched.
- Incorrect schema hash canonicalization can invalidate caches and export fingerprints.
- ndjson rename may break downstream scripts; ensure a temporary alias if needed.
- Import-time schema access may reappear if any runtime path imports build modules indirectly.
- Derived metadata tables must remain consistent with artifacts to avoid silent divergence.

## Open questions
- None (design choices are intentionally opinionated for best-in-class consolidation).
