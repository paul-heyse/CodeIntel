# Change: DAG-first artifact boundary and schema consolidation

## Why
The current architecture fragments schema authority, dataset contracts, validation, exports, and
error handling across build, storage, serving, and core layers. This increases drift risk, blocks
extensibility, and makes it hard to reason about correctness, determinism, and provenance.

## What Changes
- Establish a DAG-first, artifact-first boundary where the Hamilton global graph is the single
  source of truth for schemas, contracts, exports, and semantic registries.
- **BREAKING**: Runtime layers (storage/serving) resolve schemas and contracts exclusively from
  build artifacts; build modules are no longer importable by runtime layers.
- **BREAKING**: Introduce a DatasetCatalog artifact as the canonical contract registry; treat
  DuckDB metadata tables as derived/cacheable views.
- **BREAKING**: Canonicalize export format naming to ndjson and remove jsonl as a canonical type.
- Centralize JSON Schema generation in core and standardize schema hashing and provenance.
- Unify validation under the core ValidationRunner with profile-driven check sets.
- Standardize error envelopes on RFC 9457 ProblemDetail with a single catalog.
- Consolidate write/materialization paths into one storage writer facade.

## Impact
- Affected specs: manage-schemas, distribute-artifacts, manage-dataset-catalog,
  orchestrate-validation, standardize-errors, manage-exports, materialize-data.
- Affected code: build schemas + contracts, storage contracts + metadata bootstrap, serving
  inventory/exports/errors, core schema/validation/errors, Hamilton materializers.
