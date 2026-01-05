# Hamilton Schema Authority + Pandera/PyArrow Propagation Plan

## Goals
- Make Hamilton TableSchema the single authoritative schema source.
- Enforce schema compliance only via PyArrow (structural) and Pandera (semantic).
- Propagate validation backward to external inputs and forward to downstream outputs.
- Keep storage/serving boundary clean: build writes parquet, storage consumes parquet.

## Principles
- Single source of truth: TableSchema in Hamilton scope is authoritative.
- No parallel contracts: remove or avoid independent contract enforcement.
- Boundary enforcement only: validate at input read, target entry/exit, and storage ingest.
- Schema metadata travels with data: Arrow schema metadata carries table_key, hash, and
  extras policy.

## Scope
- External inputs (e.g., SCIP) and all Hamilton DAG outputs, including CPG.
- All build-time ingestion, transformation, and output materialization.
- Storage ingest and serving exports as boundary enforcement only.

## Key Files (current anchors)
- `src/codeintel/core/schemas/output_registry.py` (output TableSchema definitions)
- `src/codeintel/core/schemas/table_registry.py` (declared schemas + outputs registry)
- `src/codeintel/runtime/compose.py` (schema_index build + providers)
- `src/codeintel/build/meta/contract_catalog.py` (catalog composition)
- `src/codeintel/build/hamilton/native/patterns/savers.py` (output validation hooks)
- `src/codeintel/build/hamilton/data_quality.py` (Pandera validator)
- `src/codeintel/core/validation/pandera_schema.py` (Pandera schema generation)
- `src/codeintel/core/schemas/arrow_gen.py` (Arrow schema + metadata)
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py` (Arrow write)
- `src/codeintel/build/hamilton/native/patterns/loaders.py` (Arrow read)
- `src/codeintel/storage/validation/columnar.py` (storage-side validation)

## Target Architecture (boundary flow)
1) External input -> read/normalize -> Arrow schema cast -> Pandera validate
2) Hamilton compute -> output -> Arrow schema cast -> Pandera validate -> parquet write
3) Storage ingest -> Arrow schema cast -> Pandera validate -> DuckDB
4) Serving export -> Arrow schema cast -> Pandera validate -> export artifacts

## Phase 0: Inventory and Gating
Checklist:
- Enumerate all DAG outputs and verify each has a TableSchema.
- Confirm CPG outputs are defined: `graph.cpg_nodes`, `graph.cpg_edges`.
- Verify inferred outputs persist overrides so TableSchema is stable.
- Add a build-time gate to fail if a DAG output lacks a TableSchema.

File-level tasks:
- `src/codeintel/core/schemas/output_registry.py`: ensure every graph output is present.
- `config/registry/dag_output_inventory.yaml`: confirm all outputs listed.
- `src/codeintel/runtime/compose.py`: add a "schema completeness" check after
  schema_index build.

## Phase 1: Schema Authority Alignment
Checklist:
- Keep Hamilton TableSchema as the only authoritative schema.
- Contract catalog should include only declared external inputs + DAG outputs.
- Remove/disable any schema enforcement that is not Pandera or PyArrow.

File-level tasks:
- `src/codeintel/build/meta/contract_catalog.py`: keep union of declared inputs and
  `catalog.table_outputs`; optionally include views.
- `src/codeintel/storage/validation/contract.py`: limit to metadata-only checks or
  deprecate enforcement in favor of Pandera/Arrow validation.

## Phase 2: PyArrow Structural Enforcement
Checklist:
- Always cast outputs to canonical Arrow schema before write.
- Attach schema metadata: table_key, schema hash, extras policy.
- Use `pa.unify_schemas` + `Table.cast` for merge compatibility.
- Validate Arrow tables (`Table.validate(full=True)`) after casts.
- Normalize dictionaries and chunks (`Table.unify_dictionaries()`,
  `Table.combine_chunks()`).
- Stream inputs with Dataset/Scanner for scale.

File-level tasks:
- `src/codeintel/core/schemas/arrow_gen.py`: ensure metadata includes schema hash and
  extras policy consistently.
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`:
  - cast inputs to canonical schema from TableSchema
  - enforce schema metadata
  - validate and unify dictionaries/chunks before write
- `src/codeintel/build/hamilton/native/patterns/loaders.py`:
  - require schema in scan options (Arrow Dataset/Scanner)
  - cast to canonical schema on read when needed
- `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`:
  - align column order to Arrow schema; enforce extras policy

## Phase 3: Pandera Semantic Enforcement
Checklist:
- Generate Pandera schemas directly from TableSchema for every table_key.
- Enforce strictness and coercion at boundaries only.
- Use lazy validation for large datasets; allow sampling when configured.
- Enforce primary key uniqueness and nullability via Pandera checks.

File-level tasks:
- `src/codeintel/core/validation/pandera_schema.py`:
  - map TableSchema metadata to `DataFrameSchema` options
  - enforce strict or strict="filter" depending on extras policy
- `src/codeintel/build/hamilton/data_quality.py`:
  - make Pandera validation the default boundary validator
  - include row-count validation where configured
- `src/codeintel/build/hamilton/native/patterns/savers.py`:
  - ensure `ci.validate_outputs` is enabled for all outputs
  - wire Pandera validation for both Dataset and Relation outputs

## Phase 4: Boundary Enforcement Coverage
Checklist:
- External inputs validated at ingestion read points.
- Internal DAG boundaries validated at target entry/exit.
- Storage ingest validates before DuckDB load.
- Serving exports validate before artifact write.

File-level tasks:
- `src/codeintel/build/hamilton/native/ingestion/*`: wrap external input read with
  Arrow cast + Pandera validation.
- `src/codeintel/build/hamilton/native/patterns/table_target.py`:
  - ensure each TableTargetTableSpec uses a contract or validation hook
- `src/codeintel/storage/validation/columnar.py`:
  - validate Arrow tables with Pandera before ingest
- `src/codeintel/build/hamilton/native/export/serving_artifacts.py`:
  - validate outgoing tables before artifact creation

## Phase 5: Observability and Diagnostics
Checklist:
- Emit validation diagnostics with table_key and schema hash.
- Collect error details from Pandera lazy validation reports.
- Track schema mismatches by hash and table_key.

File-level tasks:
- `src/codeintel/build/hamilton/data_quality.py`: structured diagnostics payloads.
- `src/codeintel/core/schemas/arrow_metadata.py`: use schema hash for traceability.

## Phase 6: Rollout and Backfill
Checklist:
- Backfill schema overrides for inferable outputs (persist to overrides registry).
- Run a full build with validation enabled to capture first error set.
- Iterate on failing tables by adjusting TableSchema or compute logic.

File-level tasks:
- `src/codeintel/runtime/compose.py`: prefill schema_index from stored overrides.
- `src/codeintel/build/schemas/observations.py`: persist Arrow schemas for drift audit.

## Phase 7: Acceptance Criteria
Checklist:
- All DAG outputs have TableSchema (including CPG).
- Every input/output boundary uses Arrow cast + Pandera validate.
- Storage ingest rejects schema-violating parquet before DuckDB.
- Serving exports emit only validated artifacts.

## Open Questions
- Should docs views be included in the contract catalog by default?
- What is the default strictness policy for extras (retain vs filter)?
- Do we require strict Pandera validation on every target by default, or per-profile?
