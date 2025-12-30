# Hamilton Validation Shift Plan (Inference-First + Iceberg/Arrow)

## Intent

Move data-quality validation out of Hamilton DAG nodes and into the
materialization layer, using schema observation and Iceberg/Arrow metadata as
the source of truth. Enforce static, contract-like checks only for target
outputs; all non-output or intermediate schemas remain inference-driven.

## Context

Current DQ validators in `src/codeintel/build/hamilton/data_quality.py` are
attached to DAG nodes via `check_output_custom` in
`src/codeintel/build/hamilton/native/patterns/savers.py`. This creates extra
Hamilton nodes, causes saver tag collisions, and encodes static contract rules
even for outputs that are intended to be inferred at runtime.

We now have Iceberg materialization (`IcebergDatasetSaver`) and Arrow-based
schema observation (`SchemaObservationAccumulator`) that can provide stronger,
streaming validation without DAG-level nodes.

## Scope

- Shift validation to the materialization layer (Iceberg/Arrow/DuckDB savers).
- Use schema observations and Iceberg metadata for row counts, nullability, and
  drift signals.
- Enforce static checks only for target outputs (contract outputs).
- Keep intermediate or input schemas inference-driven; only require internal
  consistency checks when needed (e.g., Arrow structural validation).

## Non-goals

- No schema pinning or schema version gating for inferred outputs.
- No mandatory full-data scans for large tables (PK uniqueness is opt-in).
- No UI overhaul; validation results persist for later reporting.

## Definitions

- Target output: a materialized table declared as a contract output.
  Operationally: `output_role="contract"` and table schema is registered in
  `OUTPUT_TABLE_SCHEMAS` or the schema provider.
- Internal output: materialized tables or intermediates marked
  `output_role="internal"` (or not registered as target outputs).
- Inferred schema: schema derived from runtime data and schema observation.

## Current Behavior (Why We Change)

- Validators enforce required columns, optional non-nullability, minimum row
  count, and primary key uniqueness.
- These validators run as DAG node transforms, creating additional nodes that
  can inherit saver tags and appear as duplicate outputs.
- Static contract checks are applied whenever a schema exists, which is not
  aligned with inference-first expectations.

## Target Design (Best-in-class)

### 1) Validation at write time (no DAG nodes)

- Use the materializers to enforce structural correctness:
  - Align to contract schema using Arrow cast/unify when output is a target.
  - Call `Table.validate(full=True)` for Arrow structural integrity.
  - Avoid eager materialization; use streaming readers.

### 2) Validation from schema observations and Iceberg metadata

- Leverage `SchemaObservationAccumulator` for:
  - Row counts, null counts, basic drift summaries.
  - Derived settings (dictionary encoding, row group sizing).
- For Iceberg writes, use metadata and inspection tables:
  - `table.metadata.snapshots[-1].summary` for row counts.
  - `table.inspect.entries()` for per-file metrics and null counts.
  - Avoid full scans for large tables.

### 3) Primary key uniqueness as optional downstream validation

- Iceberg metadata does not guarantee uniqueness.
- For small tables or opt-in targets:
  - Use `DataScan.to_arrow_batch_reader()` + `pyarrow.compute` group-by and
    count-distinct checks.
- For large tables:
  - Record as skipped with a reason; allow async or external validation.

### 4) Validation scope policy (static vs inferred)

- Contract outputs:
  - Enforce column presence, nullability (strict only when configured),
    minimum row count, and optional PK uniqueness.
- Internal/inferred outputs:
  - Only structural Arrow checks and schema observation.
  - No column presence or PK enforcement unless explicitly requested.

## Implementation Plan (Phased)

### Phase 0: Policy and configuration

Goals:
- Define and codify what is a target output.

Work:
- Add a helper in `src/codeintel/build/hamilton/materializers/` (new module
  `validation_policy.py`) that resolves validation scope based on:
  - `output_role`
  - `table_key` presence in schema provider or `OUTPUT_TABLE_SCHEMAS`
- Update target specs to explicitly set `output_role` where missing.

Acceptance:
- Policy can classify outputs as contract vs internal deterministically.

### Phase 1: Materializer validation core

Goals:
- Implement materializer-level validation using Arrow and observation stats.

Work:
- Add a `materialization_validation.py` module with:
  - `validate_arrow_integrity(reader)` (always on)
  - `validate_contract_schema(table_schema, arrow_schema)` (contract only)
  - `validate_min_rows(row_count, min_rows)` (contract only)
  - `validate_nullability(observation, contract_schema)` (contract only)
  - `validate_pk_uniqueness(reader, primary_keys, max_rows)` (opt-in)
- Integrate into:
  - `IcebergDatasetSaver` (post-write, using observation + Iceberg metadata)
  - `ArrowDatasetSaver` (post-write, using observation)
  - `DuckDBRelationSaver` (post-write, using fetched batch reader)

Acceptance:
- No DAG-level validation nodes required.
- Validation results emitted consistently by each saver.

### Phase 2: Remove DAG-level data-quality nodes

Goals:
- Eliminate `check_output_custom` validators from the DAG.

Work:
- In `src/codeintel/build/hamilton/native/patterns/savers.py`, remove
  `_validation_from_config` from `save_dataset` and `save_relation_table`.
- Keep tagging logic intact.
- Update any documentation or tests that assumed DAG validation nodes exist.

Acceptance:
- DAG no longer contains validation nodes; output duplication is impossible.

### Phase 3: Persist validation results

Goals:
- Store validation outcomes for observability and audit.

Work:
- Add a storage table (e.g., `metadata.validation_results`) or extend schema
  observation records to include validation summary fields.
- Store:
  - table_key, repo, commit, target_name, snapshot_id
  - scope (contract/internal)
  - checks run, pass/fail/warn, diagnostics
- Add an accessor in `src/codeintel/storage/` for reading results.
- Attach validation status to `TableMaterializationMetadata` so builds can
  surface results without extra queries.

Acceptance:
- Validation results are queryable post-run.

### Phase 4: Test and rollout updates

Goals:
- Ensure tests reflect the new validation pipeline.

Work:
- Add tests for:
  - Contract outputs validating columns and nullability.
  - Internal outputs only performing Arrow integrity checks.
  - PK uniqueness validation threshold behavior.
- Update any Hamilton DAG tests that assert validator nodes.

Acceptance:
- Tests confirm contract vs inferred behavior and metadata persistence.

## Validation Policy Matrix (Summary)

| Output Type | Column presence | Non-nullability | Min rows | PK uniqueness | Arrow integrity |
|------------|------------------|-----------------|----------|---------------|-----------------|
| Contract   | Yes              | Configurable     | Yes      | Opt-in        | Yes             |
| Internal   | No               | No               | No       | No            | Yes             |

## Risks and Mitigations

- Risk: Missing target output classification.
  Mitigation: Enforce `output_role` in target specs and registry lookup.
- Risk: Iceberg metadata missing/null for new tables.
  Mitigation: Fall back to schema observation row counts and mark metadata
  checks as skipped with reason.
- Risk: PK uniqueness checks too expensive.
  Mitigation: Gate by row count threshold and default to async or skipped.

## Open Questions

- Do we want a hard cap on PK uniqueness checks by table size?
- Should validation results be appended to schema observations or stored in a
  dedicated table for clearer audit trails?
- Which tables are considered contract outputs beyond `OUTPUT_TABLE_SCHEMAS`?

