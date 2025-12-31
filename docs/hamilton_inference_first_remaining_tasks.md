# Hamilton Inference-First Remaining Tasks Checklist

This checklist captures the remaining scope (by phase) and ties each task to the
specific files that need work.

## Phase 3: Schema registry as inference-backed authority

- [ ] Wire SchemaService to resolve observed-first schemas by default (table + Arrow + JSON)
  using `ResolvedSchemaProvider` / `resolve_table_schema`.
  Files: `src/codeintel/build/schemas/service.py`, `src/codeintel/core/schemas/service.py`,
  `src/codeintel/core/schemas/resolution.py`.
- [ ] Update storage schema provider to use the canonical resolver/service instead of direct
  registry-only lookups.
  Files: `src/codeintel/storage/contracts/schema_provider.py`.
- [ ] Ensure Arrow schema resolution prefers observation IPC bytes when available.
  Files: `src/codeintel/core/schemas/resolution.py`, `src/codeintel/core/schemas/service.py`.

## Phase 5: Ingestion and alignment upgrades

- [ ] Align ingestion frames using observed schemas (not static ordering), and derive extras
  policy from Arrow metadata.
  Files: `src/codeintel/build/hamilton/native/ingestion/frame_utils.py`,
  `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`,
  `src/codeintel/core/columnar/schema_alignment.py`.

## Phase 6: Hamilton DAG rework for inferability

- [ ] Run the inferability audit and capture a report (commit or stash to build outputs).
  Files: `tools/diagnostics/schema_inference_inventory.py`,
  `src/codeintel/core/registry/dag_output_inventory.yaml`.
- [ ] Refactor any compute nodes that still return `DuckDBRelation` to return inferable
  tabular types and move IO into loaders/savers.
  Files: `src/codeintel/build/hamilton/native/analytics/`,
  `src/codeintel/build/hamilton/native/graphs/`,
  `src/codeintel/build/hamilton/native/ingestion/`,
  `src/codeintel/build/hamilton/native/export/serving_artifacts.py`.
- [ ] Remove relation-first compute paths from dataset loaders (no SQL/relations in compute).
  Files: `src/codeintel/build/hamilton/native/patterns/loaders.py`,
  `src/codeintel/build/hamilton/native/patterns/access.py`.
- [ ] Add tests that assert all compute outputs are inferable and do not depend on saver nodes.
  Files: `tests/build/schemas/test_inference_observation_guardrails.py` (extend),
  `tests/build/hamilton/` (new/updated).

## Phase 7: Serving and interoperability

- [ ] Remove registry/schema-service usage in serving inventory so DuckDB catalog + metadata
  are the only schema sources.
  Files: `src/codeintel/serving/semantic/inventory.py`.
- [ ] Ensure serving snapshot preparation imports only Parquet datasets and uses manifest/
  metadata for schema and validation (no registry fallback).
  Files: `src/codeintel/storage/serving/snapshot_service.py`,
  `src/codeintel/serving/semantic/datasets.py`.

## Phase 8: View migration to Hamilton + PyArrow

- [ ] Implement Hamilton-native view modules (Arrow/RecordBatchReader outputs) and tag them
  as views; remove SQLGlot-driven build-time view execution.
  Files: `src/codeintel/build/hamilton/native/views/` (new modules + `__init__.py`),
  `src/codeintel/build/hamilton/native/views/view_outputs.py`.
- [ ] Emit lineage from DAG edges instead of SQLGlot lineage; persist into metadata tables.
  Files: `src/codeintel/storage/metadata/sync.py`,
  `src/codeintel/build/hamilton/native/views/view_outputs.py`.
- [ ] Retire build-time SQLGlot schema inference and materialization orchestration.
  Files: `src/codeintel/storage/views/schema_inference.py`,
  `src/codeintel/storage/views/materialization.py`.

## Phase 9: Validation and contracts

- [ ] Ensure JSON schema generation is observed-first via SchemaService (after Phase 3 wiring).
  Files: `src/codeintel/core/schemas/service.py`,
  `src/codeintel/build/schemas/json_schema_registry.py`.
- [ ] Add tests that validate observed-first JSON schema parity and observed-first validation.
  Files: `tests/storage/` (new), `tests/build/exports/` (extend).

## Phase 10: Dataset tuning from observed stats

- [ ] Plumb manifest tuning metadata into serving imports (read and expose/write settings).
  Files: `src/codeintel/storage/serving/snapshot_service.py`,
  `src/codeintel/serving/semantic/datasets.py`,
  `src/codeintel/storage/datasets/contracts.py`.
- [ ] Add tests that assert tuning metadata is written and round-trips through serving.
  Files: `tests/build/hamilton/test_materializer.py` (extend),
  `tests/serving/` (new).

## Phase 11: Drift observability and reporting

- [ ] Add tests for drift reporting CLI and ensure summaries surface in outputs.
  Files: `src/codeintel/cli/handlers/meta.py`, `src/codeintel/cli/commands/meta.py`,
  `tests/cli/` (new).

## Phase 12: Static registry migration

- [ ] Use inferability audit results to remove inferable outputs from overrides registry.
  Files: `src/codeintel/core/schemas/output_registry.py`,
  `src/codeintel/core/schemas/table_registry.py`.
- [ ] Update any remaining consumers to use resolver/service instead of direct registry access.
  Files: `src/codeintel/core/schemas/resolution.py`,
  `src/codeintel/storage/contracts/schema_provider.py`.
- [ ] Add tests asserting inferable outputs resolve from observations, not static registries.
  Files: `tests/storage/`, `tests/core/schemas/` (new/updated).
