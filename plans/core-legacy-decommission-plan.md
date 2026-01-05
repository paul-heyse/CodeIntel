# Core Legacy Decommission Plan (src/codeintel/core)

## TL;DR
Decommission and delete legacy or unused core modules (views, query AST helpers,
compatibility shims, generated TypedDict rows, and Arrow type compat helpers).
Migrate all call sites in build/serving/storage/tests to canonical modules and
row model helpers, then remove the old files and validate with quality gates.

## Goals
- Remove dead or legacy core modules and their imports.
- Replace compatibility shims with direct canonical APIs.
- Migrate generated TypedDict row usage to canonical row model helpers.
- Ensure Ruff, pyright, and pyrefly are clean after refactor.

## Non-Goals
- No functional changes beyond module consolidation and migration.
- No new features or new dataset schemas.

## In-Scope Removals
Core modules to delete after migration:
- src/codeintel/core/views/schema_inference.py
- src/codeintel/core/views/diff.py
- src/codeintel/core/views/dependencies.py
- src/codeintel/core/views/view_tags.py
- src/codeintel/core/queries/ast.py
- src/codeintel/core/datasets/contracts.py
- src/codeintel/core/datasets/maintenance.py
- src/codeintel/core/helpers/json.py
- src/codeintel/core/helpers/payload.py
- src/codeintel/core/datasets/scanning.py
- src/codeintel/core/columnar/dataset_scanner.py
- src/codeintel/core/schemas/generated_rows/__init__.py
- src/codeintel/core/schemas/generated_rows/analytics.py
- src/codeintel/core/schemas/generated_rows/core.py
- src/codeintel/core/schemas/generated_rows/graph.py
- src/codeintel/core/validation/arrow_type_compat.py

## Replacement Map (Old -> New)
- core.datasets.scanning -> core.columnar.streaming
  - DatasetScanOptions, QueryPlanSpec, build_scanner, dataset_for_manifest,
    resolve_partitioning, unify_dataset_schema
- core.columnar.dataset_scanner -> core.columnar.streaming
  - empty_reader_from_schema, sample_reader, scan_dataset_lazyframe,
    scan_dataset_reader
- core.schemas.generated_rows.* -> core.schemas.row_models + table_registry
  - Add columns_for_table_key, row_model_for_table_key, row_struct_for_table_key
    to row_models using TABLE_SCHEMAS lookup.
- core.helpers.json -> core.serialization.json
- core.helpers.payload -> core.serialization.payload
- core.validation.arrow_type_compat -> core.validation.schema_constraints
  - Inline or move helpers into schema_constraints (or a new canonical module
    that schema_constraints imports directly).

## Implementation Plan

### Phase 0: Preflight Inventory
- Confirm import sites and call graph using rg:
  - rg -n "core\\.datasets\\.scanning|datasets\\.scanning" src tests
  - rg -n "dataset_scanner" src tests
  - rg -n "generated_rows" src tests
  - rg -n "arrow_type_compat" src tests
  - rg -n "core\\.views\\.(diff|dependencies|view_tags|schema_inference)" src tests
  - rg -n "core\\.datasets\\.(contracts|maintenance)" src tests
- Capture file lists for each migration to avoid missing edge imports.

### Phase 1: Add Canonical Row Model Helpers
Add to src/codeintel/core/schemas/row_models.py:
- columns_for_table_key(table_key: str) -> tuple[str, ...] | None
- row_model_for_table_key(table_key: str) -> type[object] | None
- row_struct_for_table_key(table_key: str) -> type[msgspec.Struct] | None
- row_binding_for_table_key(table_key: str) -> GeneratedRowBinding | None
Implementation uses TABLE_SCHEMAS and existing row_*_for_table_schema helpers.

Update exports:
- src/codeintel/core/schemas/__init__.py to re-export the new helpers.

### Phase 2: Migrate Dataset Scanning Imports
Replace re-exports with direct imports from core.columnar.streaming.

Core/build usage:
- src/codeintel/build/graphs/engine/datasets.py
- src/codeintel/build/tabular/arrow_ops.py

Storage shim and usage:
- src/codeintel/storage/datasets/scanning.py (switch to streaming imports)
- src/codeintel/serving/semantic/duckdb_relation_builder.py
- src/codeintel/serving/semantic/query_ast.py
- src/codeintel/storage/datasets/manifest_index.py
- src/codeintel/storage/datasets/arrow_store.py

Tests:
- tests/_helpers/parquet_datasets.py
- tests/serving/semantic/test_routing.py

### Phase 3: Migrate Dataset Scanner Imports
Replace dataset_scanner import sites with core.columnar.streaming equivalents.

Call sites:
- src/codeintel/build/schemas/seed_harness.py (coordinate with table-first migration)
- src/codeintel/build/hamilton/native/views/view_outputs.py (coordinate with table-first migration)
- src/codeintel/serving/semantic/duckdb_relation_builder.py
- src/codeintel/storage/serving/snapshot_service.py
- src/codeintel/storage/datasets/manifest_index.py

Coordination note:
- Do not switch the two build modules above to reader-based helpers if the
  Hamilton cache materialization plan is converting them to pa.Table. Prefer
  table-first helpers (tabular_to_arrow_table, ensure_table) or defer these
  edits until the Hamilton change lands.

### Phase 4: Migrate Generated Rows Usage
Replace generated TypedDict row classes and columns_for_table_key usage.

Columns-for-table-key usage (switch to row_models helper):
- src/codeintel/build/analytics/entrypoints/core.py
- src/codeintel/build/analytics/subsystems/cache.py
- src/codeintel/build/analytics/data_models/core.py
- src/codeintel/build/analytics/compute/dependencies/compute.py
- src/codeintel/build/analytics/compute/row_builders/core.py
- src/codeintel/build/analytics/compute/data_models/usage.py
- src/codeintel/build/graphs/validation/findings.py
- src/codeintel/build/tabular/frames.py
- src/codeintel/build/analytics/graphs/config_graph_metrics.py
- src/codeintel/build/analytics/graphs/config_data_flow.py
- src/codeintel/build/hamilton/native/graphs/cpg/_legacy.py
- src/codeintel/build/hamilton/native/graphs/goids.py

Typed row class usage (replace with row_struct_for_table_key or Mapping):
- src/codeintel/build/analytics/graphs/graph_metrics.py
- src/codeintel/build/analytics/graphs/graph_metrics_ext.py
- src/codeintel/build/analytics/graphs/module_graph_metrics_ext.py
- src/codeintel/build/analytics/functions/metrics.py
- src/codeintel/build/analytics/compute/row_builders/graph_metrics.py
- src/codeintel/build/analytics/compute/row_builders/graph_metrics_ext.py
- src/codeintel/build/graphs/compute/callgraph/collection.py
- src/codeintel/build/graphs/compute/callgraph/persistence.py
- src/codeintel/ingestion/compute/docstrings_extract.py
- src/codeintel/ingestion/compute/typing_ingest.py

Tests referencing generated rows:
- tests/analytics/test_analytics_contracts.py
- tests/docs_export/test_graph_validation_export.py
- tests/graphs/test_compute_layer.py
- tests/graphs/test_callgraph_resolution.py
- tests/storage/test_schema_roundtrip.py
- tests/_helpers/analytics_domain.py
- tests/build/hamilton/native/analytics/test_v1_scaffold.py

Guidance for row replacement:
- For row construction, prefer row_struct_builder_for_table_schema via
  row_binding_for_table_key, or construct dicts and use row_tuple_for_table.
- For annotations, use Mapping[str, object] or msgspec.Struct where practical.
  Coordinate changes in build/hamilton/native/graphs/cpg/_legacy.py and
  build/hamilton/native/graphs/goids.py with the Hamilton caching migration to
  avoid double-touching those modules.

### Phase 5: Fold Arrow Type Compatibility
Inline or move core/validation/arrow_type_compat.py helpers into
src/codeintel/core/validation/schema_constraints.py to keep a single validation
entrypoint. Update the import in schema_constraints and its tests:
- tests/storage/test_arrow_type_compat.py

### Phase 6: Delete Legacy Modules
After migrations land, remove the files listed in In-Scope Removals and update
any __all__ exports or documentation references that still point to them.

## Validation
- Run the quality report:
  - uv run python -m tools.quality_report --output build/quality-results/quality_report.json
- Run targeted tests by area (segmenting by major directories):
  - uv run pytest -q tests/analytics
  - uv run pytest -q tests/graphs
  - uv run pytest -q tests/storage
  - uv run pytest -q tests/serving

## Acceptance Criteria
- No imports remain for removed modules (rg returns zero hits).
- Ruff, pyright, and pyrefly are clean for touched files.
- Targeted test subsets pass for build/graphs/storage/serving.

## Risks and Mitigations
- Risk: Column ordering changes when replacing generated_rows.
  - Mitigation: columns_for_table_key should use TABLE_SCHEMAS column order.
- Risk: Type checking friction from removing TypedDicts.
  - Mitigation: use Mapping[str, object] or msgspec.Struct in annotations.
- Risk: Arrow type compatibility removal breaks older pyarrow behavior.
  - Mitigation: keep list_view checks inline and validate with tests.
