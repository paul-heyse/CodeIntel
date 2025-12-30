# Hamilton Best-in-Class Output Inventory

## Purpose

Capture the source-of-truth schema and tag expectations for Hamilton outputs so
schema authority, tagging, and validation sweeps are auditable.

## Schema Sources

- `output_registry`: explicit `TableSchema` overrides in `codeintel.core.schemas.output_registry`.
- `schema_index`: DAG/inferred schemas via `codeintel.build.schemas.schema_index`.
- `declared_source`: declared schemas for source tables.
- `derived_views`: view schemas inferred in `codeintel.storage.views`.

## Explicit Output Registry Tables

| table_key | output_kind | schema_source | schema_ref | tag_status |
| --- | --- | --- | --- | --- |
| `analytics.behavioral_coverage` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.cfg_block_metrics` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.cfg_function_metrics` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.cfg_function_metrics_ext` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.config_data_flow` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.config_graph_metrics_keys` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.config_graph_metrics_modules` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.config_projection_key_edges` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.config_projection_module_edges` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.config_values` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.coverage_functions` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.coverage_lines` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.data_model_fields` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.data_model_relationships` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.data_model_usage` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.data_models` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.dfg_block_metrics` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.dfg_function_metrics` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.dfg_function_metrics_ext` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.entrypoint_tests` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.entrypoints` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.external_dependencies` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.external_dependency_calls` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.file_profile` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.function_ast_features` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.function_contracts` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.function_effects` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.function_history` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.function_metrics` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.function_profile` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.function_types` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.function_validation` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.goid_risk_factors` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.graph_metrics_functions` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.graph_metrics_functions_ext` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.graph_metrics_modules` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.graph_metrics_modules_ext` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.graph_stats` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.graph_validation` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.hello_example` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.history_timeseries` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.hotspots` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.module_profile` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.semantic_roles_functions` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.semantic_roles_modules` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.static_diagnostics` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.subsystem_agreement` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.subsystem_coverage_cache` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.subsystem_graph_metrics` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.subsystem_modules` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.subsystem_profile_cache` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.subsystems` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.symbol_graph_metrics_functions` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.symbol_graph_metrics_modules` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.test_catalog` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.test_coverage_edges` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.test_graph_metrics_functions` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.test_graph_metrics_tests` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.test_profile` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `analytics.typedness` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `ci.plan_entries` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.ast_metrics` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.ast_nodes` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.cst_nodes` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.docstrings` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.file_state` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.goid_crosswalk` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.goids` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.modules` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.repo_map` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.schema_inference_errors` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.scip_diagnostics` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.scip_external_symbols` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.scip_module_state` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.scip_occurrences` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.scip_symbol_information` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.scip_symbol_relationships` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `core.scip_symbols` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `graph.call_graph_edges` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `graph.call_graph_nodes` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `graph.cfg_blocks` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `graph.cfg_edges` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `graph.dfg_edges` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `graph.import_graph_edges` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `graph.import_modules` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |
| `graph.symbol_use_edges` | table | output_registry | `OUTPUT_TABLE_SCHEMAS` | pending |

## DAG/Inferred Outputs (Pending Extraction)

Use `SchemaIndex.iter_table_schemas()` and `UnifiedSchemaProvider.iter_table_schemas()`
when the runtime bundle is available to enumerate inferred DAG outputs not listed
above. Capture the `schema_source` as `schema_index` and record the tag status
once the Phase 1 sweep is complete.

## Declared Source Tables (Pending Extraction)

Use `declared_schema_provider(runtime=...)` to list declared source schemas and
record any node tags that route to source tables.

## Derived Views (Pending Extraction)

Use view inventories from `codeintel.storage.views.inventory` and
`codeintel.storage.views.schema_inference` to list view schemas and their source
modules. Record the schema source as `derived_views`.

## Artifacts Inventory (Seed)

| artifact_name | output_kind | target | source_module | tag_status |
| --- | --- | --- | --- | --- |
| `build_decision_trace` | artifact | `decision_trace` | `src/codeintel/build/hamilton/native/export/decision_trace.py` | pending |
| `export_jsonl_summary` | artifact | `export_jsonl` | `src/codeintel/build/hamilton/native/export/export_targets.py` | pending |
| `export_parquet_summary` | artifact | `export_parquet` | `src/codeintel/build/hamilton/native/export/export_targets.py` | pending |
| `semantic_registry` | artifact | `serving_artifacts` | `src/codeintel/build/hamilton/native/export/serving_artifacts.py` | pending |
| `schema_manifest` | artifact | `serving_artifacts` | `src/codeintel/build/hamilton/native/export/serving_artifacts.py` | pending |
| `buildspec` | artifact | `serving_artifacts` | `src/codeintel/build/hamilton/native/export/serving_artifacts.py` | pending |
| `environment` | artifact | `serving_artifacts` | `src/codeintel/build/hamilton/native/export/serving_artifacts.py` | pending |
| `ci.plan.json` | artifact | `ci_plan` | `src/codeintel/build/hamilton/native/planning/plan_targets.py` | pending |
| `ci.plan.explain.md` | artifact | `ci_plan` | `src/codeintel/build/hamilton/native/planning/plan_targets.py` | pending |
| `scip_index` | artifact | `scip` | `src/codeintel/build/hamilton/native/ingestion/scip.py` | pending |
| `scip_pb2` | artifact | `scip_proto` | `src/codeintel/build/hamilton/native/ingestion/scip_proto.py` | pending |
