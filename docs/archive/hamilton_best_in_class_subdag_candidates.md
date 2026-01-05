# Hamilton SubDAG Candidate Inventory

## Purpose

Track repeated Hamilton patterns that should be consolidated via
`parameterized_subdag`/`parameterize_sources` and config-driven composition.

## Ingestion Pipelines

| pattern | modules | nodes | notes |
| --- | --- | --- | --- |
| Tool-run + ingest + table materializations + finalize | `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py` | `t__modules`, `t__config_ingest`, `t__coverage_ingest`, `t__tests_ingest`, `t__typing` | Normalize into a parameterized tool target subDAG. |
| Dynamic module records | `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py` | `module_record_inputs`, `module_record`, `module_records_dynamic` | Convert to a reusable subDAG for dynamic node fan-out. |

## Extraction Pipelines

| pattern | modules | nodes | notes |
| --- | --- | --- | --- |
| AST/CST/Docstring extraction + materialization + finalize | `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py` | `t__ast`, `t__cst`, `t__docstrings` | Consolidate per-extractor subDAG with shared tool/ingest steps. |

## Export Pipelines

| pattern | modules | nodes | notes |
| --- | --- | --- | --- |
| File artifact export (JSONL/Parquet) | `src/codeintel/build/hamilton/native/export/export_targets.py` | `t__export_jsonl`, `t__export_parquet` | Parameterize artifact name, content, and saver options. |
| Serving artifact bundle | `src/codeintel/build/hamilton/native/export/serving_artifacts.py` | `serving_artifacts__materializations_*` | Split base/views subDAGs and parameterize artifact definitions. |

## Planning Pipelines

| pattern | modules | nodes | notes |
| --- | --- | --- | --- |
| Plan artifact materialization + table record | `src/codeintel/build/hamilton/native/planning/plan_targets.py` | `t__ci_plan` | Convert into a parameterized plan artifact subDAG. |

## Analytics Pipelines

| pattern | modules | nodes | notes |
| --- | --- | --- | --- |
| Single table materialization record | `src/codeintel/build/hamilton/native/analytics/tables_*.py` | `record_from_duckdb_materialization` uses | Standardize through a shared subDAG wrapper. |
| Multi-table materialization record | `src/codeintel/build/hamilton/native/analytics/tables_dependencies.py` | `record_from_materializations` | Parameterize expected table keys and target names. |
