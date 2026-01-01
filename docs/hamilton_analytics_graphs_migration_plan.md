# Hamilton Analytics and Graphs Migration Plan

## Goals
- Migrate analytics and graphs compute into Hamilton-native DAG modules.
- Make all analytics and graph outputs dataset-backed (Parquet/Arrow) with
  inference-first schema metadata.
- Ensure `src/codeintel/build` emits a metadata-rich Parquet dataset that is the
  only source for storage/serving (DuckDB/SQLGlot).
- Provide table-by-table acceptance criteria and DAG node specs with tracking.
- Retire legacy orchestration in `src/codeintel/build/analytics` and `src/codeintel/build/graphs`
  after parity is verified.

## Scope
- Analytics tables referenced by `src/codeintel/build/analytics/**`.
- Graph tables referenced by `src/codeintel/build/graphs/**`.
- Core ingestion tables required as inputs (AST/CST/docstrings, GOIDs, modules).

## Constraints
- Batch-first (no streaming execution paths).
- Parquet dataset is the system of record; all Hamilton outputs write Parquet,
  and DuckDB/SQLGlot read only from that dataset.
- No SQLGlot view materialization; use Hamilton-native outputs.
- Current-state analytics only; decommission history/timeseries outputs.

## Inference-First Schema Policy (Parquet Boundary)
- All Hamilton outputs are inference-first; no explicit schema enforcement at
  DAG boundaries.
- Storage/serving consume explicit schemas derived from Parquet metadata, not
  from Python contracts.
- Output registry entries are treated as legacy compatibility only; they will be
  removed for analytics/graph tables once Parquet metadata coverage is
  validated.

### Parquet Metadata Contract (required)
All Parquet datasets produced by `src/codeintel/build` must include dataset-level
metadata so DuckDB/SQLGlot can build explicit schemas from the dataset alone:
- `codeintel.table_key`: fully qualified table key.
- `codeintel.domain`: `core`, `graph`, `analytics`, `docs`, etc.
- `codeintel.target`: Hamilton target name.
- `codeintel.schema_hash`: stable hash of column names + logical types.
- `codeintel.schema_digest`: full schema digest for contract parity.
- `codeintel.columns_json`: JSON mapping `{name: logical_type}`.
- `codeintel.nullability_json`: JSON mapping `{name: nullable}`.
- `codeintel.primary_keys_json`: JSON list of primary key columns (if known).
- `codeintel.partition_columns_json`: JSON list of partition columns.
- `codeintel.build_id`: unique build run identifier.
- `codeintel.repo`: repo name.
- `codeintel.commit`: commit SHA.
- `codeintel.snapshot_id`: snapshot identifier (if distinct from commit).
- `codeintel.generated_at`: ISO-8601 UTC timestamp.
- `codeintel.hamilton.node`: node name that materialized the dataset.
- `codeintel.hamilton.graph_version`: DAG graph version hash (or build spec id).
- `codeintel.inputs_json`: JSON list of `{table_key, schema_hash}` lineage.

## Tracking Legend
- [ ] not started
- [~] in progress
- [x] done
- [!] blocked

## Phase Plan (tracking)
### Phase 0: Inventory and contracts
- [x] Finalize the table inventory below against schema service and registry.
- [x] Confirm metadata coverage and remove explicit override reliance once
  Parquet metadata is authoritative (metadata contract implemented;
  non-inferable analytics overrides removed).
- [x] Adopt inference-only schema policy; treat output registry as compatibility.
- [x] Implement the Parquet metadata contract in build materializers.
- [x] Update DuckDB/SQLGlot ingestion to derive schemas from Parquet metadata only.

### Phase 1: Core ingestion prerequisites
- [~] Validate core inputs (AST/CST/docstrings/modules/GOIDs) are complete
  (pending an end-to-end validation run; tests not run yet).
- [~] Confirm typing, coverage, tests, config ingestion produce required tables
  (pending an end-to-end validation run; tests not run yet).

### Phase 2: Graph extraction in Hamilton
- [x] Implement call graph, import graph, CFG, DFG, symbol-use tables in DAG.
- [x] Add graph validation outputs and invariants checks.

### Phase 3: Function analytics core
- [x] Port function metrics, types, ast features, effects, contracts.
- [x] Implement risk factors and function validation outputs.

### Phase 4: Profiles and higher-level aggregates (snapshot-only)
- [x] Implement function/profile, file_profile, module_profile, hotspots.
- [x] Remove history-based dependencies from profiles/hotspots
  (`analytics.function_history` removed).
- [x] Decommission history_timeseries analytics (see decommission scope).

### Phase 5: Dependencies, config, semantic roles, subsystems
- [x] Port external dependency detection and config flow graphs.
- [x] Implement semantic role classification outputs.
- [x] Implement subsystem mappings and subsystem metrics.

### Phase 6: Graph metrics and CFG/DFG analytics
- [~] Port graph metrics (call/import/symbol) and stats tables
  (code migrated; quality gates pending).
- [x] Port CFG/DFG metrics tables.

### Phase 7: Test analytics
- [x] Implement test coverage edges, test profiles, and test graph metrics.
- [x] Implement behavioral coverage and entrypoint test linking.

### Phase 8: Decommission legacy packages + history analytics
- [x] Remove history_timeseries + function_history targets, schemas, CLI, docs.
- [x] Remove docs.v_* history views and view map entries.
- [x] Freeze imports of legacy analytics/graphs in build runtime
  (guarded by architecture test).
- [x] Remove unused legacy orchestration once parity is verified
  (coverage_functions moved to Polars; coverage compute + duckdb_helpers removed;
  unused AST features persistence wrapper removed).

## Reconciliation Notes (current code vs plan)
- Core ingestion tables marked [x] are backed by existing Hamilton ingestion targets
  (`src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`,
  `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`).
- `core.goids`/`core.goid_crosswalk` are surfaced via the native Hamilton target
  `src/codeintel/build/hamilton/native/graphs/goids.py` (domain: ingestion).
- Graph extraction compute exists for import/call/CFG/DFG tables plus symbol-use
  (`src/codeintel/build/hamilton/native/graphs/symbol_use.py`).
- Analytics tables are now backed by compute kernels across functions, profiles,
  dependencies, graph metrics, testing, semantic roles, data models, and subsystems.
- Inference-first policy is adopted; Parquet metadata is the authoritative schema
  source for storage/serving.
- History outputs (`analytics.function_history`, `analytics.history_timeseries`)
  have been decommissioned to keep analytics snapshot-only.
- Parquet metadata contract is wired through ArrowDatasetSaver and dataset writes;
  serving/duckdb ingestion reads schema from Parquet metadata.
- Test snapshots now emit Parquet metadata, and metadata round-trip coverage is
  exercised in dataset tests.
- coverage_functions now runs as a Polars pipeline (no DuckDB relation helper);
  legacy coverage compute + duckdb_helpers removed.
- Remaining scope: run end-to-end validation and resolve quality gates for graph
  metrics + config/subsystem migrations.
- Graph metrics orchestration now accepts DAG-provided graphs/rows
  (no gateway/runtime resolution); Hamilton graph_metric_inputs builds
  call/import/symbol graphs and filters.
- Config + subsystem graph analytics now accept DAG-provided rows/graphs
  (no gateway reads); Hamilton nodes build config bipartite + subsystem inputs.
- Quality gates still pending for graph metrics migration (ruff/pyright/pyrefly
  clean + targeted tests).

## DAG Node Conventions
- Dataset table outputs: `<table>__base -> <table>__table -> t__<target>`.
- Multi-table targets: use `make_table_materializations_collector` and a single
  `t__<target>` to record outputs.
- Preferred materializer: dataset-backed `save_dataset` (Parquet) unless a
  table must remain relational for compatibility. Use `TableContractSpec`
  for all analytics/graphs outputs.

## Inventory Review: Schema Registry Alignment
### Contract sources (confirmed)
- Explicit output schemas live in `src/codeintel/core/schemas/output_registry.py`
  and are surfaced via `codeintel.build.schemas.get_schema_provider()`.
- For analytics/graph tables, these are transitional compatibility references;
  the authoritative schema now comes from Parquet metadata, and non-inferable
  overrides have been removed.

### Explicit table schemas already declared (analytics + graph)
```
analytics.behavioral_coverage
analytics.cfg_block_metrics
analytics.cfg_function_metrics
analytics.cfg_function_metrics_ext
analytics.config_data_flow
analytics.config_graph_metrics_keys
analytics.config_graph_metrics_modules
analytics.config_projection_key_edges
analytics.config_projection_module_edges
analytics.config_values
analytics.coverage_functions
analytics.coverage_lines
analytics.data_model_fields
analytics.data_model_relationships
analytics.data_model_usage
analytics.data_models
analytics.dfg_block_metrics
analytics.dfg_function_metrics
analytics.dfg_function_metrics_ext
analytics.entrypoint_tests
analytics.entrypoints
analytics.external_dependencies
analytics.external_dependency_calls
analytics.file_profile
analytics.function_ast_features
analytics.function_contracts
analytics.function_effects
analytics.function_metrics
analytics.function_profile
analytics.function_types
analytics.function_validation
analytics.goid_risk_factors
analytics.graph_metrics_functions
analytics.graph_metrics_functions_ext
analytics.graph_metrics_modules
analytics.graph_metrics_modules_ext
analytics.graph_stats
analytics.graph_validation
analytics.hello_example
analytics.hotspots
analytics.module_profile
analytics.semantic_roles_functions
analytics.semantic_roles_modules
analytics.static_diagnostics
analytics.subsystem_agreement
analytics.subsystem_coverage_cache
analytics.subsystem_graph_metrics
analytics.subsystem_modules
analytics.subsystem_profile_cache
analytics.subsystems
analytics.symbol_graph_metrics_functions
analytics.symbol_graph_metrics_modules
analytics.test_catalog
analytics.test_coverage_edges
analytics.test_graph_metrics_functions
analytics.test_graph_metrics_tests
analytics.test_profile
analytics.typedness
graph.call_graph_edges
graph.call_graph_nodes
graph.cfg_blocks
graph.cfg_edges
graph.dfg_edges
graph.import_graph_edges
graph.import_modules
graph.symbol_use_edges
```

### Plan updates from registry review
- Added explicit plan entries for `analytics.data_models` and `analytics.subsystems`
  (declared in the registry).
- Keep `analytics.hello_example` out of scope (dev example only).
- Align graph table acceptance criteria to actual schema columns
  (e.g., `graph.call_graph_nodes` has no `repo/commit`).
- Removed from registry after decommission:
  `analytics.history_timeseries`, `analytics.function_history`,
  `docs.v_function_history_timeseries`, `docs.v_module_history_timeseries`.

## Materializer Choices (locked)
- Use `save_dataset` for all analytics/graph outputs (ArrowDatasetSaver).
- Use `save_relation_table` only for ingestion tool targets already emitting
  relation-like outputs and for multi-table collectors that are already defined.
- Partitioning: use `partition_columns=("repo", "commit")` when the schema
  includes both columns; otherwise partitioning is empty.
- Validation profile: default `lenient`; upgrade to `strict` for each table
  once unit + integration tests are green.
- Materializers must emit dataset-level Parquet metadata per the contract above;
  output registry schemas are no longer a boundary requirement.

## Parquet Boundary Implementation Checklist
- [x] Extend `save_dataset` / ArrowDatasetSaver to emit the required metadata
  into Parquet key/value metadata and a dataset-level `_metadata` or sidecar
  manifest.
- [x] Standardize the dataset directory layout so each table key has a single
  Parquet root and consistent partitioning.
- [x] Update DuckDB ingestion to read schema + metadata from Parquet only
  (no fallback to output_registry).
- [x] Update SQLGlot view building to consume only DuckDB sources derived from
  Parquet datasets.
- [x] Add tests that assert metadata presence and DuckDB schema derivation for a
  representative analytics + graph table.
- [x] Update test snapshot factory to emit Parquet metadata for dataset-backed
  snapshots.

## Initial Implementation Slice (v1)
First 10 tables to implement end-to-end in Hamilton DAG:

| Table key | DAG module (target) | Inputs | Materializer |
| --- | --- | --- | --- |
| graph.import_modules | `src/codeintel/build/hamilton/native/graphs/import_graph.py` | `core.modules`, `core.repo_map` | `save_dataset` |
| graph.import_graph_edges | `src/codeintel/build/hamilton/native/graphs/import_graph.py` | `core.modules`, AST/CST | `save_dataset` |
| graph.call_graph_nodes | `src/codeintel/build/hamilton/native/graphs/call_graph.py` | `core.goids`, AST/CST | `save_dataset` |
| graph.call_graph_edges | `src/codeintel/build/hamilton/native/graphs/call_graph.py` | `graph.call_graph_nodes`, AST/CST | `save_dataset` |
| graph.cfg_blocks | `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py` | `core.ast_nodes`, `core.goids` | `save_dataset` |
| graph.cfg_edges | `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py` | `graph.cfg_blocks` | `save_dataset` |
| graph.dfg_edges | `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py` | `core.ast_nodes`, `core.goids` | `save_dataset` |
| analytics.function_metrics | `src/codeintel/build/hamilton/native/analytics/tables_functions.py` | `core.goids`, AST metrics | `save_dataset` |
| analytics.function_types | `src/codeintel/build/hamilton/native/analytics/function_types.py` | `analytics.typedness`, `core.goids` | `save_dataset` |
| analytics.function_ast_features | `src/codeintel/build/hamilton/native/analytics/function_ast_features.py` | `core.ast_nodes`, `core.goids` | `save_dataset` |

Status: [x] Implemented in the DAG modules and tests noted below.

### Phase 1 implementation notes (completed)
- Graph compute nodes live in `src/codeintel/build/hamilton/native/graphs/*` with
  dataset materializers in `src/codeintel/build/hamilton/native/graphs/graph_targets.py`.
- Analytics v1 slice nodes live in
  `src/codeintel/build/hamilton/native/analytics/tables_functions.py`,
  `src/codeintel/build/hamilton/native/analytics/function_types.py`, and
  `src/codeintel/build/hamilton/native/analytics/function_ast_features.py`.
- Tests added: `tests/build/hamilton/native/graphs/test_v1_scaffold.py`,
  `tests/build/hamilton/native/analytics/test_v1_scaffold.py`.

## Table-by-table Acceptance Criteria and DAG Node Specs

### Core input tables (prerequisites)
#### core.modules
Status: [x]
Source logic: ingestion target `modules` in
`src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`.
DAG node spec:
- Target: `modules` (existing).
- Inputs: filesystem scan, module discovery adapters.
- Output: `core.modules` dataset.
Acceptance criteria:
- Primary key `(module, path)` unique.
- `repo`, `commit`, `language` present for all rows.
- Row count equals scanned module count for the snapshot.

#### core.repo_map
Status: [x]
Source logic: ingestion target `modules`.
DAG node spec:
- Target: `modules` (existing).
- Inputs: module discovery + overlays.
- Output: `core.repo_map` dataset.
Acceptance criteria:
- `modules` JSON length matches `core.modules` for the snapshot.
- Deterministic ordering for identical inputs.

#### core.file_state
Status: [x]
Source logic: ingestion target `modules`.
DAG node spec:
- Target: `modules` (existing).
- Inputs: file system stat + hashing.
- Output: `core.file_state` dataset.
Acceptance criteria:
- Primary key `(repo, rel_path, language)` unique.
- `content_hash` non-null for all rows.

#### core.ast_nodes
Status: [x]
Source logic: extraction target `ast` in
`src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`.
DAG node spec:
- Target: `ast` (existing).
- Inputs: module list + AST extraction step.
- Output: `core.ast_nodes` dataset.
Acceptance criteria:
- `hash` unique per node.
- `path` resolves to a module in `core.modules`.

#### core.ast_metrics
Status: [x]
Source logic: extraction target `ast`.
DAG node spec:
- Target: `ast` (existing).
- Inputs: AST extraction step.
- Output: `core.ast_metrics` dataset.
Acceptance criteria:
- One row per `rel_path`.
- `node_count`, `function_count`, `class_count` non-negative.

#### core.cst_nodes
Status: [x]
Source logic: extraction target `cst`.
DAG node spec:
- Target: `cst` (existing).
- Inputs: CST extraction step.
- Output: `core.cst_nodes` dataset.
Acceptance criteria:
- `node_id` unique.
- `path` resolves to `core.modules`.

#### core.docstrings
Status: [x]
Source logic: extraction target `docstrings`.
DAG node spec:
- Target: `docstrings` (existing).
- Inputs: docstring extraction step.
- Output: `core.docstrings` dataset.
Acceptance criteria:
- `repo`, `commit`, `module`, `qualname`, `kind` populated.
- `created_at` present for all rows.

#### core.goids
Status: [x]
Source logic: GOID inference in `src/codeintel/build/graphs/compute/goid.py`.
DAG node spec:
- Target: `goids` in `src/codeintel/build/hamilton/native/graphs/goids.py`.
- Inputs: `core.modules`, `core.ast_nodes`.
- Output: `core.goids` dataset.
Acceptance criteria:
- `goid_h128` unique.
- Rows scoped to `repo`, `commit` for the build snapshot.

#### core.goid_crosswalk
Status: [x]
Source logic: GOID inference in `src/codeintel/build/graphs/compute/goid.py`.
DAG node spec:
- Target: `goids` in `src/codeintel/build/hamilton/native/graphs/goids.py`.
- Inputs: `core.modules`, `core.ast_nodes`.
- Output: `core.goid_crosswalk` dataset.
Acceptance criteria:
- `(repo, commit, goid)` unique.
- Crosswalk references are consistent with `core.goids`.

### Graph tables
#### graph.import_modules
Status: [x]
Source logic: `src/codeintel/build/graphs/compute/imports.py`.
Target DAG module: `src/codeintel/build/hamilton/native/graphs/import_graph.py`.
DAG node spec:
- Nodes: `import_modules__base -> import_graph__modules_table -> t__import_graph`.
- Inputs: `core.modules`, `core.repo_map`.
- Output: `graph.import_modules` dataset.
Acceptance criteria:
- One row per module in `core.modules`.
- `repo`, `commit`, `module` match `core.modules`.

#### graph.import_graph_edges
Status: [x]
Source logic: `src/codeintel/build/graphs/compute/imports.py`.
Target DAG module: `src/codeintel/build/hamilton/native/graphs/import_graph.py`.
DAG node spec:
- Nodes: `import_graph_edges__base -> import_graph__edges_table -> t__import_graph`.
- Inputs: `core.modules` plus parse outputs (AST/CST) as needed.
- Output: `graph.import_graph_edges` dataset.
Acceptance criteria:
- All edges reference modules present in `graph.import_modules`.
- No duplicate `(src, dst, kind)` edges for the same snapshot.

#### graph.call_graph_nodes
Status: [x]
Source logic: `src/codeintel/build/graphs/compute/callgraph/collection.py`.
Target DAG module: `src/codeintel/build/hamilton/native/graphs/call_graph.py`.
DAG node spec:
- Nodes: `call_graph_nodes__base -> call_graph__nodes_table -> t__call_graph`.
- Inputs: `core.goids`, AST/CST nodes, optional SCIP symbols.
- Output: `graph.call_graph_nodes` dataset.
Acceptance criteria:
- Node references resolve to `core.goids`.
- No duplicate `goid_h128` rows.

#### graph.call_graph_edges
Status: [x]
Source logic: `src/codeintel/build/graphs/compute/callgraph/collection.py`.
Target DAG module: `src/codeintel/build/hamilton/native/graphs/call_graph.py`.
DAG node spec:
- Nodes: `call_graph_edges__base -> call_graph__edges_table -> t__call_graph`.
- Inputs: `graph.call_graph_nodes`, AST/CST nodes.
- Output: `graph.call_graph_edges` dataset.
Acceptance criteria:
- Edge endpoints exist in `graph.call_graph_nodes`.
- Edge count deterministic for the same inputs.

#### graph.cfg_blocks
Status: [x]
Source logic: `src/codeintel/build/graphs/compute/cfg.py`.
Target DAG module: `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`.
DAG node spec:
- Nodes: `cfg_blocks__base -> cfg__blocks_table -> t__cfg`.
- Inputs: `core.ast_nodes`, `core.goids`.
- Output: `graph.cfg_blocks` dataset.
Acceptance criteria:
- Block ids unique per `function_goid_h128`.
- `start_line` and `end_line` are within function span.

#### graph.cfg_edges
Status: [x]
Source logic: `src/codeintel/build/graphs/compute/cfg.py`.
Target DAG module: `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`.
DAG node spec:
- Nodes: `cfg_edges__base -> cfg__edges_table -> t__cfg`.
- Inputs: `graph.cfg_blocks` plus AST/CST as needed.
- Output: `graph.cfg_edges` dataset.
Acceptance criteria:
- Edge endpoints exist in `graph.cfg_blocks`.
- No duplicate edges per `(src_block, dst_block, edge_type)`.

#### graph.dfg_edges
Status: [x]
Source logic: `src/codeintel/build/graphs/compute/dfg.py`.
Target DAG module: `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`.
DAG node spec:
- Nodes: `dfg_edges__base -> dfg__edges_table -> t__dfg`.
- Inputs: `core.ast_nodes`, `core.goids`.
- Output: `graph.dfg_edges` dataset.
Acceptance criteria:
- Edge endpoints refer to valid CFG/DFG blocks or function GOIDs.
- `use_kind` and `via_phi` fields conform to expected enums.

#### graph.symbol_use_edges
Status: [x]
Source logic: `src/codeintel/build/graphs/compute/symbols.py`.
Target DAG module: `src/codeintel/build/hamilton/native/graphs/symbol_use.py`.
DAG node spec:
- Nodes: `symbol_use_edges__base -> symbol_use_edges__table -> t__symbol_use`.
- Inputs: `core.goid_crosswalk`, SCIP symbol graph, AST/CST as needed.
- Output: `graph.symbol_use_edges` dataset.
Acceptance criteria:
- Symbol endpoints resolve to `core.goid_crosswalk` or SCIP ids.
- Deterministic edge counts per snapshot.

### Analytics tables (ingestion-derived)
#### analytics.config_values
Status: [x]
Source logic: ingestion target `config_ingest`.
DAG node spec:
- Target: `config_ingest` in
  `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`.
- Output: `analytics.config_values` dataset.
Acceptance criteria:
- Key/value pairs map to discovered config files.
- `repo`, `commit` populated for all rows.

#### analytics.coverage_lines
Status: [x]
Source logic: ingestion target `coverage_ingest`.
DAG node spec:
- Target: `coverage_ingest` in ingestion targets.
- Output: `analytics.coverage_lines` dataset.
Acceptance criteria:
- Rows refer to valid files in `core.modules`.
- Coverage line numbers within file bounds.

#### analytics.test_catalog
Status: [x]
Source logic: ingestion target `tests_ingest`.
DAG node spec:
- Target: `tests_ingest` in ingestion targets.
- Output: `analytics.test_catalog` dataset.
Acceptance criteria:
- `test_id` unique per snapshot.
- `repo`, `commit` populated for all rows.

#### analytics.typedness
Status: [x]
Source logic: ingestion target `typing`.
DAG node spec:
- Target: `typing` in ingestion targets.
- Output: `analytics.typedness` dataset.
Acceptance criteria:
- Rows align to `core.goids` and/or `core.modules`.
- Typedness ratios in range `[0, 1]`.

#### analytics.static_diagnostics
Status: [x]
Source logic: ingestion target `typing`.
DAG node spec:
- Target: `typing` in ingestion targets.
- Output: `analytics.static_diagnostics` dataset.
Acceptance criteria:
- Diagnostics map to valid file paths and line spans.
- Severity values are normalized and non-null.

### Analytics tables (function analytics)
#### analytics.function_metrics
Status: [x]
Source logic: `src/codeintel/build/analytics/functions/metrics.py`,
`src/codeintel/build/analytics/compute/functions/*`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/tables_functions.py`.
DAG node spec:
- Nodes: `function_metrics__base -> function_metrics__table -> t__function_metrics`.
- Inputs: `core.goids`, AST/CST metrics, optional typedness inputs.
- Output: `analytics.function_metrics` dataset.
Acceptance criteria:
- Row count equals function GOIDs for the snapshot.
- `loc >= 0`, `cyclomatic_complexity >= 0`, `end_line >= start_line`.

#### analytics.function_types
Status: [x]
Source logic: `src/codeintel/build/analytics/compute/functions/typedness.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/function_types.py`.
DAG node spec:
- Nodes: `function_types__base -> function_types__table -> t__function_types`.
- Inputs: `analytics.typedness`, `core.goids`, typing diagnostics.
- Output: `analytics.function_types` dataset.
Acceptance criteria:
- One row per function GOID when typedness is available.
- Type coverage and diagnostic counts are consistent.

#### analytics.function_ast_features
Status: [x]
Source logic: `src/codeintel/build/analytics/ast_features/extract.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/function_ast_features.py`.
DAG node spec:
- Nodes: `function_ast_features__base -> function_ast_features__table`.
- Inputs: `core.ast_nodes`, `core.goids`, AST feature patterns.
- Output: `analytics.function_ast_features` dataset.
Acceptance criteria:
- Features are deterministic given AST inputs.
- GOID references are valid.

#### analytics.function_effects
Status: [x]
Source logic: `src/codeintel/build/analytics/functions/function_effects.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/function_effects.py`.
DAG node spec:
- Nodes: `function_effects__base -> function_effects__table`.
- Inputs: `analytics.function_metrics`, `graph.call_graph_edges`,
  `analytics.function_ast_features`.
- Output: `analytics.function_effects` dataset.
Acceptance criteria:
- Effects rows only for functions present in metrics.
- External call counts align to call graph edges.

#### analytics.function_contracts
Status: [x]
Source logic: `src/codeintel/build/analytics/functions/function_contracts.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/function_contracts.py`.
DAG node spec:
- Nodes: `function_contracts__base -> function_contracts__table`.
- Inputs: `core.docstrings`, `core.ast_nodes`, `core.goids`.
- Output: `analytics.function_contracts` dataset.
Acceptance criteria:
- Contract fields are populated when docstrings are present.
- Contracts align to GOIDs and spans.

#### analytics.function_validation
Status: [x]
Source logic: `src/codeintel/build/analytics/parsing/compute.py`,
`src/codeintel/build/analytics/functions/metrics.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/function_validation.py`.
DAG node spec:
- Nodes: `function_validation__base -> function_validation__table`.
- Inputs: `core.goids`, AST/CST metrics, typedness diagnostics.
- Output: `analytics.function_validation` dataset.
Acceptance criteria:
- Each row contains `repo`, `commit`, `rel_path`, and issue category.
- Issues are stable across identical inputs.

#### analytics.goid_risk_factors
Status: [x]
Source logic: `src/codeintel/build/analytics/subsystems/risk.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/tables_risk.py`.
DAG node spec:
- Nodes: `risk_factors__base -> risk_factors__table -> t__risk_factors`.
- Inputs: `analytics.function_metrics`, `analytics.coverage_functions`,
  `analytics.test_catalog`.
- Output: `analytics.goid_risk_factors` dataset.
Acceptance criteria:
- `risk_level` derived from `risk_score` thresholds.
- `risk_score` deterministic and non-negative.

### Analytics tables (profiles)
#### analytics.function_profile
Status: [x]
Source logic: `src/codeintel/build/analytics/profiles/functions.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/profiles.py`.
DAG node spec:
- Nodes: `function_profile__base -> function_profile__table`.
- Inputs: `analytics.function_metrics`, `analytics.function_types`,
  `analytics.function_effects`, `analytics.function_contracts`,
  `analytics.semantic_roles_functions`, `analytics.graph_metrics_functions`.
- Output: `analytics.function_profile` dataset.
Acceptance criteria:
- Composite schema matches configured composition rules.
- No missing required columns from source tables.

#### analytics.file_profile
Status: [x]
Source logic: `src/codeintel/build/analytics/profiles/files.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/profiles.py`.
DAG node spec:
- Nodes: `file_profile__base -> file_profile__table`.
- Inputs: `core.modules`, `analytics.function_profile`, `analytics.coverage_lines`.
- Output: `analytics.file_profile` dataset.
Acceptance criteria:
- One row per module file path.
- Aggregates align to function-level inputs.

#### analytics.module_profile
Status: [x]
Source logic: `src/codeintel/build/analytics/profiles/modules.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/tables_modules.py`.
DAG node spec:
- Nodes: `module_profile__base -> module_profile__table -> t__module_profile`.
- Inputs: `core.modules`, `analytics.function_profile`,
  `analytics.graph_metrics_modules`, `analytics.coverage_lines`.
- Output: `analytics.module_profile` dataset.
Acceptance criteria:
- One row per module id.
- Aggregates (counts, averages) match inputs.

#### analytics.hotspots
Status: [x]
Source logic: `src/codeintel/build/analytics/hotspots.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/profiles.py`.
DAG node spec:
- Nodes: `hotspots__base -> hotspots__table`.
- Inputs: `analytics.function_metrics`, `analytics.goid_risk_factors`.
- Output: `analytics.hotspots` dataset.
Acceptance criteria:
- Ranked outputs are deterministic per snapshot.
- Scores fall within expected ranges.

### Analytics tables (coverage and testing)
#### analytics.coverage_functions
Status: [x]
Source logic: `src/codeintel/build/hamilton/native/analytics/tables_coverage.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/tables_coverage.py`.
DAG node spec:
- Nodes: `coverage_functions__base -> coverage_functions__table`.
- Inputs: `analytics.coverage_lines`, `core.goids`.
- Output: `analytics.coverage_functions` dataset.
Acceptance criteria:
- Coverage ratio per function is between 0 and 1.
- Function GOIDs resolve to `core.goids`.

#### analytics.test_coverage_edges
Status: [x]
Source logic: `src/codeintel/build/analytics/testing/coverage/edges.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/testing.py`.
DAG node spec:
- Nodes: `test_coverage_edges__base -> test_coverage_edges__table`.
- Inputs: `analytics.test_catalog`, `analytics.coverage_lines`, `core.goids`.
- Output: `analytics.test_coverage_edges` dataset.
Acceptance criteria:
- Edges reference existing test ids and function GOIDs.
- Deterministic edge count for same inputs.

#### analytics.test_graph_metrics
Status: [x]
Source logic: `src/codeintel/build/analytics/testing/graph_metrics.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/testing.py`.
DAG node spec:
- Nodes: `test_graph_metrics_tests__base`, `test_graph_metrics_functions__base`
  -> `test_graph_metrics_tests__table`, `test_graph_metrics_functions__table`
  -> `test_graph_metrics__table_materializations -> t__test_graph_metrics`.
- Inputs: `analytics.test_coverage_edges`, `analytics.goid_risk_factors`.
- Output: `analytics.test_graph_metrics_tests`, `analytics.test_graph_metrics_functions` datasets.
Acceptance criteria:
- Graph metrics computed over bipartite test-function graph.
- All referenced tests and functions exist.

#### analytics.test_graph_metrics_functions
Status: [x]
Source logic: `src/codeintel/build/analytics/testing/graph_metrics.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/testing.py`.
DAG node spec:
- Nodes: `test_graph_metrics_functions__base -> test_graph_metrics_functions__table`.
- Inputs: `analytics.test_coverage_edges`, `analytics.function_metrics`.
- Output: `analytics.test_graph_metrics_functions` dataset.
Acceptance criteria:
- One row per function with test connectivity metrics.

#### analytics.test_graph_metrics_tests
Status: [x]
Source logic: `src/codeintel/build/analytics/testing/graph_metrics.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/testing.py`.
DAG node spec:
- Nodes: `test_graph_metrics_tests__base -> test_graph_metrics_tests__table`.
- Inputs: `analytics.test_coverage_edges`, `analytics.test_catalog`.
- Output: `analytics.test_graph_metrics_tests` dataset.
Acceptance criteria:
- One row per test id with coverage connectivity metrics.

#### analytics.test_profile
Status: [x]
Source logic: `src/codeintel/build/analytics/testing/profiles/rows.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/testing.py`.
DAG node spec:
- Nodes: `test_profile__base -> test_profile__table`.
- Inputs: `analytics.test_catalog`, `analytics.test_coverage_edges`,
  `analytics.test_graph_metrics_tests`, `analytics.behavioral_coverage`.
- Output: `analytics.test_profile` dataset.
Acceptance criteria:
- One row per test id.
- Coverage and behavioral metrics consistent with inputs.

#### analytics.behavioral_coverage
Status: [x]
Source logic: `src/codeintel/build/analytics/testing/behavioral/tags.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/testing.py`.
DAG node spec:
- Nodes: `behavioral_coverage__base -> behavioral_coverage__table`.
- Inputs: `analytics.test_catalog`, `analytics.test_coverage_edges`,
  `analytics.function_ast_features`.
- Output: `analytics.behavioral_coverage` dataset.
Acceptance criteria:
- Behavioral tags derived from AST patterns are deterministic.
- Rows reference valid tests and functions.

#### analytics.entrypoint_tests
Status: [x]
Source logic: `src/codeintel/build/analytics/entrypoints/compute.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/entrypoints.py`.
DAG node spec:
- Nodes: `entrypoint_tests__base -> entrypoint_tests__table`.
- Inputs: `analytics.entrypoints`, `analytics.test_catalog`,
  `analytics.test_coverage_edges`.
- Output: `analytics.entrypoint_tests` dataset.
Acceptance criteria:
- Links entrypoints to tests with coverage evidence.

### Analytics tables (dependencies and config)
#### analytics.external_dependency_calls
Status: [x]
Source logic: `src/codeintel/build/analytics/dependencies/compute.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/tables_dependencies.py`.
DAG node spec:
- Nodes: `external_dependency_calls__base -> external_dependency_calls__table`.
- Inputs: `analytics.function_ast_features`, `core.goids`, `core.modules`.
- Output: `analytics.external_dependency_calls` dataset.
Acceptance criteria:
- Calls map to valid GOIDs and dependency identifiers.

#### analytics.external_dependencies
Status: [x]
Source logic: `src/codeintel/build/analytics/dependencies/compute.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/tables_dependencies.py`.
DAG node spec:
- Nodes: `external_dependencies__base -> external_dependencies__table`.
- Inputs: `analytics.external_dependency_calls`.
- Output: `analytics.external_dependencies` dataset.
Acceptance criteria:
- One row per dependency signature per snapshot.

#### analytics.dependency_targets
Status: [x] (de-scoped)
Source logic: `src/codeintel/build/analytics/dependencies/core.py` (legacy helper).
Target DAG module: n/a (removed from scope).
Notes: `analytics.dependency_targets` is not declared in the schema registry and
is no longer planned for the Hamilton DAG.

#### analytics.config_data_flow
Status: [x]
Source logic: `src/codeintel/build/analytics/graphs/config_data_flow.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/config_graphs.py`.
DAG node spec:
- Nodes: `config_data_flow__base -> config_data_flow__table`.
- Inputs: `analytics.config_values`, `analytics.entrypoints`,
  `graph.call_graph_edges`, `graph.call_graph_nodes`, `core.goids`.
- Output: `analytics.config_data_flow` dataset.
Acceptance criteria:
- Config flow edges reference valid config keys and functions.

#### analytics.config_graph_metrics_keys
Status: [x]
Source logic: `src/codeintel/build/analytics/graphs/config_graph_metrics.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/config_graphs.py`.
DAG node spec:
- Nodes: `config_graph_metrics_keys__base -> config_graph_metrics_keys__table`.
- Inputs: `analytics.config_values`, `core.modules`.
- Output: `analytics.config_graph_metrics_keys` dataset.
Acceptance criteria:
- Metrics align to config key nodes in the flow graph.

#### analytics.config_graph_metrics_modules
Status: [x]
Source logic: `src/codeintel/build/analytics/graphs/config_graph_metrics.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/config_graphs.py`.
DAG node spec:
- Nodes: `config_graph_metrics_modules__base -> config_graph_metrics_modules__table`.
- Inputs: `analytics.config_values`, `core.modules`.
- Output: `analytics.config_graph_metrics_modules` dataset.
Acceptance criteria:
- Module metrics align to module nodes in config flow graph.

#### analytics.config_projection_key_edges
Status: [x]
Source logic: `src/codeintel/build/analytics/graphs/config_graph_metrics.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/config_graphs.py`.
DAG node spec:
- Nodes: `config_projection_key_edges__base -> config_projection_key_edges__table`.
- Inputs: `analytics.config_values`, `core.modules`.
- Output: `analytics.config_projection_key_edges` dataset.
Acceptance criteria:
- Projection edges represent key-to-key reachability in config graph.

#### analytics.config_projection_module_edges
Status: [x]
Source logic: `src/codeintel/build/analytics/graphs/config_graph_metrics.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/config_graphs.py`.
DAG node spec:
- Nodes: `config_projection_module_edges__base -> config_projection_module_edges__table`.
- Inputs: `analytics.config_values`, `core.modules`.
- Output: `analytics.config_projection_module_edges` dataset.
Acceptance criteria:
- Projection edges represent module-to-module config influence.

#### analytics.entrypoints
Status: [x]
Source logic: `src/codeintel/build/analytics/entrypoints/compute.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/entrypoints.py`.
DAG node spec:
- Nodes: `entrypoints__base -> entrypoints__table`.
- Inputs: `core.modules`, `analytics.function_ast_features`,
  `analytics.test_profile`, `analytics.test_coverage_edges`, `analytics.subsystems`.
- Output: `analytics.entrypoints` dataset.
Acceptance criteria:
- Entrypoint rows map to valid functions and modules.

### Analytics tables (semantic roles)
#### analytics.semantic_roles_functions
Status: [x]
Source logic: `src/codeintel/build/analytics/semantic_roles/core.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/semantic_roles.py`.
DAG node spec:
- Nodes: `semantic_roles_functions__base -> semantic_roles_functions__table`.
- Inputs: `core.modules`, `analytics.function_ast_features`,
  `analytics.function_metrics`, `analytics.function_effects`,
  `analytics.function_contracts`, `analytics.graph_metrics_functions`.
- Output: `analytics.semantic_roles_functions` dataset.
Acceptance criteria:
- Each function has at most one primary role.
- Confidence score range `[0, 1]`.

#### analytics.semantic_roles_modules
Status: [x]
Source logic: `src/codeintel/build/analytics/semantic_roles/core.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/semantic_roles.py`.
DAG node spec:
- Nodes: `semantic_roles_modules__base -> semantic_roles_modules__table`.
- Inputs: `analytics.semantic_roles_functions`, `analytics.module_profile`.
- Output: `analytics.semantic_roles_modules` dataset.
Acceptance criteria:
- Module roles aggregate function roles deterministically.

### Analytics tables (graph metrics)
#### analytics.graph_metrics_functions
Status: [x]
Source logic: `src/codeintel/build/analytics/compute/row_builders/graph_metrics.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`.
DAG node spec:
- Nodes: `graph_metrics_functions__base -> graph_metrics_functions__table`.
- Inputs: `graph.call_graph_edges`, `graph.call_graph_nodes`,
  `analytics.function_metrics`.
- Output: `analytics.graph_metrics_functions` dataset.
Acceptance criteria:
- Graph metrics computed for all functions with call graph nodes.

#### analytics.graph_metrics_functions_ext
Status: [x]
Source logic: `src/codeintel/build/analytics/compute/row_builders/graph_metrics_ext.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`.
DAG node spec:
- Nodes: `graph_metrics_functions_ext__base -> graph_metrics_functions_ext__table`.
- Inputs: `analytics.graph_metrics_functions`, `graph.call_graph_edges`.
- Output: `analytics.graph_metrics_functions_ext` dataset.
Acceptance criteria:
- Ext metrics align to base graph metrics by function id.

#### analytics.graph_metrics_modules
Status: [x]
Source logic: `src/codeintel/build/analytics/compute/row_builders/graph_metrics.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`.
DAG node spec:
- Nodes: `graph_metrics_modules__base -> graph_metrics_modules__table`.
- Inputs: `graph.import_graph_edges`, `graph.import_modules`, `core.modules`.
- Output: `analytics.graph_metrics_modules` dataset.
Acceptance criteria:
- Module metrics align to module ids and import edges.

#### analytics.graph_metrics_modules_ext
Status: [x]
Source logic: `src/codeintel/build/analytics/compute/row_builders/graph_metrics_ext.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`.
DAG node spec:
- Nodes: `graph_metrics_modules_ext__base -> graph_metrics_modules_ext__table`.
- Inputs: `analytics.graph_metrics_modules`, `graph.import_graph_edges`.
- Output: `analytics.graph_metrics_modules_ext` dataset.
Acceptance criteria:
- Ext metrics align to base module metrics by module id.

#### analytics.symbol_graph_metrics_functions
Status: [x]
Source logic: `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`.
DAG node spec:
- Nodes: `symbol_graph_metrics_functions__base -> symbol_graph_metrics_functions__table`.
- Inputs: `graph.symbol_use_edges`, `analytics.function_metrics`.
- Output: `analytics.symbol_graph_metrics_functions` dataset.
Acceptance criteria:
- Symbols resolve to functions where possible.

#### analytics.symbol_graph_metrics_modules
Status: [x]
Source logic: `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`.
DAG node spec:
- Nodes: `symbol_graph_metrics_modules__base -> symbol_graph_metrics_modules__table`.
- Inputs: `graph.symbol_use_edges`, `core.modules`.
- Output: `analytics.symbol_graph_metrics_modules` dataset.
Acceptance criteria:
- Symbols aggregate to modules deterministically.

#### analytics.graph_stats
Status: [x]
Source logic: `src/codeintel/build/analytics/graphs/graph_stats.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`.
DAG node spec:
- Nodes: `graph_stats__base -> graph_stats__table`.
- Inputs: `graph.call_graph_edges`, `graph.call_graph_nodes`,
  `graph.import_graph_edges`, `graph.import_modules`,
  `graph.symbol_use_edges`, `analytics.config_values`, `core.modules`.
- Output: `analytics.graph_stats` dataset.
Acceptance criteria:
- Stats rows include node/edge counts for call/import/symbol/config projections.

#### analytics.graph_validation
Status: [x]
Source logic: `src/codeintel/build/analytics/parsing/compute.py`,
`src/codeintel/build/graphs/validation/*`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/graph_validation.py`.
DAG node spec:
- Nodes: `graph_validation__base -> graph_validation__table`.
- Inputs: `graph.*` tables, validation rules.
- Output: `analytics.graph_validation` dataset.
Acceptance criteria:
- Each issue references a graph entity and severity.

### Analytics tables (CFG/DFG analytics)
#### analytics.cfg_function_metrics
Status: [x]
Source logic: `src/codeintel/build/analytics/cfg_dfg/compute.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/cfg_dfg_metrics.py`.
DAG node spec:
- Nodes: `cfg_function_metrics__base -> cfg_function_metrics__table`.
- Inputs: `graph.cfg_edges`, `graph.cfg_blocks`, `core.goids`.
- Output: `analytics.cfg_function_metrics` dataset.
Acceptance criteria:
- Rows align to function GOIDs with CFG graphs.

#### analytics.cfg_block_metrics
Status: [x]
Source logic: `src/codeintel/build/analytics/cfg_dfg/compute.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/cfg_dfg_metrics.py`.
DAG node spec:
- Nodes: `cfg_block_metrics__base -> cfg_block_metrics__table`.
- Inputs: `graph.cfg_blocks`, `graph.cfg_edges`.
- Output: `analytics.cfg_block_metrics` dataset.
Acceptance criteria:
- Block metrics computed for all CFG blocks.

#### analytics.cfg_function_metrics_ext
Status: [x]
Source logic: `src/codeintel/build/analytics/cfg_dfg/compute.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/cfg_dfg_metrics.py`.
DAG node spec:
- Nodes: `cfg_function_metrics_ext__base -> cfg_function_metrics_ext__table`.
- Inputs: `analytics.cfg_function_metrics`, `graph.cfg_edges`.
- Output: `analytics.cfg_function_metrics_ext` dataset.
Acceptance criteria:
- Ext metrics align to base CFG function metrics.

#### analytics.dfg_function_metrics
Status: [x]
Source logic: `src/codeintel/build/analytics/cfg_dfg/compute.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/cfg_dfg_metrics.py`.
DAG node spec:
- Nodes: `dfg_function_metrics__base -> dfg_function_metrics__table`.
- Inputs: `graph.dfg_edges`, `core.goids`.
- Output: `analytics.dfg_function_metrics` dataset.
Acceptance criteria:
- Rows align to function GOIDs with DFG graphs.

#### analytics.dfg_block_metrics
Status: [x]
Source logic: `src/codeintel/build/analytics/cfg_dfg/compute.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/cfg_dfg_metrics.py`.
DAG node spec:
- Nodes: `dfg_block_metrics__base -> dfg_block_metrics__table`.
- Inputs: `graph.dfg_edges`.
- Output: `analytics.dfg_block_metrics` dataset.
Acceptance criteria:
- Block metrics computed for all DFG blocks.

#### analytics.dfg_function_metrics_ext
Status: [x]
Source logic: `src/codeintel/build/analytics/cfg_dfg/compute.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/cfg_dfg_metrics.py`.
DAG node spec:
- Nodes: `dfg_function_metrics_ext__base -> dfg_function_metrics_ext__table`.
- Inputs: `analytics.dfg_function_metrics`, `graph.dfg_edges`.
- Output: `analytics.dfg_function_metrics_ext` dataset.
Acceptance criteria:
- Ext metrics align to base DFG function metrics.

### Analytics tables (subsystems)
#### analytics.subsystems
Status: [x]
Source logic: `src/codeintel/build/analytics/subsystems/materialize.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/subsystems.py`.
DAG node spec:
- Nodes: `subsystems__base -> subsystems__table`.
- Inputs: `core.modules`, `graph.import_graph_edges`, `analytics.goid_risk_factors`,
  `analytics.config_values` (tags).
- Output: `analytics.subsystems` dataset.
Acceptance criteria:
- `subsystem_id` unique per `(repo, commit)`.
- `module_count` equals the length of `modules_json`.
- `modules_json` entries resolve to `core.modules`.

#### analytics.subsystem_modules
Status: [x]
Source logic: `src/codeintel/build/analytics/subsystems/materialize.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/subsystems.py`.
DAG node spec:
- Nodes: `subsystem_modules__base -> subsystem_modules__table`.
- Inputs: `core.modules`, `analytics.config_values`, tag rules.
- Output: `analytics.subsystem_modules` dataset.
Acceptance criteria:
- Each module assigned to zero or one subsystem id.

#### analytics.subsystem_graph_metrics
Status: [x]
Source logic: `src/codeintel/build/analytics/compute/row_builders/subsystem_metrics.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/subsystem_metrics.py`.
DAG node spec:
- Nodes: `subsystem_graph_metrics__base -> subsystem_graph_metrics__table`.
- Inputs: `analytics.subsystem_modules`, `graph.import_graph_edges`,
  `graph.import_modules`.
- Output: `analytics.subsystem_graph_metrics` dataset.
Acceptance criteria:
- Metrics aggregated by subsystem id.

#### analytics.subsystem_agreement
Status: [x]
Source logic: `src/codeintel/build/analytics/graphs/subsystem_agreement.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/subsystem_agreement.py`.
DAG node spec:
- Nodes: `subsystem_agreement__base -> subsystem_agreement__table`.
- Inputs: `analytics.subsystem_modules`, `analytics.graph_metrics_modules_ext`.
- Output: `analytics.subsystem_agreement` dataset.
Acceptance criteria:
- Agreement score is deterministic and in `[0, 1]`.

#### analytics.subsystem_profile_cache
Status: [x]
Source logic: `src/codeintel/build/analytics/subsystems/cache.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/subsystem_cache.py`.
DAG node spec:
- Nodes: `subsystem_profile_cache__base -> subsystem_profile_cache__table`.
- Inputs: `analytics.subsystem_graph_metrics`, `analytics.module_profile`,
  `analytics.entrypoints`.
- Output: `analytics.subsystem_profile_cache` dataset.
Acceptance criteria:
- Cache rows cover all subsystems present in `analytics.subsystem_modules`.

#### analytics.subsystem_coverage_cache
Status: [x]
Source logic: `src/codeintel/build/analytics/subsystems/cache.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/subsystem_cache.py`.
DAG node spec:
- Nodes: `subsystem_coverage_cache__base -> subsystem_coverage_cache__table`.
- Inputs: `analytics.subsystem_modules`, `analytics.test_profile`,
  `analytics.coverage_functions`.
- Output: `analytics.subsystem_coverage_cache` dataset.
Acceptance criteria:
- Coverage ratios computed for all subsystems.

### Analytics tables (data models)
#### analytics.data_models
Status: [x]
Source logic: `src/codeintel/build/analytics/data_models/compute.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/data_models.py`.
DAG node spec:
- Nodes: `data_models__base -> data_models__table`.
- Inputs: `core.ast_nodes`, `core.docstrings`.
- Output: `analytics.data_models` dataset.
Acceptance criteria:
- `model_id` unique per `(repo, commit)`.
- `model_name`, `model_kind`, `module`, and `rel_path` populated.
- `goid_h128` resolves to `core.goids` when present.

#### analytics.data_model_fields
Status: [x]
Source logic: `src/codeintel/build/analytics/data_models/compute.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/data_models.py`.
DAG node spec:
- Nodes: `data_model_fields__base -> data_model_fields__table`.
- Inputs: `core.ast_nodes`, `core.docstrings`.
- Output: `analytics.data_model_fields` dataset.
Acceptance criteria:
- Field rows include type, name, and model id.

#### analytics.data_model_relationships
Status: [x]
Source logic: `src/codeintel/build/analytics/data_models/compute.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/data_models.py`.
DAG node spec:
- Nodes: `data_model_relationships__base -> data_model_relationships__table`.
- Inputs: `analytics.data_model_fields`, AST/CST nodes.
- Output: `analytics.data_model_relationships` dataset.
Acceptance criteria:
- Relationship endpoints resolve to known models/fields.

#### analytics.data_model_usage
Status: [x]
Source logic: `src/codeintel/build/analytics/compute/data_models/usage.py`.
Target DAG module: `src/codeintel/build/hamilton/native/analytics/data_models.py`.
DAG node spec:
- Nodes: `data_model_usage__base -> data_model_usage__table`.
- Inputs: `analytics.data_model_fields`, `core.ast_nodes`.
- Output: `analytics.data_model_usage` dataset.
Acceptance criteria:
- Usage rows reference valid model fields and functions.

## History + Timeseries Decommission (current-state only)
### Removal checklist
- [x] Remove history targets (`analytics.history_timeseries`,
  `analytics.function_history`) from DAG inventory and build specs.
- [x] Remove output registry + row models for history tables.
- [x] Remove CLI commands/options/results for history.timeseries.
- [x] Remove docs views that depend on history tables.
- [x] Update snapshot-only analytics to drop history inputs
  (`analytics.hotspots`, `analytics.function_profile`).
- [x] Remove tests and helpers that insert or validate history/timeseries rows.

### File scope (delete or update)
Timeseries-specific:
- [x] `src/codeintel/build/analytics/history/history_timeseries.py` (delete).
- [x] `src/codeintel/build/hamilton/native/analytics/history_timeseries.py` (delete).
- [x] `src/codeintel/build/hamilton/native/analytics/__init__.py` (drop exports).
- [x] `src/codeintel/build/hamilton/env.py` (remove HistoryTimeseriesOptions).
- [x] `src/codeintel/build/run_context.py` (remove HistoryTimeseriesOptions).
- [x] `src/codeintel/build/config.py` (remove history_timeseries config).
- [x] `src/codeintel/cli/commands/history.py` (remove history.timeseries).
- [x] `src/codeintel/cli/handlers/history.py` (remove handler logic).
- [x] `src/codeintel/cli/options/registry.py` (remove timeseries flags).
- [x] `src/codeintel/cli/core/result_types.py` (remove HistoryTimeseriesResult).
- [x] `src/codeintel/cli/handlers/__init__.py` (drop export).
- [x] `src/codeintel/core/registry/dag_output_inventory.yaml` (remove target).
- [x] `src/codeintel/core/schemas/output_registry.py` (remove history_timeseries).
- [x] `src/codeintel/core/schemas/generated_rows/analytics.py` (remove row model).
- [x] `src/codeintel/core/schemas/table_registry.py` (remove table key).
- [x] `src/codeintel/storage/views/view_ast_map.json`
  (remove docs.v_*_history_timeseries; view map already retired).
- [x] `tests/cli/test_history_timeseries_cli.py` (remove).
- [x] `tests/cli/test_history_validation.py` (remove timeseries checks).
- [x] `tests/cli/test_cli_error_parity_apps.py` (update).
- [x] `tests/cli/test_help_rendering.py` (update history help).
- [x] `tests/build/hamilton/test_pr55_final_sweep.py` (remove history_timeseries).
- [x] `tests/build/hamilton/snapshots/*` (regenerate after removal).
- [x] `docs/architecture.md` (remove history env reference).
- [x] `docs/hamilton_inference_first_implementation_plan.md`
  (remove docs.v_* history).
- [x] `docs/polars_arrow_rearchitecture_plan.md` (remove history timeseries plan).

History (multi-commit) analytics removal:
- [x] `src/codeintel/build/analytics/functions/function_history.py` (delete).
- [x] `src/codeintel/build/hamilton/native/analytics/function_history.py` (delete).
- [x] `src/codeintel/build/hamilton/native/analytics/__init__.py` (drop exports).
- [x] `src/codeintel/build/analytics/profiles/functions.py` (remove history joins).
- [x] `src/codeintel/build/hamilton/native/analytics/profiles.py` (drop inputs).
- [x] `src/codeintel/core/registry/dag_output_inventory.yaml` (remove target).
- [x] `src/codeintel/core/schemas/output_registry.py` (remove function_history).
- [x] `src/codeintel/core/schemas/generated_rows/analytics.py` (remove row model).
- [x] `src/codeintel/core/schemas/table_registry.py` (remove table key).
- [x] `src/codeintel/storage/views/view_ast_map.json`
  (remove docs.v_function_history; view map already retired).
- [x] `tests/_helpers/orchestration/history.py` (remove function_history helpers).
- [x] `tests/analytics/test_profiles_and_functions.py` (remove joins).
- [x] `tests/architecture/test_analytics_imports.py` (update allowed exports).
- [x] `docs/hamilton_best_in_class_inventory.md` (remove function_history table).

## Legacy Analytics/Graphs Decommission
Note: full deletion of `src/codeintel/build/analytics` and
`src/codeintel/build/graphs` is de-scoped. These packages now act as the
canonical compute layer feeding the
Hamilton DAG; remaining cleanup focuses on removing unused orchestration wrappers.

### Prereqs
- [~] Port any remaining compute kernels referenced by build into
  `src/codeintel/build/hamilton/native/*` (coverage_functions now Polars-based;
  coverage compute removed; remaining kernels pending audit).

### Removal checklist
- [x] Freeze imports of `codeintel.build.analytics` and `codeintel.build.graphs`
  outside the Hamilton/native and compute packages (architecture guard added).
- [x] Remove legacy orchestration modules and DuckDB helpers
  (coverage_functions migrated; coverage compute + duckdb_helpers removed;
  unused AST features persistence wrapper removed).
- [x] Update tests that enforce legacy orchestration or export lists
  (PR-52 orchestration guard updated; public exports reviewed; no changes required).
- [x] Update docs to clarify compute packages as the canonical source layer.

### File scope (retained compute layer)
- [x] `src/codeintel/build/analytics/` (compute layer retained; prune unused wrappers).
- [x] `src/codeintel/build/graphs/` (compute layer retained; prune unused wrappers).
- [x] `tests/build/hamilton/test_pr52_no_legacy_orchestrators.py` (update).
- [x] `tests/analytics/test_public_exports.py` (reviewed; no changes required).

## Orchestration Migration Checklists (Status)
These files remain in use by the Hamilton DAG today. The checklists reflect
completed migrations vs outstanding work (runtime/engine/validation). The work
focuses on making inputs explicit, aligning I/O with the Parquet boundary, and
keeping NetworkX-based analytics where they add value. Some modules are expected
to remain hybrid orchestration layers rather than pure Hamilton nodes.

### Graph metrics orchestration (NetworkX-heavy; hybrid allowed)
#### `src/codeintel/build/analytics/graphs/orchestrator.py`
- [x] Move runtime resolution into Hamilton nodes and pass `GraphRuntime`/views in.
- [x] Remove direct `StorageGateway` reads; accept graph views or row inputs.
- [x] Keep NetworkX view building here; mark as hybrid (not fully DAG-native).

#### `src/codeintel/build/analytics/graphs/symbol_orchestrator.py`
- [x] Require symbol graph inputs from DAG (no runtime lookup inside).
- [x] Keep NetworkX coupling logic in this layer (hybrid by design).
- [x] Document expected inputs and outputs for DAG wiring.

#### `src/codeintel/build/analytics/graphs/graph_metrics.py`
- [x] Replace internal runtime resolution with DAG-provided runtime or views.
- [x] Ensure all upstream tables are provided as DAG inputs (no hidden reads).
- [x] Keep NetworkX metric functions; return rows only.

#### `src/codeintel/build/analytics/graphs/graph_metrics_ext.py`
- [x] Convert to accept graph views from DAG (no gateway inside).
- [x] Keep extended metrics in NetworkX; avoid forcing Polars for algorithms.
- [x] Add explicit input contract for required graph tables.

#### `src/codeintel/build/analytics/graphs/module_graph_metrics_ext.py`
- [x] Same migration shape as `graph_metrics_ext.py`.
- [x] Ensure module graph selection is explicit in DAG wiring.
- [x] Preserve NetworkX algorithms; return rows only.

#### `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`
- [x] Replace runtime/gateway discovery with DAG-provided symbol graphs.
- [x] Keep NetworkX graph ops; avoid forcing full DAG rewrite.
- [x] Add table lineage notes in the plan (symbol edges + goids).

- [~] Validation gates pending: resolve ruff/pyright/pyrefly issues in graph
  metrics refactor and run targeted tests.

### Config graphs + subsystem orchestration (mixed SQL/NetworkX; hybrid allowed)
#### `src/codeintel/build/analytics/graphs/config_data_flow.py`
- [x] Lift all source table reads into DAG inputs.
- [x] Keep graph construction in NetworkX if needed.
- [x] Ensure outputs are schema-aligned row models only.

#### `src/codeintel/build/analytics/graphs/config_graph_metrics.py`
- [x] Replace runtime/gateway access with DAG-provided inputs.
- [x] Split graph build vs metrics into explicit helper functions.
- [x] Preserve NetworkX metrics where beneficial.

#### `src/codeintel/build/analytics/graphs/graph_stats.py`
- [x] Convert reads to DAG inputs; avoid direct gateway lookups.
- [x] Keep NetworkX/graph stats compute as a helper.
- [x] Add explicit input/row contracts for DAG wiring.

#### `src/codeintel/build/analytics/graphs/subsystem_graph_metrics.py`
- [x] Provide subsystem + graph inputs explicitly via DAG nodes.
- [x] Retain NetworkX-based metric calculations (hybrid acceptable).
- [x] Confirm outputs map directly to Parquet materializers.

#### `src/codeintel/build/analytics/graphs/subsystem_agreement.py`
- [x] Replace gateway reads with DAG-provided subsystem/graph inputs.
- [x] Keep disagreement logic (likely NetworkX) in helper layer.
- [x] Add tests for schema/lineage only (no view registry).

- [~] Validation gates pending: ruff/pyright/pyrefly clean + targeted tests for
  config/subsystem graph migrations.

### Graph runtime + engine + validation (service layer; not fully migratable)
These modules are expected to remain as a service layer because NetworkX
graph construction and validation are not a clean fit for Hamilton DAG nodes.
Migration focuses on boundary alignment (Parquet-backed inputs, no view registry).

#### `src/codeintel/build/graphs/runtime/__init__.py`
- [ ] Ensure exports reflect the supported runtime surface.
- [ ] Update docstrings to emphasize Parquet-backed graph sources.

#### `src/codeintel/build/graphs/runtime/context.py`
- [ ] Require graph inputs from Parquet-derived tables only.
- [ ] Keep context helpers pure and NetworkX-friendly.

#### `src/codeintel/build/graphs/runtime/runtime.py`
- [ ] Build graphs from Parquet-backed DuckDB scans only.
- [ ] Centralize `graph_backend` selection into runtime options.
- [ ] Keep caching and NetworkX graph construction here (hybrid by design).

#### `src/codeintel/build/graphs/engine/backend.py`
- [ ] Align backend selection with `graph_backend` config.
- [ ] Document supported backends and hybrid nature.

#### `src/codeintel/build/graphs/engine/protocol.py`
- [ ] Confirm protocol boundaries for Parquet-backed graph loads.
- [ ] Avoid leaking view registry assumptions in the interface.

#### `src/codeintel/build/graphs/engine/cache.py`
- [ ] Ensure cache invalidation keys track Parquet metadata (repo/commit/build_id).
- [ ] Avoid view registry keys or SQLGlot view identifiers.

#### `src/codeintel/build/graphs/engine/__init__.py`
- [ ] Keep exports minimal; document hybrid service role.
- [ ] Remove any references to removed resource providers.

#### `src/codeintel/build/graphs/engine/factory.py`
- [ ] Use build config to pick backend and wire Parquet-backed loaders.
- [ ] Keep the engine factory separate from Hamilton orchestration.

#### `src/codeintel/build/graphs/engine/views.py`
- [ ] Ensure all SQL reads target Parquet-backed tables only.
- [ ] Keep NetworkX conversion logic centralized here.
- [ ] Avoid reliance on legacy view registry artifacts.

#### `src/codeintel/build/graphs/engine/nx_engine.py`
- [ ] Keep NetworkX engine as the canonical implementation.
- [ ] Confirm loaders pull from Parquet-backed tables.
- [ ] No full DAG migration expected (explicitly hybrid).

#### `src/codeintel/build/graphs/validation/findings.py`
- [ ] Ensure persistence targets Parquet-backed tables only.
- [ ] Keep output schema mapping explicit for validation rows.

#### `src/codeintel/build/graphs/validation/runner.py`
- [ ] Run validations after DAG materialization, not during view builds.
- [ ] Resolve runtime via Parquet-backed engine only.

#### `src/codeintel/build/graphs/validation/checks/database.py`
- [ ] Update checks to rely on Parquet-backed tables and metadata.
- [ ] No view registry assumptions.

#### `src/codeintel/build/graphs/validation/checks/anomaly.py`
- [ ] Keep NetworkX-based anomaly checks; accept runtime graphs as inputs.
- [ ] Document required graph variants for DAG wiring.

#### `src/codeintel/build/graphs/validation/checks/structure.py`
- [ ] Keep structural checks in NetworkX; accept runtime graphs.
- [ ] Ensure checks read from Parquet-backed sources only.

#### `src/codeintel/build/graphs/validation/checks/__init__.py`
- [ ] Keep exports aligned with remaining checks.
- [ ] Document hybrid validation layer expectations.

#### `src/codeintel/build/graphs/validation/__init__.py`
- [ ] Update exports and docstrings for Parquet-backed validation.
- [ ] No DAG migration expected for the validation runner itself.

#### `src/codeintel/build/graphs/validation/context.py`
- [ ] Ensure context is constructed from Parquet-backed runtime only.
- [ ] Keep NetworkX graph references explicit.

#### `src/codeintel/build/graphs/validation/base.py`
- [ ] Keep protocol definitions stable for hybrid validation.
- [ ] Confirm contracts reflect Parquet-backed graph sources.
