# Hamilton DAG Targets and Lineage (AST, CST, SCIP)

This document explains how the Hamilton DAG ingests AST/CST/SCIP signals,
transforms them into intermediate datasets, and produces final target outputs.
It is intended to help readers trace lineage from raw code structure to
analytics tables and graph outputs.

## Scope and conventions

- Focus: Hamilton targets under `src/codeintel/build` and their materialized
  outputs (tables and artifacts).
- Lineage naming: inputs are shown as table keys (e.g. `core.ast_nodes`).
- AST refers to stdlib AST extraction; CST refers to LibCST extraction; SCIP
  refers to SCIP indexing tables and artifact.

## High-level pipeline stages

1) Repository scan and inventory
2) AST/CST and syntax extraction (LibCST and tree-sitter)
3) SCIP indexing and resolution
4) Graph construction (goids, call/import graphs, CFG/DFG/CDG/PDG, CPG)
5) Analytics outputs (functions, subsystems, graph metrics, validation)
6) Export/serving artifacts (optional targets, not core lineage)

## Stage 1: Repository scan and inventory

Targets and outputs:
- `modules` -> `core.modules`, `core.file_state`, `core.repo_map`
  - Inputs: repository scan; provides module paths and language metadata.
- `config_ingest` -> `analytics.config_values`
  - Inputs: discovered config files.
- `tests_ingest` -> `analytics.test_catalog`
  - Inputs: pytest report data.
- `typing` -> `analytics.static_diagnostics`
  - Inputs: type checking diagnostics.

These tables are foundational for path-to-module mapping and for subsequent
analytics (subsystems, entrypoints, dependency analysis).

## Stage 2: AST/CST and syntax extraction

Targets and outputs:
- `ast` -> `core.ast_nodes`, `core.ast_metrics`
  - Inputs: module paths; stdlib AST parsing.
- `cst` -> `core.cst_nodes`
  - Inputs: module paths; LibCST parsing.
- `docstrings` -> `core.docstrings`
  - Inputs: module paths; docstring extraction.
- `syntax_index` ->
  - `core.parse_manifest`
  - `core.syntax_spans`
  - `core.syntax_nodes`
  - `core.syntax_edges`
  - `core.syntax_scopes`
  - `core.syntax_defs`
  - `core.syntax_refs`
  - `core.syntax_calls`
  - `core.syntax_call_args`
  - `core.syntax_func_params`
  - `core.syntax_imports`
  - Inputs: module paths; LibCST parse manifest + syntax fact tables.
- `symtable` ->
  - `core.py_sym_scopes`
  - `core.py_sym_symbols`
  - `core.py_sym_scope_edges`
  - `core.py_sym_namespace_edges`
  - `core.py_sym_function_partitions`
  - `core.py_sym_bindings`
  - `core.py_sym_resolution_edges`
  - Inputs: module paths; CPython symtable extraction.
- `bytecode` ->
  - `core.py_bc_code_units`
  - `core.py_bc_instructions`
  - `core.py_bc_exception_table`
  - `core.py_bc_blocks`
  - `core.py_bc_cfg_edges`
  - `core.py_bc_defuse_events`
  - `core.py_compiler_metadata`
  - Inputs: module paths; CPython disassembly.
- `inspect` ->
  - `core.py_inspect_objects`
  - `core.py_inspect_members_static`
  - `core.py_inspect_class_mro`
  - `core.py_inspect_class_attrs`
  - `core.py_inspect_unwrap_hops`
  - `core.py_inspect_signatures`
  - `core.py_inspect_signature_params`
  - `core.py_inspect_annotations_kv`
  - `core.py_inspect_source`
  - `core.py_inspect_runtime_state`
  - Inputs: optional runtime inspection overlays.
- `tree_sitter_index` ->
  - `core.ts_parse_manifest`
  - `core.ts_captures`
  - `core.ts_nodes`
  - `core.ts_edges`
  - `core.ts_parse_errors`
  - `core.ts_changed_ranges`
  - `core.ts_tokens`
  - `core.ts_trivia`
  - `core.ts_language_metadata`
  - Inputs: tree-sitter query pack over module sources.

Notes:
- CST (`core.cst_nodes`) is a direct LibCST extraction. The `syntax_index`
  target uses LibCST independently to build the canonical `core.syntax_*` tables.
- AST (`core.ast_nodes`) is the primary input for `core.goids`, CFG/DFG, and
  several analytics that require function ASTs.

## Stage 3: SCIP indexing and resolution

Targets and outputs:
- `scip_proto` -> artifact `scip_pb2` at `{scip_dir}/proto/scip_pb2.py`
  - Inputs: `scip.proto` and tooling.
- `scip` -> artifact `scip_index` at `{scip_dir}/index.scip` and tables:
  - `core.scip_symbols`
  - `core.scip_occurrences`
  - `core.scip_symbol_information`
  - `core.scip_symbol_relationships`
  - `core.scip_diagnostics`
  - `core.scip_external_symbols`
  - `core.scip_module_state`
  - Inputs: module inventory and SCIP tool execution.
- `file_line_index` -> `core.file_line_index`
  - Inputs: module inventory (line mapping for files).
- `scip_resolution` ->
  - `core.scip_symbol_goid_xref`
  - `core.scip_occurrence_span_xref`
  - `core.scip_occurrence_syntax_xref`
  - Inputs: `core.scip_occurrences`, `core.scip_symbol_information`,
    `core.goids`, and `core.syntax_nodes` (span alignment).
- `syntax_enrich` ->
  - `core.syntax_defs_resolved`
  - `core.syntax_refs_resolved`
  - `core.syntax_calls_resolved`
  - `core.syntax_imports_resolved`
  - Inputs: `core.syntax_*` + `core.scip_occurrence_*_xref` tables.
- `syntax_augment` ->
  - `core.syntax_nodes_augmented`
  - `core.syntax_edges_augmented`
  - `core.ts_syntax_node_xref`
  - `core.ts_weld_coverage`
  - Inputs: `core.syntax_nodes`, `core.syntax_edges`, `core.ts_nodes`,
    `core.ts_edges`, `core.parse_manifest`.

## Stage 4: Graph construction targets

### GOID and symbol alignment

- `goids` -> `core.goids`, `core.goid_crosswalk`
  - Inputs: `core.modules`, `core.ast_nodes`.
- `symbol_uses` -> `graph.symbol_use_edges`
  - Inputs: `core.scip_occurrences`, `core.modules`, `core.goids`.

### Import and call graphs

- `import_graph` -> `graph.import_modules`, `graph.import_graph_edges`
  - Inputs: `core.modules` (AST parsing for import discovery).
- `call_graph` -> `graph.call_graph_nodes`, `graph.call_graph_edges`
  - Inputs: `core.modules`, `core.goids` (AST parsing for calls).

### CFG/DFG/CDG/PDG

- `cfg` -> `graph.cfg_blocks`, `graph.cfg_edges`
  - Inputs: `core.goids`, `core.ast_nodes`.
- `dfg` -> `graph.dfg_edges`
  - Inputs: `core.goids`, `core.ast_nodes`.
- `cdg` -> `graph.cdg_edges`
  - Inputs: `graph.cfg_blocks`, `graph.cfg_edges`.
- `pdg` -> `graph.pdg_edges`
  - Inputs: `graph.dfg_edges`, `graph.cdg_edges`.

### Call wiring (CPG auxiliary edges)

- `call_wiring` ->
  - `graph.cpg_call_targets`
  - `graph.cpg_edges_calls`
  - `graph.cpg_edges_arg_to_param`
  - `graph.cpg_edges_ret_to_call`
  - Inputs:
    - `core.syntax_calls`
    - `core.syntax_defs_resolved`
    - `core.syntax_nodes`
    - `core.syntax_call_args`
    - `core.syntax_func_params`
    - `core.scip_occurrence_span_xref`
    - `graph.cfg_blocks`

### CPG (code property graph)

- `cpg` -> `graph.cpg_nodes`, `graph.cpg_edges`
  - Inputs (grouped):
    - AST and syntax: `core.ast_nodes`, `core.syntax_nodes`, `core.syntax_calls`,
      `core.syntax_call_args`
    - SCIP: `core.scip_symbol_information`, `core.scip_symbol_relationships`,
      `core.scip_occurrence_span_xref`, `core.scip_occurrence_syntax_xref`,
      `core.scip_symbol_goid_xref`
    - GOIDs and graphs: `core.goids`, `graph.cfg_blocks`, `graph.cfg_edges`,
      `graph.dfg_edges`, `graph.cdg_edges`, `graph.call_graph_edges`,
      `graph.import_graph_edges`, `graph.cpg_edges_calls`,
      `graph.cpg_edges_arg_to_param`, `graph.cpg_edges_ret_to_call`
    - Symtable and bytecode: `core.py_sym_scopes`, `core.py_sym_bindings`,
      `core.py_sym_scope_edges`, `core.py_sym_namespace_edges`,
      `core.py_sym_resolution_edges`, `core.py_bc_code_units`,
      `core.py_bc_instructions`, `core.py_bc_blocks`, `core.py_bc_cfg_edges`,
      `core.py_bc_defuse_events`
    - Inspect overlays: `core.py_inspect_objects`, `core.py_inspect_class_mro`,
      `core.py_inspect_class_attrs`, `core.py_inspect_unwrap_hops`,
      `core.py_inspect_signatures`, `core.py_inspect_signature_params`,
      `core.py_inspect_source`, `core.py_inspect_runtime_state`
    - Tree-sitter tokenization: `core.ts_tokens`, `core.ts_trivia`

## Stage 5: Analytics targets

### Function analytics

- `function_analysis` -> in-memory `FunctionAnalyticsResult`
  - Inputs: `core.goids`.
- `function_types` -> `analytics.function_types`
  - Inputs: `FunctionAnalyticsResult`.
- `function_validation` -> `analytics.function_validation`
  - Inputs: `FunctionAnalyticsResult`.
- `function_ast_features` -> `analytics.function_ast_features`
  - Inputs: `core.goids`, `core.modules` (AST parsing per module).
- `function_effects` -> `analytics.function_effects`
  - Inputs: `core.goids`, `core.modules`, `graph.call_graph_edges`,
    `graph.call_graph_nodes`.
- `function_contracts` -> `analytics.function_contracts`
  - Inputs: `core.docstrings`, `analytics.function_types`, `core.goids`,
    `core.modules` (AST loading).
- `external_deps` ->
  - `analytics.external_dependency_calls`
  - `analytics.external_dependencies`
  - Inputs: `core.modules`, `analytics.function_ast_features`, `core.goids`,
    `analytics.config_values` (AST loading).

### Entrypoints and data models

- `entrypoints` -> `analytics.entrypoints`, `analytics.entrypoint_tests`
  - Inputs: `core.modules`, `core.goids`, `analytics.function_ast_features`,
    `analytics.test_catalog`, `analytics.subsystems`, `analytics.subsystem_modules`.
- `data_models` ->
  - `analytics.data_models`
  - `analytics.data_model_fields`
  - `analytics.data_model_relationships`
  - Inputs: `core.modules`, `core.goids`, `core.docstrings`.
- `data_model_usage` -> `analytics.data_model_usage`
  - Inputs: `analytics.data_models`, `core.modules`, `core.goids`,
    `analytics.subsystems`, `analytics.subsystem_modules`,
    `analytics.function_types`.

### Subsystems

- `subsystems` -> `analytics.subsystems`, `analytics.subsystem_modules`
  - Inputs: `core.modules`, `graph.import_graph_edges`, `graph.symbol_use_edges`,
    `analytics.config_values`.
- `subsystem_graph_metrics` -> `analytics.subsystem_graph_metrics`
  - Inputs: `analytics.subsystem_modules`, `graph.import_graph_edges`,
    `graph.import_modules`.
- `subsystem_caches` -> `analytics.subsystem_profile_cache`
  - Inputs: `analytics.subsystems`, `analytics.subsystem_graph_metrics`.
- `subsystem_agreement` -> `analytics.subsystem_agreement`
  - Inputs: `analytics.subsystem_modules`, `analytics.graph_metrics_modules_ext`.

### Graph metrics and validation

- `graph_metrics` ->
  - `analytics.graph_metrics_functions`
  - `analytics.graph_metrics_modules`
  - `analytics.graph_metrics_functions_ext`
  - `analytics.graph_metrics_modules_ext`
  - `analytics.symbol_graph_metrics_functions`
  - `analytics.symbol_graph_metrics_modules`
  - `analytics.graph_stats`
  - Inputs: `core.goids`, `core.modules`, `graph.call_graph_edges`,
    `graph.call_graph_nodes`, `graph.import_graph_edges`, `graph.import_modules`,
    `graph.symbol_use_edges`, `analytics.subsystem_modules`,
    `analytics.config_values`.
- `cfg_dfg_metrics` ->
  - `analytics.cfg_function_metrics`
  - `analytics.cfg_block_metrics`
  - `analytics.cfg_function_metrics_ext`
  - `analytics.dfg_function_metrics`
  - `analytics.dfg_block_metrics`
  - `analytics.dfg_function_metrics_ext`
  - Inputs: `graph.cfg_blocks`, `graph.cfg_edges`, `graph.dfg_edges`,
    `core.goids`, `core.modules`.
- `graph_validation` -> `analytics.graph_validation`
  - Inputs: dataset snapshot; validates existing graph tables.

### Semantic roles

- `semantic_roles` ->
  - `analytics.semantic_roles_functions`
  - `analytics.semantic_roles_modules`
  - Inputs: `core.modules`, `core.goids`, `analytics.function_ast_features`,
    `analytics.function_effects`, `analytics.function_contracts`,
    `analytics.graph_metrics_functions`.

### CPG quality

- `py_cpg_quality_report` -> `analytics.py_cpg_quality_report`
  - Inputs: `core.py_bc_instructions`, `core.py_sym_scopes`, `core.py_bc_blocks`,
    `core.py_inspect_objects`, `core.py_bc_cfg_edges`, `core.py_bc_defuse_events`,
    `graph.cpg_edges`.

## AST, CST, and SCIP lineage maps

### AST lineage

Primary AST dataset: `core.ast_nodes`.

Downstream consumers:
- `core.goids` and `core.goid_crosswalk` (GOID assignment).
- `graph.cfg_blocks`, `graph.cfg_edges`, `graph.dfg_edges` (CFG/DFG analysis).
- `graph.call_graph_nodes`, `graph.call_graph_edges` (call graph extraction).
- `graph.cpg_nodes` / `graph.cpg_edges` (core nodes and edges).
- `analytics.function_ast_features` (function feature extraction from AST).
- `analytics.function_effects` / `analytics.function_contracts` (AST loading by GOID).
- `analytics.data_models` / `analytics.data_model_fields` / `analytics.data_model_relationships`
  (AST parsing and docstring alignment).
- `analytics.external_dependency_calls` (AST-driven dependency extraction).

### CST lineage

Primary CST dataset: `core.cst_nodes` (LibCST extraction).

Downstream consumers:
- The DAG does not currently use `core.cst_nodes` as a required input for
  downstream tables. The canonical syntax tables come from `syntax_index`.
- `syntax_index` (LibCST) produces `core.syntax_*` tables used by
  `syntax_enrich`, `call_wiring`, and `cpg`.

### SCIP lineage

Primary SCIP datasets: `core.scip_*` and the `scip_index` artifact.

Downstream consumers:
- `scip_resolution` joins SCIP occurrences with GOIDs to produce xref tables.
- `syntax_enrich` welds SCIP occurrences onto syntax facts to produce resolved
  defs/refs/calls/imports tables.
- `graph.symbol_use_edges` uses `core.scip_occurrences` + GOIDs to link
  definitions to uses across modules.
- `graph.cpg_edges` and `graph.cpg_nodes` consume SCIP symbol information and
  relationship tables, plus occurrence xrefs.
- `call_wiring` uses resolved syntax tables + occurrence xrefs to wire call
  edges and argument-to-parameter edges.

## Notes and practical tracing tips

- Most analytics targets that reference AST content do so indirectly via GOID
  and module tables, then load ASTs using `FunctionAstLoadRequest`.
- `core.syntax_*` tables are the canonical per-node syntax facts (LibCST).
  Tree-sitter is used to weld or augment syntax nodes in `syntax_augment`.
- SCIP resolution tables (`core.scip_occurrence_*_xref`) are the bridge between
  SCIP symbols and syntax nodes/spans and are required for semantic wiring.
- Graph targets feed analytics targets heavily; if an analytics table is empty,
  verify its upstream graph inputs first.
