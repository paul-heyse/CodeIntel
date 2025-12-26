# DAG End-to-End Pilot Plan (Straightforward Outputs)

## Purpose
Establish a full, end-to-end Apache Hamilton DAG path for the simplest outputs first.
The intent is to create a repeatable blueprint that validates the DAG-first basis across
build, storage, CLI, and serving, then reuse the pattern for more complex outputs.

This plan complements and draws on the shared library patterns in
`docs/full_dag_basis_implementation_plan.md`, but is scoped to a single, low-risk pilot
that proves the system end to end.

## Guiding principles
- DAG outputs are the source of truth for target materializations.
- Targets are defined by `t__*` anchors and DataSaver tags, not ad hoc logic.
- Contracts and schemas are first-class and drive support node generation.
- Build produces typed artifacts and records; serving consumes those artifacts only.
- CLI behavior and flags should derive from DAG metadata, not bespoke wiring.

## Scope
In scope:
- One straightforward output (and its minimal upstream support nodes).
- End-to-end path: Hamilton DAG -> materialization -> storage -> serving -> CLI.
- Contract alignment, schema materialization, and validation behavior.
- Observability and minimal diagnostics needed to validate the path.

Out of scope:
- Migration of all outputs or all target families.
- Performance tuning beyond what is required to validate the path.
- Refactoring of unrelated CLI or serving behavior.

## Selection criteria for the pilot output
Pick the simplest output that still exercises the full pipeline:
- Table-only or artifact-only materialization (no mixed or multi-output).
- No external services or network dependencies.
- Stable contract with clear schema and low cardinality.
- Minimal downstream fan-out.
- Used by at least one existing CLI or serving path (to validate end to end).

## Output inventory artifact template (YAML/JSON)
Use a lightweight artifact to keep output discovery in sync across build, storage,
CLI, and serving.

YAML (preferred for humans):
```yaml
version: 1
generated_at: "YYYY-MM-DD"
outputs:
  - target: "function_metrics"
    domain: "analytics"
    anchor: "t__function_metrics"
    materialization: "table"
    table_keys:
      - "analytics.function_metrics"
      - "analytics.function_types"
      - "analytics.function_validation"
    contracts:
      - "analytics.function_metrics"
      - "analytics.function_types"
      - "analytics.function_validation"
    downstream_consumers:
      - "storage.views"
      - "serving.search_index"
      - "cli.datasets"
    upstream_targets:
      - "modules"
      - "goids"
    notes: "Core analytics metric set used across serving and CLI."
```

JSON (machine-first alternative):
```json
{
  "version": 1,
  "generated_at": "YYYY-MM-DD",
  "outputs": [
    {
      "target": "function_metrics",
      "domain": "analytics",
      "anchor": "t__function_metrics",
      "materialization": "table",
      "table_keys": [
        "analytics.function_metrics",
        "analytics.function_types",
        "analytics.function_validation"
      ],
      "contracts": [
        "analytics.function_metrics",
        "analytics.function_types",
        "analytics.function_validation"
      ],
      "downstream_consumers": [
        "storage.views",
        "serving.search_index",
        "cli.datasets"
      ],
      "upstream_targets": [
        "modules",
        "goids"
      ],
      "notes": "Core analytics metric set used across serving and CLI."
    }
  ]
}
```
Canonical inventory artifact: `src/codeintel/core/registry/dag_output_inventory.yaml`.

Notes on schema behavior:
- `materialization` is validated to be one of: table, artifact, mixed.
- table outputs must declare `table_keys`; artifact outputs must not.
- `contracts` defaults to `table_keys` when omitted for table outputs.
- `pilot` is an optional boolean flag for highlighting the current focus output.

## Implementation insights (from initial inventory and CLI wiring)
- The shared-flags mechanism for Command[T] commands must be explicitly flattened so Cyclopts
  can parse `--output-format` and `--json` for new command groups.
- CLI output format resolution is driven by shared flags; new commands should keep the `flags`
  field and avoid ad hoc output-format options unless the command has a distinct format.
- Inventory loading should be centralized in `RegistryService` so build, CLI, and serving share
  a single source of truth for target metadata.
- Inventory validation must fail fast on duplicate targets and mismatched materialization rules
  to avoid downstream schema mismatches.
- Tooling preflight should live alongside registry inventory loading to keep CLI and build
  resolution consistent; use runtime-resolved ToolsConfig when available and fall back to
  defaults when not.
- Tool resolution should distinguish missing tools (configured but not found) from unrecognized
  tool names (not in the ToolName enum) to keep diagnostics actionable.

## Phase 0: Inventory and candidate selection
Deliverables:
- Output inventory artifact (YAML or JSON) listing:
  - target name
  - DAG anchor node name
  - materialization type (table/artifact/mixed)
  - contract/schema references
  - downstream consumers (CLI, serving, docs)
  - current materialization path (if any)
- Ranked candidate list with rationale tied to the criteria above:
  - `analytics.function_metrics` (`t__function_metrics`)
    - Table-only output (3 tables), no external services, stable contracts.
    - Heavily used by storage views, serving search index, and CLI datasets.
    - Already uses shared save patterns, so it is a clean DAG-first baseline.
  - `analytics.function_ast_features` (`t__function_ast_features`)
    - Single table, simple compute path, no external services.
    - Lighter downstream use, useful as a secondary validation target.
  - `analytics.coverage_functions` (`t__coverage_functions`)
    - Table-only with Ibis materialization, depends on coverage_lines ingest.
    - Representative of Ibis table saver path, but higher upstream coupling.
  - `analytics.data_models` (`t__data_models`)
    - Multi-table output (models/fields/relationships), more complex but internal.
    - Good follow-on once pilot patterns are validated.
- Selected pilot output:
  - `analytics.function_metrics` (`t__function_metrics`)
    - Most representative without external dependencies.
    - End-to-end coverage: build -> storage -> serving -> CLI paths already exist.

Acceptance criteria:
- Inventory is checked in and reviewed.
- Pilot output selection is documented and agreed.
Status:
- Completed: inventory artifact created at `src/codeintel/core/registry/dag_output_inventory.yaml`
  with analytics, graphs, ingestion, and export targets.
- Completed: pilot output remains `analytics.function_metrics` (`t__function_metrics`).

## Phase 1: DAG-first build path for the pilot
Tasks:
- Ensure the pilot output is produced via a `t__*` anchor.
- Use shared pattern helpers (`native/patterns/*`) for materialization records:
  - shared hash node
  - `record_from_materializations(...)`
  - target run record structure
- Ensure support nodes use contract-driven inputs when possible.
- Remove any direct env/gateway access from compute nodes for this output.
- Add or align dataset contract/schema for the pilot output.

Deliverables:
- DAG definition for the pilot output is canonical and uses shared patterns.
- Dataset contract and schema are aligned with the materialized output.
- Target run record includes all required metadata for downstream use.

Acceptance criteria:
- `t__*` anchor exists and uses shared record builder.
- Contract validation passes in lenient mode; strict mode available in CI.

## Phase 2: Storage and serving integration
Tasks:
- Ensure storage registration for the pilot output uses shared services.
- Confirm the serving layer reads only from published artifacts/tables.
- Add or update storage repository methods (if needed) for the pilot output.
- Update serving snapshot or materialization pipeline to reference the pilot output.

Deliverables:
- Storage uses shared boundary services (no direct DuckDB use outside storage).
- Serving snapshot references the pilot output via storage contracts.

Acceptance criteria:
- Serving path works with only the materialized output present.
- No direct access to build internals from serving for the pilot output.

## Phase 3: CLI and UX alignment
Tasks:
- Ensure CLI commands referencing the pilot output are DAG-driven.
- Derive CLI help and flags from shared metadata (where feasible).
- Add or update CLI validation and messaging aligned with new contracts.

Deliverables:
- CLI can read or export the pilot output with stable help/flags.
- CLI uses validation mode controls consistently.

Acceptance criteria:
- CLI snapshot tests for the pilot output are stable.
- CLI exit codes and errors reflect validation mode behavior.

Recent updates:
- Added `registry` CLI group with `registry outputs` and `registry validate` to expose the
  inventory and filtering via shared output formatting.
- Registered the command group in the root CLI app so it is available in the standard entrypoint.
- Added options for registry filters (domain, materialization, targets, table_key, pilot_only).
- Build decommission cleanup complete:
  - Output inventory and target catalog caches removed; DAG-derived outputs and target graph are
    the only sources of truth.
  - Contract resolution now defaults to the enriched, DAG-backed service; declared-only mode is
    removed.
  - Support nodes are always derived from DAG saver tags (no runtime toggle).
  - ExecutionResult compatibility shim removed; graph targets now emit ExecutionResult directly or
    use explicit conversions where intermediate results are required.

## Phase 4: End-to-end tests and artifacts
Tasks:
- Add a focused integration test for the pilot output path.
- Capture golden outputs or fixtures for serving validation.
- Update targeted tests where contracts or schema changes occur.

Deliverables:
- Pilot end-to-end test with deterministic artifacts.
- Updated goldens/fixtures for the pilot output.

Acceptance criteria:
- Pilot test passes in isolation and in CI subsets.
- No regressions in related analytics/build/serving tests.

## Phase 5: Replication checklist for next outputs
Create a short checklist that can be applied to the next output:
- Anchor `t__*` node exists and uses shared materialization patterns.
- Contract/schema aligned and referenced by support nodes.
- Storage/serving boundary uses shared services only.
- CLI entry points resolve via DAG metadata.
- Tests and goldens updated for the output.

Deliverable:
- A reusable migration checklist with concrete "done" criteria.

### Phase 5A: Simple follow-on target (function_ast_features)
Target:
- `analytics.function_ast_features` (`t__function_ast_features`)

Checklist:
- Inventory metadata is complete and accurate (table key, contract, upstream targets).
- DAG nodes use shared materialization helpers; no direct gateway access in compute nodes.
- Contract/schema registry includes `analytics.function_ast_features` with consistent names.
- Storage read path uses shared storage services (no direct DuckDB use outside storage).
- CLI selection includes the target via metadata-driven paths (`build targets`, `build plan`).
- Add a focused harness test validating row count + schema (single table).

Acceptance criteria:
- `t__function_ast_features` runs from the harness and materializes the table.
- Schema validation passes for `analytics.function_ast_features`.
- CLI output listing includes the target and honors filters.

### Phase 5B: Complex follow-on target (profiles or data_models)
Candidates:
- `analytics.profiles` (`t__profiles`)
- `analytics.data_models` (`t__data_models`)

Checklist:
- Inventory metadata includes all table keys, contracts, and downstream consumers.
- Contract/schema coverage is complete for all output tables in the target.
- DAG nodes separate pure compute from I/O; materialization uses shared patterns.
- Storage repositories (if any) use shared services and validate contracts.
- Serving dependencies are explicitly listed; no build-internal access from serving.
- CLI output flags and error handling align with validation mode rules.
- Add an integration test validating multi-table writes and downstream usage.

Acceptance criteria:
- Target runs end-to-end with all tables materialized and recorded.
- Downstream consumers (serving or CLI) operate with only materialized outputs.
- Contract validation behavior is stable in lenient and strict modes.

## Ingestion replication pattern and implementation plan
Ingestion targets are heterogeneous (filesystem scan, parser extractors, and external tools),
but we can keep them DAG-first by standardizing on the tool-target pattern, shared
materialization helpers, and contract-driven schemas.

### Ingestion scope inventory (current targets)
- modules: core.modules, core.file_state, core.repo_map (foundation for downstream ingestion).
- config_ingest: analytics.config_values.
- coverage_ingest: analytics.coverage_lines.
- tests_ingest: analytics.test_catalog.
- typing: analytics.typedness, analytics.static_diagnostics.
- ast: core.ast_nodes, core.ast_metrics.
- cst: core.cst_nodes.
- docstrings: core.docstrings.
- scip_proto: artifact scip_pb2 (generated scip_pb2.py).
- scip: artifact scip_index plus core.scip_symbols, core.scip_occurrences,
  core.scip_symbol_information, core.scip_symbol_relationships, core.scip_diagnostics,
  core.scip_external_symbols, core.scip_module_state.

### Ingestion tooling inventory artifact
Maintain tool dependencies separately from output inventory so build, CLI, and tests
have a single source of truth for tool availability, config keys, and install hints.

Canonical artifact: `src/codeintel/core/registry/ingestion_tooling_inventory.yaml`.

Template:
```yaml
version: 1
generated_at: "YYYY-MM-DD"
tools:
  - tool_name: "scip-python"
    display_name: "scip-python"
    kind: "binary"
    cli: "scip"
    config_key: "scip_python_bin"
    required_by:
      - "ingestion.scip"
    packages:
      - "scip-python"
    version_probe: "scip --version"
    notes: "CLI entrypoint provided by scip-python package."
```
Notes:
- Loader validates tool kinds, detects duplicates, and normalizes display fields.
- CLI preflight (`registry tools`) reports status: available, missing, unrecognized.

### Standard ingestion target template (DAG-first)
1) Inputs: module inventory, scan profile, tool options, and hash options.
2) Tool step: run_tool_step/run_tool_and_ingest with ToolRunContext (skip via manifest plus
   options hash).
3) Ingest step: pure transformation from tool output to row payloads (IngestStep[Payload]).
4) Materialization: save_rows for tables and save_artifact for tool outputs.
5) Anchor: finalize_target_from_materializations with record_from_materializations and
   change_delta when available.

Notes:
- Pure Python steps (repo scan, AST/CST extraction) still fit the tool-step template; treat the
  compute step as the "tool" so skip and error handling is consistent.
- Multi-table outputs should use TableSaveSpec in tool_target to eliminate per-table decorators.
- Artifact outputs should be declared once and reused by downstream nodes (avoid duplicate
  path logic).

### Supporting functionality to fold into the standard framework
- Tool registry and availability: canonical tool names (use scip-python CLI consistently),
  preflight availability checks, and explicit failure reasons when tools are missing.
- Hash and skip semantics: use InputHashOptions (file state plus options hash) for all ingestion
  targets; avoid ad hoc skip logic.
- Shared module inventory nodes: use module_paths and module_records as the single source for
  module lists; avoid filesystem scans in downstream targets.
- Uniform artifact paths: use saver helpers to resolve paths under the build artifact root and
  record them in TargetRunRecord.
- Contract-first row building: row serializers and schema services drive output column sets and
  validation behavior (lenient default, strict in CI).

### Recommended migration order (simple to complex)
1) Foundation: modules (enables consistent inputs for all other ingestion targets).
2) Simple pilot: docstrings (single table, pure Python, minimal dependencies).
3) Parser extractors: ast then cst (multi-table then single table).
4) Multi-table ingestion: coverage_ingest, tests_ingest, typing (file-based inputs, diagnostics).
5) External tool: scip_proto then scip (artifact plus multi-table, external tool execution).

### Phase I: Ingestion metadata and prerequisites
Tasks:
- Ensure ingestion outputs are complete in the inventory and flagged with upstream targets.
- Document tool dependencies per target in the ingestion tooling inventory artifact.
- Confirm contract/schema entries exist for each ingestion table.

Deliverables:
- Inventory entries for all ingestion targets with correct table keys and anchor names.
- Tool dependency list for ingestion targets with config keys and CLI entrypoints.
- Ingestion tooling inventory artifact checked in at the canonical path.

Acceptance criteria:
- Inventory validation passes.
- No ingestion target references missing contracts or schemas.

### Phase II: Core ingestion helpers aligned to tool_target
Tasks:
- Replace per-target SaveToObjectMetadataDecorator usage with save_rows/save_artifact and
  table or artifact collectors from native/patterns/tool_target.py.
- Normalize tool execution and ingest steps to return ToolStepOutput/IngestStep.
- Standardize skip logic via ToolRunContext plus InputHashOptions.

Deliverables:
- Shared helper nodes for inputs (options, hash, module records) reused across ingestion targets.
- Updated ingestion targets emit TargetRunRecord via finalize_target_from_materializations.

Acceptance criteria:
- Ingestion targets produce materializations via shared savers only.
- Skip behavior is consistent across ingestion targets.

### Phase III: Simple ingestion pilot migration
Tasks:
- Migrate docstrings to the tool_target template.
- Add a focused harness test validating table materialization and contract compliance.
- Validate CLI listing and inventory alignment for the target.

Deliverables:
- Target anchor uses tool_target patterns end-to-end.
- Test proves output rows and schema for the pilot.

Acceptance criteria:
- Pilot target runs via build harness with deterministic output.
- Contract validation passes in lenient mode for the pilot.

### Phase IV: Parser extractors and multi-table ingestion
Tasks:
- Migrate ast and cst to the shared tool_target template.
- Migrate coverage_ingest, tests_ingest, typing with consistent tool and ingest steps.
- Ensure diagnostics and warnings are surfaced via ExecutionResult warnings.

Deliverables:
- AST/CST and ingestion targets fully aligned with shared materialization helpers.
- Consistent table counts and materialization metadata in TargetRunRecord.

Acceptance criteria:
- Harness tests validate each target individually.
- No direct filesystem scanning outside shared discovery helpers.

### Phase V: External tool ingestion (scip_proto, scip)
Tasks:
- Align scip_proto with tool_target artifact saver patterns.
- Align scip tool execution to the tool registry with canonical scip-python CLI naming.
- Ensure artifact and table materializations are recorded via shared collectors.
- Capture tool telemetry and per-table row counts in TargetRunRecord metadata.

Deliverables:
- scip_proto and scip targets use the same run/ingest/finalize pattern.
- Tool availability errors are explicit and actionable.

Acceptance criteria:
- scip targets run end-to-end when the tool is available.
- Missing tool results in a clear, deterministic failure or skip as configured.

### Phase VI: Integration and replication checklist
Tasks:
- Add a small ingestion end-to-end test that runs modules -> docstrings (or config_ingest).
- Update CLI help or registry output if ingestion metadata is exposed.
- Refresh goldens or fixtures if ingestion schema changes occur.

Replication checklist for each ingestion target:
- Anchor t__* node uses tool_target patterns and record_from_materializations.
- Inputs derive from module inventory or explicit tool options nodes.
- Materializations use save_rows/save_artifact with schema-driven columns.
- TargetRunRecord includes tool metadata and table counts when relevant.
- Harness test exists and validates contract compliance.

## Graphs replication pattern and implementation plan
Graph targets should be fully DAG-first, fed only by upstream ingestion outputs and
graph-specific helpers. The goal is to standardize materialization patterns and
eliminate direct gateway or filesystem access inside compute nodes.

### Graphs scope inventory (current targets)
- goids: core.goids, core.goid_crosswalk (foundation for all graph outputs).
- import_graph: graph.import_modules, graph.import_graph_edges.
- call_graph: graph.call_graph_nodes, graph.call_graph_edges.
- call_graph_views: graph.v_function_call_counts, graph.v_call_depth_stats.
- graph_metrics: analytics.graph_metrics_functions, analytics.graph_metrics_functions_ext,
  analytics.graph_metrics_modules, analytics.graph_metrics_modules_ext, analytics.graph_stats.
- cfg: graph.cfg_blocks, graph.cfg_edges.
- dfg: graph.dfg_edges.
- graph_validation: analytics.graph_validation.
- symbol_uses: graph.symbol_use_edges.

### Standard graph target template (DAG-first)
1) Inputs: module inventory, goid map, or SCIP occurrences via explicit helper nodes.
2) Compute: pure graph construction/transforms that return row payloads or domain objects.
3) Ingest: wrap outputs as IngestStep payloads with ExecutionResult warnings/errors.
4) Materialization: save_rows with TableSaveSpec for each table key.
5) Anchor: finalize_target_from_materializations with record_from_materializations.

Notes:
- Avoid direct env.gateway usage in compute nodes; read inputs via helper nodes.
- Prefer shared graph helpers for node/edge normalization and row serialization.
- Validation nodes (graph_validation) should consume only materialized graph outputs.

### Recommended migration order (simple to complex)
1) Foundation: goids (core.goids, core.goid_crosswalk).
2) Simple pilot: import_graph (single module-level graph, low fan-out).
3) Complex pilot: call_graph + call_graph_views + graph_metrics (multi-table, views).
4) Dataflow: cfg then dfg then graph_validation (dependent chain).
5) External tool dependency: symbol_uses (depends on SCIP occurrences).

### Phase I: Graph metadata and dependencies
Tasks:
- Confirm output inventory entries for graph targets are complete and correct.
- Confirm dataset contracts and schemas exist for all graph tables.
- Document upstream target dependencies for each graph output (modules, goids, scip).

Deliverables:
- Inventory entries for all graph targets with table keys and anchors.
- Contract coverage for graph tables with consistent column sets.

Acceptance criteria:
- Inventory validation passes for graph targets.
- No graph target references a missing contract/schema.

File-level breakdown (Phase I):
- `src/codeintel/core/registry/dag_output_inventory.yaml`: verify graph targets, table_keys,
  upstream_targets, and downstream_consumers.
- `src/codeintel/core/schemas/generated_rows/graph.py`: confirm row models align with contract
  columns for all graph.* tables and views.
- `src/codeintel/build/hamilton/contracts/schemas/pandera_schemas.py`: verify graph table
  schemas, view schemas, and descriptions are complete.
- `src/codeintel/build/hamilton/native/options/graphs.py`: ensure graph option models cover
  scope filters and defaults used by new DAG-first nodes.

### Phase II: Core graph helpers aligned to shared patterns
Tasks:
- Centralize graph input helpers (module inventory, goid map, scip occurrences).
- Ensure compute nodes are pure and return row payloads or domain objects only.
- Standardize materialization with save_rows and TableSaveSpec per table key.

Deliverables:
- Shared helper nodes reused across graph targets.
- Graph targets emit TargetRunRecord via finalize_target_from_materializations.

Acceptance criteria:
- No direct gateway access in graph compute nodes.
- Table materializations use shared saver helpers only.

File-level breakdown (Phase II):
- `src/codeintel/build/hamilton/native/patterns/tool_target.py`: confirm shared helpers are
  sufficient (IngestStep, ToolRunContext, save_rows, TableSaveSpec).
- `src/codeintel/build/hamilton/native/graphs/graph_targets.py`: remove `gateway` helper,
  replace SaveToObjectMetadataDecorator with save_rows + TableSaveSpec for goids, symbol_uses,
  call_graph_views, graph_metrics, graph_validation; add hash/finalize helpers.
- `src/codeintel/build/hamilton/native/graphs/graph_targets.py`: add shared helper nodes for
  module inventory and goid maps (or move into a new helpers module).
- `src/codeintel/build/hamilton/native/graphs/__init__.py`: update exports to include new
  run/ingest nodes and helper nodes.

### Phase III: Simple pilot migration (import_graph)
Tasks:
- Migrate import_graph to the standard template end-to-end.
- Add a focused harness test validating materialization and contract compliance.
- Ensure CLI graph commands read from the materialized output.

Deliverables:
- import_graph anchor uses shared patterns and records materializations.
- Test coverage for import_graph target output.

Acceptance criteria:
- import_graph runs via build harness with deterministic output.
- CLI graph listing or export for import graph remains stable.

File-level breakdown (Phase III):
- `src/codeintel/build/hamilton/native/graphs/import_graph.py`: split run/ingest/finalize nodes,
  replace _materialize_import_graph with save_rows, and remove direct gateway access.
- `src/codeintel/build/hamilton/native/graphs/import_graph.py`: use module inventory helpers
  instead of querying core.modules via Ibis.
- `tests/graphs/test_compute_imports_extended.py`: add/adjust assertions for row output shape.
- `tests/build/hamilton/test_pr14_graph_exports.py`: refresh expectations if export metadata
  changes for import_graph.

### Phase IV: Complex graph migration (call_graph, views, metrics)
Tasks:
- Migrate call_graph to standard template with normalized edge/node rows.
- Align call_graph_views to consume call_graph materializations only.
- Migrate graph_metrics to consume call_graph materializations only.

Deliverables:
- call_graph, call_graph_views, graph_metrics fully DAG-first.
- Consistent row counts and materialization metadata.

Acceptance criteria:
- Graph metrics tests pass and match goldens.
- Graph views query through storage boundaries only.

File-level breakdown (Phase IV):
- `src/codeintel/build/hamilton/native/graphs/call_graph.py`: convert to run/ingest/finalize,
  remove SaveToObjectMetadataDecorator and direct gateway usage, and persist via save_rows.
- `src/codeintel/build/hamilton/native/graphs/graph_targets.py`: migrate call_graph_views to
  tool_target pattern using save_rows for view tables.
- `src/codeintel/build/hamilton/native/graphs/graph_targets.py`: migrate graph_metrics to
  tool_target pattern with table-specific save_rows and table materialization collectors.
- `tests/graphs/test_callgraph_builder.py`: validate node/edge semantics after migration.
- `tests/graphs/test_engine_nx.py`: ensure graph targets still map to expected outputs.
- `tests/graphs/test_compute_metrics.py`: validate graph_metrics outputs and counts.

### Phase V: Dataflow graphs and validation
Tasks:
- Migrate cfg and dfg to standard template (cfg -> dfg chain).
- Migrate graph_validation to consume cfg/dfg outputs only.
- Ensure validation results surface via ExecutionResult warnings.

Deliverables:
- cfg, dfg, graph_validation targets fully aligned.
- Validation outputs materialized via shared save_rows.

Acceptance criteria:
- CFG/DFG tests pass in isolation.
- Validation table exists and is queryable through storage.

File-level breakdown (Phase V):
- `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`: convert cfg/dfg to run/ingest/finalize,
  replace SaveToObjectMetadataDecorator with save_rows, and remove direct gateway access.
- `src/codeintel/build/hamilton/native/graphs/graph_targets.py`: migrate graph_validation to
  tool_target pattern, consuming cfg/dfg tables only.
- `tests/graphs/test_compute_metrics_cfg.py`: validate cfg outputs.
- `tests/graphs/test_compute_metrics_dfg.py`: validate dfg outputs.
- `tests/graphs/test_validation.py`: verify validation output and warnings.

### Phase VI: Symbol uses (SCIP-dependent)
Tasks:
- Migrate symbol_uses to standard template with SCIP occurrences input.
- Ensure scip availability errors are explicit and consistent with tool registry.

Deliverables:
- symbol_uses target aligned with scip ingestion outputs.

Acceptance criteria:
- symbol_uses runs end-to-end when SCIP tooling is available.
- Clear failure behavior when SCIP is missing.

File-level breakdown (Phase VI):
- `src/codeintel/build/hamilton/native/graphs/graph_targets.py`: migrate symbol_uses to
  run/ingest/finalize pattern and use save_rows for graph.symbol_use_edges.
- `src/codeintel/build/hamilton/native/graphs/graph_targets.py`: ensure symbol_uses inputs read
  from core.scip_occurrences and core.goids without direct gateway access.
- `tests/graphs/test_compute_symbols_extended.py`: validate symbol_uses rows and counts.
- `tests/graphs/test_span_consistency_integration.py`: verify cross-component alignment.

### Phase VII: Integration and replication checklist
Tasks:
- Add a small end-to-end test: modules -> goids -> import_graph.
- Validate CLI graph commands operate with materialized graph outputs only.
- Refresh goldens if graph schema or output changes.

Replication checklist for each graph target:
- Anchor t__* uses shared materialization patterns.
- Inputs derive from upstream materialized outputs (no direct scans).
- Row serialization uses contract-driven schemas.
- TargetRunRecord includes table counts and warnings.
- Harness test validates contract compliance.

File-level breakdown (Phase VII):
- `tests/build/hamilton/test_graphs_end_to_end.py`: new end-to-end harness test for modules ->
  goids -> import_graph (or add to existing build harness test file if preferred).
- `tests/graphs/test_resources.py`: ensure storage boundary usage remains localized.
- `tests/cli/test_graph_cli.py`: verify CLI graph commands still resolve outputs via DAG metadata.

## Work completed so far (implementation status)
- Added canonical inventory artifact: `src/codeintel/core/registry/dag_output_inventory.yaml`.
- Added inventory loader and validation types: `src/codeintel/core/registry/service.py`.
- Added registry CLI commands: `src/codeintel/cli/commands/registry.py`.
- Added registry CLI option specs: `src/codeintel/cli/options/registry.py`.
- Registered registry app in the CLI entrypoint: `src/codeintel/cli/__init__.py`.
- Added tests for loader and CLI wiring:
  `tests/core/test_dag_output_inventory.py`, `tests/cli/test_registry_cli.py`.
- Added shared-flags flattening for Command[T] commands so `--output-format` works
  consistently across new command groups.
- Added ingestion tooling inventory artifact:
  `src/codeintel/core/registry/ingestion_tooling_inventory.yaml`.
- Added ingestion tooling loader and validation helpers:
  `src/codeintel/core/registry/service.py`.
- Added `registry tools` CLI preflight with resolution status and filters:
  `src/codeintel/cli/commands/registry.py`.
- Added tests for tooling inventory loading and CLI output:
  `tests/core/test_ingestion_tooling_inventory.py`, `tests/cli/test_registry_cli.py`.

## Open follow-ups and validation
- Resolve the OpenTelemetry import dependency used during CLI/test import
  (`opentelemetry.propagators.baggage`) so pytest can run without import errors.
- Re-run the focused tests:
  `uv run pytest -q tests/core/test_dag_output_inventory.py tests/cli/test_registry_cli.py
  tests/core/test_ingestion_tooling_inventory.py`.

## Risks and mitigations
- Contract/schema drift: enforce contract-first updates and tighten CI checks.
- DAG output discovery gaps: keep inventory in sync via tooling or CI guard.
- CLI snapshot churn: derive help text from stable metadata.
- Serving coupling: ensure serving only consumes materialized outputs.

## Next steps after the pilot
- Use the checklist to migrate the next-most-complex output.
- Refresh the broader DAG basis plan based on pilot learnings.
- Expand inventory to include all targets and prioritize by impact/risk.
