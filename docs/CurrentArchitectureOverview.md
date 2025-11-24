At a high level, everything under `src/codeintel` is one big pipeline that:

1. Scans a repo snapshot.
2. Ingests code / tests / config / coverage into DuckDB tables.
3. Builds graphs (GOIDs, call graph, import graph, CFG/DFG, symbol uses).
4. Derives analytics (complexity, hotspots, typedness, coverage, risk factors).
5. Exports those tables as JSONL/Parquet (the files you’ve attached).
6. Exposes them via CLI, a FastAPI server, and MCP tools.

I’ll walk through the architecture package‑by‑package, and then map that to the concrete `*.jsonl` / `*.json` you see in the project.

---

## 1. Top‑level layout

Under `src/codeintel/codeintel` you have:

* `cli/` – command‑line entrypoint & arg parsing.
* `config/` – Pydantic config models + DuckDB schema registry.
* `storage/` – DuckDB connection, schema creation, and docs.* views.
* `models/` – typed “row” models that mirror the DuckDB tables & JSONL shapes.
* `utils/` – path normalization helpers.
* `ingestion/` – all the raw data ingestion (AST/CST, coverage, typing, config, tests, etc.).
* `graphs/` – GOIDs + call graph, import graph, CFG/DFG, symbol uses, validations.
* `analytics/` – complexity/hotspots, per‑function metrics, coverage analytics, risk factors.
* `docs_export/` – export tables to JSONL / Parquet into “Document Output”.
* `orchestration/` – pipeline orchestration (plain Python + Prefect 3 flow + step graph).
* `server/` – FastAPI server serving docs.* views via SQL templates.
* `mcp/` – Model Context Protocol server/tools for querying CodeIntel datasets.
* `stubs/` – type-checking stubs.
* `types.py` – shared TypedDicts/protocols for tool outputs (pytest, pyrefly, SCIP, etc.).
* `__init__.py` – just a package docstring.

Everything is glued together by DuckDB: ingestion and graph builders insert into DuckDB tables; analytics read from those; docs_export and server query from them.

---

## 2. Configuration & schemas (`config/`)

### `config/models.py`

Defines Pydantic models used everywhere:

* **RepoConfig**
  Canonical identity of a run: repo name, commit, etc. These fields are embedded into GOIDs and most exported rows (`repo`, `commit` in JSONL like `goids.jsonl`, `coverage_lines.jsonl`, etc.).

* **PathsConfig**
  “Filesystem contract” for a single run. Docstring explicitly assumes a layout like:

  ```text
  repo_root/
    src/
    Document Output/
    build/
      db/codeintel.duckdb
  ```

* **ToolsConfig**
  Paths/binaries for external tools: `scip-python`, `pyrefly`, `pyright`, `ruff`, `coverage`, etc. Ingestion modules read this to know what to run.

* **CodeIntelConfig**
  Top‑level config consumed by the CLI. Bundles `RepoConfig`, `PathsConfig`, `ToolsConfig`, and per‑step configs.

* **Per‑step configs** – one model per pipeline stage, for example:

  * `RepoScanConfig`
  * `CoverageIngestConfig`
  * `TypingIngestConfig`
  * `DocstringsIngestConfig`
  * `ConfigIngestConfig`
  * `GoidBuilderConfig`, `CallGraphConfig`, `CFGBuilderConfig`, `ImportGraphConfig`, `SymbolUsesConfig`
  * `HotspotsConfig`, `CoverageAnalyticsConfig`, `FunctionAnalyticsConfig`, `TestCoverageConfig`, etc.

These config types are what you see being passed into ingestion functions and steps, and they encode time‑invariant assumptions about repo layout and tools.

### `config/schemas/tables.py`

This is the central **DuckDB schema registry**. It uses three helper classes:

* `Column` – name, type, `nullable` flag.
* `TableSchema` – table name, schema, columns, indexes.
* `Index` – secondary index definitions.

Then it defines a mapping from **fully qualified table names** to `TableSchema`, e.g.:

* `core.ast_nodes`

* `core.cst_nodes`

* `core.docstrings`

* `core.goids`

* `core.goid_crosswalk`

* `core.modules`

* `core.repo_map`

* `core.ast_metrics`

* `analytics.config_values`

* `analytics.coverage_functions`

* `analytics.coverage_lines`

* `analytics.function_metrics`

* `analytics.function_types`

* `analytics.goid_risk_factors`

* `analytics.hotspots`

* `analytics.static_diagnostics`

* `analytics.tags_index`

* `analytics.test_catalog`

* `analytics.test_coverage_edges`

* `analytics.typedness`

* `graph.call_graph_edges`

* `graph.call_graph_nodes`

* `graph.cfg_blocks`

* `graph.cfg_edges`

* `graph.dfg_edges`

* `graph.import_graph_edges`

* `graph.symbol_use_edges`

These match your JSONL/JSON artifacts almost one‑to‑one (I’ll map them explicitly in §7).

### `config/schemas/ingestion_sql.py`

Holds **column orders** used during ingestion, e.g.:

* `AST_NODES_COLUMNS = ["path", "node_type", "name", ...]`
* `COVERAGE_LINES_COLUMNS = ["repo", "commit", "rel_path", "line", ...]`
* `TEST_CATALOG_COLUMNS`, etc.

These arrays are used with the row helpers in `models/rows.py` so that `INSERT` column order stays consistent with the DuckDB schema and JSONL exports.

### `config/schemas/sql_builder.py`

Utility functions for DDL and prepared statements:

* `prepared_statements(table_key)` – given `"analytics.coverage_lines"`, returns pre‑built `INSERT` and optional `DELETE` SQL.
* `ensure_schema(con, table_key)` – compares a live DuckDB table with the registered `TableSchema` and raises if there’s drift.

---

## 3. Storage & row models

### `models/rows.py`

This file defines **TypedDict row types** and helper functions to turn them into ordered tuples for DuckDB inserts. Key row types include:

* `CoverageLineRow` → `analytics.coverage_lines` → `coverage_lines.jsonl`
  Keys: `repo`, `commit`, `rel_path`, `line`, `is_executable`, `is_covered`, `hits`, `context_count`, `created_at`.

* `DocstringRow` → `core.docstrings` → `docstrings.jsonl`
  Keys: `repo`, `commit`, `rel_path`, `module`, `qualname`, `kind`, `lineno`, `raw_docstring`, `short_desc`, `long_desc`, `params`, `returns`, etc.

* `SymbolUseRow` → `graph.symbol_use_edges` → `symbol_use_edges.jsonl`
  Keys: `symbol`, `def_path`, `use_path`, `same_file`, `same_module`.

* `ConfigValueRow` → `analytics.config_values` → `config_values.jsonl`
  Keys: `config_path`, `format`, `key`, `reference_paths`, `reference_modules`, `reference_count`.

* `GoidRow` → `core.goids` → `goids.jsonl`
  Keys: `goid_h128`, `urn`, `repo`, `commit`, `rel_path`, `language`, `kind`, `qualname`, `start_line`, `end_line`, `created_at`.

* `GoidCrosswalkRow` → `core.goid_crosswalk` → `goid_crosswalk.jsonl`
  Keys: `goid`, `lang`, `module_path`, `file_path`, `start_line`, `end_line`, `scip_symbol`, `ast_qualname`, `cst_node_id`, `chunk_id`, `symbol_id`, `updated_at`.

* `TypednessRow` → `analytics.typedness` → `typedness.jsonl`
  Keys: `path`, `type_error_count`, `annotation_ratio`, `untyped_defs`, `overlay_needed`.

* `StaticDiagnosticRow` → `analytics.static_diagnostics` → `static_diagnostics.jsonl`.

* `HotspotRow` → `analytics.hotspots` → `hotspots.jsonl`
  Keys: `rel_path`, `commit_count`, `author_count`, `lines_added`, `lines_deleted`, `complexity`, `score`.

* **Graph tables:**

  * `CallGraphNodeRow` → `graph.call_graph_nodes` → `call_graph_nodes.jsonl`.
  * `CallGraphEdgeRow` → `graph.call_graph_edges` → `call_graph_edges.jsonl`.
  * `ImportEdgeRow` → `graph.import_graph_edges` → `import_graph_edges.jsonl`.
  * `CFGBlockRow` → `graph.cfg_blocks` → `cfg_blocks.jsonl`.
  * `CFGEdgeRow` → `graph.cfg_edges` → `cfg_edges.jsonl`.
  * `DFGEdgeRow` → `graph.dfg_edges` → `dfg_edges.jsonl`.

* **Test/coverage tables:**

  * `TestCatalogRowModel` → `analytics.test_catalog` (not exported as JSONL in your snapshot but used in analytics).
  * `TestCoverageEdgeRow` → `analytics.test_coverage_edges` → `test_coverage_edges.jsonl`.

Each has a `*_to_tuple(row)` helper that enforces column order. These same shapes are what you see line‑by‑line in the JSONL outputs.

### `storage/schemas.py` & `storage/duckdb_client.py`

* `create_schemas(con)` – ensures logical schemas (`core`, `graph`, `analytics`, `docs`) exist.
* `apply_all_schemas(con)` – creates all known tables using the registry in `config/schemas/tables.py`.
* `DuckDBConfig` / `DuckDBClient` – small wrapper for connecting to the DuckDB file (db path, read‑only flags) with `get_connection(config)`.

### `storage/views.py`

Defines AI‑/docs‑oriented **views in the `docs` schema**, notably:

* `docs.v_function_summary`
  Joins `analytics.goid_risk_factors` with per‑function metrics, docstrings, and tags/owners to produce a single “function summary” row per GOID. This is the table that backs a lot of downstream consumption.

* `docs.v_call_graph_enriched`
  Joins call graph edges with caller/callee GOIDs and risk scores, so you can ask questions like “show me high‑risk callers of a given function”.

These views are what the FastAPI server and MCP backend query by default.

---

## 4. Utilities & types

### `utils/paths.py`

Path normalization helpers:

* `ensure_repo_root(path)` – makes sure a repo root is an absolute `Path`.
* `normalize_rel_path(path)` – canonical POSIX relative path.
* `repo_relpath(root, path)` – repo‑relative path with forward slashes.
* `relpath_to_module(path)` – turns `src/codeintel/utils/paths.py` → `src.codeintel.utils.paths`.

These functions are used throughout ingestion, graphs, and analytics so that everything agrees on `rel_path` and module naming (which feeds into GOIDs and module maps).

### `types.py`

Shared TypedDicts / Protocols for external tool outputs:

* `PytestCallEntry`, `PytestTestEntry` – shapes of `pytest-json-report`.
* `PyreflyError` – static error entries from pyrefly.
* `ScipOccurrence`, `ScipDocument` – shapes of SCIP’s JSON output (`index.scip.json`).

…and `HasModelDump` protocol for Pydantic models used in MCP responses.

---

## 5. Ingestion pipeline (`ingestion/`)

`ingestion/__init__.py` sums it up nicely: “Ingestion stages that parse repositories into normalized DuckDB tables for analytics.”

### 5.1. Shared plumbing

* `ingestion/common.py`

  * `ModuleRecord`, `BatchResult`.
  * Functions like `log_progress`, `load_module_map`, `iter_modules`, helpers for batched inserts and schema guards.

* `ingestion/source_scanner.py`

  * `ScanConfig`, `ScanRecord`, `SourceScanner` – generic scanner for walking Python sources given `repo_root` and inclusion/exclusion rules.

* `ingestion/ast_utils.py`

  * `parse_python_module`, `timed_parse` – parse Python to `ast` with timing.
  * `AstSpanIndex` – index AST nodes by `(start_line, end_line)`.

* `ingestion/tool_runner.py`

  * `ToolRunner` – structured execution of CLIs like `scip-python`, `pyrefly`, `pyright`, `ruff`, `coverage`, with caching and typed results (using the TypedDicts in `types.py`).

* `ingestion/runner.py`

  * `BuildPaths`, `IngestionContext` – capture repo root, build dir, tool paths.
  * `run_repo_scan`, `run_cst_extract`, `run_ast_extract`, `run_coverage_ingest`, `run_tests_ingest`, `run_typing_ingest`, `run_docstrings_ingest`, `run_config_ingest`.
    These are the “one‑function per ingestion phase” building blocks the orchestration layer uses.

### 5.2. Repo structure & modules

* `ingestion/repo_scan.py`

  * Scans the repo and populates:

    * `core.modules` → `modules.jsonl`
      Example row: `{ "module": "src.codeintel.__init__", "path": "src/codeintel/__init__.py", ... }`
    * `core.repo_map` → `repo_map.json`
      Includes the `modules` dict mapping module names to paths and timestamps.
    * `analytics.tags_index` – optional tagging of paths (e.g., test, docs, generated, etc.).

### 5.3. AST / CST and docstrings

* `ingestion/cst_utils.py`

  * `CstCaptureConfig`, `LineIndexedSource`, `CstCaptureVisitor` – generic LibCST capture logic (node IDs, spans, qnames).

* `ingestion/cst_extract.py`

  * Parses modules listed in `core.modules` with LibCST.
  * Writes `core.cst_nodes` → `cst_nodes.jsonl`.
    Example row: `{ "path": "src/codeintel/__init__.py", "node_id": "...:Module:1:0:2:0", "kind": "Module", "span": "{...}" }`.

* `ingestion/py_ast_extract.py`

  * Uses `ast` (stdlib) to extract declarations and perhaps other AST‑based info.

* `ingestion/docstrings_ingest.py`

  * Docstring describes: “Extract structured docstrings with AST and docstring‑parser …”.
  * Walks ASTs, uses `docstring_parser` to structure text, and persists to `core.docstrings` → `docstrings.jsonl`.
    Example row includes `short_desc`, `long_desc`, `params`, `returns`, `raises`, etc.

* `core.ast_nodes` / `ast_nodes.jsonl`
  Populated by AST extractors:
  Example row:

  ```json
  {
    "path": "src/codeintel/__init__.py",
    "node_type": "Module",
    "name": "__init__",
    "qualname": "src.codeintel.__init__",
    "lineno": null,
    "end_lineno": null,
    "decorators": "[]",
    "docstring": "CodeIntel package exposing ingestion..."
  }
  ```

### 5.4. Coverage, tests, typing, config

* `ingestion/coverage_ingest.py`

  * Reads `.coverage` or JSON output via `coverage.py` API/CLI.
  * Populates `analytics.coverage_lines` → `coverage_lines.jsonl`.

* `ingestion/tests_ingest.py`

  * Ingests `pytest-json-report` output into `analytics.test_catalog` (ID, nodeid, status, duration, etc.).
  * These catalog entries are later joined to GOIDs and coverage edges.

* `ingestion/typing_ingest.py`

  * Runs `pyrefly`, `pyright`, and `ruff`.
  * Computes annotation ratios from AST.
  * Populates:

    * `analytics.typedness` → `typedness.jsonl` (file‑level typedness & overlay flags).
    * `analytics.static_diagnostics` → `static_diagnostics.jsonl` (per file: pyrefly/pyright/ruff error counts).

* `ingestion/config_ingest.py`

  * Walks configuration files (e.g., JSON, YAML, TOML, INI, env).
  * Flattens them into keypaths and reference metadata.
  * Populates `analytics.config_values` → `config_values.jsonl`.

* `ingestion/scip_ingest.py`

  * Runs `scip-python` to produce `index.scip.json`.
  * Registers that SCIP index within DuckDB so `graphs/symbol_uses.py` can build symbol edges.

---

## 6. Graphs (`graphs/`)

This package builds the higher‑level code graphs on top of AST/CST and SCIP.

* `graphs/goid_builder.py`

  * Uses AST nodes (`core.ast_nodes`) and module map to build:

    * `core.goids` → `goids.jsonl`
    * `core.goid_crosswalk` → `goid_crosswalk.jsonl`
  * GOID = hashed ID (128‑bit decimal) + URN string like:
    `goid:repo/path#language:kind:qualname?s=start_line&e=end_line`

* `graphs/callgraph_builder.py`

  * Uses LibCST (`cst_nodes`), GOIDs, and import resolution to build:

    * `graph.call_graph_nodes` → `call_graph_nodes.jsonl`

      * Keys: `goid_h128`, `language`, `kind` (module/class/function), `arity`, `is_public`, `rel_path`.
    * `graph.call_graph_edges` → `call_graph_edges.jsonl`

      * Keys: `caller_goid_h128`, `callee_goid_h128` (nullable), `callsite_path`, `callsite_line`, `callsite_col`, `kind`, `resolved_via`, `confidence`, `evidence_json`.

* `graphs/import_graph.py`

  * Builds the module import graph (`graph.import_graph_edges` → `import_graph_edges.jsonl`), with:

    * `src_module`, `dst_module`, `src_fan_out`, `dst_fan_in`, `cycle_group`.

* `graphs/cfg_builder.py`

  * Builds:

    * `graph.cfg_blocks` → `cfg_blocks.jsonl`
      Basic block metadata: `function_goid_h128`, `block_idx`, `block_id`, `label`, `file_path`, `start_line`, `end_line`, `kind`, `stmts_json`, `in_degree`, `out_degree`.
    * `graph.cfg_edges` → `cfg_edges.jsonl`
      `function_goid_h128`, `src_block_id`, `dst_block_id`, `edge_kind`.
    * `graph.dfg_edges` → `dfg_edges.jsonl`
      Data flow edges across CFG blocks, with `src_var`, `dst_var`, `edge_kind`.

* `graphs/symbol_uses.py`

  * Consumes SCIP JSON (`index.scip.json`) and crosswalk to produce `graph.symbol_use_edges` → `symbol_use_edges.jsonl`.

* `graphs/function_index.py` / `graphs/function_catalog.py`

  * Helpers for mapping spans → GOIDs and building a “function catalog” backing docs views.

* `graphs/import_resolver.py`

  * Canonical import resolution utilities used by the call graph builder.

* `graphs/validation.py`

  * Runs quality checks over graph outputs and writes `analytics.graph_validation` → `graph_validation.jsonl`:

    * Missing GOIDs for functions found in AST.
    * Callsite positions outside caller spans.
    * Modules with no GOIDs (orphans).
  * Example row: `{ "check_name": "callsite_span_mismatch", "severity": "warning", "path": "...", "detail": "...", "context": "{...}" }`.

---

## 7. Analytics (`analytics/`)

### 7.1. AST metrics & hotspots

* `analytics/ast_metrics.py`

  * Computes per‑file structural metrics from AST + simple git churn.
  * Populates `core.ast_metrics` → `ast_metrics.jsonl`:

    * `rel_path`, `node_count`, `function_count`, `class_count`, `avg_depth`, `max_depth`, `complexity`, `generated_at`.
  * Also drives `analytics.hotspots` → `hotspots.jsonl`:

    * `rel_path`, `commit_count`, `author_count`, `lines_added`, `lines_deleted`, `complexity`, `score` (a combined hotness score).

### 7.2. Per‑function metrics & typing

* `analytics/functions.py`

  * Reads GOIDs and AST to compute per‑function metrics.
  * Populates:

    * `analytics.function_metrics` → `function_metrics.jsonl`
      Includes `function_goid_h128`, `urn`, `repo`, `commit`, `rel_path`, `language`, `kind`, `qualname`, `start_line`, `end_line`, `loc`, `logical_loc`, `param_count`, varargs flags, `cyclomatic_complexity`, nesting depth, `stmt_count`, `decorator_count`, `has_docstring`, `complexity_bucket`, `created_at`.
    * `analytics.function_types` → `function_types.jsonl`
      Adds typedness by function: `total_params`, `annotated_params`, `return_type`, `param_types`, `fully_typed`/`partial_typed`/`untyped`, `typedness_bucket`, `typedness_source`, `created_at`.

### 7.3. Coverage analytics

* `analytics/coverage_analytics.py`

  * Aggregates `analytics.coverage_lines` and GOIDs to compute:

    * `analytics.coverage_functions` → `coverage_functions.jsonl`
      Keys: `function_goid_h128`, `urn`, `repo`, `commit`, `rel_path`, `language`, `kind`, `qualname`, `start_line`, `end_line`, `executable_lines`, `covered_lines`, `coverage_ratio`, `tested`, `untested_reason`, `created_at`.
    * `analytics.test_coverage_edges` → `test_coverage_edges.jsonl`
      Connects tests to functions: `test_id`, `test_goid_h128`, `function_goid_h128`, `urn`, `repo`, `rel_path`, `qualname`, `covered_lines`, `executable_lines`, `coverage_ratio`, `last_status`, `created_at`.

  * Also has helpers to backfill `test_goid_h128` in `test_catalog` using GOIDs.

### 7.4. Tests analytics

* `analytics/tests_analytics.py`

  * Summarizes test catalog information (pass/fail/flaky, durations, etc.) used for risk scoring and docs views.
  * Works on `analytics.test_catalog` and `analytics.test_coverage_edges`.

### 7.5. Risk factors

There isn’t a dedicated `analytics/risk_factors.py`; instead, `orchestration/steps.RiskFactorsStep` runs a **SQL join** over analytics tables to create:

* `analytics.goid_risk_factors` → `goid_risk_factors.jsonl`
  This is the per‑function “risk summary” row:

  * from function metrics: `loc`, `logical_loc`, `cyclomatic_complexity`, `complexity_bucket`
  * from function types: `typedness_bucket`, `typedness_source`
  * from hotspots: `hotspot_score`, `file_typed_ratio`
  * from static diagnostics: `static_error_count`, `has_static_errors`
  * from coverage: `executable_lines`, `covered_lines`, `coverage_ratio`, `tested`, `test_count`, `failing_test_count`, `last_test_status`
  * plus a computed `risk_score`, `risk_level`, `tags`, `owners`, `created_at`.

`docs.v_function_summary` then pivots on this table to add docstrings and other metadata.

---

## 8. Orchestration (`orchestration/`) & CLI (`cli/`)

### Orchestration

* `orchestration/steps.py`

  * Defines the **logical dependency graph** of pipeline steps via classes like:

    * `RepoScanStep`
    * `SCIPIngestStep`
    * `CSTStep`
    * `AstStep`
    * `DocstringsStep`
    * `GOIDStep`
    * `ImportGraphStep`
    * `CallGraphStep`
    * `CFGStep`
    * `SymbolUsesStep`
    * `CoverageStep`
    * `TestsStep`
    * `TypingStep`
    * `AstMetricsStep`
    * `HotspotsStep`
    * `CoverageAnalyticsStep`
    * `FunctionAnalyticsStep`
    * `TestCoverageEdgesStep`
    * `RiskFactorsStep`
    * `ExportDocsStep`
  * Each step has a `name`, `deps` and a `run(ctx, con)` method that calls into the corresponding `ingestion.*`, `graphs.*`, or `analytics.*` functions.

* `orchestration/prefect_flow.py`

  * Wraps the pipeline into a Prefect 3 flow, with `ExportArgs` as input.
  * Handles connecting to DuckDB, running steps in dependency order, and then exporting docs.

### CLI (`cli/main.py`)

* `main(argv=None)` is the actual entrypoint.
* Builds an `argparse` parser with subcommands like:

  * `pipeline run` – runs the full ingestion → graphs → analytics → risk → export pipeline for a given repo/commit.
  * `docs export` – exports JSONL/Parquet only, assuming tables already populated.
* `_open_connection` calls into `storage` to ensure schemas and views exist before running steps.
* `_build_config_from_args` converts CLI flags → `CodeIntelConfig`.

So your typical flow from the outside world is: **CLI → Orchestration Steps → Ingestion/Graphs/Analytics → DuckDB → Docs Export/Server/MCP**.

---

## 9. Docs export, server & MCP

### `docs_export/export_jsonl.py` & `export_parquet.py`

* These modules contain mappings from logical dataset names (like `"core.ast_nodes"`, `"analytics.function_metrics"`, `"graph.call_graph_edges"`) to filenames like `ast_nodes.jsonl`, `function_metrics.jsonl`, `call_graph_edges.jsonl`.
* `export_jsonl_for_table(con, table_name, output_path)` – generic JSONL export.
* `export_all_jsonl(con, document_output_dir)` – exports all configured datasets beneath `Document Output/`.
* Same for Parquet via DuckDB’s `COPY ... TO ... (FORMAT PARQUET)`.

The JSONL you uploaded (e.g. `ast_nodes.jsonl`, `goids.jsonl`, `coverage_functions.jsonl`, `goid_risk_factors.jsonl`, etc.) are exactly the outputs of these exporters.

### `server/` (FastAPI)

* `server/fastapi.py`

  * `create_app(config_loader, backend_factory)` – wires up a FastAPI app using:

    * `storage.views` for `docs.*` views.
    * `server/query_templates.py` – parameterized SQL templates for:

      * Function summaries
      * Coverage gaps
      * Graph neighborhoods (callers/callees)
* `openapi_codeintel.json` – a frozen OpenAPI schema used both as config and as a doc source (note it appears in `config_values.jsonl`).

### `mcp/`

* `mcp/backend.py`

  * Backend that abstracts “read from DuckDB” or “read from HTTP” for MCP tools.
* `mcp/models.py`

  * Typed models for MCP request/response payloads and errors.
* `mcp/server.py`

  * MCP server exposing CodeIntel tools (e.g. query function risk, find callers, etc.).
* `mcp/tools.py`

  * Declares the MCP tools (parameters, what SQL they execute under the hood).

Together, these let an LLM or other client query your code intelligence datasets live, rather than just offline JSONL.

---

## 10. How the JSONL/JSON objects line up with the architecture

Here’s a quick map of key files you see to their source modules and tables:

**Core / structure**

* `modules.jsonl` → table `core.modules` → produced by `ingestion/repo_scan.py`.
* `repo_map.json` → table `core.repo_map` → produced by `ingestion/repo_scan.py`.
* `ast_nodes.jsonl` → table `core.ast_nodes` → produced by AST extractors (`ingestion/py_ast_extract.py`, `ingestion/ast_utils.py`).
* `cst_nodes.jsonl` → table `core.cst_nodes` → produced by `ingestion/cst_extract.py`.
* `docstrings.jsonl` → table `core.docstrings` → produced by `ingestion/docstrings_ingest.py`.
* `goids.jsonl` → table `core.goids` → produced by `graphs/goid_builder.py`.
* `goid_crosswalk.jsonl` → table `core.goid_crosswalk` → produced by `graphs/goid_builder.py`.
* `ast_metrics.jsonl` → table `core.ast_metrics` → produced by `analytics/ast_metrics.py`.
* `index.scip.json` → raw SCIP index produced by `ingestion/scip_ingest.py`.

**Graph**

* `call_graph_nodes.jsonl` → `graph.call_graph_nodes` → `graphs/callgraph_builder.py`.
* `call_graph_edges.jsonl` → `graph.call_graph_edges` → `graphs/callgraph_builder.py`.
* `import_graph_edges.jsonl` → `graph.import_graph_edges` → `graphs/import_graph.py`.
* `cfg_blocks.jsonl` → `graph.cfg_blocks` → `graphs/cfg_builder.py`.
* `cfg_edges.jsonl` → `graph.cfg_edges` → `graphs/cfg_builder.py`.
* `dfg_edges.jsonl` → `graph.dfg_edges` → `graphs/cfg_builder.py`.
* `symbol_use_edges.jsonl` → `graph.symbol_use_edges` → `graphs/symbol_uses.py`.
* `graph_validation.jsonl` → `analytics.graph_validation` → `graphs/validation.py`.

**Analytics**

* `coverage_lines.jsonl` → `analytics.coverage_lines` → `ingestion/coverage_ingest.py`.
* `coverage_functions.jsonl` → `analytics.coverage_functions` → `analytics/coverage_analytics.py`.
* `function_metrics.jsonl` → `analytics.function_metrics` → `analytics/functions.py`.
* `function_types.jsonl` → `analytics.function_types` → `analytics/functions.py`.
* `hotspots.jsonl` → `analytics.hotspots` → `analytics/ast_metrics.py`.
* `typedness.jsonl` → `analytics.typedness` → `ingestion/typing_ingest.py`.
* `static_diagnostics.jsonl` → `analytics.static_diagnostics` → `ingestion/typing_ingest.py`.
* `test_coverage_edges.jsonl` → `analytics.test_coverage_edges` → `analytics/coverage_analytics.py`.
* `config_values.jsonl` → `analytics.config_values` → `ingestion/config_ingest.py`.
* `goid_risk_factors.jsonl` → `analytics.goid_risk_factors` → SQL join in `orchestration/steps.RiskFactorsStep`.

(There are also `analytics.tags_index` and `analytics.test_catalog` tables that exist in DuckDB but don’t appear as JSONL in your snapshot.)

---

## 11. Mental model / summary

If you want a compact mental picture:

* **Config** defines what repo/commit you’re looking at, how the filesystem is laid out, and where external tools live.
* **Ingestion** turns raw files, configs, test reports, coverage, and static analysis into canonical DuckDB tables (`core.*`, `analytics.coverage_lines`, `analytics.typedness`, `analytics.static_diagnostics`, `analytics.config_values`, etc.).
* **Graphs** build code‑level relationships: GOIDs, call graph, import graph, CFG/DFG, and symbol uses (`graph.*` tables).
* **Analytics** sit on top of that to compute structure & risk metrics per file and per function (`ast_metrics`, `function_metrics`, `function_types`, `coverage_functions`, `goid_risk_factors`, etc.).
* **Storage** standardizes how those tables are created, connected, and exposed via views (`docs.v_function_summary`, `docs.v_call_graph_enriched`).
* **Docs export** takes those tables and turns them into the JSONL/Parquet artifacts you’ve shared.
* **CLI / Orchestration** provide the “do everything for this repo snapshot” control plane.
* **Server / MCP** are your live query surfaces against the same underlying DuckDB & views.

# Call graph edge resolution and validation deep dive #

Here’s how call graph edges are *resolved* and then *validated* in your `src/codeintel` stack.

I’ll walk through the pipeline in the same order the code does it.

---

## 1. Where call graph edges fit in the pipeline

The main entrypoint is:

```python
# graphs/callgraph_builder.py
def build_call_graph(con: duckdb.DuckDBPyConnection, cfg: CallGraphConfig) -> None:
```

High‑level steps:

1. Build **call graph nodes** from GOIDs (`_build_call_graph_nodes`).
2. Load a **FunctionCatalog** (spans + metadata for each function) via `load_function_catalog`.
3. Build a **global callee map**: `qualname` and short name → GOID (`_callee_map`).
4. Load **SCIP-based symbol candidates** for unresolved calls (`_load_scip_candidates`).
5. **Collect edges** via LibCST / AST visitors (`_collect_edges`).
6. **Deduplicate** edges (`_dedupe_edges`).
7. **Persist edges** to `graph.call_graph_edges` (`_persist_call_graph_edges`).
8. Run **graph validations** (`run_graph_validations` in `graphs/validation.py`).

So “how edges are resolved” lives mostly in the LibCST/AST visitors and their resolution heuristics; “how they’re validated” lives in `graphs/validation.py`.

---

## 2. Inputs: function spans and name maps

### Function spans (callers + callees)

`FunctionCatalog` (`graphs/function_catalog.py`) wraps:

* A list of `FunctionMeta` (GOID, rel_path, qualname, URN, start/end lines, etc.)
* A `FunctionSpanIndex`, which is the core lookup structure (`graphs/function_index.py`).

`FunctionSpan` holds:

* `goid`
* `rel_path` (normalized)
* `qualname` (e.g. `pkg.module.Class.method`)
* `start_line`, `end_line`

`FunctionSpanIndex` exposes:

* `spans_for_path(path)` → all `FunctionSpan` for that file.

* `local_name_map(path)` → for that file:

  ```python
  mapping[local_name] = goid        # short name, e.g. "foo"
  mapping[qualname]  = goid        # full qualname, e.g. "pkg.module.foo"
  ```

* `lookup(rel_path, start_line, end_line)` → resolve a GOID for a given span, using a multi‑step strategy:

  * Exact span match (start & end lines).
  * Qualname match overlapping the span.
  * Any span enclosing the line.
  * Fallback to functions starting on the same line.

This index is used for:

* Determining **who the caller is** (which function a callsite belongs to).
* Providing **local callee maps** for resolution.

### Local & global callee maps

From the catalog’s function spans, the builder constructs:

* **Global map** `_callee_map(func_rows)`:

  * For each function span:

    * `mapping[qualname] = goid`
    * `mapping[suffix_of_qualname] = goid` (e.g. `foo` from `pkg.mod.foo`)

* **Local map** per file via `function_index.local_name_map(rel_path)`:

  * For each span in that file:

    * `mapping[local_name] = goid`
    * `mapping[qualname] = goid`

These two maps are the core primitives for callee resolution.

### Import aliases

From `graphs/import_resolver.py`:

```python
def collect_aliases(module: cst.Module, current_module: str | None = None) -> dict[str, str]:
```

This walks `import` / `from ... import ...` statements and produces:

* `alias` → fully‑qualified module or symbol string

Examples:

* `import numpy as np` → `{"np": "numpy"}`
* `from pkg.sub import mod as m` → `{"m": "pkg.sub.mod"}`

These aliases are used to interpret `np.foo()` or `m.bar()` as references to underlying fully‑qualified names.

### SCIP-based symbol evidence (for unresolved calls)

`_load_scip_candidates` in `callgraph_builder.py`:

* First tries DuckDB: `SELECT def_path, use_path FROM graph.symbol_use_edges`.
* If that’s empty or missing:

  * Falls back to reading the SCIP index JSON via `symbol_uses.default_scip_json_path` and `symbol_uses.load_scip_documents`.
  * Uses `build_def_map` and `build_use_def_mapping` to derive:

    ```python
    use_path (normalized) -> {def_path1, def_path2, ...}
    ```

So for each file path, you end up with a tuple of candidate definition paths, used only as *evidence* when GOID resolution fails.

---

## 3. Collecting edges: the LibCST path

### Per-file context

For each file in `catalog.functions_by_path`:

```python
context = EdgeResolutionContext(
    function_index=function_index,
    local_callees=callee_by_name,            # function_index.local_name_map(rel_path)
    global_callees=global_callee_by_name,    # _callee_map across repo
    import_aliases=alias_collector,          # collect_aliases(...)
    scip_candidates_by_use_path=scip_candidates_by_use,
)
```

This compact `EdgeResolutionContext` object is passed into the visitor and also reused by the AST fallback.

### Visitor: `_FileCallGraphVisitor`

`_FileCallGraphVisitor` is a `libcst.CSTVisitor` tailored to call edge extraction.

Key behavior:

1. **Track current caller GOID**

   * `METADATA_DEPENDENCIES = (metadata.PositionProvider,)` – uses LibCST’s `PositionProvider` to get accurate start/end line/col for nodes.

   * In `visit`:

     ```python
     if isinstance(node, FUNCTION_NODE_TYPES):
         span = self._pos(node)
         start, end = span
         self.current_function_goid = self.context.function_index.lookup(
             self.rel_path, start.line, end.line
         )
     ```

   * `FUNCTION_NODE_TYPES = (cst.FunctionDef, cst.AsyncFunctionDef)`.

   * In `leave`, it resets `current_function_goid` when leaving a function definition.

2. **Handle calls**

   ```python
   def _handle_call(self, node: cst.Call) -> None:
       if self.current_function_goid is None:
           # Fallback: single-function module or span mismatch
           spans = self.context.function_index.spans_for_path(self.rel_path)
           if spans:
               self.current_function_goid = spans[0].goid
       if self.current_function_goid is None:
           return
   ```

   * If we don’t know the current function, we attempt a fallback:

     * If the file has any function spans, assume the first one (covers single‑function modules or minor span mismatches).
   * If still no caller GOID, we bail: no edge recorded.

   Then:

   * Get the callsite position via `PositionProvider`.

   * Extract callee expression details:

     ```python
     callee_name, attr_chain = self._extract_callee(node.func)
     ```

     `_extract_callee` handles:

     * `Name`: returns (`"foo"`, `["foo"]`).
     * `Attribute` chains like `obj.foo.bar(...)`: returns

       * `callee_name = "bar"`
       * `attr_chain = ["obj", "foo", "bar"]`
     * Other cases: `("", [])`.

   * Resolve callee GOID:

     ```python
     callee_goid, resolved_via, confidence = _resolve_callee(
         callee_name,
         attr_chain,
         self.context.local_callees,
         self.context.global_callees,
         self.context.import_aliases,
     )
     ```

   * Determine the edge *kind*:

     * `"direct"` if `callee_goid` is not `None`.
     * `"unresolved"` otherwise.

   * Add SCIP‑based evidence if unresolved:

     ```python
     scip_paths = context.scip_candidates_by_use_path.get(rel_path)
     evidence = {
         "callee_name": callee_name,
         "attr_chain": attr_chain or None,
         "resolved_via": resolved_via,
     }
     if callee_goid is None and scip_paths:
         evidence["scip_candidates"] = list(scip_paths)
     ```

   * Finally, append a `CallGraphEdgeRow`:

     ```python
     self.edges.append(
         CallGraphEdgeRow(
             caller_goid_h128=self.current_function_goid,
             callee_goid_h128=callee_goid,
             callsite_path=self.rel_path,
             callsite_line=start.line,
             callsite_col=start.column,
             language="python",
             kind=kind,
             resolved_via=resolved_via,
             confidence=confidence,
             evidence_json=evidence,
         )
     )
     ```

   So each edge carries:

   * **IDs**: caller GOID (always), callee GOID (if resolved).
   * **Location**: path, line, col.
   * **Resolution metadata**: `kind`, `resolved_via`, `confidence`.
   * **Evidence**: raw callee name + attr chain + optional SCIP candidates.

---

## 4. Resolution heuristics (`_resolve_callee` / `resolve_callee`)

The “brain” for mapping calls to GOIDs is `resolve_callee` (shared between CST/AST collectors via `call_resolution.py`).

Inputs:

* `callee_name` (string, often final attribute name).
* `attr_chain` (list of names representing `obj.attr1.attr2`).
* `local_callees`: per‑file map from names/qualnames to GOIDs.
* `global_callees`: repo‑wide map from qualnames & short names to GOIDs.
* `import_aliases`: alias → fully qualified module/name.

Resolution steps (with associated `resolved_via` and `confidence`):

1. **Local name in same file**

   ```python
   if callee_name in local_callees:
       goid = local_callees[callee_name]
       resolved_via = "local_name"
       confidence = 0.8
   ```

   This handles calls like `foo()` defined in the same file.

2. **Local attribute chain**

   If there is an attribute chain, try more specific names:

   ```python
   joined = ".".join(attr_chain)
   goid = local_callees.get(joined) or local_callees.get(attr_chain[-1])
   if goid is not None:
       resolved_via = "local_attr"
       confidence = 0.75
   ```

   This catches cases like:

   * `self.method()` where the GOID qualname is something like `pkg.mod.Class.method`.
   * Or module-level attributes that ended up in `local_callees` as qualified names.

3. **Import alias expansion**

   If not resolved yet and there’s an attribute chain:

   * Take the root (`attr_chain[0]`) and see if it’s an alias:

     ```python
     root = attr_chain[0]
     alias_target = import_aliases.get(root)
     if alias_target:
         qualified = (
             alias_target
             if len(attr_chain) == 1
             else ".".join([alias_target, *attr_chain[1:]])
         )
         goid = local_callees.get(qualified) or global_map.get(qualified)
         if goid is not None:
             resolved_via = "import_alias"
             confidence = 0.7
     ```

   This supports things like:

   ```python
   import package.module as m
   m.func()   # attr_chain = ["m", "func"] ⇒ "package.module.func"
   ```

4. **Global fallback by name**

   If still unresolved:

   ```python
   if goid is None and callee_name in global_map:
       goid = global_map[callee_name]
       resolved_via = "global_name"
       confidence = 0.6
   ```

   And in the standalone function variant, for attr chains:

   ```python
   qualified = ".".join(attr_chain)
   goid = global_callees.get(qualified) or global_callees.get(attr_chain[-1])
   if goid is not None:
       resolved_via = "global_name"
       confidence = 0.6
   ```

   This is the “best effort” cross‑module resolution: if a function name is unique (or at least present) globally, we link the edge using that.

5. **Unresolved**

   If all of the above fail:

   * `goid = None`
   * `resolved_via = "unresolved"`
   * `confidence = 0.0`
   * Edge `kind = "unresolved"`.

   SCIP `use_path -> def_path` candidates are attached in `evidence_json` so downstream consumers can still correlate symbol usage, even if GOID resolution fails.

---

## 5. AST fallback path

If LibCST parsing or metadata fails, `_collect_edges_ast` kicks in:

```python
def _collect_edges_ast(rel_path: str, file_path: Path, context: EdgeResolutionContext) -> list[CallGraphEdgeRow]:
```

It:

1. Reads the source and parses with Python’s built‑in `ast.parse`.
2. Uses a nested `_AstVisitor(ast.NodeVisitor)` with:

   * `visit_FunctionDef` / `visit_AsyncFunctionDef`:

     * Determine `current_goid` via `context.function_index.lookup(rel_path, start, end)`, same as the CST visitor.
   * `visit_Call`:

     * If `current_goid` is set:

       * `_extract_callee_ast`:

         * `Name` → (`id`, `[id]`).
         * `Attribute` → flatten to `["obj", "foo", "bar"]` via `_flatten_attribute`.
       * Call `_resolve_callee` with the same local/global/alias maps.
       * Build `CallGraphEdgeRow` exactly as in the CST version, using `node.lineno`, `node.col_offset`, and SCIP evidence.

The AST path is only a fallback; the primary behavior is always LibCST with metadata.

---

## 6. Edge deduplication & persistence

Before writing to DuckDB:

```python
def _dedupe_edges(edges: list[CallGraphEdgeRow]) -> list[CallGraphEdgeRow]:
    seen = set()
    unique_edges = []
    for row in edges:
        key = (
            row["caller_goid_h128"],
            row["callee_goid_h128"],
            row["callsite_path"],
            row["callsite_line"],
            row["callsite_col"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique_edges.append(row)
    return unique_edges
```

Duplicate edges (same caller, callee, and callsite coordinates) are collapsed; in the refactor this logic lives in `call_persist.dedupe_edges`.

Persistence uses `run_batch` to insert into `graph.call_graph_edges` with repo/commit scoping. `CallGraphEdgeRow` → tuple is via `models/rows.py`, in the column order that matches the `graph.call_graph_edges` table.

---

## 7. Validation: how edges are checked

Once nodes and edges are persisted, `build_call_graph` calls:

```python
run_graph_validations(con, repo=cfg.repo, commit=cfg.commit, logger=log)
```

In `graphs/validation.py`, this function runs several checks and records findings into `analytics.graph_validation`.

### 7.1. Missing function GOIDs

`_warn_missing_function_goids`:

* Computes per‑file:

  * `function_count` = number of AST functions (from `core.ast_nodes` where `node_type` in `('FunctionDef','AsyncFunctionDef')`).
  * `goid_count` = number of GOID records in `core.goids` with `kind` in `('function','method')` for that file.

* For any file where `function_count > goid_count`:

  * Logs a warning: `"%d file(s) have functions without GOIDs"`.
  * Adds a finding:

    ```json
    {
      "check_name": "missing_function_goids",
      "severity": "warning",
      "path": "<file>",
      "detail": "<function_count> functions, <goid_count> GOIDs",
      "context": { "function_count": ..., "goid_count": ... }
    }
    ```

This indirectly validates call graph edges by ensuring the underlying GOID set is complete enough; otherwise there will be missing nodes for potential callees or callers.

### 7.2. Callsite span mismatches (directly about call graph edges)

`_warn_callsite_span_mismatches` is the key “edge validation” pass.

Steps:

1. Build a `spans_by_goid` dict from the `FunctionCatalog`:

   ```python
   spans_by_goid = {span.goid: span for span in catalog.function_spans}
   ```

2. Pull all edges that have a callsite line:

   ```python
   rows = con.execute("""
       SELECT
           e.caller_goid_h128,
           e.callsite_path,
           e.callsite_line
       FROM graph.call_graph_edges e
       WHERE e.callsite_line IS NOT NULL
   """).fetchall()
   ```

3. For each `(goid, path, line)`:

   * Look up the caller span: `span = spans_by_goid.get(int(goid))`.
   * If no span, skip (this might be covered by other validations).
   * If the callsite line lies *outside* (`line < span.start_line` or `line > span.end_line`), record mismatch:

     ```python
     mismatches.append((path, line, span.start_line, span.end_line))
     ```

4. If any mismatches exist:

   * Log a warning:

     > `Validation: N call graph edges fall outside caller spans (sample: path:line, ...)`

   * Emit findings:

     ```json
     {
       "check_name": "callsite_span_mismatch",
       "severity": "warning",
       "path": "<path>",
       "detail": "callsite <line> outside span <start>-<end>",
       "context": {
         "callsite_line": <line>,
         "start_line": <start>,
         "end_line": <end>
       }
     }
     ```

These are exactly the `graph_validation.jsonl` rows you’re seeing with context like `{"callsite_line": ..., "start_line": ..., "end_line": ...}`.

**Semantically:** this check enforces that *every call graph edge’s callsite* is consistent with its *caller function span*. If the visitor mis‑assigned `current_function_goid`, or spans are mis‑computed, you get a mismatch.

### 7.3. Orphan modules (no GOIDs)

`_warn_orphan_modules`:

* Looks at `core.modules` for `(repo, commit)`.
* LEFT JOINs against `core.goids` on `rel_path`, `repo`, `commit`.
* Finds modules where there are **no GOIDs at all** (i.e., no functions, classes, etc.).

For each orphan module:

```json
{
  "check_name": "orphan_module",
  "severity": "warning",
  "path": "<module path>",
  "detail": "module has no GOIDs",
  "context": {}
}
```

While this isn’t strictly about edges, it’s part of the set of “graph integrity” signals stored in `analytics.graph_validation` alongside the callsite mismatch check.

### 7.4. Findings persistence

Everything funnels into `_persist_findings`, which:

* Ensures `analytics.graph_validation` table exists.
* `INSERT`s each finding with:

  * `repo`
  * `commit`
  * `check_name`
  * `severity`
  * `path`
  * `detail`
  * `context` (JSON)
  * `created_at` (timestamp)

That’s what you see in `graph_validation.jsonl`.

---

## 8. How this shows up in your exported data

* **`call_graph_edges.jsonl`** rows are basically serialized `CallGraphEdgeRow`s:

  * `caller_goid_h128`
  * `callee_goid_h128` (nullable)
  * `callsite_path`, `callsite_line`, `callsite_col`
  * `language`
  * `kind` (`"direct"` or `"unresolved"`)
  * `resolved_via` (`"local_name"`, `"local_attr"`, `"import_alias"`, `"global_name"`, `"unresolved"`)
  * `confidence`
  * `evidence_json` (JSON string: callee name / attr chain / optional SCIP candidates)

* **`graph_validation.jsonl`** rows with `check_name = "callsite_span_mismatch"` are your concrete “edge validity” issues: callsites outside caller spans.

---

# SCIP evidence path deep dive #


Here’s a zoomed‑in walkthrough of the **SCIP evidence path** in your `src/codeintel` stack—i.e., how `scip-python` output gets turned into structured evidence that flows into GOIDs, symbol uses, and call graph edges.

I’ll go stage by stage so you can see exactly where SCIP comes in and what artifacts it produces.

---

## 0. What “SCIP evidence” means in this codebase

In your system, SCIP is used in **three distinct ways**:

1. **Definition evidence**
   Mapping `(file, start_line)` → **SCIP symbol** and attaching that symbol to GOIDs (`core.goid_crosswalk.scip_symbol`).

2. **Use/definition evidence**
   Mapping **SCIP symbols** → `(def_path, use_path)` pairs, stored in `graph.symbol_use_edges` and exported as `symbol_use_edges.jsonl`.

3. **Fallback call‑graph evidence**
   For unresolved call graph edges, you attach a list of **candidate definition files** derived from SCIP, stored in `graph.call_graph_edges.evidence_json` → exported as `call_graph_edges.jsonl` (with `scip_candidates`).

All of this flows from a single SCIP JSON index file: `build/scip/index.scip.json` (also copied into “Document Output” and exported as `/mnt/data/index.scip.json`).

---

## 1. Generating and loading the SCIP index

### 1.1 Config + entrypoint

**Module:** `codeintel.ingestion.scip_ingest`
**Config type:** `ScipIngestConfig` (from `codeintel.config.models`)

Key config fields used in `ingest_scip`:

* `repo_root`: filesystem root of the repo
* `repo`: `org/repo` string
* `commit`: git SHA
* `build_dir`: build output dir (e.g., `repo_root/build`)
* `document_output_dir`: where human/MCP‑visible artifacts go
* `scip_python_bin`: usually `"scip-python"`
* `scip_bin`: usually `"scip"`

### 1.2 `ingest_scip`: running scip‑python + scip print

```python
def ingest_scip(con: duckdb.DuckDBPyConnection, cfg: ScipIngestConfig) -> ScipIngestResult:
    """
    Run scip-python + scip print, register view, and backfill SCIP symbols.
    """
```

High‑level flow:

1. **Preflight: require git + binaries**

   * Requires `repo_root/.git` to exist; if missing, returns status `"unavailable"` with a reason.
   * `_probe_binaries(cfg)`:

     * Uses `shutil.which` to ensure `cfg.scip_python_bin` and `cfg.scip_bin` exist.
     * Runs `<binary> --version` for each; on failure, returns an `"unavailable"` result.

2. **Choose indexing target**

   * `_run_scip_python(binary: str, repo_root: Path, output_path: Path)`:

     * Prefers `repo_root / "src"` if it exists, else `repo_root`.

     * Executes:

       ```bash
       scip-python index <target_dir> --output <build/scip/index.scip>
       ```

     * On success, it logs debug `stderr` (if any) and returns `True`.

     * On `MISSING_BINARY_EXIT_CODE`, logs a warning and returns `False`.

     * Any other non‑zero exit: logs failure and returns `False`.

3. **Convert `.scip` → JSON**

   * `_run_scip_print(binary: str, index_scip: Path, output_json: Path)`:

     ```bash
     scip print --json <build/scip/index.scip> > <build/scip/index.scip.json>
     ```

     * On success, writes `stdout` to `index.scip.json` and logs debug `stderr` if present.
     * Missing binary or non‑zero exit → logs + returns `False`.

4. **Persist artifacts for docs/UI**

   On success:

   * Copies both files into `document_output_dir`:

     ```python
     shutil.copy2(index_scip, doc_dir / "index.scip")
     shutil.copy2(index_json, doc_dir / "index.scip.json")
     ```

   These copies are what you’ve exported as `/mnt/data/index.scip.json`.

5. **Register `scip_index_view` in DuckDB**

   ```python
   docs_table = con.execute(
       "SELECT unnest(documents, recursive:=true) AS document FROM read_json(?)",
       [str(index_json)],
   ).fetch_arrow_table()

   con.execute("DROP VIEW IF EXISTS scip_index_view")
   con.register("scip_index_view_temp", docs_table)
   con.execute("CREATE VIEW scip_index_view AS SELECT * FROM scip_index_view_temp")
   ```

   This gives you a convenient DuckDB view where each row is a single SCIP **document**:

   * Each document roughly matches the `ScipDocument` TypedDict in `codeintel.types`:

     ```python
     class ScipDocument(TypedDict, total=False):
         """SCIP JSON document emitted by scip-python."""
         relative_path: str
         occurrences: list[ScipOccurrence]
     ```

### 1.3 Backfilling GOIDs with SCIP symbols

**Goal:** attach a SCIP symbol to each GOID that has a matching `(rel_path, start_line)` in the SCIP index.

Function: `_update_scip_symbols(con, index_json: Path) -> None`

Steps:

1. **Ensure table exists**

   ```python
   ensure_schema(con, "core.goid_crosswalk")
   ```

2. **Load raw docs**

   ```python
   docs = _load_scip_documents(index_json)
   ```

   `_load_scip_documents`:

   * `json.loads` the file.
   * Accepts both:

     * `{ "documents": [...] }`, or
     * `[...]` (list root).
   * Returns `list[dict]`.

3. **Build definition map: `(rel_path, start_line) -> symbol`**

   `_build_definition_map(docs)`:

   * For each doc:

     * Reads `relative_path` (string) as `rel_path_obj`.
   * For each `occurrence` in `doc["occurrences"]`:

     * Only consider dict occurrences.

     * `roles = occurrence["symbol_roles"]`. The code uses:

       ```python
       if roles & 1 == 0:
           continue  # only keep definitions (bit 1)
       ```

     * `rng = occurrence["range"]` (SCIP “range” is a list, `[startLine, startCol, endCol?]`).

       * The code takes `start_line = int(rng[0]) + 1`, converting 0‑based to 1‑based.

     * `symbol = occurrence["symbol"]` (string).
   * Fills:

     ```python
     def_map[(rel_path_obj, start_line)] = symbol
     ```

4. **Fetch GOIDs**

   `_fetch_goids(con)`:

   ```python
   SELECT urn, rel_path, start_line FROM core.goids
   ```

   This corresponds to entries you see in `/mnt/data/goids.jsonl`, e.g.:

   ```json
   {
     "urn": "goid:paul-heyse/CodeIntel/src/codeintel/analytics/ast_metrics.py#python:method:src.codeintel.analytics.ast_metrics.FileChurn.to_summary?s=38&e=52",
     "rel_path": "src/codeintel/analytics/ast_metrics.py",
     "start_line": 38,
     ...
   }
   ```

5. **Construct `(symbol, goid)` updates**

   `_build_symbol_updates(def_map, goids)`:

   * For each `(urn, rel_path, start_line)` row from GOIDs:

     * Looks up `symbol = def_map.get((rel_path, int(start_line)))`.
     * If found, appends `(symbol, urn)` to `updates`.

6. **Write into `core.goid_crosswalk.scip_symbol`**

   Uses the SQL literal from `codeintel.config.schemas.sql_builder`:

   ```python
   GOID_CROSSWALK_UPDATE_SCIP = (
       "UPDATE core.goid_crosswalk SET scip_symbol = ? WHERE goid = ?"
   )
   ```

   So `scip_symbol` becomes the SCIP symbol string (e.g.
   `"scip-python python CodeIntel 0.1.0 `codeintel.config.models`/CallGraphConfig#"`).

This gives you **definition‑level SCIP evidence attached to your canonical GOIDs**.

---

## 2. Building symbol‑use edges from SCIP

Now we use the same SCIP JSON to derive **“who uses whom” at the symbol level.**

**Module:** `codeintel.graphs.symbol_uses`
**Config:** `SymbolUsesConfig` (in `codeintel.config.models`)
**Output table:** `graph.symbol_use_edges` → `/mnt/data/symbol_use_edges.jsonl`

### 2.1 Locating & loading SCIP JSON

Two helper functions:

```python
def default_scip_json_path(repo_root: Path, build_dir: Path | None) -> Path | None:
    base = build_dir if build_dir is not None else repo_root / "build"
    scip_path = (base / "scip" / "index.scip.json").resolve()
    return scip_path if scip_path.exists() else None
```

```python
def load_scip_documents(scip_path: Path) -> list[ScipDocument] | None:
    if not scip_path.exists():
        log.warning("SCIP JSON not found at %s; skipping symbol_use_edges", scip_path)
        return None

    with scip_path.open("r", encoding="utf-8") as f:
        docs_raw = json.load(f)

    # Handle dict root with 'documents' or direct list
    if isinstance(docs_raw, dict):
        docs_raw = docs_raw.get("documents", [])

    if not isinstance(docs_raw, list):
        log.warning("SCIP JSON root (or 'documents' key) is not a list; aborting symbol_use_edges build.")
        return None

    return [cast("ScipDocument", doc) for doc in docs_raw if isinstance(doc, dict)]
```

So symbol_uses can work in two modes:

* From a pre‑configured `SymbolUsesConfig.scip_json_path`, or
* By discovering `build/scip/index.scip.json` automatically.

### 2.2 Mapping symbols → definition paths

`build_def_map(docs: list[ScipDocument]) -> dict[str, str]`:

* Iterates over all docs.
* For each doc:

  * `rel_path = str(doc["relative_path"]).replace("\\", "/")`.
* For each occurrence:

  * `symbol = occ["symbol"]`.
  * `roles = int(occ.get("symbol_roles", 0))`.
  * `is_def = bool(roles & 1)` (definition bit).
  * First time we see a definition, we capture:

    ```python
    def_path_by_symbol[symbol] = rel_path
    ```

This produces a map:

```text
SCIP symbol → defining file path (def_path)
```

### 2.3 Mapping use paths → definition paths

`build_use_def_mapping(docs, def_path_by_symbol) -> dict[str, set[str]]`:

* For each doc:

  * Treats its `relative_path` as the **use_path** (where the symbol is used):

    ```python
    use_path = str(doc["relative_path"]).replace("\\", "/")
    ```

* For each occurrence in that doc:

  * `symbol = occ["symbol"]`.

  * `roles = occ["symbol_roles"]`.

  * Considers as a “use” if:

    ```python
    is_ref = bool(roles & (2 | 4 | 8))
    # 2 = Import, 4 = WriteAccess, 8 = ReadAccess
    ```

  * Looks up `def_path = def_path_by_symbol.get(symbol)`.

  * If both exist, adds:

    ```python
    mapping.setdefault(use_path, set()).add(def_path)
    ```

Result: a mapping

```text
use_path → {def_path₁, def_path₂, ...}
```

### 2.4 Persisting `graph.symbol_use_edges`

The main builder (not fully shown in your snippet) uses these mappings and the function catalog to create rows of `SymbolUseRow`, stored in `graph.symbol_use_edges` and exported in `/mnt/data/symbol_use_edges.jsonl`.

Example rows from your export:

```json
{
  "symbol": "scip-python python CodeIntel 0.1.0 `codeintel.types`/PytestCallEntry#",
  "def_path": "src/codeintel/types.py",
  "use_path": "src/codeintel/types.py",
  "same_file": true,
  "same_module": true
},
{
  "symbol": "scip-python python CodeIntel 0.1.0 `codeintel.config.models`/HotspotsConfig#",
  "def_path": "src/codeintel/config/models.py",
  "use_path": "src/codeintel/analytics/ast_metrics.py",
  "same_file": false,
  "same_module": false
}
```

So this is **SCIP evidence path #2**: raw SCIP index → `graph.symbol_use_edges`.

---

## 3. Feeding SCIP into the call graph (fallback evidence)

Now we use those symbol‑use edges to enrich **call graph edges**.

**Module:** `codeintel.graphs.callgraph_builder`
**Config:** `CallGraphConfig`
**Output table:** `graph.call_graph_edges` → `/mnt/data/call_graph_edges.jsonl`

### 3.1 Precomputing SCIP candidates per use_path

```python
def _load_scip_candidates(
    con: duckdb.DuckDBPyConnection, repo_root: Path
) -> dict[str, tuple[str, ...]]:
    """
    Map `use_path` -> candidate `def_path` values from symbol_use_edges.

    The result is used to enrich unresolved call edges with SCIP-derived
    evidence so downstream consumers can correlate callsites to symbols even
    when GOID resolution fails.
    """
```

Steps:

1. **Try to read from `graph.symbol_use_edges`**

   ```python
   rows = con.execute(
       "SELECT def_path, use_path FROM graph.symbol_use_edges"
   ).fetchall()
   ```

   If the table/view is missing or any DuckDB error occurs, it catches `duckdb.Error` and uses `rows = []`.

2. **Build `mapping: use_path -> set(def_path)`**

   ```python
   mapping: dict[str, set[str]] = {}
   for def_path, use_path in rows:
       if def_path is None or use_path is None:
           continue
       use_norm = normalize_rel_path(str(use_path))
       mapping.setdefault(use_norm, set()).add(
           normalize_rel_path(str(def_path))
       )
   ```

3. **Fallback: directly read `index.scip.json` if no rows**

   If `mapping` is empty:

   ```python
   scip_path = symbol_uses.default_scip_json_path(repo_root, None)
   docs = symbol_uses.load_scip_documents(scip_path) if scip_path is not None else None
   if docs:
       def_map = symbol_uses.build_def_map(docs)
       mapping = symbol_uses.build_use_def_mapping(docs, def_map)
   ```

4. **Return normalized mapping**

   ```python
   return {path: tuple(sorted(defs)) for path, defs in mapping.items()}
   ```

So by the time we start edge collection, we have:

```python
scip_candidates_by_use: dict[str, tuple[str, ...]]
# use_path (rel_path of callsite file) → candidate definition paths
```

### 3.2 Passing SCIP candidates into the resolution context

In `_collect_edges(...)`:

```python
scip_candidates_by_use = _load_scip_candidates(con, repo_root)

context = EdgeResolutionContext(
    function_index=function_index,
    local_callees=callee_by_name,
    global_callees=global_callee_by_name,
    import_aliases=alias_collector,
    scip_candidates_by_use_path=scip_candidates_by_use,
)
visitor = _FileCallGraphVisitor(rel_path=rel_path, context=context)
```

`EdgeResolutionContext` gives the visitor:

* `function_index`: mapping `(path, lines) -> GOID`
* `local_callees`: names→GOID for functions in the current file
* `global_callees`: names→GOID across the repo
* `import_aliases`: alias mapping collected via LibCST
* `scip_candidates_by_use_path`: the mapping we just discussed

### 3.3 Using SCIP candidates when resolution fails

Inside `_FileCallGraphVisitor._handle_call` (LibCST path) and the AST fallback `_collect_edges_ast`, the code roughly does:

1. Extracts `callee_name` and an `attr_chain` from the call expression (`ast.Name`, `ast.Attribute`, etc.).

2. Attempts to resolve:

   * local function
   * attribute on a local/alias
   * global function via fully qualified name
   * etc.

3. When `callee_goid_h128` stays `None`, it marks the edge as unresolved:

   * `kind="unresolved"`
   * `resolved_via="unresolved"`
   * `confidence=0.0`

4. Then it attaches **SCIP candidates** for the current file:

   ```python
   scip_paths = context.scip_candidates_by_use_path.get(rel_path)
   if callee_goid is None and scip_paths:
       evidence["scip_candidates"] = list(scip_paths)
   ```

This evidence is serialized into `evidence_json` on each `CallGraphEdgeRow`.

From your exported `/mnt/data/call_graph_edges.jsonl`, an unresolved edge looks like:

```json
{
  "caller_goid_h128": 3.302982762e+37,
  "callee_goid_h128": null,
  "callsite_path": "src/codeintel/analytics/ast_metrics.py",
  "callsite_line": 48,
  "callsite_col": 28,
  "language": "python",
  "kind": "unresolved",
  "resolved_via": "unresolved",
  "confidence": 0.0,
  "evidence_json": {
    "callee_name": "len",
    "attr_chain": ["len"],
    "resolved_via": "unresolved",
    "scip_candidates": [
      "src/codeintel/analytics/ast_metrics.py",
      "src/codeintel/analytics/coverage_analytics.py",
      "src/codeintel/analytics/functions.py",
      "src/codeintel/config/models.py",
      "src/codeintel/ingestion/common.py",
      "src/codeintel/ingestion/tool_runner.py",
      "src/codeintel/models/rows.py"
    ]
  }
}
```

So even when the call graph can’t map `len(...)` to a specific GOID, downstream tools know **where SCIP says `len` is used/defined** and can use these paths as hints.

That’s **SCIP evidence path #3**.

---

## 4. Surfacing SCIP evidence to consumers

**Module:** `codeintel.storage.views`
**View:** `docs.v_call_graph_enriched`

This view joins call graph edges with GOID metadata and risk scores, but crucially it keeps the `e.evidence_json` column intact:

```sql
CREATE OR REPLACE VIEW docs.v_call_graph_enriched AS
    SELECT
        e.caller_goid_h128,
        gc.urn           AS caller_urn,
        gc.rel_path      AS caller_rel_path,
        gc.qualname      AS caller_qualname,
        rc.risk_level    AS caller_risk_level,
        rc.risk_score    AS caller_risk_score,
        e.callee_goid_h128,
        gcallee.urn      AS callee_urn,
        gcallee.rel_path AS callee_rel_path,
        gcallee.qualname AS callee_qualname,
        rcallee.risk_level AS callee_risk_level,
        rcallee.risk_score AS callee_risk_score,
        e.callsite_path,
        e.callsite_line,
        e.callsite_col,
        e.language,
        e.kind,
        e.resolved_via,
        e.confidence,
        e.evidence_json
    FROM graph.call_graph_edges e
    ...
```

So any MCP tool / API client querying `docs.v_call_graph_enriched` gets:

* Structured caller/callee GOID info.
* Risk scores (from `analytics.goid_risk_factors`).
* The full `evidence_json` blob, which for unresolved edges includes `scip_candidates` built from SCIP.

That’s how SCIP evidence becomes **user‑facing context** in your call‑graph view.

---

## 5. End‑to‑end SCIP evidence story (at a glance)

Putting it all together:

1. **SCIP indexing**

   * `ScipIngestConfig` + `ingest_scip` run `scip-python index` and `scip print --json`, producing:

     * `build/scip/index.scip`
     * `build/scip/index.scip.json` (also copied to “Document Output”)

2. **SCIP documents → DuckDB**

   * `ingest_scip` loads `index.scip.json` into DuckDB via `read_json(...)` and exposes:

     * View `scip_index_view` (each row is a SCIP document).

3. **Definitions backfilled into GOIDs**

   * `_update_scip_symbols`:

     * Builds `(rel_path, start_line) → symbol` from SCIP definitions.
     * Joins with `core.goids` to set `core.goid_crosswalk.scip_symbol` via `GOID_CROSSWALK_UPDATE_SCIP`.

4. **Symbol definition/use edges**

   * `codeintel.graphs.symbol_uses`:

     * `build_def_map(docs)` → `symbol → def_path`
     * `build_use_def_mapping(docs, def_map)` → `use_path → {def_path}`
     * Persists into `graph.symbol_use_edges` → exported as `symbol_use_edges.jsonl`.

5. **Call graph with SCIP fallback evidence**

   * `codeintel.graphs.callgraph_builder`:

     * `_load_scip_candidates`:

       * First from `graph.symbol_use_edges`, otherwise directly from `index.scip.json`.
       * Produces `use_path → tuple(def_paths)`.
     * `EdgeResolutionContext` passes this to visitors.
     * When an edge can’t be resolved to a GOID:

       * It remains `kind="unresolved"`, but `evidence_json.scip_candidates` is filled.

6. **Exposed to clients**

   * `docs.v_call_graph_enriched` emits `e.evidence_json`, so your MCP tools / UI can:

     * See unresolved edges,
     * Inspect SCIP‑derived candidate locations, and
     * Potentially present “best guess” navigation or debugging hints.

---

