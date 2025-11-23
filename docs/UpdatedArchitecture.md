Here’s the consolidated architecture spec for the whole system as we’ve designed it so far. I’ll treat this as a blueprint for an AI agent to implement the full program with minimal ambiguity.

The README you provided is the **ground truth for datasets and columns**; everything below is designed to produce exactly those artifacts under `Document Output/`. 

---

# 0. High‑level Overview

## 0.1 Goals

* **Single‑repo metadata warehouse** for a Python repo:

  * Central DuckDB database (`build/db/codeintel.duckdb`)
  * Tables grouped by schema: `core`, `graph`, `analytics`, `docs`
* **Deterministic enrichment pipeline** that:

  * Ingests AST/CST, SCIP, coverage, tests, typedness, git churn, config
  * Builds GOIDs, callgraph, CFG/DFG, import graph, symbol uses
  * Computes per‑function metrics, coverage, typedness, risk
* **Export layer** that writes:

  * Parquet + JSONL per dataset, matching the README
  * `repo_map.json`, `index.scip`/`index.scip.json`
* **Query surfaces** for AI:

  * DuckDB views under `docs.*`
  * FastAPI “CodeIntel Metadata API”
  * MCP server (local first, remote‑ready) exposing tools like `get_function_summary`, `get_callgraph_neighbors`, etc.

Everything is built so that **GOID (goid_h128 + URN)** is the primary cross‑dataset key, and **repo+commit** is embedded everywhere for stability and future multi‑repo support. 

---

# 1. Repository & Package Layout

```text
repo_root/
  src/
    codeintel/
      cli/
        __init__.py
        main.py              # `codeintel` CLI entrypoint
      config/
        __init__.py
        models.py            # Pydantic configs: repo/paths/tools
      storage/
        __init__.py
        duckdb_client.py     # connection + apply_all_schemas()
        schemas.py           # DDL for core/graph/analytics/docs
      ingestion/
        __init__.py
        repo_scan.py         # core.modules, core.repo_map, tags_index
        scip_ingest.py       # scip-python → index.scip + index.scip.json
        cst_extract.py       # core.cst_nodes
        py_ast_extract.py    # core.ast_nodes, core.ast_metrics
        docstrings_ingest.py # core.docstrings
        coverage_ingest.py   # analytics.coverage_lines
        tests_ingest.py      # analytics.test_catalog (raw)
        typing_ingest.py     # analytics.typedness, analytics.static_diagnostics
        config_ingest.py     # analytics.config_values (raw)
      graphs/
        __init__.py
        goid_builder.py      # core.goids, core.goid_crosswalk
        callgraph_builder.py # graph.call_graph_nodes, graph.call_graph_edges
        cfg_builder.py       # graph.cfg_blocks, graph.cfg_edges, graph.dfg_edges
        import_graph.py      # graph.import_graph_edges
        symbol_uses.py       # graph.symbol_use_edges
      analytics/
        __init__.py
        ast_metrics.py       # analytics.hotspots
        functions.py         # analytics.function_metrics, analytics.function_types
        coverage_analytics.py# analytics.coverage_functions
        risk_factors.py      # analytics.goid_risk_factors (SQL lives here if not already)
      config/
        views.py             # docs.v_* view definitions over DuckDB
      docs_export/
        __init__.py
        export_parquet.py    # writes *.parquet to Document Output/
        export_jsonl.py      # writes *.jsonl + repo_map.json, index.json
      orchestration/
        __init__.py
        steps.py             # PipelineStep classes & registry
        pipeline.py          # DAG runner (toposort + execute)
        run_profiles.py      # (optional) named profiles (full, incremental)
      server/
        __init__.py
        api.py               # FastAPI server over docs.* views
      mcp/
        __init__.py
        backend.py           # QueryBackend, DuckDBBackend (and later HttpBackend)
        server.py            # MCP server using FastMCP
  Document Output/           # final published Parquet + JSONL + repo_map.json
  build/
    db/
      codeintel.duckdb       # main DuckDB database
    logs/
    scip/
      index.scip             # raw SCIP index
      index.scip.json        # SCIP JSON index
  scripts/
    generate_documents.sh    # wrapper over `codeintel pipeline run --target export_docs`
```

---

# 2. Data Model & Storage (DuckDB)

## 2.1 Schemas

* `core`: raw, mostly 1:1 with direct ingestion

  * `modules`, `repo_map`, `ast_nodes`, `ast_metrics`, `cst_nodes`, `goids`, `goid_crosswalk`
* `graph`: graph‑structured derived tables

  * `call_graph_nodes`, `call_graph_edges`, `cfg_blocks`, `cfg_edges`, `dfg_edges`, `import_graph_edges`, `symbol_use_edges`
* `analytics`: higher‑level analytics

  * `hotspots`, `typedness`, `static_diagnostics`, `function_metrics`, `function_types`, `coverage_lines`, `coverage_functions`, `test_catalog`, `test_coverage_edges`, `config_values`, `goid_risk_factors`
* `docs`: AI‑facing materialized views

  * `v_function_summary`, `v_call_graph_enriched`, `v_test_to_function`, `v_file_summary`

The **column sets** must align with the README metadata for each dataset. 

## 2.3 Common Utilities

`codeintel.utils.paths`:

* `repo_relpath(repo_root, path)`: consistently computes POSIX relative paths.
* `relpath_to_module(path)`: converts file paths to dotted module names.

`codeintel.ingestion.common`:

* `iter_modules(module_map, ...)`: efficient iteration over module files with progress logging.
* `load_module_map(...)`: fetches `path -> module` mapping from DuckDB.
* `run_batch(...)`: handles idempotent inserts with `DELETE` + `INSERT` logic and logging.

## 2.4 Schema Implementation

`storage/schemas.py`:

* Provide `apply_all_schemas(con: duckdb.DuckDBPyConnection) -> None` that:

  * Creates schemas if missing (`CREATE SCHEMA IF NOT EXISTS core;`, etc.).
  * Recreates tables (`DROP TABLE IF EXISTS` + `CREATE TABLE`) to ensure schema consistency during development.
  * Creates indexes.
  * Is **idempotent** (safe to run multiple times), though currently destructive to table data.

Rules:

* Use DuckDB types:

  * `BIGINT` or `DECIMAL(38,0)` for `goid_h128`.
  * `VARCHAR` for string columns.
  * `BOOLEAN`, `INTEGER`, `DOUBLE`, `TIMESTAMP`, `JSON` as needed.
* Put indices on hot query keys:

  * `core.goids(goid_h128)`, `core.goids(urn)`
  * `analytics.goid_risk_factors(function_goid_h128)`, `analytics.coverage_functions(function_goid_h128)`
  * `analytics.test_catalog(test_id)`, `analytics.test_coverage_edges(function_goid_h128)`
  * `core.modules(path)` & `core.modules(module)`.

---

# 3. Configuration Models (Pydantic)

`config/models.py` defines:

* `RepoConfig`:

  * `repo: str`, `commit: str`
* `PathsConfig`:

  * `repo_root: Path`
  * `build_dir: Path` (defaults to `repo_root / "build"`)
  * `db_path: Path` (defaults to `build_dir / "db" / "codeintel.duckdb"`)
  * `document_output_dir: Path` (defaults to `repo_root / "Document Output"`)
  * Derived properties: `db_dir`, `logs_dir`, `scip_dir`
* `ToolsConfig`:

  * `scip_python_bin: str = "scip-python"`
  * `scip_bin: str = "scip"`
  * `pyright_bin: str = "pyright"`
  * `coverage_file: Optional[Path]`
  * `pytest_report_path: Optional[Path]`
* `CodeIntelConfig`:

  * `repo: RepoConfig`
  * `paths: PathsConfig`
  * `tools: ToolsConfig`
  * `default_targets: List[str] = ["export_docs"]`
  * Helper `from_cli_args(...)` to construct from CLI params.

All path fields normalize with `Path.expanduser()` and resolve relative to `repo_root` or `build_dir` appropriately.

---

# 4. Storage / Connection Layer

`storage/duckdb_client.py`:

* `DuckDBConfig`:

  * `db_path: Path`
  * `read_only: bool`
  * `threads: Optional[int]`
* `DuckDBClient`:

  * Lazily opens a DuckDB connection on first `connect()`.
  * For `read_only=False`, creates `db_path.parent` and calls `apply_all_schemas(con)`.
  * Applies `PRAGMA threads` if provided.
  * Usable as context manager.
* `get_connection(db_path: Path, read_only: bool = False)`:

  * Convenience wrapper returning `DuckDBPyConnection`.

**Invariant**: Each DB file is for a single repo+commit (current design). The repo+commit values are still stored in tables for future multi‑repo support but queries generally assume 1:1.

---

# 5. Ingestion Layer

### Common patterns

* Each ingestion module exposes a single main function:

  * Signature: `ingest_<thing>(con: DuckDBPyConnection, repo_root: Path, repo: str, commit: str, ...) -> None`
* Ingestion functions:

  * **Delete** rows for this repo+commit before inserting new ones (where appropriate).
  * Are **idempotent per repo+commit**.
  * Rely on external tools when needed (scip, pyright, coverage.py, git).

## 5.1 `repo_scan.py`

Responsibilities:

* Walk `repo_root` and:

  * Populate `core.modules`:
    * Scopes scanning to `src/` directory if present, falling back to `repo_root`.
    * Uses `repo_relpath` for consistent path normalization.
    * `module`: dotted path from repo root (e.g., `pkg.mod`)
    * `path`: repo‑relative path (forward slashes)
    * `repo`, `commit`, `language="python"`, `tags`, `owners`
  * Populate `core.repo_map`:

    * Single row per (repo, commit)
    * `modules`: JSON mapping `{module: path}`
    * `overlays`: JSON dict (empty for now)
    * `generated_at`: timestamp
  * Populate `analytics.tags_index` from `tags_index.yaml`:

    * `tag`, `description`, `includes`, `excludes`, `matches`

Rules:

* Ignore `.git`, `.hg`, `.venv`, `__pycache__`, `build`, etc.
* Tags:

  * For each file, compute `tags` by matching its repo‑relative path against `includes`/`excludes` patterns and `matches` lists.

## 5.2 `scip_ingest.py`

Responsibilities:

* Run `scip-python index <repo_root> --output <build/scip/index.scip>`.
* Run `scip print --format json index.scip > index.scip.json`.
* Copy both to:

  * `build/scip/` (source of truth)
  * `Document Output/` (exported artifacts)
* Register a **DuckDB view**:

  ```sql
  CREATE OR REPLACE VIEW scip_index_view AS
  SELECT unnest(documents, recursive:=true) FROM read_json('<path/to/index.scip.json>');
  ```

  * Handles SCIP JSON structure where the root is a dictionary containing a `documents` list.

If scip binaries are missing, module logs a warning and returns without error.

## 5.3 `py_ast_extract.py`

Responsibilities:

* For each `core.modules` row (language=python):
  * Parse source with standard library `ast`.
  * Visit tree with `AstVisitor` to populate `core.ast_nodes`:
    * Entities: `Module`, `ClassDef`, `FunctionDef`, `AsyncFunctionDef`.
    * Fields: `path`, `node_type`, `name`, `qualname`, `lineno`, `end_lineno`, `col_offset`, `end_col_offset`, `parent_qualname`, `decorators`, `docstring=None`, `hash` (stable hash).
  * Compute and populate `core.ast_metrics`:
    * `rel_path`, `node_count`, `function_count`, `class_count`, `avg_depth`, `max_depth`, `complexity`, `generated_at`.
    * Complexity is a heuristic count of decision points (if/for/while/try/with/…).

## 5.4 `cst_extract.py`

Responsibilities:

* For each `core.modules` row (language=python):
  * Parse source with LibCST.
  * Visit tree with `CstVisitor` to populate `core.cst_nodes`:
    * `path`, `node_id`, `kind`, `span`, `text_preview`, `parents`, `qnames`.
  * Filters nodes to capture only high-value structural elements (Module, Class, Function, Assign, Call, Import, Control Flow, etc.) to optimize performance.

## 5.5 `coverage_ingest.py`

Responsibilities:

* Read Coverage.py data from `.coverage` (or explicit path).
* For each measured file under `repo_root`:

  * Use `coverage.Coverage` API (`analysis2` / `get_data().contexts_by_lineno`) to compute:

    * `is_executable`, `is_covered`, `hits`, `context_count` per line.
* Insert into `analytics.coverage_lines` with fields from README: repo, commit, rel_path, line, is_executable, is_covered, hits, context_count, created_at. 

## 5.6 `tests_ingest.py`

Responsibilities:

* Locate a pytest JSON report (`pytest-json-report`), defaulting to common filenames.
  * If missing, log warning and skip (leaving `analytics.test_catalog` empty).
* Load `tests` list from it.
* For each test:

  * Extract `nodeid`, outcome, duration, markers/keywords.
  * Parse `nodeid` into `rel_path` + `qualname` (`tests/test_app.py::TestFoo::test_bar[param]` → `tests/test_app.py`, `TestFoo::test_bar[param]`).
  * Insert into `analytics.test_catalog` as per README. 

Does **not** compute `test_coverage_edges`; that’s an analytics step combining coverage contexts with GOIDs.

## 5.7 `typing_ingest.py`

Responsibilities:

* Per `.py` file:

  * Use `ast` to compute:

    * Annotation ratios for params & returns.
    * `untyped_defs` (top-level functions not fully annotated).
  * Run `pyright --outputjson` and aggregate error counts per file.
* Populate:

  * `analytics.typedness` with fields: `path`, `type_error_count`, `annotation_ratio (JSON)`, `untyped_defs`, `overlay_needed`.
  * `analytics.static_diagnostics` with `rel_path`, `pyrefly_errors=0`, `pyright_errors`, `total_errors`, `has_errors`. 

## 5.8 `config_ingest.py`

Responsibilities:

* Discover config files by extension: `.yaml`, `.yml`, `.toml`, `.json`, `.ini`, `.cfg`, `.env`.
* Scopes discovery to `src/` directory if present.
* Parse each into a dictionary using appropriate library (PyYAML, `tomllib`, `json`, `configparser` or manual `.env` parsing).
* Flatten nested structure into keypaths:
  * `service.database.host`, `service.database.ports.0`, etc.
* Insert into `analytics.config_values`:
  * `config_path`, `format`, `key`, `reference_paths=[]`, `reference_modules=[]`, `reference_count=0`. 

Later analytics can fill references by scanning AST/symbol uses.

## 5.9 `docstrings_ingest.py`

Responsibilities:

* Iterate over modules in `core.modules`.
* Parse each file using Python's `ast` module.
* Use `docstring_parser` to parse raw docstrings into structured fields (params, returns, raises, examples).
* Populate `core.docstrings`:
  * `repo`, `commit`, `rel_path`, `module`, `qualname`, `kind`, `lineno`.
  * Structured fields: `short_desc`, `long_desc`, `params` (JSON), `returns` (JSON), etc.

*Dependencies*: Uses standard library `ast` and `docstring-parser`, removing the need for `griffe`.

Responsibilities:

* Discover config files by extension: `.yaml`, `.yml`, `.toml`, `.json`, `.ini`, `.cfg`, `.env`.
* Parse each into a dictionary using appropriate library (PyYAML, `tomllib`, `json`, `configparser` or manual `.env` parsing).
* Flatten nested structure into keypaths:

  * `service.database.host`, `service.database.ports.0`, etc.
* Insert into `analytics.config_values`:

  * `config_path`, `format`, `key`, `reference_paths=[]`, `reference_modules=[]`, `reference_count=0`. 

Later analytics can fill references by scanning AST/symbol uses.

---

# 6. Graph Layer

## 6.1 `goid_builder.py`

Responsibilities:

* For each AST node representing a module/class/function/method:

  * Compute GOID:

    ```text
    input: repo, commit, language, rel_path, kind, qualname, start_line, end_line
    hash: BLAKE2b (digest_size=16 bytes) → 128-bit integer -> `goid_h128`
    ```
  * Build URN:

    ```text
    goid:<repo>/<rel_path>#<language>:<kind>:<qualname>?s=<start>&e=<end>
    ```
* Populate:

  * `core.goids` with columns as in README’s GOID Registry. 
  * `core.goid_crosswalk`:

    * `goid` (URN string), `lang`, `module_path`, `file_path`, `start_line`, `end_line`, `scip_symbol (NULL)`, `ast_qualname`, `cst_node_id (NULL)`, `chunk_id (NULL)`, `symbol_id (NULL)`, `updated_at`.

## 6.2 `callgraph_builder.py`

Responsibilities:

* Build **call graph nodes**:

  * From `core.goids` where `kind IN ('function', 'method', 'class', 'module')`:

    * Insert into `graph.call_graph_nodes`:

      * `goid_h128`, `language`, `kind`, `arity` (=-1 for unknown initially), `is_public`, `rel_path`. 

* Build **call graph edges**:

  * For each file with function GOIDs:

    * Map each function GOID by `(start_line, end_line)` and by local name.
    * Parse source with LibCST and walk with `_FileCallGraphVisitor`:

      * Track current function GOID (by span).
      * For each `Call` node:

        * Extract callee name/attr chain.
        * Resolve to callee GOID by local name if possible.
        * Append edge: `(caller_goid_h128, callee_goid_h128?, callsite_path, callsite_line, callsite_col, language, kind, resolved_via, confidence, evidence_json)`.
  * Deduplicate edges per `(caller_goid_h128, callee_goid_h128, callsite_path, callsite_line, callsite_col)`.

Insert into `graph.call_graph_edges` with columns from README. 

## 6.3 `cfg_builder.py`

Minimal implementation (extendable later):

* For each function GOID:

  * Create a single block:

    * `block_idx=0`, `block_id="<goid>:block0"`, `label="body:0"`, `kind="body"`, `start_line`, `end_line`, `stmts_json="[]"`, `in_degree=0`, `out_degree=0`.
  * Insert into `graph.cfg_blocks`.
* `graph.cfg_edges` and `graph.dfg_edges` remain empty for now (placeholders).

This gives a usable table schema; CFG/DFG logic can be enriched later to match the full spec in the README. 

## 6.4 `import_graph.py`

Responsibilities:

* From `core.modules`:

  * For each `.py` file:

    * Parse with LibCST and `_ImportCollector`, collecting `(src_module, dst_module)` pairs (derived from `Import`/`ImportFrom`).
* Build adjacency map and compute Tarjan SCCs to assign `cycle_group` per module.
* Compute:

  * `src_fan_out` (number of outgoing edges)
  * `dst_fan_in` (number of incoming edges)
* Insert into `graph.import_graph_edges` with columns from README. 

## 6.5 `symbol_uses.py`

Responsibilities:

* From `index.scip.json`:

  * First pass: map `symbol → def_path` for occurrence roles marked as definitions.
  * Second pass: for each reference occurrence (role bit 2, 4, or 8 - Import, Write, Read):

    * Determine `use_path`.
    * Lookup `def_path` for symbol.
    * Determine `same_file` and `same_module` (by joining to `core.modules`).
    * Deduplicate edges to prevent primary key violations.
* Insert into `graph.symbol_use_edges`:

  * `symbol`, `def_path`, `use_path`, `same_file`, `same_module`. 

---

# 7. Analytics Layer

## 7.1 `ast_metrics.py` – hotspots

Responsibilities:

* From `core.ast_metrics`: get `rel_path`, `complexity`.

* From `git log --numstat` (max N commits):

  * For each file, compute:

    * `commit_count`, `author_count`, `lines_added`, `lines_deleted`.

* For each file, compute `score`:

  ```text
  score = 0.4*log1p(commit_count) +
          0.3*log1p(author_count) +
          0.2*log1p(lines_added + lines_deleted) +
          0.1*log1p(complexity + 1.0)
  ```

* Insert into `analytics.hotspots` with fields from README. 

## 7.2 `functions.py` – function_metrics + function_types

Responsibilities:

* For each file, group function GOIDs (`core.goids` where `kind IN ('function','method')`).
* Parse each file once into Python `ast`.
* Map AST function nodes by `(lineno, end_lineno)` to GOIDs.
* For each GOID/function:

  * Structural metrics:

    * `loc`, `logical_loc`
    * Parameter counts, varargs/varkw
    * `is_async`, `is_generator`
    * `return_count`, `yield_count`, `raise_count`
    * `cyclomatic_complexity`, `max_nesting_depth`, `stmt_count`, `decorator_count`, `has_docstring`, `complexity_bucket`
  * Type metrics:

    * For all params (excluding self/cls):

      * total, annotated, unannotated
      * `param_types: {name -> annotation string|null}`
    * Return annotation: `return_type`, `has_return_annotation`, `return_type_source`
    * Derived booleans: `fully_typed`, `partial_typed`, `untyped`
    * `param_typed_ratio`
    * `typedness_bucket`, `typedness_source`

Insert into:

* `analytics.function_metrics` and
* `analytics.function_types` as per README. 

## 7.3 `coverage_analytics.py` – coverage_functions

Responsibilities:

* Join `core.goids` (functions/methods) to `analytics.coverage_lines` by:

  * `repo`, `commit`, `rel_path`
  * line range `[start_line, end_line]`
* Aggregate per function:

  * `executable_lines = count(is_executable)`
  * `covered_lines = count(is_executable AND is_covered)`
  * `coverage_ratio = covered_lines / executable_lines` (NULL if `executable_lines=0`)
  * `tested = covered_lines > 0`
  * `untested_reason`:

    * `no_executable_code` if `executable_lines=0`
    * `no_tests` if `executable_lines>0` and `covered_lines=0`
    * "" otherwise

Insert into `analytics.coverage_functions` as per README. 

## 7.4 Test coverage edges (implementation location)

We haven’t fully implemented `test_coverage_edges` yet, but the spec is:

* Use coverage with `dynamic_context = test_function`, where each context corresponds to a `test_id` (pytest nodeid).
* For each `(test_id, function_goid_h128)` pair:

  * `covered_lines`, `executable_lines`, `coverage_ratio`, `last_status`, `repo`, `commit`, `urn`, `rel_path`, `qualname`.
* Insert into `analytics.test_coverage_edges`. 

Implementation can live in `analytics/tests_analytics.py`.

## 7.5 `risk_factors.py` – goid_risk_factors

Responsibilities:

* SQL join across:

  * `analytics.function_metrics`
  * `analytics.function_types`
  * `analytics.coverage_functions`
  * `analytics.hotspots`
  * `analytics.typedness`
  * `analytics.static_diagnostics`
  * `analytics.test_coverage_edges` + `analytics.test_catalog` for test stats
  * `core.modules` for tags/owners

* Compute `risk_score` as a weighted heuristic:

  ```sql
  risk_score =
      COALESCE(1.0 - cf.coverage_ratio, 1.0) * 0.4 +
      CASE fm.complexity_bucket
          WHEN 'high'   THEN 0.4
          WHEN 'medium' THEN 0.2
          ELSE 0.0
      END +
      CASE WHEN sd.has_errors THEN 0.2 ELSE 0.0 END +
      CASE WHEN h.score > 0 THEN 0.1 ELSE 0.0 END
  ```

* `risk_level` buckets:

  * `high` if `risk_score >= 0.7`
  * `medium` if `risk_score >= 0.4`
  * `low` otherwise

Insert into `analytics.goid_risk_factors` with full column set as per README. 

---

# 8. AI‑Facing Views (`config/views.py`)

Define four core views in `docs` schema:

## 8.1 `docs.v_function_summary`

* One row per function GOID, joining:

  * `analytics.goid_risk_factors` (base)
  * `analytics.function_metrics` (structural details)
* Fields:

  * Identity: `function_goid_h128`, `urn`, `repo`, `commit`, `rel_path`, `language`, `kind`, `qualname`
  * Metrics: loc, logical_loc, cyclomatic_complexity, complexity_bucket
  * Structural details: params, async/generator, returns/yields/raises
  * Typedness: `typedness_bucket`, `typedness_source`
  * Hotspot & diagnostics: `hotspot_score`, `file_typed_ratio`, `static_error_count`, `has_static_errors`
  * Coverage & tests: `executable_lines`, `covered_lines`, `coverage_ratio`, `tested`, `test_count`, `failing_test_count`, `last_test_status`
  * Risk: `risk_score`, `risk_level`
  * Ownership: `tags`, `owners`

## 8.2 `docs.v_call_graph_enriched`

* Base: `graph.call_graph_edges` joined to:

  * `core.goids` (caller/callee metadata)
  * `analytics.goid_risk_factors` (risk info)
* Columns for caller and callee URNs, qualnames, risk, plus callsite context.

## 8.3 `docs.v_test_to_function`

* Base: `analytics.test_coverage_edges` joined to:

  * `analytics.test_catalog`
  * `analytics.goid_risk_factors`
* Columns: test metadata + function metadata + per‑edge coverage.

## 8.4 `docs.v_file_summary`

* Base: `core.modules` joined to:

  * `core.ast_metrics`
  * `analytics.hotspots`
  * `analytics.typedness`
  * `analytics.static_diagnostics`
  * Aggregated function risk stats per file.

Views should be (re)created at startup in both CLI pipeline and FastAPI server.

---

# 9. Export Layer

## 9.1 `docs_export/export_parquet.py`

* Map DuckDB tables to Parquet filenames in `Document Output/`:

  * `core.goids` → `goids.parquet`
  * `core.goid_crosswalk` → `goid_crosswalk.parquet`
  * `graph.call_graph_nodes` → `call_graph_nodes.parquet`
  * `graph.call_graph_edges` → `call_graph_edges.parquet`
  * etc., matching README dataset names exactly. 

* For each mapping:

  ```sql
  COPY (SELECT * FROM <table>) TO ? (FORMAT PARQUET);
  ```

* Log and continue on missing tables.

## 9.2 `docs_export/export_jsonl.py`

* Mirror table → JSONL mapping (`*.jsonl`).

* Use:

  ```sql
  COPY (SELECT * FROM <table>) TO ? (FORMAT JSON, ARRAY FALSE);
  ```

  to produce newline‑delimited JSON objects (JSONL). 

* Additionally:

  * Export `core.repo_map` to `repo_map.json` (first row only, assuming single repo).
  * Write `index.json`:

    ```json
    {
      "generated_at": "ISO8601",
      "files": ["goids.jsonl", "..."]
    }
    ```

---

# 10. CLI & Pipeline Orchestration

## 10.1 PipelineContext & Steps

`orchestration/steps.py`:

* `PipelineContext`:

  * `repo_root`, `db_path`, `build_dir`, `repo`, `commit`, `extra: Dict`
* `PipelineStep` Protocol:

  * `name: str`
  * `deps: Sequence[str]`
  * `run(ctx: PipelineContext, con: DuckDBPyConnection) -> None`

Define step classes:

* Ingestion:

  * `RepoScanStep`, `SCIPIngestStep`, `AstCstStep`, `CoverageIngestStep`, `TestsIngestStep`, `TypingIngestStep`, `ConfigIngestStep`
* Graphs:

  * `GoidsStep`, `CallGraphStep`, `CFGStep`, `ImportGraphStep`, `SymbolUsesStep`
* Analytics:

  * `HotspotsStep`, `FunctionAnalyticsStep`, `CoverageAnalyticsStep`, `RiskFactorsStep`
* Export:

  * `ExportDocsStep` (calls `export_all_parquet` + `export_all_jsonl`)

Maintain `PIPELINE_STEPS: Dict[str, PipelineStep]`.

## 10.2 DAG Runner

`orchestration/pipeline.py`:

* `_toposort_steps(targets: Iterable[str]) -> List[PipelineStep]`:

  * DFS with cycle detection.
  * Starts from each target, recursively visits `deps`.
* `run_pipeline(ctx, con, targets)`:

  * Toposort steps.
  * Execute `step.run(ctx, con)` in order.

### Typical full pipeline target

`export_docs` depends on `risk_factors`, which depends on metrics, coverage, hotspots, etc. Running:

```bash
codeintel pipeline run --repo-root . --repo <slug> --commit <sha> --target export_docs
```

should produce all analytics and export datasets.

## 10.3 CLI (`cli/main.py`)

Commands:

* `codeintel pipeline run`:

  * Common args: `--repo-root`, `--repo`, `--commit`, `--db-path`, `--build-dir`, `--document-output-dir`.
  * `--target` (repeatable); defaults to `["export_docs"]`.
  * Constructs `CodeIntelConfig`, opens DB (read/write), builds `PipelineContext`, and calls `run_pipeline`.
* `codeintel docs export`:

  * Same common args.
  * Opens DB (read‑only) and calls `export_all_parquet` + `export_all_jsonl`.

Logging controlled by `-v/--verbose` (0=WARNING, 1=INFO, 2+=DEBUG).

---

# 11. FastAPI “CodeIntel Metadata API”

`server/api.py`:

* Initialize at startup:

  * `con = duckdb.connect(DB_PATH, read_only=False)`
  * `apply_all_schemas(con)`
  * `create_all_views(con)`
* Maintain connection on `app.state.con`.

Endpoints (OpenAPI 3.0; FastAPI generates spec):

1. `GET /function/summary/by-urn` → `FunctionSummary`
2. `GET /function/summary/by-goid` → `FunctionSummary`
3. `GET /function/callgraph` → `List[CallGraphEdge]`
4. `GET /tests/for-function` → `List[TestToFunctionEdge]`

Schemas:

* `FunctionSummary`, `CallGraphEdge`, `TestToFunctionEdge` Pydantic models, matching the DuckDB view columns and OpenAPI JSON schema described earlier.

Standards:

* OpenAPI 3.0 JSON spec auto‑exposed at `/openapi.json`.
* Use Pydantic types (lists, nested objects) so JSON Schema is fully explicit.

The HTTP API is optional for local AI, but it’s the foundation for **remote QueryBackend** later.

---

# 12. MCP Layer

## 12.1 `QueryBackend` & `DuckDBBackend`

`mcp/backend.py` defines:

* `QueryBackend` Protocol:

  ```python
  class QueryBackend(Protocol):
      def get_function_summary(...): ...
      def list_high_risk_functions(...): ...
      def get_callgraph_neighbors(...): ...
      def get_tests_for_function(...): ...
      def get_file_summary(...): ...
      def list_datasets(...): ...
      def read_dataset_rows(...): ...
  ```

* `DuckDBBackend(QueryBackend)`:

  * Holds `con: DuckDBPyConnection`, `repo: str`, `commit: str`.
  * Methods implemented as parameterized SQL queries over `docs.v_*` and analytics tables, using helper `_fetch_one_dict` and `_fetch_all_dicts`.

Later you can add `HttpBackend(QueryBackend)` calling FastAPI (`/function/summary/by-urn`, `/function/callgraph`, etc.).

## 12.2 MCP Server (`mcp/server.py`)

Uses the **FastMCP** Python MCP SDK.

* On module import:

  * Resolve config from env:

    * `CODEINTEL_REPO_ROOT`, `CODEINTEL_DB_PATH`, `CODEINTEL_REPO`, `CODEINTEL_COMMIT`
  * Open DuckDB connection `read_only=True`.
  * Instantiate `DuckDBBackend`.
  * Create MCP server:

    ```python
    mcp = FastMCP("CodeIntel", json_response=True)
    ```

* Tools (`@mcp.tool()`):

  * `get_function_summary(urn?, goid_h128?, rel_path?, qualname?) -> {"found": bool, "summary": dict}`
  * `list_high_risk_functions(min_risk=0.7, limit=50, tested_only=False) -> List[dict]`
  * `get_callgraph_neighbors(goid_h128, direction="both", limit=50) -> {"incoming": [...], "outgoing": [...]}`
  * `get_tests_for_function(goid_h128?, urn?) -> List[dict]`
  * `get_file_summary(rel_path) -> {"found": bool, "file": dict}`
  * `list_datasets() -> List[dict]`
  * `read_dataset_rows(dataset_name, limit=50, offset=0) -> List[dict]`

* Entry point:

  ```python
  def main():
      mcp.run()  # stdio transport
  ```

* Also provide `__main__.py` to allow `python -m codeintel.mcp`.

### MCP Standards

* FastMCP uses JSON Schema behind the scenes:

  * Tool parameters and results must be JSON‑serializable.
  * Type hints + docstrings define the schema.
* Tools should avoid returning huge datasets; enforce `limit`/`offset` parameters and keep defaults modest.

### Cursor Manifest Example

Per‑project `.cursor/mcp.json`:

```jsonc
{
  "mcpServers": {
    "codeintel": {
      "command": "python",
      "args": ["-m", "codeintel.mcp"],
      "env": {
        "CODEINTEL_REPO_ROOT": "/abs/path/to/repo",
        "CODEINTEL_REPO": "my-org/my-repo",
        "CODEINTEL_COMMIT": "deadbeef",
        "CODEINTEL_DB_PATH": "/abs/path/to/repo/build/db/codeintel.duckdb"
      }
    }
  }
}
```

Cursor (and OpenAI CLI MCP) will then expose these tools to the LLM for local querying.

---

# 13. Standards & Design Conventions

* **OpenAPI 3.0**:

  * FastAPI app exposes `/openapi.json`.
  * Pydantic models (`FunctionSummary`, etc.) map 1:1 to DB view columns.
  * Use meaningful descriptions and examples where helpful.

* **JSON Schema**:

  * Automatically derived from Pydantic and FastMCP.
  * Parameters/results for MCP tools must be strictly typed.
  * Avoid `Any` except where truly necessary; prefer explicit `Dict[str, Any]`.

* **File formats**:

  * Parquet: columnar, DuckDB‑compatible, for analytics.
  * JSONL: one JSON row per line, no wrapping arrays; produced by `COPY ... FORMAT JSON, ARRAY FALSE`.

* **ID & Key design**:

  * GOIDs are **128‑bit BLAKE2b hashes** of entity descriptors.
  * GOID URN is the canonical human‑readable identifier.
  * Cross‑joins use `goid_h128` or `urn`.

* **Repo + commit**:

  * Always included on per‑entity rows (`repo`, `commit`).
  * Current assumption is **1 repo+commit per DB file**; spec is future‑proof to multi‑repo.

* **Idempotency & determinism**:

  * Each pipeline step deletes any prior data for `repo, commit` before inserting.
  * No randomness; deterministic given repo state and tool versions.
  * Edges sorted by stable keys when deduplicating.

* **Error handling**:

  * Missing external tools (scip, pyright) should **log warnings** and leave corresponding tables/views empty—not crash the whole pipeline.
  * Parsers (LibCST, Python ast) should catch syntax errors; those files are skipped with logged warnings.

* **Performance**:

  * Avoid scanning entire DB in MCP tools; always require filters or `LIMIT`.
  * Use DuckDB indexes for heavy joins (especially on `goid_h128` and `rel_path`).
  * Gate expensive analytics (e.g., git churn) behind configuration if needed.

---

# 14. How an AI Agent Should Use This Spec

If an AI coding agent is tasked with implementing or extending this system, it should:

1. **Treat the README datasets as the source of truth** for column names and types. 
2. **Implement modules and functions exactly as specified** in this architecture:

   * Use the described signatures.
   * Respect idempotency and repo+commit semantics.
3. **Keep Cross‑layer contracts consistent**:

   * GOID generation stable and deterministic.
   * Views in `docs.*` must match both FastAPI models and MCP tool outputs.
4. **Ensure OpenAPI/JSON Schema accuracy**:

   * FastAPI endpoints and MCP tools should use Pydantic models so JSON schemas are precise.
   * Any evolution of views/tables must be reflected in both the API and MCP tool schemas.
5. **Preserve local‑first, remote‑ready design**:

   * All query logic should live in `QueryBackend`; MCP server and FastAPI just route to it (or its HTTP equivalent).
   * When adding remote serving, create `HttpBackend` that maps method calls → HTTP calls with matching OpenAPI contracts.

If you want, the next layer we could specify is a test strategy: e.g., fixtures for small toy repos, golden JSONL snapshots for Document Output, and integration tests that spin up MCP + FastAPI against the same DB and validate consistent answers.
