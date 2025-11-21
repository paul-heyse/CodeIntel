Here’s a full architecture design for “v2 from scratch” that reproduces (and slightly improves) the datasets described in your README, with a clean DuckDB-centric core, clear module boundaries, and an AI‑friendly query layer. 

---

## 1. Goals & Scope

**Primary goal:**
Build a fresh, maintainable code-intel metadata pipeline whose ultimate outputs are the Parquet/JSONL artifacts described in `README_METADATA.md` (GOIDs, callgraph, CFG/DFG, coverage, tests, risk factors, etc.), all stitched via DuckDB.

**Key requirements:**

* Use a small set of well-chosen Python libraries:

  * **LibCST** for Python AST/CST
  * **tree-sitter** for future multi-language or complementary parsing
  * **DuckDB** as the analytics + storage engine
  * **PyArrow/Parquet** for file formats
  * **scip-python** for symbol graph
  * **coverage.py** & **pytest-json-report** for coverage and tests
  * **Pyright/Pyrefly** for typedness/diagnostics
* Produce the datasets described in the README and stitch them with consistent keys (primarily GOIDs). 
* Keep the codebase modular and navigable.
* Provide a **canonical query system** and a set of **DuckDB views** + **JSONL “docs”** that are easy for AI agents to consume.

---

## 2. High‑Level Architecture

At a high level:

1. **Ingestion & Parsing layer**

   * Scan repo & Git history
   * Run LibCST/tree-sitter, scip-python, type checkers, coverage, pytest
   * Output “raw” structured events (AST/CST nodes, coverage lines, test reports, diagnostics) into DuckDB.

2. **Core Graph layer**

   * Build GOID registry + crosswalk
   * Build call graph, CFG, DFG, import graph, symbol uses, config graph
   * All intermediate graph structures exist as DuckDB tables.

3. **Analytics & Risk layer**

   * Compute metrics per file/function, hotspots, typedness, coverage, test edges, risk factors
   * These are mostly SQL transforms over the graph + raw tables.

4. **Document Output layer**

   * From DuckDB, export **Parquet + JSONL** into `Document Output/` exactly as your spec describes. 

5. **AI Access layer**

   * A set of **canonical DuckDB views** + **query templates**
   * Optional thin Python/HTTP service that exposes a small, well‑typed query API for agents.

---

## 3. Technology Stack

### Core Python stack

* **Python 3.11+**
* **LibCST**: Python CST/AST extraction, import metadata, docstrings, etc.
* **tree-sitter / tree_sitter_languages**:

  * Optional for non-Python; also useful for low-level CFG/DFG experiments.
* **DuckDB**:

  * Main analytical DB, used for joins, aggregations, and exports.
* **PyArrow & pyarrow.parquet**:

  * In-memory tables; efficient transfer between Python and DuckDB; Parquet I/O.
* **pydantic / dataclasses**:

  * Structured models for pipeline steps and configs.
* **Typer / Click**:

  * CLI entrypoints.

### External tooling & integrations

* **scip-python**:

  * Repository symbol index → `index.scip` / `index.scip.json`.
* **GitPython**:

  * Git history for hotspots (commit counts, churn).
* **coverage.py**:

  * Line-level and context coverage.
* **pytest + pytest-json-report**:

  * Test events → test_catalog + test coverage edges.
* **Pyright & Pyrefly**:

  * Static diag + typedness for files.
* **YAML (ruamel.yaml or PyYAML)**:

  * For tags index.

---

## 4. Repository Layout

A concrete, opinionated layout:

```text
repo_root/
  src/
    codeintel/
      cli/
        __init__.py
        main.py              # `codeintel` CLI
      config/
        __init__.py
        models.py            # pydantic configs: paths, repo info, etc.
      storage/
        duckdb_client.py     # connection + migration helpers
        schemas.py           # schema definitions & DDL for all tables
      ingestion/
        repo_scan.py         # repo_map, modules, tags_index
        scip_ingest.py       # scip-python integration
        ast_cst_extract.py   # LibCST, tree-sitter, ast_nodes, cst_nodes
        coverage_ingest.py   # coverage_lines
        tests_ingest.py      # test_catalog, test_coverage_edges (raw)
        typing_ingest.py     # typedness, static_diagnostics
        config_ingest.py     # config_values (raw)
      graphs/
        goid_builder.py      # goids, goid_crosswalk
        callgraph_builder.py # call_graph_nodes, call_graph_edges
        cfg_builder.py       # cfg_blocks, cfg_edges, dfg_edges
        import_graph.py      # import_graph_edges
        symbol_uses.py       # symbol_use_edges
      analytics/
        ast_metrics.py       # ast_metrics, hotspots
        functions.py         # function_metrics, function_types
        coverage_analytics.py# coverage_functions
        tests_analytics.py   # normalized test_catalog/test_coverage_edges
        risk_factors.py      # goid_risk_factors
      docs_export/
        export_parquet.py    # writes *.parquet under Document Output/
        export_jsonl.py      # COPY to JSONL via DuckDB
      orchestration/
        pipeline.py          # Pipeline orchestration graph
        steps.py             # Step interface & registry
        run_profiles.py      # configs for full vs. incremental runs
  Document Output/           # final datasets (Parquet + JSONL)
  build/
    db/
      codeintel.duckdb       # main DuckDB database
    logs/
  scripts/
    generate_documents.sh    # small wrapper over `codeintel docs export`
```

This layout intentionally:

* Groups logic by concern (ingestion, graphs, analytics).
* Keeps DuckDB-specific code central in `storage/` and `docs_export/`.
* Keeps orchestration logic independent of any particular step implementation.

---

## 5. DuckDB Schema & Stitching Strategy

### 5.1 Schema namespaces

Adopt schemas (DuckDB supports `schema.table`):

* `core` – repo-level, GOIDs, crosswalk, AST/CST, modules, repo_map.
* `graph` – call graph, CFG/DFG, import graph, symbol uses, config graph.
* `analytics` – hotspots, typedness, metrics, coverage, tests, risk factors, diagnostics.
* `docs` – optional denormalized views for AI consumption.

Each table maps directly to the datasets defined in the README. 

Example DDL snippet:

```sql
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS graph;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS docs;

CREATE TABLE IF NOT EXISTS core.goids (
  goid_h128        DECIMAL(38,0),
  urn              TEXT,
  repo             TEXT,
  commit           TEXT,
  rel_path         TEXT,
  language         TEXT,
  kind             TEXT,
  qualname         TEXT,
  start_line       INTEGER,
  end_line         INTEGER,
  created_at       TIMESTAMP
);
```

Repeat similarly for:

* `core.goid_crosswalk`, `core.ast_nodes`, `core.cst_nodes`, `core.ast_metrics`, `core.modules`, `core.repo_map`
* `graph.call_graph_nodes`, `graph.call_graph_edges`, `graph.cfg_blocks`, `graph.cfg_edges`, `graph.dfg_edges`, `graph.import_graph_edges`, `graph.symbol_use_edges`, `graph.config_values`
* `analytics.hotspots`, `analytics.typedness`, `analytics.function_metrics`, `analytics.function_types`, `analytics.static_diagnostics`, `analytics.coverage_lines`, `analytics.coverage_functions`, `analytics.test_catalog`, `analytics.test_coverage_edges`, `analytics.goid_risk_factors`, `analytics.line_coverage` (alias for coverage_lines if desired), `analytics.tags_index` etc.

> **Key design choice:**
> DuckDB tables are the **single source of truth**. Parquet/JSONL are just export formats, not the primary store.

### 5.2 Key stitching mechanisms

* **GOID join**:

  * The canonical foreign key: `goid_h128` / `function_goid_h128` used across graphs & analytics tables. 
  * `core.goid_crosswalk` links GOIDs to `rel_path`, SCIP symbols, chunk ids, CST node ids.

* **Path join**:

  * Files-level analytics (hotspots, typedness, static_diagnostics, coverage_lines) join via `rel_path`. 

* **SCIP symbol join**:

  * `graph.symbol_use_edges.symbol` ⇔ `index.scip.json` ⇔ `core.goid_crosswalk.scip_symbol`. 

* **Module join**:

  * `graph.import_graph_edges.src_module/dst_module` ⇔ `core.modules.module`. 

This makes the stitched datasets essentially a graph-of-graphs centered on GOIDs and paths.

### 5.3 Inside DuckDB vs. in Python

* **Compute in Python** for:

  * LibCST/tree-sitter walking
  * building GOIDs (custom hashing, URNs)
  * CFG/DFG construction
  * calling external tools (scip-python, Pyright, coverage, pytest)

* **Compute in DuckDB** for:

  * Aggregations (function coverage from coverage_lines + goids)
  * Risk factors (join metrics, coverage, tests, hotspots, typedness, diagnostics)
  * Derived views and doc exports for AI

Example: `coverage_functions` computed purely in DuckDB:

```sql
INSERT INTO analytics.coverage_functions
SELECT
  g.goid_h128  AS function_goid_h128,
  g.urn,
  g.repo,
  g.commit,
  g.rel_path,
  g.language,
  g.kind,
  g.qualname,
  g.start_line,
  g.end_line,
  COUNT(*) FILTER (WHERE c.is_executable) AS executable_lines,
  COUNT(*) FILTER (WHERE c.is_executable AND c.is_covered) AS covered_lines,
  CASE
    WHEN COUNT(*) FILTER (WHERE c.is_executable) = 0 THEN NULL
    ELSE CAST(
      COUNT(*) FILTER (WHERE c.is_executable AND c.is_covered)
      AS DOUBLE
    ) / COUNT(*) FILTER (WHERE c.is_executable)
  END AS coverage_ratio,
  COUNT(*) FILTER (WHERE c.is_executable AND c.is_covered) > 0 AS tested,
  CASE
    WHEN COUNT(*) FILTER (WHERE c.is_executable) = 0 THEN 'no_executable_code'
    WHEN COUNT(*) FILTER (WHERE c.is_executable AND c.is_covered) = 0 THEN 'no_tests'
    ELSE ''
  END AS untested_reason,
  NOW() AS created_at
FROM core.goids g
LEFT JOIN analytics.coverage_lines c
  ON c.rel_path = g.rel_path
 AND c.line BETWEEN g.start_line AND g.end_line
WHERE g.kind IN ('function','method')
GROUP BY ALL;
```

(Logic mirrors your spec; we just centralize it in DuckDB. )

---

## 6. Pipeline Stages & Orchestration

### 6.1 Pipeline model

Define a small pipeline abstraction:

```python
@dataclass
class PipelineContext:
    repo_root: Path
    db_path: Path
    repo: str
    commit: str
    config: PipelineConfig
    logger: logging.Logger

class PipelineStep(Protocol):
    name: str
    deps: list[str]

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        ...

PIPELINE_STEPS: dict[str, PipelineStep] = {
    "scan_repo": RepoScanStep(),
    "ast_cst": AstCstStep(),
    "scip": ScipStep(),
    "goids": GoidBuilderStep(),
    "callgraph": CallGraphStep(),
    "cfg_dfg": CfgDfgStep(),
    "imports": ImportGraphStep(),
    "symbol_uses": SymbolUsesStep(),
    "config_values": ConfigValuesStep(),
    "ast_metrics": AstMetricsStep(),
    "hotspots": HotspotsStep(),
    "typedness": TypednessStep(),
    "static_diagnostics": StaticDiagnosticsStep(),
    "function_metrics": FunctionMetricsStep(),
    "function_types": FunctionTypesStep(),
    "coverage_lines": CoverageLinesStep(),
    "coverage_functions": CoverageFunctionsStep(),
    "test_catalog": TestCatalogStep(),
    "test_edges": TestCoverageEdgesStep(),
    "risk_factors": RiskFactorsStep(),
    "export_docs": DocsExportStep(),
}
```

The orchestrator (in `orchestration/pipeline.py`) topologically sorts these steps and runs them with a shared DuckDB connection.

### 6.2 Stage A – Repo Scan & Base Metadata

**Step: `scan_repo`**

* **Libraries:** `GitPython`, `pathlib`, `yaml`.
* **Outputs (DuckDB):**

  * `core.repo_map` (repo name, commit, module→path map) 
  * `core.modules` (module metadata, tags, owners) 
  * `analytics.tags_index` (normalized version of `tags_index.yaml`) 

Parsing and writing:

1. Walk repo root, infer modules for `.py` files.
2. Load `tags_index.yaml` if present and resolve matches.
3. Insert into DuckDB via parameterized `INSERT` or Arrow.

### 6.3 Stage B – AST/CST, SCIP & Raw Signals

**Step: `ast_cst`**

* **Libraries:** LibCST, `ast` (optional), tree-sitter.
* **Outputs:**

  * `core.ast_nodes` (fields as in README) 
  * `core.ast_metrics` 
  * `core.cst_nodes` 

Implementation:

* For each Python file:

  * Parse with LibCST; walk nodes; collect:

    * Qualname, node type, spans, docstrings, decorators, parent relations. 
  * Compute AST metrics (node count, depth, complexity).
  * Optionally parse with tree-sitter, saving structural or token-level info as needed (MVP may store only minimal tree-sitter data or skip it until multi-language).

**Step: `scip`**

* **Libraries:** `subprocess` to call scip-python.
* **Outputs:**

  * `build/scip/index.scip`
  * `Document Output/index.scip.json` (copied)
* **DuckDB:**

  * Optional: load a subset of SCIP data into `core.scip_occurrences` and `core.scip_symbols` for easier joins.

**Step: `coverage_lines`**

* **Libraries:** coverage.py.
* **Outputs:**

  * `analytics.coverage_lines` (line-level coverage as described). 

**Step: `test_catalog` & `test_edges` (raw)**

* **Libraries:** pytest + pytest-json-report; coverage contexts.
* **Outputs:**

  * `analytics.test_catalog` (nodeids, status, markers) 
  * `analytics.test_coverage_edges_raw` (raw mapping test → line coverage contexts)

**Step: `typing_ingest`**

* **Libraries:** Pyright, Pyrefly.
* **Outputs:**

  * `analytics.typedness` (per-file) 
  * `analytics.static_diagnostics` (file-level error counts). 

**Step: `config_values`**

* **Libraries:** `json`, `yaml`, `toml`, config indexer logic.
* **Outputs:**

  * `graph.config_values` exactly as spec. 

### 6.4 Stage C – GOIDs & Graphs

**Step: `goids`**

* **Inputs:** `core.ast_nodes`, `core.repo_map`.
* **Libraries:** custom GOID hashing logic (we’ll replicate the semantics: repo+commit+path+kind+qualname).
* **Outputs:**

  * `core.goids` (canonical ID registry) 
  * `core.goid_crosswalk` (mapping GOID ↔ AST/CST/SCIP/chunk ids). 

**Step: `callgraph`**

* **Inputs:** LibCST analysis of imports & callsites, GOIDs, optional SCIP.
* **Outputs:**

  * `graph.call_graph_nodes`
  * `graph.call_graph_edges` with evidence & confidence. 

**Step: `cfg_dfg`**

* **Inputs:** AST node spans w/ GOIDs.
* **Libraries:** custom CFG builder; optional tree-sitter for more precise edges.
* **Outputs:**

  * `graph.cfg_blocks`, `graph.cfg_edges`, `graph.dfg_edges`. 

**Step: `imports`**

* **Inputs:** LibCST import metadata.
* **Outputs:**

  * `graph.import_graph_edges` (module-level imports). 

**Step: `symbol_uses`**

* **Inputs:** `index.scip.json`, optionally `core.goid_crosswalk`.
* **Outputs:**

  * `graph.symbol_use_edges` (symbol def→use edges). 

### 6.5 Stage D – Analytics

**Step: `ast_metrics`**

* Already partially computed in `ast_cst`; here we just ensure `core.ast_metrics` is materialized. 

**Step: `hotspots`**

* **Inputs:** Git history, `core.ast_metrics`.
* **Outputs:** `analytics.hotspots` (score per file). 

**Step: `function_metrics` & `function_types`**

* **Inputs:** `core.ast_nodes`, `core.goids`.
* **Outputs:**

  * `analytics.function_metrics` (per-function structural metrics) 
  * `analytics.function_types` (parameter/return annotations). 

**Step: `coverage_functions` (in DuckDB)**

* As in the SQL example above, uses `core.goids` + `analytics.coverage_lines`. 

**Step: `tests_analytics`**

* **Inputs:** `analytics.test_catalog`, `coverage_lines` with `dynamic_context = test_function`, `core.goids`.
* **Outputs:**

  * `analytics.test_coverage_edges` normalized to your spec, with function GOIDs & per-edge coverage ratios. 

**Step: `risk_factors` (in DuckDB)**

* **Inputs:**

  * `analytics.function_metrics`, `analytics.function_types`
  * `analytics.hotspots`, `analytics.typedness`, `analytics.static_diagnostics`
  * `analytics.coverage_functions`
  * `analytics.test_coverage_edges`, `analytics.test_catalog`
  * `core.goids`, `core.modules`
* **Output:** `analytics.goid_risk_factors` with joined metrics & heuristic risk scoring. 

The risk SQL is mostly a big `LEFT JOIN` chain with some CASE expressions for risk buckets, mirroring the README fields. 

### 6.6 Stage E – Document Output

**Step: `export_docs`**

* For each target dataset:

  * `COPY (SELECT * FROM schema.table) TO 'Document Output/<name>.parquet' (FORMAT PARQUET)`
  * `COPY (SELECT * FROM schema.table) TO 'Document Output/<name>.jsonl' (FORMAT JSON);`
* Names match exactly:

  * `goids.{parquet,jsonl}`, `goid_crosswalk.{parquet,jsonl}`, `call_graph_nodes.*`, `call_graph_edges.*`, …, `goid_risk_factors.*`, etc. 
* Copy static files like `index.scip.json`, `repo_map.json`, `tags_index.yaml` into `Document Output/`.

---

## 7. Orchestration Profiles & Incrementality

Support multiple run profiles via CLI (Typer):

* `codeintel pipeline full`
  Runs all steps end-to-end.

* `codeintel pipeline graphs-only`
  Skips heavy analytics (coverage/tests) – for faster iteration.

* `codeintel pipeline analytics-only`
  Reuses existing `ast_nodes/goids/graphs` tables.

* `codeintel docs export`
  Only re-export Parquet/JSONL from DuckDB, no recompute.

Incrementality options:

* Each step can:

  * Check for existing rows for the current `(repo, commit)` and either skip or truncate & reinsert.
  * Read a `--since-commit` config and filter Git history, but v1 can be “full recompute per commit” for simplicity.

---

## 8. Accessing the Stitched Datasets

There are three main access patterns:

1. **Direct DuckDB SQL** (for power users).
2. **JSONL/Parquet artifacts** (for LLM ingestion / offline analysis).
3. **High-level query API** (for AI agents and tools).

### 8.1 Canonical DuckDB Views

Create views in schema `docs` to hide raw joins and present a friendly schema:

#### `docs.v_function_summary`

One row per function GOID, merging core, metrics, types, coverage, risk:

```sql
CREATE OR REPLACE VIEW docs.v_function_summary AS
SELECT
  g.goid_h128            AS function_goid_h128,
  g.urn,
  g.repo,
  g.commit,
  g.rel_path,
  g.language,
  g.kind,
  g.qualname,
  fm.loc,
  fm.logical_loc,
  fm.cyclomatic_complexity,
  fm.complexity_bucket,
  ft.typedness_bucket,
  ft.fully_typed,
  cf.coverage_ratio,
  cf.tested,
  rf.risk_score,
  rf.risk_level,
  rf.test_count,
  rf.failing_test_count,
  rf.last_test_status,
  rf.tags,
  rf.owners
FROM core.goids g
LEFT JOIN analytics.function_metrics fm
  ON fm.function_goid_h128 = g.goid_h128
LEFT JOIN analytics.function_types ft
  ON ft.function_goid_h128 = g.goid_h128
LEFT JOIN analytics.coverage_functions cf
  ON cf.function_goid_h128 = g.goid_h128
LEFT JOIN analytics.goid_risk_factors rf
  ON rf.function_goid_h128 = g.goid_h128
WHERE g.kind IN ('function','method');
```

#### `docs.v_call_graph_enriched`

Call edges plus human-readable names and risk info for caller/callee:

```sql
CREATE OR REPLACE VIEW docs.v_call_graph_enriched AS
SELECT
  e.caller_goid_h128,
  gc.urn             AS caller_urn,
  gc.qualname        AS caller_qualname,
  e.callee_goid_h128,
  gcallee.urn        AS callee_urn,
  gcallee.qualname   AS callee_qualname,
  e.callsite_path,
  e.callsite_line,
  e.kind,
  e.resolved_via,
  e.confidence,
  rc.risk_level      AS caller_risk_level,
  rcallee.risk_level AS callee_risk_level
FROM graph.call_graph_edges e
LEFT JOIN core.goids gc
  ON gc.goid_h128 = e.caller_goid_h128
LEFT JOIN core.goids gcallee
  ON gcallee.goid_h128 = e.callee_goid_h128
LEFT JOIN analytics.goid_risk_factors rc
  ON rc.function_goid_h128 = e.caller_goid_h128
LEFT JOIN analytics.goid_risk_factors rcallee
  ON rcallee.function_goid_h128 = e.callee_goid_h128;
```

#### Other useful views

* `docs.v_file_summary` – join hotspots, typedness, diagnostics per file.
* `docs.v_test_to_function` – tests ↔ functions from `test_coverage_edges` + `test_catalog`.
* `docs.v_module_imports` – import graph edges with tags & owners.

These views form the **contract** agents rely on.

### 8.2 JSONL “doc rows” for AI retrieval

For LLM context, it’s often easiest to have **one JSON object per function/module** that already embeds the important data. Generate these from DuckDB with a small Python helper or pure SQL:

Example: `Document Output/function_docs.jsonl`:

Each row:

```json
{
  "urn": "goid:repo/path#python:function:pkg.mod.Class.method",
  "summary": {
    "repo": "myrepo",
    "path": "pkg/mod.py",
    "qualname": "Class.method",
    "risk_level": "high",
    "coverage_ratio": 0.33,
    "typedness": "partial",
    "complexity_bucket": "high"
  },
  "metrics": {...},
  "coverage": {...},
  "tests": [...],
  "callers": [...],
  "callees": [...],
  "owners": [...],
  "tags": [...]
}
```

These doc rows can be fed into embedding pipelines and used as retrieval units.

Implementation strategy:

* Use DuckDB’s JSON functions (`json_object`, `json_group_array`) to build nested JSON directly in SQL, and `COPY` to JSONL.

---

## 9. Query System for AI Agents

### 9.1 Query vocabulary / “schema description”

Ship a **machine-readable schema description** (e.g., `Document Output/schema_metadata.json`) that:

* Lists tables and views that agents are expected to use:

  * `docs.v_function_summary`, `docs.v_call_graph_enriched`, `docs.v_file_summary`, `docs.v_test_to_function`, etc.
* For each:

  * Column names
  * Types
  * English description
* High-level “recipes” like:

  * “To find tests covering a function, use v_test_to_function filtered by function_goid_h128.”
  * “To get the risk profile of a function, use v_function_summary.”

This schema description becomes the **system prompt** or grounding doc for AI agents.

### 9.2 Canonical query templates (examples)

You can codify a few core templates in a Python module, which an LLM-guided agent can fill in with parameters:

1. **Get function summary by URN**

```sql
SELECT * FROM docs.v_function_summary WHERE urn = $urn;
```

2. **Find high-risk untested functions in a module**

```sql
SELECT *
FROM docs.v_function_summary
WHERE rel_path = $rel_path
  AND risk_level IN ('medium','high')
  AND tested = FALSE;
```

3. **Find tests that exercise a function**

```sql
SELECT t.*
FROM docs.v_test_to_function v
JOIN analytics.test_catalog t
  ON t.test_id = v.test_id
WHERE v.function_goid_h128 = $goid;
```

4. **Get call graph neighborhood around a function**

```sql
SELECT *
FROM docs.v_call_graph_enriched
WHERE caller_goid_h128 = $goid
   OR callee_goid_h128 = $goid;
```

These live in `codeintel/ai_queries.py` as parameterized templates so an agent doesn’t have to invent raw SQL from scratch.

### 9.3 Optional HTTP / RPC layer

To make this easy for external tools:

* Provide a small service (e.g., FastAPI) in `src/codeintel/server.py` that exposes endpoints:

  * `GET /function/summary?urn=...`
  * `GET /function/callgraph?urn=...`
  * `GET /module/files?path=...`
  * `GET /tests/for-function?urn=...`
  * `POST /sql` (optional, restricted)

* Service uses DuckDB in-process for low latency; results returned as JSON.

AI agents can then call these endpoints instead of running SQL directly.

---

## 10. Summary

This architecture:

* Treats your README spec as the **contract** for final artifacts and faithfully reproduces:

  * GOID registry + crosswalk, AST/CST, call graph, CFG/DFG, import graph, SCIP symbol uses, config values, hotspots, typedness, function metrics/types, coverage, tests, risk factors, etc. 
* Centers all state and stitching inside **DuckDB**, using:

  * `core.*` for identifiers & AST/CST
  * `graph.*` for structural graphs
  * `analytics.*` for metrics and risk
  * `docs.*` for agent-friendly views
* Uses a **simple, explicit pipeline** of steps with clear inputs/outputs and minimal cross-coupling.
* Provides a **clean query contract** for AI agents via:

  * Documented views (`docs.v_*`)
  * JSONL function/module docs
  * Query templates and optional HTTP endpoints.

If you’d like, I can next draft:

* A concrete `schemas.py` with DDL for all tables, or
* The `PipelineStep` skeletons for a few key steps (GOIDs, callgraph, risk_factors) to kickstart implementation.
