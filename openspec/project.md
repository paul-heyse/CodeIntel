# Project Context

## Purpose
CodeIntel is a Python-based code intelligence platform that ingests repository snapshots
(repo + commit) and builds structured datasets and artifacts for analysis, search, and LLM
consumption. The Hamilton-first build system produces DuckDB tables (core/graphs/analytics/docs),
graph artifacts (call, import, CFG/DFG), metrics/profiles, and external artifacts (SCIP index,
schema manifests, buildspecs, semantic registry). Outputs are stored in DuckDB and optionally
exported as Parquet/JSONL with dataset manifests and incremental markers. A CLI orchestrates the
pipeline, and a serving layer (FastAPI + FastMCP) exposes semantic catalogs, queries, explain,
exports, search, and snapshot metadata from published serving snapshots via an atomic
`current.json` pointer.

## Tech Stack
- Python 3.13; `uv` for env/tooling; Cyclopts CLI
- Parsing/analysis: LibCST and stdlib AST; SCIP tooling for symbol indexes
- Graph/analytics: NetworkX (optional nx-cugraph GPU backend), NumPy/Pandas, Polars
- Data/storage: DuckDB, Ibis 11, SQLGlot, Pandera, PyArrow
- Pipelines: Hamilton (sf-hamilton), Hamilton tag-based target discovery
- Serving: FastAPI, FastMCP, Starlette, Pydantic
- Observability: structured logging, optional OpenTelemetry instrumentation
- Quality: Ruff, pyright (strict), pyrefly, pytest (+cov/xdist), pydoclint/pydocstyle

## Project Conventions

### Code Style
- Ruff is canonical; 100-char lines, double quotes, absolute imports only.
- Every module includes `from __future__ import annotations`.
- Type-only imports go under `if TYPE_CHECKING:`; heavy deps gated via
  `CodeIntel_common.typing.gate_import`.
- Prefer `pathlib.Path`; no print debugging (use logging).
- NumPy-style docstrings are required for public APIs.

### Architecture Patterns
- Hamilton-first build system: target graph and dependencies are derived from Hamilton tags;
  `OutputTarget` contracts define produced table keys/artifacts and execution metadata.
- Snapshot-first pipeline: `SnapshotRef` (repo, commit, repo_root, branch) is the canonical identity
  for all build, graph, analytics, storage, and serving operations.
- Contracts and schemas: Pandera `DataFrameSchema` is the source of truth; JSON Schema 2020-12 and
  OpenAPI 3.2 govern external boundaries. `buildspec.json`, `schema_manifest.json`, and
  `semantic_registry.json` are compiled artifacts used for validation and serving.
- Serving snapshot publishing copies the build DuckDB plus serving artifacts into `serve_dir`,
  writes `current.json` atomically, and retains a configurable number of prior snapshots.
- Storage access: reads via `StorageGateway`/`IbisGateway` and qualified name helpers; writes via
  `DuckDBPolicyBackend` (bulk insert/upsert, snapshot-scoped deletes).
- Ports/adapters: ingestion, graphs, and analytics keep pure compute logic in `*/compute` with I/O
  isolated behind ports/resources.
- Serving architecture: semantic registry compiled from Hamilton tags; semantic kernel uses view
  registry + query templates; search uses DuckDB FTS with LIKE fallback; HTTP and MCP share a
  transport-agnostic operations layer and RFC 9457 Problem Details errors.
- Plugins: unified plugin protocol with capability metadata; entry points exist for ingest and
  graph plugins and CLI scaffolding.

### Testing Strategy
- Pytest with `integration` and `benchmark` markers; CPU-only.
- Hexagonal test architecture using `tests/_helpers` contexts and seed packs.
- Architecture boundary tests enforce import constraints (duckdb/networkx/faiss layer rules).
- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
  before `uv run pytest -q`.
- Docstring coverage >= 90% and NumPy docstring structure enforced.

### Git Workflow
- Do not use `git checkout` or destructive git commands.
- Do not assume a clean worktree; ignore unrelated changes.
- Do not amend commits unless explicitly requested.
- Ask if branch/commit conventions are needed.

## Domain Context
- GOID registry provides stable IDs for modules, functions, classes, and CFG blocks, with crosswalk
  tables tying GOIDs to AST/SCIP symbols.
- Ingestion produces core tables (modules, file state, AST/CST, SCIP symbols/occurrences, tests,
  coverage, config) scoped by snapshot.
- Graphs layer builds call/import/symbol-use graphs plus CFG/DFG structures and metrics.
- Analytics layer computes function/file metrics, profiles, entrypoints, subsystems, risk/hotspot
  signals, data model usage, and test/coverage analytics.
- `docs.*` views provide denormalized summaries for serving and LLM exports, including
  `docs.search_documents` for search.

### Build Targets (Hamilton)
- Ingestion targets: modules, ast, cst, docstrings, scip, config_ingest, coverage_ingest,
  tests_ingest, typing.
- Graph targets: goids, call_graph, import_graph, cfg, dfg, symbol_uses, call_graph_views,
  graph_metrics, graph_validation.
- Analytics targets: function_metrics, function_history, history_timeseries, function_contracts,
  function_effects, function_ast_features, entrypoints, data_models, data_model_usage, profiles,
  semantic_roles, risk_factors, hotspots, subsystems, subsystem_graph_metrics, subsystem_agreement,
  symbol_graph_metrics, test_graph_metrics, test_profile, coverage_functions, coverage_test_edges,
  behavioral_coverage, config_data_flow, cfg_dfg_metrics, external_deps.
- Export targets: export_jsonl, export_parquet, serving_artifacts.

### Table Families and Producers
- core.*: ingestion targets populate snapshot-scoped source, symbol, test, coverage, and config
  tables (modules/ast/cst/docstrings/scip/config_ingest/coverage_ingest/tests_ingest/typing).
- graph.*: graph targets emit edges/nodes and derived views (call_graph/import_graph/cfg/dfg/
  symbol_uses, plus `graph.v_*` from call_graph_views).
- analytics.*: analytics targets compute metrics, profiles, entrypoints, subsystems, risk/hotspots,
  data model usage, and testing analytics from core/graph inputs.
- docs.*: Ibis view builders in `codeintel.storage.views.ibis_views` define denormalized views for
  exports/serving; `docs.search_documents` is populated during serving snapshot publish and
  indexed with DuckDB FTS.

### Serving Tools and Resources
- MCP tools: semantic_catalog, semantic_describe, semantic_query, semantic_explain, semantic_export,
  code_search, serving_meta.
- HTTP surface mirrors the semantic tools and exposes health/meta endpoints; errors use RFC 9457.

### CLI Command Surface (Build-Tied)
- `codeintel build run` executes targets by name/module with dependency resolution; `build spec`
  compiles buildspecs; `build schema` compiles schema manifests; other build subcommands inspect
  plans/lineage/impact and build status.
- `codeintel graph` lists graph-module targets and their execution plans.
- `codeintel dataset` inspects schemas/constraints/flows and verifies data against Pandera;
  `codeintel datasets` lints, snapshots, diffs, and scaffolds dataset specs.
- `codeintel docs export` writes Document Output artifacts (JSONL/Parquet + manifests).
- `codeintel serve http|mcp` runs the FastAPI and MCP surfaces over published snapshots.
- `codeintel storage`, `codeintel history`, `codeintel jobs`, `codeintel plugins` provide
  operational utilities for storage, history, job runner, and plugin management.

## Important Constraints
- Python 3.13 only; strict linting and type gates (ruff, pyright, pyrefly) are zero-error.
- Absolute imports only; heavy-type imports must be type-gated.
- Ibis 11 API patterns and qualified schema handling are required (database param for schema names).
- Use schema registry + validation helpers for dataset writes and serving artifacts.
- Import boundaries: DuckDB usage confined to storage, NetworkX to graphs/analytics, FAISS to
  serving/cli.

## External Dependencies
- DuckDB database files and DuckDB FTS extension for serving search.
- SCIP tooling (`scip-python`, `scip`) for symbol indexing artifacts.
- Serving snapshots on disk (serve_dir with `current.json`, `schema_manifest.json`,
  `semantic_registry.json`, `buildspec.json`).
- Optional GPU graph backend via NetworkX + nx-cugraph when enabled.
- OpenTelemetry exporters if CLI telemetry is configured.
