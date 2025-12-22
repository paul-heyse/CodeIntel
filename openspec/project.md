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
