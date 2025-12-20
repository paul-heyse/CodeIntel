# Project Context

## Purpose
CodeIntel is a Python-based code intelligence platform that analyzes repositories to produce
structured datasets (AST/SCIP/GOID/CFG/DFG/call graph, metrics, profiles) stored in DuckDB and
exported as Parquet/JSONL for LLMs and tooling. It provides a CLI and serving layer (FastMCP and
FastAPI) for querying semantic views, exporting artifacts, and operating the pipeline.

## Tech Stack
- Python 3.13, uv for environment/tooling
- Core analysis: LibCST, Tree-sitter, NetworkX, NumPy/Pandas/Polars, SciPy
- Data/storage: DuckDB, Ibis 11, SQLGlot, Pandera, PyArrow
- Pipelines: Hamilton (sf-hamilton), Python CLI tooling (cyclopts)
- Serving: FastAPI, FastMCP, Starlette, Pydantic, msgspec
- Observability: Prometheus client libraries and OpenTelemetry
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
- Schema-first data contracts: Pandera `DataFrameSchema` is source of truth; JSON Schema 2020-12
  and OpenAPI 3.2 at boundaries.
- DuckDB access via Ibis 11 patterns and qualified `schema.table` handling; bulk writes through
  `DuckDBPolicyBackend`.
- Pipeline stages generate GOIDs, graphs (call/CFG/DFG), metrics, profiles, and subsystem views.
- Serving layer exposes semantic views and exports over FastMCP/FastAPI.
- Plugin discovery via entry points for ingest and graph plugins.

### Testing Strategy
- Pytest with `integration` and `benchmark` markers; CPU-only.
- Hexagonal test architecture using `tests/_helpers` contexts and seed packs.
- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
  before `uv run pytest -q`.
- Docstring coverage >= 90% and NumPy docstring structure enforced.

### Git Workflow
- Do not use `git checkout` or destructive git commands.
- Do not assume a clean worktree; ignore unrelated changes.
- Do not amend commits unless explicitly requested.
- Ask if branch/commit conventions are needed.

## Domain Context
- GOID registry provides stable IDs for modules, functions, classes, and CFG blocks.
- Enrichment pipeline builds AST/SCIP indexes, graphs, analytics, and exports.
- DuckDB `docs.*` views provide denormalized architecture and subsystem summaries.

## Important Constraints
- Python 3.13 only; strict linting and type gates (ruff, pyright, pyrefly) are zero-error.
- Absolute imports only; heavy-type imports must be type-gated.
- Ibis 11 API patterns and qualified schema handling are required.
- Use schema registry and validation helpers for dataset writes.

## External Dependencies
- DuckDB catalog and Parquet/JSONL exports.
- FastMCP/FastAPI serving surface for tool access.
- Observability via Prometheus and OpenTelemetry.
- Optional GPU graph backend via NetworkX + nx-cugraph when enabled.
