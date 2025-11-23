"""Ingestion stages that parse repositories into normalized DuckDB tables for analytics."""

from codeintel.ingestion.runner import (
    IngestionContext,
    run_ast_extract,
    run_config_ingest,
    run_coverage_ingest,
    run_cst_extract,
    run_docstrings_ingest,
    run_repo_scan,
    run_scip_ingest,
    run_tests_ingest,
    run_typing_ingest,
)

__all__ = [
    "IngestionContext",
    "run_ast_extract",
    "run_config_ingest",
    "run_coverage_ingest",
    "run_cst_extract",
    "run_docstrings_ingest",
    "run_repo_scan",
    "run_scip_ingest",
    "run_tests_ingest",
    "run_typing_ingest",
]
