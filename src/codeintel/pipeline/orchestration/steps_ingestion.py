"""Pipeline ingestion step implementations."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from codeintel.ingestion.runner import (
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
from codeintel.ingestion.scip_ingest import ScipIngestResult
from codeintel.pipeline.orchestration.core import (
    PipelineContext,
    PipelineStep,
    _ingestion_ctx,
    _log_step,
)

log = logging.getLogger(__name__)


@dataclass
class SchemaBootstrapStep:
    """Apply schemas and create views before ingestion."""

    name: str = "schema_bootstrap"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext) -> None:  # noqa: ARG002
        """No-op here; actual bootstrap is handled in the Prefect task."""
        _log_step(self.name)


@dataclass
class RepoScanStep:
    """Ingest repository modules and repo_map."""

    name: str = "repo_scan"
    deps: Sequence[str] = ("schema_bootstrap",)

    def run(self, ctx: PipelineContext) -> None:
        """Execute repository scan ingestion."""
        _log_step(self.name)
        tracker = run_repo_scan(_ingestion_ctx(ctx))
        ctx.change_tracker = tracker


@dataclass
class SCIPIngestStep:
    """Run scip-python and register SCIP artifacts/view."""

    name: str = "scip_ingest"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext) -> None:
        """Register SCIP artifacts and populate SCIP symbols in crosswalk."""
        _log_step(self.name)
        ingest_ctx = _ingestion_ctx(ctx)
        result: ScipIngestResult = run_scip_ingest(ingest_ctx)
        ctx.extra["scip_ingest"] = result
        if result.status != "success":
            log.info(
                "SCIP ingestion %s: %s",
                result.status,
                result.reason or "no reason provided",
            )


@dataclass
class CSTStep:
    """Parse CST and persist rows."""

    name: str = "cst_extract"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext) -> None:
        """Extract CST rows into core.cst_nodes."""
        _log_step(self.name)
        run_cst_extract(_ingestion_ctx(ctx))


@dataclass
class AstStep:
    """Parse stdlib AST and persist rows/metrics."""

    name: str = "ast_extract"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext) -> None:
        """Extract AST rows and metrics into core tables."""
        _log_step(self.name)
        run_ast_extract(_ingestion_ctx(ctx))


@dataclass
class CoverageIngestStep:
    """Load coverage.py data into analytics.coverage_lines."""

    name: str = "coverage_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext) -> None:
        """Ingest line-level coverage signals."""
        _log_step(self.name)
        run_coverage_ingest(_ingestion_ctx(ctx))


@dataclass
class TestsIngestStep:
    """Load pytest JSON report into analytics.test_catalog."""

    name: str = "tests_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext) -> None:
        """Ingest pytest test catalog."""
        _log_step(self.name)
        run_tests_ingest(_ingestion_ctx(ctx))


@dataclass
class TypingIngestStep:
    """Collect typedness/static diagnostics."""

    name: str = "typing_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext) -> None:
        """Ingest typing signals from ast + pyright."""
        _log_step(self.name)
        run_typing_ingest(_ingestion_ctx(ctx))


@dataclass
class DocstringsIngestStep:
    """Extract and persist structured docstrings."""

    name: str = "docstrings_ingest"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext) -> None:
        """Ingest docstrings for all Python modules."""
        _log_step(self.name)
        run_docstrings_ingest(_ingestion_ctx(ctx))


@dataclass
class ConfigIngestStep:
    """Flatten config files into analytics.config_values."""

    name: str = "config_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext) -> None:
        """Ingest configuration files from repo root."""
        _log_step(self.name)
        run_config_ingest(_ingestion_ctx(ctx))


INGESTION_STEPS: dict[str, PipelineStep] = {
    "schema_bootstrap": SchemaBootstrapStep(),
    "repo_scan": RepoScanStep(),
    "scip_ingest": SCIPIngestStep(),
    "cst_extract": CSTStep(),
    "ast_extract": AstStep(),
    "coverage_ingest": CoverageIngestStep(),
    "tests_ingest": TestsIngestStep(),
    "typing_ingest": TypingIngestStep(),
    "docstrings_ingest": DocstringsIngestStep(),
    "config_ingest": ConfigIngestStep(),
}


__all__ = [
    "INGESTION_STEPS",
    "AstStep",
    "CSTStep",
    "ConfigIngestStep",
    "CoverageIngestStep",
    "DocstringsIngestStep",
    "RepoScanStep",
    "SCIPIngestStep",
    "SchemaBootstrapStep",
    "TestsIngestStep",
    "TypingIngestStep",
]
