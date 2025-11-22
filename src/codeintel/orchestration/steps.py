"""Pipeline step definitions for ingestion, graphs, analytics, and export."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import duckdb

from codeintel.analytics.ast_metrics import HotspotsConfig, build_hotspots
from codeintel.analytics.coverage_analytics import (
    CoverageAnalyticsConfig,
    compute_coverage_functions,
)
from codeintel.analytics.functions import (
    FunctionAnalyticsConfig,
    compute_function_metrics_and_types,
)
from codeintel.analytics.tests_analytics import (
    TestCoverageConfig,
    compute_test_coverage_edges,
)
from codeintel.docs_export.export_jsonl import export_all_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet
from codeintel.graphs.callgraph_builder import CallGraphConfig, build_call_graph
from codeintel.graphs.cfg_builder import CFGBuilderConfig, build_cfg_and_dfg
from codeintel.graphs.goid_builder import GoidBuilderConfig, build_goids
from codeintel.graphs.import_graph import ImportGraphConfig, build_import_graph
from codeintel.graphs.symbol_uses import SymbolUsesConfig, build_symbol_use_edges
from codeintel.ingestion import (
    ast_cst_extract,
    config_ingest,
    coverage_ingest,
    docstrings_ingest,
    repo_scan,
    scip_ingest,
    tests_ingest,
    typing_ingest,
)
from codeintel.storage.views import create_all_views

log = logging.getLogger(__name__)


def _log_step(name: str) -> None:
    """Log step execution at debug level."""
    log.debug("Running pipeline step: %s", name)


@dataclass
class PipelineContext:
    """
    Shared context passed to every pipeline step.

    This matches the repo layout described in your architecture:
      repo_root/
        src/
        Document Output/
        build/
    """

    repo_root: Path
    db_path: Path
    build_dir: Path
    repo: str
    commit: str
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def document_output_dir(self) -> Path:
        """Document Output directory resolved under repo root."""
        return self.repo_root / "Document Output"


class PipelineStep(Protocol):
    """Contract for pipeline steps."""

    name: str
    deps: Sequence[str]

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Execute the step using shared context and DuckDB connection."""


# ---------------------------------------------------------------------------
# Ingestion steps
# ---------------------------------------------------------------------------


@dataclass
class RepoScanStep:
    """Ingest repository modules and repo_map."""

    name: str = "repo_scan"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Execute repository scan ingestion."""
        _log_step(self.name)
        repo_scan.ingest_repo(
            con=con,
            repo_root=ctx.repo_root,
            repo=ctx.repo,
            commit=ctx.commit,
        )


@dataclass
class SCIPIngestStep:
    """Run scip-python and register SCIP artifacts/view."""

    name: str = "scip_ingest"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Register SCIP artifacts and populate SCIP symbols in crosswalk."""
        _log_step(self.name)
        cfg = scip_ingest.ScipIngestConfig(
            repo_root=ctx.repo_root,
            repo=ctx.repo,
            commit=ctx.commit,
            build_dir=ctx.build_dir,
            document_output_dir=ctx.document_output_dir,
        )
        scip_ingest.ingest_scip(con=con, cfg=cfg)


@dataclass
class AstCstStep:
    """Parse AST/CST and compute per-file metrics."""

    name: str = "ast_cst"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Extract AST/CST rows and metrics into core schema."""
        _log_step(self.name)
        ast_cst_extract.ingest_ast_and_cst(
            con=con,
            repo_root=ctx.repo_root,
            repo=ctx.repo,
            commit=ctx.commit,
        )


@dataclass
class CoverageIngestStep:
    """Load coverage.py data into analytics.coverage_lines."""

    name: str = "coverage_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Ingest line-level coverage signals."""
        _log_step(self.name)
        coverage_ingest.ingest_coverage_lines(
            con=con,
            repo_root=ctx.repo_root,
            repo=ctx.repo,
            commit=ctx.commit,
        )


@dataclass
class TestsIngestStep:
    """Load pytest JSON report into analytics.test_catalog."""

    name: str = "tests_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Ingest pytest test catalog."""
        _log_step(self.name)
        tests_ingest.ingest_tests(
            con=con,
            repo_root=ctx.repo_root,
            repo=ctx.repo,
            commit=ctx.commit,
        )


@dataclass
class TypingIngestStep:
    """Collect typedness/static diagnostics."""

    name: str = "typing_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Ingest typing signals from ast + pyright."""
        _log_step(self.name)
        typing_ingest.ingest_typing_signals(
            con=con,
            repo_root=ctx.repo_root,
            repo=ctx.repo,
            commit=ctx.commit,
        )


@dataclass
class DocstringsIngestStep:
    """Extract and persist structured docstrings."""

    name: str = "docstrings_ingest"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Ingest docstrings for all Python modules."""
        _log_step(self.name)
        cfg = docstrings_ingest.DocstringConfig(
            repo_root=ctx.repo_root,
            repo=ctx.repo,
            commit=ctx.commit,
        )
        docstrings_ingest.ingest_docstrings(con, cfg)


@dataclass
class ConfigIngestStep:
    """Flatten config files into analytics.config_values."""

    name: str = "config_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Ingest configuration files from repo root."""
        _log_step(self.name)
        config_ingest.ingest_config_values(
            con=con,
            repo_root=ctx.repo_root,
        )


# ---------------------------------------------------------------------------
# Graph steps
# ---------------------------------------------------------------------------


@dataclass
class GoidsStep:
    """Build core.goids and core.goid_crosswalk from AST."""

    name: str = "goids"
    deps: Sequence[str] = ("ast_cst",)

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Build GOID registry and crosswalk tables."""
        _log_step(self.name)
        cfg = GoidBuilderConfig(repo=ctx.repo, commit=ctx.commit, language="python")
        build_goids(con, cfg)


@dataclass
class CallGraphStep:
    """Build graph.call_graph_nodes and graph.call_graph_edges."""

    name: str = "callgraph"
    deps: Sequence[str] = ("goids", "ast_cst", "repo_scan")

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Construct static call graph nodes and edges."""
        _log_step(self.name)
        cfg = CallGraphConfig(repo=ctx.repo, commit=ctx.commit, repo_root=ctx.repo_root)
        build_call_graph(con, cfg)


@dataclass
class CFGStep:
    """Build graph.cfg_blocks, graph.cfg_edges, and graph.dfg_edges."""

    name: str = "cfg"
    deps: Sequence[str] = ("function_metrics",)  # falls back to GOIDs if needed

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Create minimal CFG/DFG scaffolding."""
        _log_step(self.name)
        cfg = CFGBuilderConfig(repo=ctx.repo, commit=ctx.commit)
        build_cfg_and_dfg(con, cfg)


@dataclass
class ImportGraphStep:
    """Build graph.import_graph_edges from LibCST imports."""

    name: str = "import_graph"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Construct module import graph edges."""
        _log_step(self.name)
        cfg = ImportGraphConfig(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        build_import_graph(con, cfg)


@dataclass
class SymbolUsesStep:
    """Build graph.symbol_use_edges from index.scip.json."""

    name: str = "symbol_uses"
    deps: Sequence[str] = ("repo_scan", "scip_ingest")

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Derive symbol definition→use edges from SCIP JSON."""
        _log_step(self.name)
        scip_json = ctx.build_dir / "scip" / "index.scip.json"
        cfg = SymbolUsesConfig(
            repo_root=ctx.repo_root,
            scip_json_path=scip_json,
        )
        build_symbol_use_edges(con, cfg)


# ---------------------------------------------------------------------------
# Analytics steps
# ---------------------------------------------------------------------------


@dataclass
class HotspotsStep:
    """Build analytics.hotspots from core.ast_metrics plus git churn."""

    name: str = "hotspots"
    deps: Sequence[str] = ("ast_cst",)

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Compute file-level hotspot scores."""
        _log_step(self.name)
        cfg = HotspotsConfig(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        build_hotspots(con, cfg)


@dataclass
class FunctionAnalyticsStep:
    """Build analytics.function_metrics and analytics.function_types."""

    name: str = "function_metrics"
    deps: Sequence[str] = ("goids",)

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Compute per-function metrics and typedness."""
        _log_step(self.name)
        cfg = FunctionAnalyticsConfig(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        compute_function_metrics_and_types(con, cfg)


@dataclass
class CoverageAnalyticsStep:
    """Build analytics.coverage_functions from GOIDs and coverage_lines."""

    name: str = "coverage_functions"
    deps: Sequence[str] = ("goids", "coverage_ingest")

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Aggregate line coverage to function spans."""
        _log_step(self.name)
        cfg = CoverageAnalyticsConfig(repo=ctx.repo, commit=ctx.commit)
        compute_coverage_functions(con, cfg)


@dataclass
class TestCoverageEdgesStep:
    """Build analytics.test_coverage_edges from coverage contexts."""

    name: str = "test_coverage_edges"
    deps: Sequence[str] = ("coverage_ingest", "tests_ingest", "goids")

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Derive test-to-function edges using coverage contexts."""
        _log_step(self.name)
        cfg = TestCoverageConfig(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        compute_test_coverage_edges(con, cfg)


@dataclass
class RiskFactorsStep:
    """Aggregate analytics into analytics.goid_risk_factors."""

    name: str = "risk_factors"
    deps: Sequence[str] = (
        "function_metrics",
        "coverage_functions",
        "hotspots",
        "typing_ingest",
        "tests_ingest",
        "test_coverage_edges",
        "config_ingest",  # indirectly for tags/owners via modules
    )

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Compute risk factors by joining analytics tables."""
        _log_step(self.name)
        log.info("Computing risk_factors for %s@%s", ctx.repo, ctx.commit)

        # Clear previous rows for this repo/commit
        con.execute(
            "DELETE FROM analytics.goid_risk_factors WHERE repo = ? AND commit = ?",
            [ctx.repo, ctx.commit],
        )

        risk_sql = """
        INSERT INTO analytics.goid_risk_factors
        SELECT
            fm.function_goid_h128,
            fm.urn,
            fm.repo,
            fm.commit,
            fm.rel_path,
            fm.language,
            fm.kind,
            fm.qualname,
            fm.loc,
            fm.logical_loc,
            fm.cyclomatic_complexity,
            fm.complexity_bucket,
            ft.typedness_bucket,
            ft.typedness_source,
            h.score                       AS hotspot_score,
            ty.annotation_ratio->>'params'::DOUBLE AS file_typed_ratio,
            sd.total_errors               AS static_error_count,
            sd.has_errors                 AS has_static_errors,
            cf.executable_lines,
            cf.covered_lines,
            cf.coverage_ratio,
            cf.tested,
            COALESCE(t_stats.test_count, 0)         AS test_count,
            COALESCE(t_stats.failing_test_count, 0) AS failing_test_count,
            COALESCE(t_stats.last_test_status, 'unknown') AS last_test_status,
            (
                COALESCE(1.0 - cf.coverage_ratio, 1.0) * 0.4 +
                CASE fm.complexity_bucket
                    WHEN 'high' THEN 0.4
                    WHEN 'medium' THEN 0.2
                    ELSE 0.0
                END +
                CASE WHEN sd.has_errors THEN 0.2 ELSE 0.0 END +
                CASE WHEN h.score > 0 THEN 0.1 ELSE 0.0 END
            ) AS risk_score,
            CASE
                WHEN (
                    COALESCE(1.0 - cf.coverage_ratio, 1.0) * 0.4 +
                    CASE fm.complexity_bucket
                        WHEN 'high' THEN 0.4
                        WHEN 'medium' THEN 0.2
                        ELSE 0.0
                    END +
                    CASE WHEN sd.has_errors THEN 0.2 ELSE 0.0 END +
                    CASE WHEN h.score > 0 THEN 0.1 ELSE 0.0 END
                ) >= 0.7 THEN 'high'
                WHEN (
                    COALESCE(1.0 - cf.coverage_ratio, 1.0) * 0.4 +
                    CASE fm.complexity_bucket
                        WHEN 'high' THEN 0.4
                        WHEN 'medium' THEN 0.2
                        ELSE 0.0
                    END +
                    CASE WHEN sd.has_errors THEN 0.2 ELSE 0.0 END +
                    CASE WHEN h.score > 0 THEN 0.1 ELSE 0.0 END
                ) >= 0.4 THEN 'medium'
                ELSE 'low'
            END AS risk_level,
            m.tags,
            m.owners,
            NOW() AS created_at
        FROM analytics.function_metrics fm
        LEFT JOIN analytics.function_types ft
            ON ft.function_goid_h128 = fm.function_goid_h128
        LEFT JOIN analytics.coverage_functions cf
            ON cf.function_goid_h128 = fm.function_goid_h128
        LEFT JOIN analytics.hotspots h
            ON h.rel_path = fm.rel_path
        LEFT JOIN analytics.typedness ty
            ON ty.path = fm.rel_path
        LEFT JOIN analytics.static_diagnostics sd
            ON sd.rel_path = fm.rel_path
        LEFT JOIN (
            SELECT
                e.function_goid_h128,
                COUNT(DISTINCT e.test_id) AS test_count,
                COUNT(DISTINCT CASE WHEN t.status IN ('failed','error') THEN e.test_id END) AS failing_test_count,
                CASE
                    WHEN COUNT(DISTINCT e.test_id) = 0 THEN 'untested'
                    WHEN COUNT(DISTINCT CASE WHEN t.status IN ('failed','error') THEN e.test_id END) > 0
                        THEN 'some_failing'
                    WHEN COUNT(DISTINCT CASE WHEN t.status = 'passed' THEN e.test_id END) > 0
                        THEN 'all_passing'
                    ELSE 'unknown'
                END AS last_test_status
            FROM analytics.test_coverage_edges e
            LEFT JOIN analytics.test_catalog t
                ON t.test_id = e.test_id
            GROUP BY e.function_goid_h128
        ) AS t_stats
            ON t_stats.function_goid_h128 = fm.function_goid_h128
        LEFT JOIN core.modules m
            ON m.path = fm.rel_path
        WHERE fm.repo = ?
          AND fm.commit = ?;
        """

        con.execute(risk_sql, [ctx.repo, ctx.commit])

        count_row = con.execute(
            "SELECT COUNT(*) FROM analytics.goid_risk_factors WHERE repo = ? AND commit = ?",
            [ctx.repo, ctx.commit],
        ).fetchone()
        n = int(count_row[0]) if count_row is not None else 0
        log.info("risk_factors populated: %d rows for %s@%s", n, ctx.repo, ctx.commit)


# ---------------------------------------------------------------------------
# Export step
# ---------------------------------------------------------------------------


@dataclass
class ExportDocsStep:
    """Export all Parquet + JSONL datasets into Document Output/."""

    name: str = "export_docs"
    deps: Sequence[str] = (
        "risk_factors",
        "callgraph",
        "cfg",
        "import_graph",
        "symbol_uses",
        "scip_ingest",
    )

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        """Create views and export Parquet/JSONL artifacts."""
        _log_step(self.name)
        create_all_views(con)
        export_all_parquet(con, ctx.document_output_dir)
        export_all_jsonl(con, ctx.document_output_dir)
        log.info("Document Output refreshed at %s", ctx.document_output_dir)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PIPELINE_STEPS: dict[str, PipelineStep] = {
    # ingestion
    "repo_scan": RepoScanStep(),
    "scip_ingest": SCIPIngestStep(),
    "ast_cst": AstCstStep(),
    "coverage_ingest": CoverageIngestStep(),
    "tests_ingest": TestsIngestStep(),
    "typing_ingest": TypingIngestStep(),
    "docstrings_ingest": DocstringsIngestStep(),
    "config_ingest": ConfigIngestStep(),
    # graphs
    "goids": GoidsStep(),
    "callgraph": CallGraphStep(),
    "cfg": CFGStep(),
    "import_graph": ImportGraphStep(),
    "symbol_uses": SymbolUsesStep(),
    # analytics
    "hotspots": HotspotsStep(),
    "function_metrics": FunctionAnalyticsStep(),
    "coverage_functions": CoverageAnalyticsStep(),
    "test_coverage_edges": TestCoverageEdgesStep(),
    "risk_factors": RiskFactorsStep(),
    # export
    "export_docs": ExportDocsStep(),
}
