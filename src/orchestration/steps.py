# src/codeintel/orchestration/steps.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Protocol, Sequence

import duckdb

# Ingestion builders
from codeintel.ingestion import (
    repo_scan,
    scip_ingest,
    ast_cst_extract,
    coverage_ingest,
    tests_ingest,
    typing_ingest,
    config_ingest,
)

# Graph builders
from codeintel.graphs.goid_builder import GoidBuilderConfig, build_goids
from codeintel.graphs.callgraph_builder import CallGraphConfig, build_call_graph
from codeintel.graphs.cfg_builder import CFGBuilderConfig, build_cfg_and_dfg
from codeintel.graphs.import_graph import ImportGraphConfig, build_import_graph
from codeintel.graphs.symbol_uses import SymbolUsesConfig, build_symbol_use_edges

# Analytics builders
from codeintel.analytics.ast_metrics import HotspotsConfig, build_hotspots
from codeintel.analytics.functions import FunctionAnalyticsConfig, compute_function_metrics_and_types
from codeintel.analytics.coverage_analytics import CoverageAnalyticsConfig, compute_coverage_functions


log = logging.getLogger(__name__)


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
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def document_output_dir(self) -> Path:
        return self.repo_root / "Document Output"


class PipelineStep(Protocol):
    name: str
    deps: Sequence[str]

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        ...


# ---------------------------------------------------------------------------
# Ingestion steps
# ---------------------------------------------------------------------------


@dataclass
class RepoScanStep:
    name: str = "repo_scan"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        repo_scan.ingest_repo(
            con=con,
            repo_root=ctx.repo_root,
            repo=ctx.repo,
            commit=ctx.commit,
        )


@dataclass
class SCIPIngestStep:
    name: str = "scip_ingest"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        scip_ingest.ingest_scip(
            con=con,
            repo_root=ctx.repo_root,
            repo=ctx.repo,
            commit=ctx.commit,
            build_dir=ctx.build_dir,
            document_output_dir=ctx.document_output_dir,
        )


@dataclass
class AstCstStep:
    name: str = "ast_cst"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        ast_cst_extract.ingest_ast_and_cst(
            con=con,
            repo_root=ctx.repo_root,
            repo=ctx.repo,
            commit=ctx.commit,
        )


@dataclass
class CoverageIngestStep:
    name: str = "coverage_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        coverage_ingest.ingest_coverage_lines(
            con=con,
            repo_root=ctx.repo_root,
            repo=ctx.repo,
            commit=ctx.commit,
        )


@dataclass
class TestsIngestStep:
    name: str = "tests_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        tests_ingest.ingest_tests(
            con=con,
            repo_root=ctx.repo_root,
            repo=ctx.repo,
            commit=ctx.commit,
        )


@dataclass
class TypingIngestStep:
    name: str = "typing_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        typing_ingest.ingest_typing_signals(
            con=con,
            repo_root=ctx.repo_root,
            repo=ctx.repo,
            commit=ctx.commit,
        )


@dataclass
class ConfigIngestStep:
    name: str = "config_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        config_ingest.ingest_config_values(
            con=con,
            repo_root=ctx.repo_root,
        )


# ---------------------------------------------------------------------------
# Graph steps
# ---------------------------------------------------------------------------


@dataclass
class GoidsStep:
    """
    Build core.goids + core.goid_crosswalk from AST. :contentReference[oaicite:5]{index=5}
    """

    name: str = "goids"
    deps: Sequence[str] = ("ast_cst",)

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        cfg = GoidBuilderConfig(repo=ctx.repo, commit=ctx.commit, language="python")
        build_goids(con, cfg)


@dataclass
class CallGraphStep:
    """
    Build graph.call_graph_nodes + graph.call_graph_edges. :contentReference[oaicite:6]{index=6}
    """

    name: str = "callgraph"
    deps: Sequence[str] = ("goids", "ast_cst", "repo_scan")

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        cfg = CallGraphConfig(repo=ctx.repo, commit=ctx.commit, repo_root=ctx.repo_root)
        build_call_graph(con, cfg)


@dataclass
class CFGStep:
    """
    Build graph.cfg_blocks + graph.cfg_edges + graph.dfg_edges (minimal). :contentReference[oaicite:7]{index=7}
    """

    name: str = "cfg"
    deps: Sequence[str] = ("function_metrics",)  # falls back to GOIDs if needed

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        cfg = CFGBuilderConfig(repo=ctx.repo, commit=ctx.commit)
        build_cfg_and_dfg(con, cfg)


@dataclass
class ImportGraphStep:
    """
    Build graph.import_graph_edges from LibCST imports. :contentReference[oaicite:8]{index=8}
    """

    name: str = "import_graph"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        cfg = ImportGraphConfig(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        build_import_graph(con, cfg)


@dataclass
class SymbolUsesStep:
    """
    Build graph.symbol_use_edges from index.scip.json. :contentReference[oaicite:9]{index=9}
    """

    name: str = "symbol_uses"
    deps: Sequence[str] = ("repo_scan", "scip_ingest")

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
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
    """
    Build analytics.hotspots from core.ast_metrics + git. :contentReference[oaicite:10]{index=10}
    """

    name: str = "hotspots"
    deps: Sequence[str] = ("ast_cst",)

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        cfg = HotspotsConfig(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        build_hotspots(con, cfg)


@dataclass
class FunctionAnalyticsStep:
    """
    Build analytics.function_metrics + analytics.function_types. :contentReference[oaicite:11]{index=11}
    """

    name: str = "function_metrics"
    deps: Sequence[str] = ("goids",)

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        cfg = FunctionAnalyticsConfig(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        compute_function_metrics_and_types(con, cfg)


@dataclass
class CoverageAnalyticsStep:
    """
    Build analytics.coverage_functions from GOIDs + coverage_lines. :contentReference[oaicite:12]{index=12}
    """

    name: str = "coverage_functions"
    deps: Sequence[str] = ("goids", "coverage_ingest")

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        cfg = CoverageAnalyticsConfig(repo=ctx.repo, commit=ctx.commit)
        compute_coverage_functions(con, cfg)


@dataclass
class RiskFactorsStep:
    """
    Aggregate analytics into analytics.goid_risk_factors:

      - function_metrics
      - function_types
      - coverage_functions
      - hotspots
      - typedness
      - static_diagnostics
      - test_coverage_edges + test_catalog
      - modules (tags/owners) :contentReference[oaicite:13]{index=13}
    """

    name: str = "risk_factors"
    deps: Sequence[str] = (
        "function_metrics",
        "coverage_functions",
        "hotspots",
        "typing_ingest",
        "tests_ingest",  # test_coverage_edges builder can be separate
        "config_ingest",  # indirectly for tags/owners via modules
    )

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
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

        n = con.execute(
            "SELECT COUNT(*) FROM analytics.goid_risk_factors WHERE repo = ? AND commit = ?",
            [ctx.repo, ctx.commit],
        ).fetchone()[0]
        log.info("risk_factors populated: %d rows for %s@%s", n, ctx.repo, ctx.commit)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PIPELINE_STEPS: Dict[str, PipelineStep] = {
    # ingestion
    "repo_scan": RepoScanStep(),
    "scip_ingest": SCIPIngestStep(),
    "ast_cst": AstCstStep(),
    "coverage_ingest": CoverageIngestStep(),
    "tests_ingest": TestsIngestStep(),
    "typing_ingest": TypingIngestStep(),
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
    "risk_factors": RiskFactorsStep(),
}

