"""Graph-oriented docs views."""

from __future__ import annotations

from duckdb import DuckDBPyConnection

GRAPH_VIEW_NAMES: tuple[str, ...] = (
    "docs.v_call_graph_enriched",
    "docs.v_symbol_module_graph",
    "docs.v_validation_summary",
)


def create_graph_views(con: DuckDBPyConnection) -> None:
    """Create or replace graph docs views."""
    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_call_graph_enriched AS
        SELECT
            e.caller_goid_h128,
            gc.repo         AS caller_repo,
            gc.commit       AS caller_commit,
            gc.urn           AS caller_urn,
            gc.rel_path      AS caller_rel_path,
            gc.qualname      AS caller_qualname,
            rc.risk_level    AS caller_risk_level,
            rc.risk_score    AS caller_risk_score,
            e.callee_goid_h128,
            gcallee.repo     AS callee_repo,
            gcallee.commit   AS callee_commit,
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
        LEFT JOIN core.goids gc
          ON gc.goid_h128 = e.caller_goid_h128
         AND gc.repo = e.repo
         AND gc.commit = e.commit
        LEFT JOIN core.goids gcallee
          ON gcallee.goid_h128 = e.callee_goid_h128
         AND gcallee.repo = e.repo
         AND gcallee.commit = e.commit
        LEFT JOIN analytics.goid_risk_factors rc
          ON rc.function_goid_h128 = e.caller_goid_h128
         AND rc.repo = e.repo
         AND rc.commit = e.commit
        LEFT JOIN analytics.goid_risk_factors rcallee
          ON rcallee.function_goid_h128 = e.callee_goid_h128
         AND rcallee.repo = e.repo
         AND rcallee.commit = e.commit;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_symbol_module_graph AS
        SELECT
            repo,
            commit,
            module,
            symbol_betweenness,
            symbol_closeness,
            symbol_eigenvector,
            symbol_harmonic,
            symbol_k_core,
            symbol_constraint,
            symbol_effective_size,
            symbol_community_id,
            symbol_component_id,
            symbol_component_size,
            created_at
        FROM analytics.symbol_graph_metrics_modules;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_validation_summary AS
        SELECT
            'function' AS domain,
            repo,
            commit,
            CAST(function_goid_h128 AS VARCHAR) AS entity_id,
            issue,
            detail
        FROM analytics.function_validation
        UNION ALL
        SELECT
            'graph' AS domain,
            repo,
            commit,
            CAST(entity_id AS VARCHAR) AS entity_id,
            issue,
            detail
        FROM analytics.graph_validation;
        """
    )
