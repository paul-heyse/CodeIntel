"""
DuckDB schema definitions for the CodeIntel metadata warehouse.

These DDLs are derived from README_METADATA.md ("CodeIntel Metadata Outputs")
and cover all exported datasets (goids, call graph, CFG/DFG, coverage, tests,
risk factors, etc.).
"""

from __future__ import annotations

from collections.abc import Iterable

from duckdb import DuckDBPyConnection

SCHEMAS = ("core", "graph", "analytics", "docs")

# One CREATE TABLE per dataset; views go into a separate module (e.g. views.py).
TABLE_DDL: dict[str, str] = {
    # -------------------------------------------------------------------------
    # core schema: identifiers, AST/CST, modules, repo metadata
    # -------------------------------------------------------------------------
    "core.goids": """
    CREATE TABLE IF NOT EXISTS core.goids (
        goid_h128  DECIMAL(38,0),
        urn        TEXT,
        repo       TEXT,
        commit     TEXT,
        rel_path   TEXT,
        language   TEXT,
        kind       TEXT,
        qualname   TEXT,
        start_line INTEGER,
        end_line   INTEGER,
        created_at TIMESTAMP
    );
    """,
    "core.goid_crosswalk": """
    CREATE TABLE IF NOT EXISTS core.goid_crosswalk (
        goid         TEXT,
        lang         TEXT,
        module_path  TEXT,
        file_path    TEXT,
        start_line   INTEGER,
        end_line     INTEGER,
        scip_symbol  TEXT,
        ast_qualname TEXT,
        cst_node_id  TEXT,
        chunk_id     TEXT,
        symbol_id    TEXT,
        updated_at   TIMESTAMP
    );
    """,
    "core.ast_nodes": """
    CREATE TABLE IF NOT EXISTS core.ast_nodes (
        path            TEXT,
        node_type       TEXT,
        name            TEXT,
        qualname        TEXT,
        lineno          INTEGER,
        end_lineno      INTEGER,
        col_offset      INTEGER,
        end_col_offset  INTEGER,
        parent_qualname TEXT,
        decorators      JSON,
        docstring       TEXT,
        hash            TEXT
    );
    """,
    "core.ast_metrics": """
    CREATE TABLE IF NOT EXISTS core.ast_metrics (
        rel_path    TEXT,
        node_count  INTEGER,
        function_count INTEGER,
        class_count INTEGER,
        avg_depth   DOUBLE,
        max_depth   INTEGER,
        complexity  DOUBLE,
        generated_at TIMESTAMP
    );
    """,
    "core.modules": """
    CREATE TABLE IF NOT EXISTS core.modules (
        module   TEXT,
        path     TEXT,
        repo     TEXT,
        commit   TEXT,
        language TEXT,
        tags     JSON,
        owners   JSON
    );
    """,
    "core.repo_map": """
    CREATE TABLE IF NOT EXISTS core.repo_map (
        repo         TEXT,
        commit       TEXT,
        modules      JSON,
        overlays     JSON,
        generated_at TIMESTAMP
    );
    """,
    "core.cst_nodes": """
    CREATE TABLE IF NOT EXISTS core.cst_nodes (
        path         TEXT,
        node_id      TEXT,
        kind         TEXT,
        span         JSON,
        text_preview TEXT,
        parents      JSON,
        qnames       JSON
    );
    """,
    # -------------------------------------------------------------------------
    # graph schema: call graph, CFG, DFG, imports, symbol uses
    # -------------------------------------------------------------------------
    "graph.call_graph_nodes": """
    CREATE TABLE IF NOT EXISTS graph.call_graph_nodes (
        goid_h128 DECIMAL(38,0),
        language  TEXT,
        kind      TEXT,
        arity     INTEGER,
        is_public BOOLEAN,
        rel_path  TEXT
    );
    """,
    "graph.call_graph_edges": """
    CREATE TABLE IF NOT EXISTS graph.call_graph_edges (
        caller_goid_h128 DECIMAL(38,0),
        callee_goid_h128 DECIMAL(38,0),
        callsite_path    TEXT,
        callsite_line    INTEGER,
        callsite_col     INTEGER,
        language         TEXT,
        kind             TEXT,
        resolved_via     TEXT,
        confidence       DOUBLE,
        evidence_json    JSON
    );
    """,
    "graph.cfg_blocks": """
    CREATE TABLE IF NOT EXISTS graph.cfg_blocks (
        function_goid_h128 DECIMAL(38,0),
        block_idx          INTEGER,
        block_id           TEXT,
        label              TEXT,
        file_path          TEXT,
        start_line         INTEGER,
        end_line           INTEGER,
        kind               TEXT,
        stmts_json         JSON,
        in_degree          INTEGER,
        out_degree         INTEGER
    );
    """,
    "graph.cfg_edges": """
    CREATE TABLE IF NOT EXISTS graph.cfg_edges (
        function_goid_h128 DECIMAL(38,0),
        src_block_idx      INTEGER,
        dst_block_idx      INTEGER,
        edge_type          TEXT,
        cond_json          JSON,
        src                TEXT,
        dst                TEXT
    );
    """,
    "graph.dfg_edges": """
    CREATE TABLE IF NOT EXISTS graph.dfg_edges (
        function_goid_h128 DECIMAL(38,0),
        src_block_idx      INTEGER,
        dst_block_idx      INTEGER,
        src_symbol         TEXT,
        dst_symbol         TEXT,
        via_phi            BOOLEAN,
        use_kind           TEXT
    );
    """,
    "graph.import_graph_edges": """
    CREATE TABLE IF NOT EXISTS graph.import_graph_edges (
        src_module  TEXT,
        dst_module  TEXT,
        src_fan_out INTEGER,
        dst_fan_in  INTEGER,
        cycle_group INTEGER
    );
    """,
    "graph.symbol_use_edges": """
    CREATE TABLE IF NOT EXISTS graph.symbol_use_edges (
        symbol      TEXT,
        def_path    TEXT,
        use_path    TEXT,
        same_file   BOOLEAN,
        same_module BOOLEAN
    );
    """,
    # -------------------------------------------------------------------------
    # analytics schema: metrics, typedness, coverage, tests, risk
    # -------------------------------------------------------------------------
    "analytics.hotspots": """
    CREATE TABLE IF NOT EXISTS analytics.hotspots (
        rel_path     TEXT,
        commit_count INTEGER,
        author_count INTEGER,
        lines_added  INTEGER,
        lines_deleted INTEGER,
        complexity   DOUBLE,
        score        DOUBLE
    );
    """,
    "analytics.typedness": """
    CREATE TABLE IF NOT EXISTS analytics.typedness (
        path             TEXT,
        type_error_count INTEGER,
        annotation_ratio JSON,
        untyped_defs     INTEGER,
        overlay_needed   BOOLEAN
    );
    """,
    "analytics.function_metrics": """
    CREATE TABLE IF NOT EXISTS analytics.function_metrics (
        function_goid_h128     DECIMAL(38,0),
        urn                    TEXT,
        repo                   TEXT,
        commit                 TEXT,
        rel_path               TEXT,
        language               TEXT,
        kind                   TEXT,
        qualname               TEXT,
        start_line             INTEGER,
        end_line               INTEGER,
        loc                    INTEGER,
        logical_loc            INTEGER,
        param_count            INTEGER,
        positional_params      INTEGER,
        keyword_only_params    INTEGER,
        has_varargs            BOOLEAN,
        has_varkw              BOOLEAN,
        is_async               BOOLEAN,
        is_generator           BOOLEAN,
        return_count           INTEGER,
        yield_count            INTEGER,
        raise_count            INTEGER,
        cyclomatic_complexity  INTEGER,
        max_nesting_depth      INTEGER,
        stmt_count             INTEGER,
        decorator_count        INTEGER,
        has_docstring          BOOLEAN,
        complexity_bucket      TEXT,
        created_at             TIMESTAMP
    );
    """,
    "analytics.function_types": """
    CREATE TABLE IF NOT EXISTS analytics.function_types (
        function_goid_h128   DECIMAL(38,0),
        urn                  TEXT,
        repo                 TEXT,
        commit               TEXT,
        rel_path             TEXT,
        language             TEXT,
        kind                 TEXT,
        qualname             TEXT,
        start_line           INTEGER,
        end_line             INTEGER,
        total_params         INTEGER,
        annotated_params     INTEGER,
        unannotated_params   INTEGER,
        param_typed_ratio    DOUBLE,
        has_return_annotation BOOLEAN,
        return_type          TEXT,
        return_type_source   TEXT,
        type_comment         TEXT,
        param_types          JSON,
        fully_typed          BOOLEAN,
        partial_typed        BOOLEAN,
        untyped              BOOLEAN,
        typedness_bucket     TEXT,
        typedness_source     TEXT,
        created_at           TIMESTAMP
    );
    """,
    "analytics.tags_index": """
    CREATE TABLE IF NOT EXISTS analytics.tags_index (
        tag         TEXT,
        description TEXT,
        includes    JSON,
        excludes    JSON,
        matches     JSON
    );
    """,
    "analytics.config_values": """
    CREATE TABLE IF NOT EXISTS analytics.config_values (
        config_path       TEXT,
        format            TEXT,
        key               TEXT,
        reference_paths   JSON,
        reference_modules JSON,
        reference_count   INTEGER
    );
    """,
    "analytics.static_diagnostics": """
    CREATE TABLE IF NOT EXISTS analytics.static_diagnostics (
        rel_path       TEXT,
        pyrefly_errors INTEGER,
        pyright_errors INTEGER,
        total_errors   INTEGER,
        has_errors     BOOLEAN
    );
    """,
    "analytics.coverage_lines": """
    CREATE TABLE IF NOT EXISTS analytics.coverage_lines (
        repo          TEXT,
        commit        TEXT,
        rel_path      TEXT,
        line          INTEGER,
        is_executable BOOLEAN,
        is_covered    BOOLEAN,
        hits          INTEGER,
        context_count INTEGER,
        created_at    TIMESTAMP
    );
    """,
    "analytics.coverage_functions": """
    CREATE TABLE IF NOT EXISTS analytics.coverage_functions (
        function_goid_h128 DECIMAL(38,0),
        urn                TEXT,
        repo               TEXT,
        commit             TEXT,
        rel_path           TEXT,
        language           TEXT,
        kind               TEXT,
        qualname           TEXT,
        start_line         INTEGER,
        end_line           INTEGER,
        executable_lines   INTEGER,
        covered_lines      INTEGER,
        coverage_ratio     DOUBLE,
        tested             BOOLEAN,
        untested_reason    TEXT,
        created_at         TIMESTAMP
    );
    """,
    "analytics.test_catalog": """
    CREATE TABLE IF NOT EXISTS analytics.test_catalog (
        test_id        TEXT,
        test_goid_h128 DECIMAL(38,0),
        urn            TEXT,
        repo           TEXT,
        commit         TEXT,
        rel_path       TEXT,
        qualname       TEXT,
        kind           TEXT,
        status         TEXT,
        duration_ms    DOUBLE,
        markers        JSON,
        parametrized   BOOLEAN,
        flaky          BOOLEAN,
        created_at     TIMESTAMP
    );
    """,
    "analytics.test_coverage_edges": """
    CREATE TABLE IF NOT EXISTS analytics.test_coverage_edges (
        test_id            TEXT,
        test_goid_h128     DECIMAL(38,0),
        function_goid_h128 DECIMAL(38,0),
        urn                TEXT,
        repo               TEXT,
        commit             TEXT,
        rel_path           TEXT,
        qualname           TEXT,
        covered_lines      INTEGER,
        executable_lines   INTEGER,
        coverage_ratio     DOUBLE,
        last_status        TEXT,
        created_at         TIMESTAMP
    );
    """,
    "analytics.goid_risk_factors": """
    CREATE TABLE IF NOT EXISTS analytics.goid_risk_factors (
        function_goid_h128   DECIMAL(38,0),
        urn                  TEXT,
        repo                 TEXT,
        commit               TEXT,
        rel_path             TEXT,
        language             TEXT,
        kind                 TEXT,
        qualname             TEXT,
        loc                  INTEGER,
        logical_loc          INTEGER,
        cyclomatic_complexity INTEGER,
        complexity_bucket    TEXT,
        typedness_bucket     TEXT,
        typedness_source     TEXT,
        hotspot_score        DOUBLE,
        file_typed_ratio     DOUBLE,
        static_error_count   INTEGER,
        has_static_errors    BOOLEAN,
        executable_lines     INTEGER,
        covered_lines        INTEGER,
        coverage_ratio       DOUBLE,
        tested               BOOLEAN,
        test_count           INTEGER,
        failing_test_count   INTEGER,
        last_test_status     TEXT,
        risk_score           DOUBLE,
        risk_level           TEXT,
        tags                 JSON,
        owners               JSON,
        created_at           TIMESTAMP
    );
    """,
}

INDEX_DDL: tuple[str, ...] = (
    # core
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_core_goids_h128 ON core.goids(goid_h128);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_core_goids_urn ON core.goids(urn);",
    "CREATE INDEX IF NOT EXISTS idx_core_goids_path ON core.goids(rel_path);",
    "CREATE INDEX IF NOT EXISTS idx_core_gcw_goid ON core.goid_crosswalk(goid);",
    "CREATE INDEX IF NOT EXISTS idx_core_modules_path ON core.modules(path);",
    "CREATE INDEX IF NOT EXISTS idx_core_modules_module ON core.modules(module);",
    # graph
    "CREATE INDEX IF NOT EXISTS idx_graph_call_edges_caller ON graph.call_graph_edges(caller_goid_h128);",
    "CREATE INDEX IF NOT EXISTS idx_graph_call_edges_callee ON graph.call_graph_edges(callee_goid_h128);",
    "CREATE INDEX IF NOT EXISTS idx_graph_cfg_blocks_fn ON graph.cfg_blocks(function_goid_h128);",
    "CREATE INDEX IF NOT EXISTS idx_graph_cfg_edges_fn ON graph.cfg_edges(function_goid_h128);",
    "CREATE INDEX IF NOT EXISTS idx_graph_dfg_edges_fn ON graph.dfg_edges(function_goid_h128);",
    "CREATE INDEX IF NOT EXISTS idx_graph_import_edges_src ON graph.import_graph_edges(src_module);",
    "CREATE INDEX IF NOT EXISTS idx_graph_import_edges_dst ON graph.import_graph_edges(dst_module);",
    "CREATE INDEX IF NOT EXISTS idx_graph_symbol_use_symbol ON graph.symbol_use_edges(symbol);",
    # analytics
    "CREATE INDEX IF NOT EXISTS idx_analytics_function_metrics_goid ON analytics.function_metrics(function_goid_h128);",
    "CREATE INDEX IF NOT EXISTS idx_analytics_function_types_goid ON analytics.function_types(function_goid_h128);",
    "CREATE INDEX IF NOT EXISTS idx_analytics_coverage_functions_goid ON analytics.coverage_functions(function_goid_h128);",
    "CREATE INDEX IF NOT EXISTS idx_analytics_gorf_goid ON analytics.goid_risk_factors(function_goid_h128);",
    "CREATE INDEX IF NOT EXISTS idx_analytics_test_cov_edges_goid ON analytics.test_coverage_edges(function_goid_h128);",
    "CREATE INDEX IF NOT EXISTS idx_analytics_test_catalog_id ON analytics.test_catalog(test_id);",
)


def create_schemas(con: DuckDBPyConnection) -> None:
    """Ensure logical schemas (core, graph, analytics, docs) exist."""
    for schema in SCHEMAS:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")


def apply_all_schemas(
    con: DuckDBPyConnection,
    extra_ddl: Iterable[str] | None = None,
) -> None:
    """
    Create all known tables in the current DuckDB database.

    Call this once at startup before running any pipeline steps that
    insert into these tables.
    """
    create_schemas(con)

    for ddl in TABLE_DDL.values():
        con.execute(ddl)

    for ddl in INDEX_DDL:
        con.execute(ddl)

    if extra_ddl:
        for stmt in extra_ddl:
            con.execute(stmt)
