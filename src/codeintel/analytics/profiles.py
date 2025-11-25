"""
Build denormalized profiles for functions, files, and modules.

These helpers layer on top of existing analytics outputs (risk factors,
coverage, tests, call graph, docstrings) to provide single-hop profiles that
are easy for downstream tools and agents to consume.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import duckdb

from codeintel.config.models import ProfilesAnalyticsConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.function_catalog_service import (
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

SLOW_TEST_THRESHOLD_MS = 1000.0


def _seed_catalog_modules(
    con: duckdb.DuckDBPyConnection,
    catalog_provider: FunctionCatalogProvider | None,
    repo: str,
    commit: str,
) -> bool:
    """
    Create or refresh a temp module mapping table from a catalog provider.

    Returns
    -------
    bool
        True when a catalog provided a module map and the temp table exists.
    """
    if catalog_provider is None:
        return False

    module_by_path = catalog_provider.catalog().module_by_path
    if not module_by_path:
        return False

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE temp.catalog_modules (
            path VARCHAR,
            module VARCHAR,
            repo VARCHAR,
            commit VARCHAR,
            language VARCHAR,
            tags JSON,
            owners JSON
        )
        """
    )
    con.executemany(
        "INSERT INTO temp.catalog_modules VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                path,
                module,
                repo,
                commit,
                "python",
                "[]",
                "[]",
            )
            for path, module in module_by_path.items()
        ],
    )
    return True


def build_function_profile(
    gateway: StorageGateway,
    cfg: ProfilesAnalyticsConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> None:
    """
    Populate `analytics.function_profile` for a repo/commit snapshot.

    The profile denormalizes risk factors, coverage, tests, docstrings, and call
    graph degrees into a single row per function GOID.
    """
    con = gateway.con
    ensure_schema(con, "analytics.function_profile")
    ensure_schema(con, "analytics.goid_risk_factors")
    use_catalog_modules = _seed_catalog_modules(con, catalog_provider, cfg.repo, cfg.commit)
    _ = catalog_provider or FunctionCatalogService.from_db(
        gateway,
        repo=cfg.repo,
        commit=cfg.commit,
    )

    con.execute(
        "DELETE FROM analytics.function_profile WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    now = datetime.now(tz=UTC)

    sql_core = """
        INSERT INTO analytics.function_profile (
            function_goid_h128,
            urn,
            repo,
            commit,
            rel_path,
            module,
            language,
            kind,
            qualname,
            start_line,
            end_line,
            loc,
            logical_loc,
            cyclomatic_complexity,
            complexity_bucket,
            param_count,
            positional_params,
            keyword_params,
            vararg,
            kwarg,
            max_nesting_depth,
            stmt_count,
            decorator_count,
            has_docstring,
            total_params,
            annotated_params,
            return_type,
            param_types,
            fully_typed,
            partial_typed,
            untyped,
            typedness_bucket,
            typedness_source,
            file_typed_ratio,
            static_error_count,
            has_static_errors,
            executable_lines,
            covered_lines,
            coverage_ratio,
            tested,
            untested_reason,
            tests_touching,
            failing_tests,
            slow_tests,
            flaky_tests,
            last_test_status,
            dominant_test_status,
            slow_test_threshold_ms,
            created_in_commit,
            created_at_history,
            last_modified_commit,
            last_modified_at,
            age_days,
            commit_count,
            author_count,
            lines_added,
            lines_deleted,
            churn_score,
            stability_bucket,
            call_fan_in,
            call_fan_out,
            call_edge_in_count,
            call_edge_out_count,
            call_is_leaf,
            call_is_entrypoint,
            call_is_public,
            risk_score,
            risk_level,
            risk_component_coverage,
            risk_component_complexity,
            risk_component_static,
            risk_component_hotspot,
            is_pure,
            uses_io,
            touches_db,
            uses_time,
            uses_randomness,
            modifies_globals,
            modifies_closure,
            spawns_threads_or_tasks,
            has_transitive_effects,
            purity_confidence,
            param_nullability_json,
            return_nullability,
            has_preconditions,
            has_postconditions,
            has_raises,
            contract_confidence,
            role,
            framework,
            role_confidence,
            role_sources_json,
            tags,
            owners,
            doc_short,
            doc_long,
            doc_params,
            doc_returns,
            created_at
        )
        WITH rf AS (
            SELECT *
            FROM analytics.goid_risk_factors
            WHERE repo = ? AND commit = ?
        ),
        fm AS (
            SELECT * FROM analytics.function_metrics
        ),
        ft AS (
            SELECT * FROM analytics.function_types
        ),
        effects AS (
            SELECT *
            FROM analytics.function_effects
            WHERE repo = ? AND commit = ?
        ),
        contracts AS (
            SELECT
                repo,
                commit,
                function_goid_h128,
                preconditions_json,
                postconditions_json,
                raises_json,
                param_nullability_json,
                return_nullability,
                contract_confidence,
                COALESCE(json_array_length(preconditions_json), 0) > 0 AS has_preconditions,
                COALESCE(json_array_length(postconditions_json), 0) > 0 AS has_postconditions,
                COALESCE(json_array_length(raises_json), 0) > 0 AS has_raises
            FROM analytics.function_contracts
            WHERE repo = ? AND commit = ?
        ),
        roles AS (
            SELECT
                repo,
                commit,
                function_goid_h128,
                role,
                framework,
                role_confidence,
                role_sources_json
            FROM analytics.semantic_roles_functions
            WHERE repo = ? AND commit = ?
        ),
        doc AS (
            SELECT
                repo,
                commit,
                rel_path,
                qualname,
                kind,
                short_desc    AS doc_short,
                long_desc     AS doc_long,
                params        AS doc_params,
                returns       AS doc_returns
            FROM core.docstrings
        ),
        t_stats AS (
            SELECT
                e.function_goid_h128,
                COUNT(DISTINCT e.test_id) AS tests_touching,
                COUNT(DISTINCT CASE WHEN tc.status IN ('failed', 'error') THEN e.test_id END)
                    AS failing_tests,
                COUNT(DISTINCT CASE WHEN tc.duration_ms > ? THEN e.test_id END) AS slow_tests,
                COUNT(DISTINCT CASE WHEN tc.flaky THEN e.test_id END) AS flaky_tests,
                MODE() WITHIN GROUP (ORDER BY tc.status) AS dominant_test_status
            FROM analytics.test_coverage_edges AS e
            LEFT JOIN analytics.test_catalog AS tc
              ON e.test_id = tc.test_id
             AND e.repo = tc.repo
             AND e.commit = tc.commit
            GROUP BY e.function_goid_h128
        ),
        cg_out AS (
            SELECT
                caller_goid_h128 AS function_goid_h128,
                COUNT(*) AS call_edge_out_count,
                COUNT(DISTINCT callee_goid_h128) AS call_fan_out
            FROM graph.call_graph_edges
            WHERE repo = ? AND commit = ?
            GROUP BY caller_goid_h128
        ),
        cg_in AS (
            SELECT
                callee_goid_h128 AS function_goid_h128,
                COUNT(*) AS call_edge_in_count,
                COUNT(DISTINCT caller_goid_h128) AS call_fan_in
            FROM graph.call_graph_edges
            WHERE callee_goid_h128 IS NOT NULL
              AND repo = ? AND commit = ?
            GROUP BY callee_goid_h128
        ),
        cg_nodes AS (
            SELECT
                goid_h128 AS function_goid_h128,
                is_public
            FROM graph.call_graph_nodes
        ),
        cg_degrees AS (
            SELECT
                COALESCE(co.function_goid_h128, ci.function_goid_h128, cn.function_goid_h128)
                    AS function_goid_h128,
                COALESCE(ci.call_fan_in, 0) AS call_fan_in,
                COALESCE(co.call_fan_out, 0) AS call_fan_out,
                COALESCE(ci.call_edge_in_count, 0) AS call_edge_in_count,
                COALESCE(co.call_edge_out_count, 0) AS call_edge_out_count,
                COALESCE(co.call_fan_out, 0) = 0 AS call_is_leaf,
                COALESCE(ci.call_fan_in, 0) = 0 AND COALESCE(co.call_fan_out, 0) > 0
                    AS call_is_entrypoint,
                cn.is_public AS call_is_public
            FROM cg_out AS co
            FULL OUTER JOIN cg_in AS ci USING (function_goid_h128)
            FULL OUTER JOIN cg_nodes AS cn USING (function_goid_h128)
        ),
        fh AS (
            SELECT
                repo,
                commit,
                function_goid_h128,
                created_in_commit,
                created_at,
                last_modified_commit,
                last_modified_at,
                age_days,
                commit_count,
                author_count,
                lines_added,
                lines_deleted,
                churn_score,
                stability_bucket
            FROM analytics.function_history
            WHERE repo = ? AND commit = ?
        )
        SELECT
            rf.function_goid_h128,
            rf.urn,
            rf.repo,
            rf.commit,
            rf.rel_path,
            m.module,
            rf.language,
            rf.kind,
            rf.qualname,
            fm.start_line,
            fm.end_line,
            rf.loc,
            rf.logical_loc,
            rf.cyclomatic_complexity,
            rf.complexity_bucket,
            fm.param_count,
            fm.positional_params,
            fm.keyword_only_params AS keyword_params,
            fm.has_varargs AS vararg,
            fm.has_varkw AS kwarg,
            fm.max_nesting_depth,
            fm.stmt_count,
            fm.decorator_count,
            fm.has_docstring,
            ft.total_params,
            ft.annotated_params,
            ft.return_type,
            ft.param_types,
            ft.fully_typed,
            ft.partial_typed,
            ft.untyped,
            rf.typedness_bucket,
            rf.typedness_source,
            rf.file_typed_ratio,
            rf.static_error_count,
            rf.has_static_errors,
            rf.executable_lines,
            rf.covered_lines,
            rf.coverage_ratio,
            rf.tested,
            cf.untested_reason,
            COALESCE(t_stats.tests_touching, 0),
            COALESCE(t_stats.failing_tests, 0),
            COALESCE(t_stats.slow_tests, 0),
            COALESCE(t_stats.flaky_tests, 0),
            rf.last_test_status,
            COALESCE(t_stats.dominant_test_status, rf.last_test_status),
            ?,
            fh.created_in_commit,
            fh.created_at,
            fh.last_modified_commit,
            fh.last_modified_at,
            fh.age_days,
            COALESCE(fh.commit_count, 0),
            COALESCE(fh.author_count, 0),
            COALESCE(fh.lines_added, 0),
            COALESCE(fh.lines_deleted, 0),
            COALESCE(fh.churn_score, 0.0),
            COALESCE(fh.stability_bucket, 'unknown'),
            COALESCE(cg.call_fan_in, 0),
            COALESCE(cg.call_fan_out, 0),
            COALESCE(cg.call_edge_in_count, 0),
            COALESCE(cg.call_edge_out_count, 0),
            cg.call_is_leaf,
            cg.call_is_entrypoint,
            cg.call_is_public,
            rf.risk_score,
            rf.risk_level,
            COALESCE(1.0 - rf.coverage_ratio, 1.0) * 0.4 AS risk_component_coverage,
            CASE rf.complexity_bucket
                WHEN 'high' THEN 0.4
                WHEN 'medium' THEN 0.2
                ELSE 0.0
            END AS risk_component_complexity,
            CASE WHEN rf.has_static_errors THEN 0.2 ELSE 0.0 END AS risk_component_static,
            CASE WHEN rf.hotspot_score > 0 THEN 0.1 ELSE 0.0 END AS risk_component_hotspot,
            COALESCE(fe.is_pure, FALSE),
            COALESCE(fe.uses_io, FALSE),
            COALESCE(fe.touches_db, FALSE),
            COALESCE(fe.uses_time, FALSE),
            COALESCE(fe.uses_randomness, FALSE),
            COALESCE(fe.modifies_globals, FALSE),
            COALESCE(fe.modifies_closure, FALSE),
            COALESCE(fe.spawns_threads_or_tasks, FALSE),
            COALESCE(fe.has_transitive_effects, FALSE),
            fe.purity_confidence,
            fc.param_nullability_json,
            fc.return_nullability,
            COALESCE(fc.has_preconditions, FALSE),
            COALESCE(fc.has_postconditions, FALSE),
            COALESCE(fc.has_raises, FALSE),
            fc.contract_confidence,
            fr.role,
            fr.framework,
            fr.role_confidence,
            fr.role_sources_json,
            rf.tags,
            rf.owners,
            doc.doc_short,
            doc.doc_long,
            doc.doc_params,
            doc.doc_returns,
            ?
        FROM rf
        LEFT JOIN analytics.function_metrics AS fm
          ON rf.function_goid_h128 = fm.function_goid_h128
         AND rf.repo = fm.repo
         AND rf.commit = fm.commit
        LEFT JOIN analytics.function_types AS ft
          ON rf.function_goid_h128 = ft.function_goid_h128
         AND rf.repo = ft.repo
         AND rf.commit = ft.commit
        LEFT JOIN analytics.coverage_functions AS cf
          ON rf.function_goid_h128 = cf.function_goid_h128
         AND rf.repo = cf.repo
         AND rf.commit = cf.commit
        LEFT JOIN t_stats
          ON rf.function_goid_h128 = t_stats.function_goid_h128
        LEFT JOIN cg_degrees AS cg
          ON rf.function_goid_h128 = cg.function_goid_h128
        LEFT JOIN effects AS fe
          ON rf.function_goid_h128 = fe.function_goid_h128
         AND rf.repo = fe.repo
         AND rf.commit = fe.commit
        LEFT JOIN contracts AS fc
          ON rf.function_goid_h128 = fc.function_goid_h128
         AND rf.repo = fc.repo
         AND rf.commit = fc.commit
        LEFT JOIN roles AS fr
          ON rf.function_goid_h128 = fr.function_goid_h128
         AND rf.repo = fr.repo
         AND rf.commit = fr.commit
        LEFT JOIN fh
          ON fh.function_goid_h128 = rf.function_goid_h128
         AND fh.repo = rf.repo
         AND fh.commit = rf.commit
        LEFT JOIN core.modules AS m
          ON m.path = rf.rel_path
         AND (m.repo IS NULL OR m.repo = rf.repo)
         AND (m.commit IS NULL OR m.commit = rf.commit)
        LEFT JOIN doc
          ON doc.repo = rf.repo
         AND doc.commit = rf.commit
         AND doc.rel_path = rf.rel_path
         AND doc.qualname = rf.qualname
         AND doc.kind = rf.kind;
        """
    sql_temp = sql_core.replace("core.modules", "temp.catalog_modules")
    con.execute(
        sql_temp if use_catalog_modules else sql_core,
        [
            cfg.repo,
            cfg.commit,
            cfg.repo,
            cfg.commit,
            cfg.repo,
            cfg.commit,
            cfg.repo,
            cfg.commit,
            SLOW_TEST_THRESHOLD_MS,
            cfg.repo,
            cfg.commit,
            cfg.repo,
            cfg.commit,
            cfg.repo,
            cfg.commit,
            SLOW_TEST_THRESHOLD_MS,
            now,
        ],
    )

    count_row = con.execute(
        "SELECT COUNT(*) FROM analytics.function_profile WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    ).fetchone()
    count = int(count_row[0]) if count_row is not None else 0
    log.info("function_profile populated: %s rows for %s@%s", count, cfg.repo, cfg.commit)


def build_file_profile(
    gateway: StorageGateway,
    cfg: ProfilesAnalyticsConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> None:
    """
    Populate `analytics.file_profile` by aggregating function_profile rows.

    File-level rows blend AST metrics, typedness, hotspots, risk buckets, and
    coverage summaries to give a holistic view of each path.
    """
    con = gateway.con
    ensure_schema(con, "analytics.file_profile")
    use_catalog_modules = _seed_catalog_modules(con, catalog_provider, cfg.repo, cfg.commit)

    con.execute(
        "DELETE FROM analytics.file_profile WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    now = datetime.now(tz=UTC)

    sql_core = """
        INSERT INTO analytics.file_profile (
            repo,
            commit,
            rel_path,
            module,
            language,
            node_count,
            function_count,
            class_count,
            avg_depth,
            max_depth,
            ast_complexity,
            hotspot_score,
            commit_count,
            author_count,
            lines_added,
            lines_deleted,
            annotation_ratio,
            untyped_defs,
            overlay_needed,
            type_error_count,
            static_error_count,
            has_static_errors,
            total_functions,
            public_functions,
            avg_loc,
            max_loc,
            avg_cyclomatic_complexity,
            max_cyclomatic_complexity,
            high_risk_function_count,
            medium_risk_function_count,
            max_risk_score,
            file_coverage_ratio,
            tested_function_count,
            untested_function_count,
            tests_touching,
            tags,
            owners,
            created_at
        )
        WITH fm AS (
            SELECT
                repo,
                commit,
                rel_path,
                COUNT(*) AS total_functions,
                COUNT(*) FILTER (WHERE call_is_public) AS public_functions,
                AVG(loc) AS avg_loc,
                MAX(loc) AS max_loc,
                AVG(cyclomatic_complexity) AS avg_cyclomatic_complexity,
                MAX(cyclomatic_complexity) AS max_cyclomatic_complexity,
                SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END)
                    AS high_risk_function_count,
                SUM(CASE WHEN risk_level = 'medium' THEN 1 ELSE 0 END)
                    AS medium_risk_function_count,
                MAX(risk_score) AS max_risk_score,
                SUM(covered_lines) AS sum_covered_lines,
                SUM(executable_lines) AS sum_exec_lines,
                SUM(CASE WHEN tested THEN 1 ELSE 0 END) AS tested_function_count,
                SUM(CASE WHEN NOT tested THEN 1 ELSE 0 END) AS untested_function_count,
                SUM(tests_touching) AS tests_touching
            FROM analytics.function_profile
            WHERE repo = ? AND commit = ?
            GROUP BY repo, commit, rel_path
        ),
        ast AS (
            SELECT * FROM core.ast_metrics
        ),
        hs AS (
            SELECT * FROM analytics.hotspots
        ),
        ty AS (
            SELECT * FROM analytics.typedness
        ),
        sd AS (
            SELECT
                rel_path,
                total_errors AS static_error_count,
                has_errors AS has_static_errors
            FROM analytics.static_diagnostics
        ),
        mod AS (
            SELECT repo, commit, path, module, language, tags, owners
            FROM core.modules
        )
        SELECT
            fm.repo,
            fm.commit,
            fm.rel_path,
            mod.module,
            mod.language,
            ast.node_count,
            ast.function_count,
            ast.class_count,
            ast.avg_depth,
            ast.max_depth,
            ast.complexity AS ast_complexity,
            hs.score AS hotspot_score,
            hs.commit_count,
            hs.author_count,
            hs.lines_added,
            hs.lines_deleted,
            CAST(ty.annotation_ratio->>'params' AS DOUBLE) AS annotation_ratio,
            ty.untyped_defs,
            ty.overlay_needed,
            ty.type_error_count,
            sd.static_error_count,
            sd.has_static_errors,
            fm.total_functions,
            fm.public_functions,
            fm.avg_loc,
            fm.max_loc,
            fm.avg_cyclomatic_complexity,
            fm.max_cyclomatic_complexity,
            fm.high_risk_function_count,
            fm.medium_risk_function_count,
            fm.max_risk_score,
            CASE
                WHEN fm.sum_exec_lines > 0 THEN fm.sum_covered_lines * 1.0 / fm.sum_exec_lines
                ELSE NULL
            END AS file_coverage_ratio,
            fm.tested_function_count,
            fm.untested_function_count,
            fm.tests_touching,
            mod.tags,
            mod.owners,
            ?
        FROM fm
        LEFT JOIN ast
          ON fm.rel_path = ast.rel_path
        LEFT JOIN hs
          ON fm.rel_path = hs.rel_path
        LEFT JOIN ty
          ON fm.rel_path = ty.path
        LEFT JOIN sd
          ON fm.rel_path = sd.rel_path
        LEFT JOIN mod
          ON fm.repo = mod.repo
         AND fm.commit = mod.commit
         AND fm.rel_path = mod.path;
        """
    sql_temp = sql_core.replace("core.modules", "temp.catalog_modules")
    con.execute(sql_temp if use_catalog_modules else sql_core, [cfg.repo, cfg.commit, now])

    count_row = con.execute(
        "SELECT COUNT(*) FROM analytics.file_profile WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    ).fetchone()
    count = int(count_row[0]) if count_row is not None else 0
    log.info("file_profile populated: %s rows for %s@%s", count, cfg.repo, cfg.commit)


def build_module_profile(
    gateway: StorageGateway,
    cfg: ProfilesAnalyticsConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> None:
    """Populate `analytics.module_profile` by aggregating file and function profiles."""
    con = gateway.con
    ensure_schema(con, "analytics.module_profile")
    use_catalog_modules = _seed_catalog_modules(con, catalog_provider, cfg.repo, cfg.commit)

    con.execute(
        "DELETE FROM analytics.module_profile WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    now = datetime.now(tz=UTC)

    sql_core = """
        INSERT INTO analytics.module_profile (
            repo,
            commit,
            module,
            path,
            language,
            file_count,
            total_loc,
            total_logical_loc,
            function_count,
            class_count,
            avg_file_complexity,
            max_file_complexity,
            high_risk_function_count,
            medium_risk_function_count,
            low_risk_function_count,
            max_risk_score,
            avg_risk_score,
            module_coverage_ratio,
            tested_function_count,
            untested_function_count,
            import_fan_in,
            import_fan_out,
            cycle_group,
            in_cycle,
            role,
            role_confidence,
            role_sources_json,
            tags,
            owners,
            created_at
        )
        WITH func_stats AS (
            SELECT
                repo,
                commit,
                module,
                COUNT(*) AS function_count,
                SUM(COALESCE(loc, 0)) AS total_loc,
                SUM(COALESCE(logical_loc, 0)) AS total_logical_loc,
                SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) AS high_risk_function_count,
                SUM(CASE WHEN risk_level = 'medium' THEN 1 ELSE 0 END)
                    AS medium_risk_function_count,
                SUM(CASE WHEN risk_level = 'low' THEN 1 ELSE 0 END) AS low_risk_function_count,
                MAX(risk_score) AS max_risk_score,
                AVG(risk_score) AS avg_risk_score,
                SUM(CASE WHEN tested THEN 1 ELSE 0 END) AS tested_function_count,
                SUM(CASE WHEN NOT tested THEN 1 ELSE 0 END) AS untested_function_count
            FROM analytics.function_profile
            WHERE repo = ? AND commit = ?
            GROUP BY repo, commit, module
        ),
        files AS (
            SELECT
                repo,
                commit,
                module,
                COUNT(*) AS file_count,
                SUM(class_count) AS class_count,
                AVG(ast_complexity) AS avg_file_complexity,
                MAX(ast_complexity) AS max_file_complexity
            FROM analytics.file_profile
            WHERE repo = ? AND commit = ?
            GROUP BY repo, commit, module
        ),
        mod AS (
            SELECT repo, commit, module, path, language, tags, owners
            FROM core.modules
        ),
        imports AS (
            SELECT
                repo,
                commit,
                src_module AS module,
                MAX(src_fan_out) AS import_fan_out,
                MAX(dst_fan_in) FILTER (WHERE dst_module = src_module) AS import_fan_in,
                MAX(cycle_group) AS cycle_group,
                MAX(CASE WHEN cycle_group IS NOT NULL THEN 1 ELSE 0 END) AS in_cycle_flag
            FROM graph.import_graph_edges
            WHERE repo = ? AND commit = ?
            GROUP BY repo, commit, src_module
        ),
        roles AS (
            SELECT repo, commit, module, role, role_confidence, role_sources_json
            FROM analytics.semantic_roles_modules
            WHERE repo = ? AND commit = ?
        )
        SELECT
            mod.repo,
            mod.commit,
            mod.module,
            mod.path,
            mod.language,
            COALESCE(files.file_count, 0),
            COALESCE(func_stats.total_loc, 0),
            COALESCE(func_stats.total_logical_loc, 0),
            COALESCE(func_stats.function_count, 0),
            COALESCE(files.class_count, 0),
            files.avg_file_complexity,
            files.max_file_complexity,
            COALESCE(func_stats.high_risk_function_count, 0),
            COALESCE(func_stats.medium_risk_function_count, 0),
            COALESCE(func_stats.low_risk_function_count, 0),
            func_stats.max_risk_score,
            func_stats.avg_risk_score,
            CASE
                WHEN COALESCE(func_stats.tested_function_count, 0)
                     + COALESCE(func_stats.untested_function_count, 0) > 0
                THEN
                    CAST(func_stats.tested_function_count AS DOUBLE)
                    / (func_stats.tested_function_count + func_stats.untested_function_count)
                ELSE NULL
            END AS module_coverage_ratio,
            func_stats.tested_function_count,
            func_stats.untested_function_count,
            COALESCE(imports.import_fan_in, 0),
            COALESCE(imports.import_fan_out, 0),
            imports.cycle_group,
            imports.in_cycle_flag > 0 AS in_cycle,
            roles.role,
            roles.role_confidence,
            roles.role_sources_json,
            mod.tags,
            mod.owners,
            ?
        FROM mod
        LEFT JOIN func_stats
          ON func_stats.module = mod.module
         AND func_stats.repo = mod.repo
         AND func_stats.commit = mod.commit
        LEFT JOIN files
          ON files.module = mod.module
         AND files.repo = mod.repo
         AND files.commit = mod.commit
        LEFT JOIN imports
          ON imports.module = mod.module
         AND imports.repo = mod.repo
         AND imports.commit = mod.commit
        LEFT JOIN roles
          ON roles.module = mod.module
         AND roles.repo = mod.repo
         AND roles.commit = mod.commit
        WHERE mod.repo = ?
          AND mod.commit = ?;
        """
    sql_temp = sql_core.replace("core.modules", "temp.catalog_modules")
    con.execute(
        sql_temp if use_catalog_modules else sql_core,
        [
            cfg.repo,
            cfg.commit,
            cfg.repo,
            cfg.commit,
            cfg.repo,
            cfg.commit,
            cfg.repo,
            cfg.commit,
            now,
            cfg.repo,
            cfg.commit,
        ],
    )

    count_row = con.execute(
        "SELECT COUNT(*) FROM analytics.module_profile WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    ).fetchone()
    count = int(count_row[0]) if count_row is not None else 0
    log.info("module_profile populated: %s rows for %s@%s", count, cfg.repo, cfg.commit)
