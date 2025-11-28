"""Helpers to seed a minimal architecture dataset for gateway-backed tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway
from codeintel.storage.gateway import open_memory_gateway as _open_memory_gateway


def open_seeded_architecture_gateway(
    *,
    repo: str,
    commit: str,
    db_path: Path | None = None,
    strict_schema: bool = True,
) -> StorageGateway:
    """
    Open a gateway (file-backed or in-memory) and seed architecture tables.

    Parameters
    ----------
    repo : str
        Repository identifier to seed.
    commit : str
        Commit hash to seed.
    db_path : Path | None
        Optional on-disk location for the DuckDB file. When omitted, an in-memory
        gateway is created.
    strict_schema : bool
        When True, schemas/views/validation are applied before seeding.

    Returns
    -------
    StorageGateway
        Gateway with schema, views, and architecture seed data applied.
    """
    if db_path is None:
        gateway = _open_memory_gateway(
            apply_schema=True,
            ensure_views=strict_schema,
            validate_schema=strict_schema,
        )
    else:
        cfg = StorageConfig(
            db_path=db_path,
            apply_schema=True,
            ensure_views=strict_schema,
            validate_schema=strict_schema,
        )
        gateway = open_gateway(cfg)
    return seed_architecture(gateway=gateway, repo=repo, commit=commit)


def seed_architecture(*, gateway: StorageGateway, repo: str, commit: str) -> StorageGateway:
    """
    Populate the minimal set of architecture tables required by docs views.

    Parameters
    ----------
    gateway : StorageGateway
        Gateway to seed with architecture tables and views.
    repo : str
        Repository identifier to attach to seed data.
    commit : str
        Commit hash anchoring the seeded rows.

    Returns
    -------
    StorageGateway
        Gateway with architecture tables populated for tests.
    """
    now = datetime.now(UTC)
    now_iso = now.isoformat()

    gateway.core.insert_repo_map([(repo, commit, "{}", "{}", now_iso)])
    gateway.core.insert_modules(
        [
            ("pkg.mod", "pkg/mod.py", repo, commit),
            ("pkg.alpha", "pkg/alpha.py", repo, commit),
            ("pkg.beta", "pkg/beta.py", repo, commit),
        ]
    )
    gateway.core.insert_goids(
        [
            (
                1,
                "goid:demo/repo#python:function:pkg.mod.func",
                repo,
                commit,
                "pkg/mod.py",
                "python",
                "function",
                "pkg.mod.func",
                1,
                2,
                now_iso,
            )
        ]
    )
    gateway.analytics.insert_function_metrics(
        [
            (
                1,
                "goid:demo/repo#python:function:pkg.mod.func",
                repo,
                commit,
                "pkg/mod.py",
                "python",
                "function",
                "pkg.mod.func",
                1,
                2,
                2,
                2,
                0,
                0,
                0,
                False,
                False,
                False,
                False,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                False,
                "low",
                now_iso,
            )
        ]
    )
    gateway.analytics.insert_goid_risk_factors(
        [
            (
                1,
                "goid:demo/repo#python:function:pkg.mod.func",
                repo,
                commit,
                "pkg/mod.py",
                "python",
                "function",
                "pkg.mod.func",
                10,
                10,
                1,
                "low",
                "full",
                "annotations",
                0.1,
                1.0,
                0,
                False,
                2,
                2,
                1.0,
                True,
                1,
                0,
                "passed",
                0.2,
                "low",
                "[]",
                "[]",
                now_iso,
            )
        ]
    )
    gateway.graph.insert_call_graph_edges(
        [
            (
                repo,
                commit,
                1,
                1,
                "pkg/mod.py",
                1,
                1,
                "python",
                "direct",
                "local",
                0.9,
                "{}",
            )
        ]
    )
    gateway.analytics.insert_graph_metrics_functions(
        [(repo, commit, 1, 2, 3, 2, 3, 0.5, 0.1, 0.2, False, 0, 1, now_iso)]
    )
    gateway.analytics.insert_graph_metrics_modules(
        [
            (
                repo,
                commit,
                "pkg.mod",
                3,
                2,
                3,
                2,
                0.4,
                0.2,
                0.3,
                False,
                0,
                1,
                5,
                4,
                now_iso,
            )
        ]
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.graph_metrics_functions_ext (
            repo, commit, function_goid_h128, call_betweenness, call_closeness, call_eigenvector,
            call_harmonic, call_core_number, call_clustering_coeff, call_triangle_count,
            call_is_articulation, call_is_bridge_endpoint, call_component_id, call_component_size,
            call_scc_id, call_scc_size, created_at
        ) VALUES (?, ?, ?, 0.1, 0.2, 0.3, 0.4, 1, 0.5, 1, FALSE, FALSE, 1, 1, 1, 1, ?)
        """,
        [repo, commit, 1, now_iso],
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.test_graph_metrics_functions (
            repo, commit, function_goid_h128, tests_degree, tests_weighted_degree,
            tests_degree_centrality, proj_degree, proj_weight, proj_clustering,
            proj_betweenness, created_at
        ) VALUES (?, ?, ?, 1, 1, 0.1, 1, 1, 0.1, 0.1, ?)
        """,
        [repo, commit, 1, now_iso],
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.cfg_function_metrics (
            repo, commit, function_goid_h128, rel_path, module, qualname, cfg_block_count,
            cfg_edge_count, cfg_has_cycles, cfg_scc_count, cfg_longest_path_len,
            cfg_avg_shortest_path_len, cfg_branching_factor_mean, cfg_branching_factor_max,
            cfg_linear_block_fraction, cfg_dom_tree_height, cfg_dominance_frontier_size_mean,
            cfg_dominance_frontier_size_max, cfg_loop_count, cfg_loop_nesting_depth_max,
            cfg_bc_betweenness_max, cfg_bc_betweenness_mean, cfg_bc_closeness_mean,
            cfg_bc_eigenvector_max, created_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, 1, 1, FALSE, 1, 1, 1.0, 1.0, 1.0, 0.1, 1, 0.1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, ?
        )
        """,
        [repo, commit, 1, "pkg/mod.py", "pkg.mod", "pkg.mod.func", now_iso],
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.dfg_function_metrics (
            repo, commit, function_goid_h128, rel_path, module, qualname, dfg_block_count,
            dfg_edge_count, dfg_phi_edge_count, dfg_symbol_count, dfg_component_count,
            dfg_scc_count, dfg_has_cycles, dfg_longest_chain_len, dfg_avg_shortest_path_len,
            dfg_avg_in_degree, dfg_avg_out_degree, dfg_max_in_degree, dfg_max_out_degree,
            dfg_branchy_block_fraction, dfg_bc_betweenness_max, dfg_bc_betweenness_mean,
            dfg_bc_eigenvector_max, created_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, 1, 1, 0, 1, 1, 1, FALSE, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, ?
        )
        """,
        [repo, commit, 1, "pkg/mod.py", "pkg.mod", "pkg.mod.func", now_iso],
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.graph_metrics_modules_ext (
            repo, commit, module, import_betweenness, import_closeness, import_eigenvector,
            import_harmonic, import_k_core, import_constraint, import_effective_size,
            import_community_id, import_component_id, import_component_size, import_scc_id,
            import_scc_size, created_at
        ) VALUES (?, ?, ?, 0.1, 0.1, 0.1, 0.1, 1, 0.1, 0.1, 1, 1, 1, 1, 1, ?)
        """,
        [repo, commit, "pkg.mod", now_iso],
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.symbol_graph_metrics_modules (
            repo, commit, module, symbol_betweenness, symbol_closeness, symbol_eigenvector,
            symbol_harmonic, symbol_k_core, symbol_constraint, symbol_effective_size,
            symbol_community_id, symbol_component_id, symbol_component_size, created_at
        ) VALUES (?, ?, ?, 0.1, 0.1, 0.1, 0.1, 1, 0.1, 0.1, 1, 1, 1, ?)
        """,
        [repo, commit, "pkg.mod", now_iso],
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.config_graph_metrics_modules (
            repo, commit, module, community_id, degree, weighted_degree, betweenness, closeness, created_at
        ) VALUES (?, ?, ?, 1, 1, 1, 0.1, 0.1, ?)
        """,
        [repo, commit, "pkg.mod", now_iso],
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.subsystem_graph_metrics (
            repo, commit, subsystem_id, import_in_degree, import_out_degree, import_pagerank,
            import_betweenness, import_closeness, import_layer, created_at
        ) VALUES (?, ?, ?, 1, 1, 0.1, 0.1, 0.1, 0, ?)
        """,
        [repo, commit, "subsysdemo", now_iso],
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.subsystem_agreement (
            repo, commit, module, import_community_id, agrees, created_at
        ) VALUES (?, ?, ?, 1, TRUE, ?)
        """,
        [repo, commit, "pkg.mod", now_iso],
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.function_profile (
            function_goid_h128, repo, commit, urn, rel_path, module, language, kind, qualname,
            loc, logical_loc, cyclomatic_complexity, param_count, total_params, annotated_params,
            return_type, typedness_bucket, file_typed_ratio, coverage_ratio, tested, tests_touching,
            failing_tests, slow_tests, risk_score, risk_level, tags, owners, created_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, 'python', 'function', ?, 2, 2, 1, 0, 0, 0,
            'int', 'typed', 1.0, 1.0, TRUE, 1, 0, 0, 0.1, 'low', '[]', '[]', ?
        )
        """,
        [
            1,
            repo,
            commit,
            "goid:demo/repo#python:function:pkg.mod.func",
            "pkg/mod.py",
            "pkg.mod",
            "pkg.mod.func",
            now_iso,
        ],
    )
    gateway.graph.insert_import_graph_edges(
        [
            (repo, commit, "pkg.alpha", "pkg.beta", 1, 1, 0),
            (repo, commit, "pkg.beta", "pkg.alpha", 1, 1, 0),
        ]
    )
    gateway.analytics.insert_subsystem_modules(
        [
            (repo, commit, "sub1", "pkg.alpha", "core"),
            (repo, commit, "sub2", "pkg.beta", "core"),
        ]
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.module_profile (
            repo, commit, module, avg_risk_score, max_risk_score, module_coverage_ratio,
            tested_function_count, untested_function_count, import_fan_in, import_fan_out,
            in_cycle, cycle_group, created_at
        ) VALUES (?, ?, ?, 0.1, 0.2, 1.0, 1, 0, 1, 1, FALSE, 0, ?)
        """,
        [repo, commit, "pkg.mod", now_iso],
    )
    gateway.analytics.insert_subsystems(
        [
            (
                repo,
                commit,
                "subsysdemo",
                "api_pkg",
                "Subsystem api_pkg covering 1 modules",
                1,
                '["pkg.mod"]',
                "[]",
                1,
                0,
                0,
                0,
                1,
                0.1,
                0.1,
                0,
                "low",
                now_iso,
            )
        ]
    )
    gateway.analytics.insert_subsystem_modules([(repo, commit, "subsysdemo", "pkg.mod", "api")])
    gateway.analytics.insert_test_catalog(
        [
            (
                "pkg/mod.py::test_func",
                10,
                "goid:demo/repo#python:function:pkg.mod.test_func",
                repo,
                commit,
                "pkg/mod.py",
                "pkg.mod.test_func",
                "test",
                "passed",
                1,
                "[]",
                False,
                False,
                now_iso,
            )
        ]
    )
    gateway.analytics.insert_test_coverage_edges(
        [
            (
                "pkg/mod.py::test_func",
                10,
                1,
                "goid:demo/repo#python:function:pkg.mod.func",
                repo,
                commit,
                "pkg/mod.py",
                "pkg.mod.func",
                2,
                2,
                1.0,
                "passed",
                now_iso,
            )
        ]
    )
    gateway.analytics.insert_typedness(
        [(repo, commit, "pkg/mod.py", 0, '{"params":1.0}', 0, False)]
    )
    gateway.analytics.insert_static_diagnostics([(repo, commit, "pkg/mod.py", 0, 0, 0, 0, False)])
    gateway.con.execute(
        """
        INSERT INTO analytics.hotspots VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("pkg/mod.py", 1, 1, 1, 1, 1.0, 0.1),
    )
    gateway.con.execute(
        """
        INSERT INTO core.ast_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("pkg/mod.py", 1, 1, 0, 1.0, 1, 0.1, now),
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.function_validation (
            repo, commit, function_goid_h128, rel_path, qualname, issue, detail, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            repo,
            commit,
            1,
            "pkg/mod.py",
            "pkg.mod.func",
            "span_not_found",
            "Span 1-2",
            now,
        ),
    )
    return gateway
