"""Helpers to seed a minimal architecture dataset for gateway-backed tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.analytics.utilities.datasets import (
    get_analytics_dataset_contract,
    insert_analytics_rows,
)
from codeintel.analytics.utilities.persistence import DeleteScope
from codeintel.core.schemas.generated_types import (
    GraphMetricsFunctionsExtRow,
    GraphMetricsFunctionsRow,
    GraphMetricsModulesExtRow,
    GraphMetricsModulesRow,
)
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.schema import apply_all_schemas
from tests._helpers.configs import CoverageSeedConfig
from tests._helpers.gateway import GatewayFactory
from tests._helpers.orchestration import seed_coverage_rows

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def _clear_architecture_seed(*, gateway: StorageGateway, repo: str, commit: str) -> None:
    """Remove previously seeded rows for a repo/commit to allow idempotent seeding."""
    con = gateway.con
    delete_statements = [
        ("DELETE FROM core.repo_map WHERE repo = ? AND commit = ?", [repo, commit]),
        (
            "DELETE FROM core.modules WHERE repo = ? AND commit = ? AND module IN (?, ?, ?)",
            [repo, commit, "pkg.mod", "pkg.alpha", "pkg.beta"],
        ),
        ("DELETE FROM core.goids WHERE repo = ? AND commit = ?", [repo, commit]),
        ("DELETE FROM graph.call_graph_edges WHERE repo = ? AND commit = ?", [repo, commit]),
        ("DELETE FROM analytics.function_metrics WHERE repo = ? AND commit = ?", [repo, commit]),
        ("DELETE FROM analytics.goid_risk_factors WHERE repo = ? AND commit = ?", [repo, commit]),
        (
            "DELETE FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
            [repo, commit],
        ),
        (
            "DELETE FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?",
            [repo, commit],
        ),
        (
            "DELETE FROM analytics.graph_metrics_functions_ext WHERE repo = ? AND commit = ?",
            [repo, commit],
        ),
        (
            "DELETE FROM analytics.graph_metrics_modules_ext WHERE repo = ? AND commit = ?",
            [repo, commit],
        ),
        (
            "DELETE FROM analytics.test_graph_metrics_functions WHERE repo = ? AND commit = ?",
            [repo, commit],
        ),
        (
            "DELETE FROM analytics.cfg_function_metrics WHERE repo = ? AND commit = ?",
            [repo, commit],
        ),
        (
            "DELETE FROM analytics.dfg_function_metrics WHERE repo = ? AND commit = ?",
            [repo, commit],
        ),
        (
            "DELETE FROM analytics.symbol_graph_metrics_modules WHERE repo = ? AND commit = ?",
            [repo, commit],
        ),
        (
            "DELETE FROM analytics.config_graph_metrics_modules WHERE repo = ? AND commit = ?",
            [repo, commit],
        ),
        (
            "DELETE FROM analytics.subsystem_graph_metrics WHERE repo = ? AND commit = ?",
            [repo, commit],
        ),
        ("DELETE FROM analytics.subsystems WHERE repo = ? AND commit = ?", [repo, commit]),
        ("DELETE FROM analytics.subsystem_modules WHERE repo = ? AND commit = ?", [repo, commit]),
        ("DELETE FROM analytics.subsystem_agreement WHERE repo = ? AND commit = ?", [repo, commit]),
        ("DELETE FROM analytics.function_profile WHERE repo = ? AND commit = ?", [repo, commit]),
        ("DELETE FROM analytics.module_profile WHERE repo = ? AND commit = ?", [repo, commit]),
        ("DELETE FROM analytics.test_catalog WHERE repo = ? AND commit = ?", [repo, commit]),
        ("DELETE FROM analytics.test_coverage_edges WHERE repo = ? AND commit = ?", [repo, commit]),
        ("DELETE FROM analytics.typedness WHERE repo = ? AND commit = ?", [repo, commit]),
        ("DELETE FROM analytics.static_diagnostics WHERE repo = ? AND commit = ?", [repo, commit]),
        ("DELETE FROM analytics.hotspots WHERE rel_path LIKE ?", ["%"]),
        ("DELETE FROM core.ast_metrics WHERE rel_path LIKE ?", ["%"]),
        ("DELETE FROM analytics.function_validation WHERE repo = ? AND commit = ?", [repo, commit]),
    ]
    for statement, statement_params in delete_statements:
        con.execute(statement, statement_params)


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
        factory = GatewayFactory()
        factory = factory.with_views() if strict_schema else factory.without_validation()
        gateway = factory.open()
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
    apply_all_schemas(gateway.con)
    _clear_architecture_seed(gateway=gateway, repo=repo, commit=commit)
    now = datetime.now(UTC)
    now_iso = now.isoformat()
    seed = CoverageSeedConfig(test_goid=10)
    rel_path = Path(seed.module_import.replace(".", "/")).with_suffix(".py").as_posix()

    gateway.core.insert_repo_map([(repo, commit, "{}", "{}", now_iso)])
    seed_coverage_rows(
        gateway=gateway,
        rel_path=rel_path,
        seed=seed,
        include_test_catalog=False,
    )
    gateway.core.insert_modules(
        [
            ("pkg.alpha", "pkg/alpha.py", repo, commit),
            ("pkg.beta", "pkg/beta.py", repo, commit),
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
    function_contract = get_analytics_dataset_contract(gateway, "analytics.graph_metrics_functions")
    module_contract = get_analytics_dataset_contract(gateway, "analytics.graph_metrics_modules")
    function_ext_contract = get_analytics_dataset_contract(
        gateway, "analytics.graph_metrics_functions_ext"
    )
    module_ext_contract = get_analytics_dataset_contract(
        gateway, "analytics.graph_metrics_modules_ext"
    )
    insert_analytics_rows(
        gateway,
        function_contract,
        [
            GraphMetricsFunctionsRow(
                repo=repo,
                commit=commit,
                function_goid_h128=1,
                call_fan_in=2,
                call_fan_out=3,
                call_in_degree=2,
                call_out_degree=3,
                call_pagerank=0.5,
                call_betweenness=0.1,
                call_closeness=0.2,
                call_cycle_member=False,
                call_cycle_id=0,
                call_layer=1,
                created_at=now,
            )
        ],
        delete_scope=DeleteScope(repo=repo, commit=commit),
        scope=f"{repo}@{commit}",
    )
    insert_analytics_rows(
        gateway,
        module_contract,
        [
            GraphMetricsModulesRow(
                repo=repo,
                commit=commit,
                module="pkg.mod",
                import_fan_in=3,
                import_fan_out=2,
                import_in_degree=3,
                import_out_degree=2,
                import_pagerank=0.4,
                import_betweenness=0.2,
                import_closeness=0.3,
                import_cycle_member=False,
                import_cycle_id=0,
                import_layer=1,
                symbol_fan_in=5,
                symbol_fan_out=4,
                created_at=now,
            )
        ],
        delete_scope=DeleteScope(repo=repo, commit=commit),
        scope=f"{repo}@{commit}",
    )
    insert_analytics_rows(
        gateway,
        function_ext_contract,
        [
            GraphMetricsFunctionsExtRow(
                repo=repo,
                commit=commit,
                function_goid_h128=1,
                call_betweenness=0.1,
                call_closeness=0.2,
                call_eigenvector=0.3,
                call_harmonic=0.4,
                call_core_number=1,
                call_clustering_coeff=0.5,
                call_triangle_count=1,
                call_is_articulation=False,
                call_articulation_impact=None,
                call_is_bridge_endpoint=False,
                call_component_id=1,
                call_component_size=1,
                call_scc_id=1,
                call_scc_size=1,
                call_ancestor_count=None,
                call_descendant_count=None,
                call_community_id=None,
                created_at=now,
            )
        ],
        delete_scope=DeleteScope(repo=repo, commit=commit),
        scope=f"{repo}@{commit}",
    )
    insert_analytics_rows(
        gateway,
        module_ext_contract,
        [
            GraphMetricsModulesExtRow(
                repo=repo,
                commit=commit,
                module="pkg.mod",
                import_betweenness=0.1,
                import_closeness=0.1,
                import_eigenvector=0.1,
                import_harmonic=0.1,
                import_k_core=1,
                import_constraint=0.1,
                import_effective_size=0.1,
                import_rich_club=None,
                import_shell_index=None,
                import_community_id=1,
                import_component_id=1,
                import_component_size=1,
                import_scc_id=1,
                import_scc_size=1,
                created_at=now,
            )
        ],
        delete_scope=DeleteScope(repo=repo, commit=commit),
        scope=f"{repo}@{commit}",
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
    # Module graph metrics extensions are seeded via insert_analytics_rows above; no manual inserts.
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
