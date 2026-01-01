"""Helpers to seed a minimal architecture dataset for gateway-backed tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.analytics.utilities.datasets import (
    get_analytics_dataset_contract,
    insert_analytics_rows,
)
from codeintel.build.analytics.utilities.persistence import DeleteScope
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.schema import apply_all_schemas
from codeintel.storage.warehouse import MaterializeOptions, Warehouse
from tests._helpers.assertions import ModulesAssertions
from tests._helpers.columnar_tables import materialize_table_from_rows
from tests._helpers.configs import CoverageSeedConfig
from tests._helpers.fixtures.rows import RowFactory
from tests._helpers.gateway import GatewayFactory, seed_contract_catalog
from tests._helpers.modules_expectations import modules_expected_from_repo_tree
from tests._helpers.orchestration import seed_coverage_rows
from tests._helpers.orchestration.coverage_orchestration import CoverageSeedOptions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.storage.gateway import StorageGateway


def _table_columns(gateway: StorageGateway, table_key: str) -> tuple[str, ...]:
    schema_provider = gateway.policy.schema_provider
    if schema_provider is None:
        msg = "StorageGateway.policy requires schema_provider for test seeding"
        raise RuntimeError(msg)
    return tuple(col.name for col in schema_provider.require_table_schema(table_key).columns)


def _row_mapping(columns: Sequence[str], values: Sequence[object]) -> dict[str, object]:
    return dict(zip(columns, values, strict=True))


@dataclass(frozen=True)
class _ArchitectureSeedContext:
    gateway: StorageGateway
    repo: str
    commit: str
    now: datetime
    warehouse: Warehouse
    append: MaterializeOptions
    repo_root: Path | None
    rel_path: str
    module_import: str
    module_map: dict[str, str]


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
    repo_root: Path | None = None,
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
    repo_root : Path | None
        Optional repo root for modules-first inventory derivation.

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
        gateway = open_gateway(cfg, seed_contract_catalog=seed_contract_catalog)
    return seed_architecture(gateway=gateway, repo=repo, commit=commit, repo_root=repo_root)


def seed_architecture(
    *,
    gateway: StorageGateway,
    repo: str,
    commit: str,
    repo_root: Path | None = None,
) -> StorageGateway:
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
    repo_root : Path | None
        Optional repo root for modules-first inventory derivation.

    Returns
    -------
    StorageGateway
        Gateway with architecture tables populated for tests.
    """
    apply_all_schemas(gateway.con)
    _clear_architecture_seed(gateway=gateway, repo=repo, commit=commit)
    now = datetime.now(UTC)
    seed = CoverageSeedConfig(test_goid=10)
    rel_path = Path(seed.module_import.replace(".", "/")).with_suffix(".py").as_posix()
    module_map = _resolve_architecture_module_map(repo_root, rel_path, seed.module_import)
    context = _ArchitectureSeedContext(
        gateway=gateway,
        repo=repo,
        commit=commit,
        now=now,
        warehouse=Warehouse(gateway),
        append=MaterializeOptions(mode="append"),
        repo_root=repo_root,
        rel_path=rel_path,
        module_import=seed.module_import,
        module_map=module_map,
    )
    _seed_core_tables(context, seed)
    _seed_function_metrics(context)
    _seed_graph_metrics(context)
    _seed_additional_analytics(context)
    _seed_graph_tables(context)
    return gateway


def _seed_core_tables(context: _ArchitectureSeedContext, seed: CoverageSeedConfig) -> None:
    repo_map_columns = _table_columns(context.gateway, "core.repo_map")
    materialize_table_from_rows(
        context.warehouse,
        "core.repo_map",
        [
            _row_mapping(
                repo_map_columns,
                (context.repo, context.commit, context.module_map, {}, context.now),
            )
        ],
        columns=repo_map_columns,
        options=context.append,
    )
    seed_coverage_rows(
        gateway=context.gateway,
        rel_path=context.rel_path,
        seed=seed,
        options=CoverageSeedOptions(
            include_test_catalog=False,
            seed_repo_map=False,
        ),
    )
    core_module_map = dict(context.module_map)
    core_module_map.pop(context.module_import, None)
    context.warehouse.materialize_mappings(
        "core.modules",
        [
            {
                "module": module,
                "path": path,
                "repo": context.repo,
                "commit": context.commit,
                "language": "python",
                "tags": [],
                "owners": [],
            }
            for module, path in sorted(core_module_map.items())
        ],
        options=context.append,
    )
    snapshot = SnapshotRef(
        repo=context.repo,
        commit=context.commit,
        repo_root=context.repo_root or Path.cwd(),
    )
    ModulesAssertions(context.gateway, snapshot).inventory_consistent()


def _seed_function_metrics(context: _ArchitectureSeedContext) -> None:
    function_metrics_columns = _table_columns(context.gateway, "analytics.function_metrics")
    materialize_table_from_rows(
        context.warehouse,
        "analytics.function_metrics",
        [
            _row_mapping(
                function_metrics_columns,
                (
                    1,
                    "goid:demo/repo#python:function:pkg.mod.func",
                    context.repo,
                    context.commit,
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
                    context.now,
                ),
            )
        ],
        columns=function_metrics_columns,
        options=context.append,
    )
    goid_risk_columns = _table_columns(context.gateway, "analytics.goid_risk_factors")
    materialize_table_from_rows(
        context.warehouse,
        "analytics.goid_risk_factors",
        [
            _row_mapping(
                goid_risk_columns,
                (
                    1,
                    context.repo,
                    context.commit,
                    1,
                    "low",
                    1,
                    0,
                    0,
                    False,
                ),
            )
        ],
        columns=goid_risk_columns,
        options=context.append,
    )
    call_graph_columns = _table_columns(context.gateway, "graph.call_graph_edges")
    materialize_table_from_rows(
        context.warehouse,
        "graph.call_graph_edges",
        [
            _row_mapping(
                call_graph_columns,
                (
                    context.repo,
                    context.commit,
                    1,
                    1,
                    "pkg/mod.py",
                    1,
                    1,
                    "python",
                    "direct",
                    "local",
                    0.9,
                    {},
                ),
            )
        ],
        columns=call_graph_columns,
        options=context.append,
    )


def _seed_graph_metrics(context: _ArchitectureSeedContext) -> None:
    function_contract = get_analytics_dataset_contract(
        context.gateway,
        "analytics.graph_metrics_functions",
    )
    module_contract = get_analytics_dataset_contract(
        context.gateway,
        "analytics.graph_metrics_modules",
    )
    function_ext_contract = get_analytics_dataset_contract(
        context.gateway,
        "analytics.graph_metrics_functions_ext",
    )
    module_ext_contract = get_analytics_dataset_contract(
        context.gateway,
        "analytics.graph_metrics_modules_ext",
    )
    delete_scope = DeleteScope(repo=context.repo, commit=context.commit)
    scope = f"{context.repo}@{context.commit}"
    insert_analytics_rows(
        context.gateway,
        function_contract,
        [
            RowFactory.row_for(
                "analytics.graph_metrics_functions",
                repo=context.repo,
                commit=context.commit,
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
                created_at=context.now,
            )
        ],
        delete_scope=delete_scope,
        scope=scope,
    )
    insert_analytics_rows(
        context.gateway,
        module_contract,
        [
            RowFactory.row_for(
                "analytics.graph_metrics_modules",
                repo=context.repo,
                commit=context.commit,
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
                created_at=context.now,
            )
        ],
        delete_scope=delete_scope,
        scope=scope,
    )
    insert_analytics_rows(
        context.gateway,
        function_ext_contract,
        [
            RowFactory.row_for(
                "analytics.graph_metrics_functions_ext",
                repo=context.repo,
                commit=context.commit,
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
                created_at=context.now,
            )
        ],
        delete_scope=delete_scope,
        scope=scope,
    )
    insert_analytics_rows(
        context.gateway,
        module_ext_contract,
        [
            RowFactory.row_for(
                "analytics.graph_metrics_modules_ext",
                repo=context.repo,
                commit=context.commit,
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
                created_at=context.now,
            )
        ],
        delete_scope=delete_scope,
        scope=scope,
    )


def _seed_additional_analytics(context: _ArchitectureSeedContext) -> None:
    repo = context.repo
    commit = context.commit
    now = context.now
    con = context.gateway.con
    con.execute(
        """
        INSERT INTO analytics.test_graph_metrics_functions (
            repo, commit, function_goid_h128, tests_degree, tests_weighted_degree,
            tests_degree_centrality, proj_degree, proj_weight, proj_clustering,
            proj_betweenness, created_at
        ) VALUES (?, ?, ?, 1, 1, 0.1, 1, 1, 0.1, 0.1, ?)
        """,
        [repo, commit, 1, now],
    )
    con.execute(
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
            ?, ?, ?, ?, ?, ?, 1, 1, FALSE, 1, 1, 1.0, 1.0, 1.0, 0.1, 1,
            0.1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, ?
        )
        """,
        [repo, commit, 1, "pkg/mod.py", "pkg.mod", "pkg.mod.func", now],
    )
    con.execute(
        """
        INSERT INTO analytics.dfg_function_metrics (
            repo, commit, function_goid_h128, rel_path, module, qualname, dfg_block_count,
            dfg_edge_count, dfg_phi_edge_count, dfg_symbol_count, dfg_component_count,
            dfg_scc_count, dfg_has_cycles, dfg_longest_chain_len, dfg_avg_shortest_path_len,
            dfg_avg_in_degree, dfg_avg_out_degree, dfg_max_in_degree, dfg_max_out_degree,
            dfg_branchy_block_fraction, dfg_bc_betweenness_max, dfg_bc_betweenness_mean,
            dfg_bc_eigenvector_max, created_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, 1, 1, 0, 1, 1, 1, FALSE, 1, 1.0, 1.0, 1.0,
            1.0, 1.0, 0.1, 0.1, 0.1, 0.1, ?
        )
        """,
        [repo, commit, 1, "pkg/mod.py", "pkg.mod", "pkg.mod.func", now],
    )
    con.execute(
        """
        INSERT INTO analytics.subsystem_graph_metrics (
            repo, commit, subsystem_id, import_in_degree, import_out_degree, import_pagerank,
            import_betweenness, import_closeness, import_layer, created_at
        ) VALUES (?, ?, ?, 1, 1, 0.1, 0.1, 0.1, 0, ?)
        """,
        [repo, commit, "subsysdemo", now],
    )
    con.execute(
        """
        INSERT INTO analytics.subsystem_agreement (
            repo, commit, module, import_community_id, agrees, created_at
        ) VALUES (?, ?, ?, 1, TRUE, ?)
        """,
        [repo, commit, "pkg.mod", now],
    )
    context.warehouse.materialize_mappings(
        "analytics.function_profile",
        [
            RowFactory.row_for(
                "analytics.function_profile",
                function_goid_h128=1,
                repo=repo,
                commit=commit,
                urn="goid:demo/repo#python:function:pkg.mod.func",
                rel_path="pkg/mod.py",
                module="pkg.mod",
                language="python",
                kind="function",
                qualname="pkg.mod.func",
                loc=2,
                logical_loc=2,
                cyclomatic_complexity=1,
                param_count=0,
                total_params=0,
                annotated_params=0,
                return_type="int",
                typedness_bucket="typed",
                file_typed_ratio=1.0,
                coverage_ratio=1.0,
                tested=True,
                tests_touching=1,
                failing_tests=0,
                slow_tests=0,
                risk_score=0.1,
                risk_level="low",
                tags=[],
                owners=[],
                created_at=now,
            )
        ],
        options=context.append,
    )


def _seed_graph_tables(context: _ArchitectureSeedContext) -> None:
    repo = context.repo
    commit = context.commit
    now = context.now
    con = context.gateway.con
    import_graph_columns = _table_columns(context.gateway, "graph.import_graph_edges")
    materialize_table_from_rows(
        context.warehouse,
        "graph.import_graph_edges",
        [
            _row_mapping(
                import_graph_columns,
                (repo, commit, "pkg.alpha", "pkg.beta", 1, 1, 0, None),
            ),
            _row_mapping(
                import_graph_columns,
                (repo, commit, "pkg.beta", "pkg.alpha", 1, 1, 0, None),
            ),
        ],
        columns=import_graph_columns,
        options=context.append,
    )
    subsystem_modules_columns = _table_columns(context.gateway, "analytics.subsystem_modules")
    materialize_table_from_rows(
        context.warehouse,
        "analytics.subsystem_modules",
        [
            _row_mapping(subsystem_modules_columns, (repo, commit, "sub1", "pkg.alpha", "core")),
            _row_mapping(subsystem_modules_columns, (repo, commit, "sub2", "pkg.beta", "core")),
        ],
        columns=subsystem_modules_columns,
        options=context.append,
    )
    con.execute(
        """
        INSERT INTO analytics.module_profile (
            repo, commit, module, avg_risk_score, max_risk_score, module_coverage_ratio,
            tested_function_count, untested_function_count, import_fan_in, import_fan_out,
            in_cycle, cycle_group, created_at
        ) VALUES (?, ?, ?, 0.1, 0.2, 1.0, 1, 0, 1, 1, FALSE, 0, ?)
        """,
        [repo, commit, "pkg.mod", now],
    )
    subsystem_columns = _table_columns(context.gateway, "analytics.subsystems")
    materialize_table_from_rows(
        context.warehouse,
        "analytics.subsystems",
        [
            _row_mapping(
                subsystem_columns,
                (
                    repo,
                    commit,
                    "subsysdemo",
                    "api_pkg",
                    "Subsystem api_pkg covering 1 modules",
                    1,
                    ["pkg.mod"],
                    [],
                    1,
                    0,
                    0,
                    0,
                    1,
                    0.1,
                    0.1,
                    0,
                    "low",
                    now,
                ),
            )
        ],
        columns=subsystem_columns,
        options=context.append,
    )
    materialize_table_from_rows(
        context.warehouse,
        "analytics.subsystem_modules",
        [_row_mapping(subsystem_modules_columns, (repo, commit, "subsysdemo", "pkg.mod", "api"))],
        columns=subsystem_modules_columns,
        options=context.append,
    )
    test_catalog_columns = _table_columns(context.gateway, "analytics.test_catalog")
    materialize_table_from_rows(
        context.warehouse,
        "analytics.test_catalog",
        [
            _row_mapping(
                test_catalog_columns,
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
                    [],
                    False,
                    False,
                    now,
                ),
            )
        ],
        columns=test_catalog_columns,
        options=context.append,
    )
    test_coverage_columns = _table_columns(context.gateway, "analytics.test_coverage_edges")
    materialize_table_from_rows(
        context.warehouse,
        "analytics.test_coverage_edges",
        [
            _row_mapping(
                test_coverage_columns,
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
                    now,
                ),
            )
        ],
        columns=test_coverage_columns,
        options=context.append,
    )
    typedness_columns = _table_columns(context.gateway, "analytics.typedness")
    materialize_table_from_rows(
        context.warehouse,
        "analytics.typedness",
        [
            _row_mapping(
                typedness_columns,
                (repo, commit, "pkg/mod.py", 0, {"params": 1.0}, 0, False),
            )
        ],
        columns=typedness_columns,
        options=context.append,
    )
    diagnostics_columns = _table_columns(context.gateway, "analytics.static_diagnostics")
    materialize_table_from_rows(
        context.warehouse,
        "analytics.static_diagnostics",
        [_row_mapping(diagnostics_columns, (repo, commit, "pkg/mod.py", 0, 0, 0, 0, False))],
        columns=diagnostics_columns,
        options=context.append,
    )
    con.execute(
        """
        INSERT INTO analytics.hotspots VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("pkg/mod.py", 1, 1, 1, 1, 1.0, 0.1),
    )
    con.execute(
        """
        INSERT INTO core.ast_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("pkg/mod.py", 1, 1, 0, 1.0, 1, 0.1, now),
    )
    con.execute(
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


def _resolve_architecture_module_map(
    repo_root: Path | None,
    rel_path: str,
    module_import: str,
) -> dict[str, str]:
    module_map: dict[str, str] = {}
    if repo_root is not None:
        path_map = modules_expected_from_repo_tree(repo_root)
        module_map = {module: path for path, module in path_map.items()}
    if not module_map:
        module_map = {
            "pkg.alpha": "pkg/alpha.py",
            "pkg.beta": "pkg/beta.py",
        }
    module_map.setdefault(module_import, rel_path)
    return module_map
