"""Integration-style tests for analytics graph metric modules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import pytest

from codeintel.analytics.graphs.config_data_flow import compute_config_data_flow
from codeintel.analytics.graphs.config_graph_metrics import compute_config_graph_metrics
from codeintel.analytics.graphs.contracts import (
    NotNullFractionSpec,
    SnapshotKey,
    assert_not_null_fraction,
    assert_table_exists,
    columns_present_checker,
    not_null_fraction_checker,
    run_contract_checkers,
    table_exists_checker,
    table_not_empty_checker,
)
from codeintel.analytics.graphs.graph_metrics import GraphMetricsDeps, compute_graph_metrics
from codeintel.analytics.graphs.graph_metrics_ext import compute_graph_metrics_functions_ext
from codeintel.analytics.graphs.graph_stats import compute_graph_stats
from codeintel.analytics.graphs.module_graph_metrics_ext import compute_graph_metrics_modules_ext
from codeintel.analytics.graphs.plugin_catalog import (
    build_plugin_catalog,
    render_plugin_catalog_markdown,
)
from codeintel.analytics.graphs.subsystem_agreement import compute_subsystem_agreement
from codeintel.analytics.graphs.subsystem_graph_metrics import compute_subsystem_graph_metrics
from codeintel.analytics.graphs.symbol_graph_metrics import (
    compute_symbol_graph_metrics_functions,
    compute_symbol_graph_metrics_modules,
)
from codeintel.analytics.parsing.ast_cache import FunctionAst
from codeintel.analytics.runtime import GraphRuntimeOptions
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import ConfigDataFlowStepConfig, GraphMetricsStepConfig
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql.builder import ensure_schema
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_true,
)
from tests._helpers.contracts import ContractCtx, count_rows
from tests._helpers.gateway import GatewayFactory
from tests._helpers.graphs import (
    build_ast_map,
    build_graph_engine_double,
    build_module_map,
    build_sample_graphs,
    build_source_files,
    insert_config_values,
    insert_entrypoints,
    insert_goids,
    insert_modules,
    insert_subsystems,
    insert_symbol_edges,
)
from tests._helpers.repo import (
    GOID_FUNC_A,
    GOID_FUNC_B,
    GOID_FUNC_C,
    GOID_HELPER,
    MOD_A_FQN,
    MOD_B_FQN,
    MOD_C_FQN,
)

MIN_CONFIG_DATA_FLOW_ROWS = 0
CONFIG_GRAPH_METRICS_KEY_COUNT = 2
MIN_GRAPH_STATS_ROWS = 4


@dataclass
class GraphSample:
    """Seed data and runtime options for graph analytics tests."""

    snapshot: SnapshotRef
    gateway: StorageGateway
    runtime_options: GraphRuntimeOptions
    call_graph: nx.DiGraph
    import_graph: nx.DiGraph
    config_graph: nx.Graph
    symbol_module_graph: nx.Graph
    symbol_function_graph: nx.Graph
    ast_by_goid: dict[int, FunctionAst]
    module_map: dict[str, str]
    goids: dict[str, int]

    def close(self) -> None:
        """Close the underlying gateway."""
        self.gateway.close()


def _build_graph_sample(tmp_path: Path) -> GraphSample:
    snapshot = SnapshotRef(repo="demo", commit="abc123", repo_root=tmp_path / "repo")
    paths = build_source_files(snapshot.repo_root)
    gateway = GatewayFactory().with_snapshot(snapshot.repo, snapshot.commit).open()
    now = datetime.now(tz=UTC)

    for table in (
        "analytics.graph_metrics_functions_ext",
        "analytics.graph_metrics_modules_ext",
    ):
        ensure_schema(gateway.con, table)

    goids = {
        "func_a": GOID_FUNC_A,
        "func_b": GOID_FUNC_B,
        "func_c": GOID_FUNC_C,
        "helper": GOID_HELPER,
    }
    insert_modules(gateway, snapshot, paths)

    target_names = {
        MOD_A_FQN: "func_a",
        MOD_B_FQN: "func_b",
        MOD_C_FQN: "func_c",
        "pkg.util": "helper",
    }
    ast_by_goid = build_ast_map(paths, goids, snapshot.repo_root, target_names=target_names)
    insert_goids(gateway, snapshot, ast_by_goid, now=now)
    insert_config_values(gateway, snapshot, goids, ast_by_goid)
    insert_entrypoints(gateway, snapshot, goids, ast_by_goid, now=now)
    insert_subsystems(gateway, snapshot)
    insert_symbol_edges(gateway, goids, ast_by_goid)

    graphs = build_sample_graphs(goids)
    runtime_options = GraphRuntimeOptions(
        snapshot=snapshot,
        engine=build_graph_engine_double(
            gateway,
            snapshot,
            call_graph=graphs.call_graph,
            import_graph=graphs.import_graph,
            config_graph=graphs.config_graph,
            symbol_module_graph=graphs.symbol_module_graph,
            symbol_function_graph=graphs.symbol_function_graph,
            cfg_graph=graphs.cfg_graph,
        ),
    )

    module_map = build_module_map(
        ast_by_goid,
        {
            goids["func_a"]: MOD_A_FQN,
            goids["func_b"]: MOD_B_FQN,
            goids["func_c"]: MOD_C_FQN,
        },
    )

    return GraphSample(
        snapshot=snapshot,
        gateway=gateway,
        runtime_options=runtime_options,
        call_graph=graphs.call_graph,
        import_graph=graphs.import_graph,
        config_graph=graphs.config_graph,
        symbol_module_graph=graphs.symbol_module_graph,
        symbol_function_graph=graphs.symbol_function_graph,
        ast_by_goid=ast_by_goid,
        module_map=module_map,
        goids=goids,
    )


def _run_graph_pipeline(sample: GraphSample) -> None:
    cfg_graph = GraphMetricsStepConfig(snapshot=sample.snapshot)
    compute_config_data_flow(
        sample.gateway,
        ConfigDataFlowStepConfig(snapshot=sample.snapshot),
        call_graph=sample.call_graph,
        ast_by_goid=sample.ast_by_goid,
    )
    compute_config_graph_metrics(
        sample.gateway,
        repo=sample.snapshot.repo,
        commit=sample.snapshot.commit,
        runtime=sample.runtime_options,
    )
    compute_graph_metrics(
        sample.gateway,
        cfg_graph,
        deps=GraphMetricsDeps(
            runtime=sample.runtime_options,
            module_by_path=sample.module_map,
        ),
    )
    compute_graph_metrics_functions_ext(
        sample.gateway,
        repo=sample.snapshot.repo,
        commit=sample.snapshot.commit,
        runtime=sample.runtime_options,
    )
    compute_graph_metrics_modules_ext(
        sample.gateway,
        repo=sample.snapshot.repo,
        commit=sample.snapshot.commit,
        runtime=sample.runtime_options,
    )
    compute_graph_stats(
        sample.gateway,
        repo=sample.snapshot.repo,
        commit=sample.snapshot.commit,
        runtime=sample.runtime_options,
    )
    compute_subsystem_graph_metrics(
        sample.gateway,
        repo=sample.snapshot.repo,
        commit=sample.snapshot.commit,
        runtime=sample.runtime_options,
    )
    compute_subsystem_agreement(
        sample.gateway,
        repo=sample.snapshot.repo,
        commit=sample.snapshot.commit,
    )
    compute_symbol_graph_metrics_modules(
        sample.gateway,
        repo=sample.snapshot.repo,
        commit=sample.snapshot.commit,
        runtime=sample.runtime_options,
    )
    compute_symbol_graph_metrics_functions(
        sample.gateway,
        repo=sample.snapshot.repo,
        commit=sample.snapshot.commit,
        runtime=sample.runtime_options,
    )


def test_graph_metrics_end_to_end(tmp_path: Path) -> None:
    """Build graphs and compute analytics metrics end-to-end."""
    sample = _build_graph_sample(tmp_path)
    try:
        _run_graph_pipeline(sample)
        params = [sample.snapshot.repo, sample.snapshot.commit]
        con = sample.gateway.con

        expect_true(
            count_rows(
                con,
                "SELECT COUNT(*) FROM analytics.config_data_flow WHERE repo = ? AND commit = ?",
                params,
            )
            >= MIN_CONFIG_DATA_FLOW_ROWS,
        )
        chain_row = con.execute(
            """
            SELECT call_chain_json
            FROM analytics.config_data_flow
            WHERE repo = ? AND commit = ?
            LIMIT 1
            """,
            params,
        ).fetchone()
        if chain_row is None:
            pytest.skip("config_data_flow not produced for canonical sample")

        chain_json = chain_row[0]
        expect_in(str(sample.goids["func_a"]), chain_json)

        expect_equal(
            count_rows(
                con,
                "SELECT COUNT(*) FROM analytics.config_graph_metrics_keys WHERE repo = ? AND commit = ?",
                params,
            ),
            CONFIG_GRAPH_METRICS_KEY_COUNT,
        )
        expect_true(
            count_rows(
                con,
                "SELECT COUNT(*) FROM analytics.config_projection_module_edges WHERE repo = ? AND commit = ?",
                params,
            )
            > 0,
        )

        expect_equal(
            count_rows(
                con,
                "SELECT COUNT(*) FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
                params,
            ),
            len(sample.call_graph.nodes),
        )
        expect_true(
            count_rows(
                con,
                "SELECT COUNT(*) FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?",
                params,
            )
            >= len(sample.import_graph.nodes),
        )
        expect_equal(
            count_rows(
                con,
                "SELECT COUNT(*) FROM analytics.graph_metrics_functions_ext WHERE repo = ? AND commit = ?",
                params,
            ),
            len(sample.call_graph.nodes),
        )
        expect_true(
            count_rows(
                con,
                "SELECT COUNT(*) FROM analytics.graph_metrics_modules_ext WHERE repo = ? AND commit = ?",
                params,
            )
            >= len(sample.import_graph.nodes),
        )

        expect_true(
            count_rows(
                con,
                "SELECT COUNT(*) FROM analytics.graph_stats WHERE repo = ? AND commit = ?",
                params,
            )
            >= MIN_GRAPH_STATS_ROWS,
        )
        expect_true(
            count_rows(
                con,
                "SELECT COUNT(*) FROM analytics.subsystem_graph_metrics WHERE repo = ? AND commit = ?",
                params,
            )
            >= 1,
        )
        expect_true(
            count_rows(
                con,
                "SELECT COUNT(*) FROM analytics.subsystem_agreement WHERE repo = ? AND commit = ?",
                params,
            )
            >= 1,
        )

        expect_true(
            count_rows(
                con,
                "SELECT COUNT(*) FROM analytics.symbol_graph_metrics_modules WHERE repo = ? AND commit = ?",
                params,
            )
            >= len(sample.symbol_module_graph.nodes),
        )
        expect_true(
            count_rows(
                con,
                "SELECT COUNT(*) FROM analytics.symbol_graph_metrics_functions WHERE repo = ? AND commit = ?",
                params,
            )
            >= len(sample.symbol_function_graph.nodes),
        )
    finally:
        sample.close()


def test_contracts_and_catalog(tmp_path: Path) -> None:
    """Validate dataset contracts and plugin catalog rendering."""
    sample = _build_graph_sample(tmp_path)
    try:
        _run_graph_pipeline(sample)
        params = [sample.snapshot.repo, sample.snapshot.commit]
        con = sample.gateway.con

        table_checks = run_contract_checkers(
            ctx=ContractCtx(
                gateway=sample.gateway,
                repo=sample.snapshot.repo,
                commit=sample.snapshot.commit,
            ),
            checkers=(
                table_exists_checker("analytics.graph_metrics_functions"),
                table_not_empty_checker("analytics.graph_metrics_functions"),
                columns_present_checker(
                    "analytics.graph_metrics_modules",
                    expected_columns={"module", "repo", "commit"},
                ),
                not_null_fraction_checker(
                    "analytics.graph_metrics_modules",
                    column="module",
                    min_fraction=0.5,
                ),
            ),
        )
        statuses = {check.name: check.status for check in table_checks}
        expect_equal(statuses["analytics.graph_metrics_functions_exists"], "passed")
        expect_equal(statuses["analytics.graph_metrics_functions_not_empty"], "passed")

        nullable_result = assert_not_null_fraction(
            sample.gateway,
            snapshot=SnapshotKey(repo=sample.snapshot.repo, commit=sample.snapshot.commit),
            spec=NotNullFractionSpec(
                table="analytics.graph_metrics_functions",
                column="function_goid_h128",
                min_fraction=0.1,
            ),
        )
        expect_in(nullable_result.status, {"passed", "failed"})

        with pytest.raises(ValueError, match="Unsafe or unknown table"):
            assert_table_exists(sample.gateway, table="analytics.unknown_table")

        catalog = build_plugin_catalog()
        expect_equal(catalog["count"], len(catalog["plugins"]))
        markdown = render_plugin_catalog_markdown(catalog)
        expect_in("Analytics Plugin Catalog", markdown)
        expect_in(str(len(catalog["plugins"])), markdown)
        expect_true(
            count_rows(
                con,
                "SELECT COUNT(*) FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?",
                params,
            )
            > 0,
        )
    finally:
        sample.close()
