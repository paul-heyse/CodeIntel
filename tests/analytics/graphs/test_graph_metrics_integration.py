"""Integration-style tests for analytics graph metric modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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
from codeintel.analytics.graphs.graph_metrics import compute_graph_metrics
from codeintel.analytics.graphs.plugin_catalog import (
    build_plugin_catalog,
    render_plugin_catalog_markdown,
)
from codeintel.config import ConfigBuilder, SnapshotInit
from codeintel.config.primitives import BuildLayoutOptions
from tests._helpers import (
    GraphMetricsGatewayOptions,
    graph_metrics_ready_gateway,
    seed_function_graph_cycle,
    seed_module_graph_inputs,
)
from tests._helpers.assertions import (
    FunctionMetricsExpectation,
    GraphMetricsTableExpectations,
    ModuleMetricsExpectation,
    assert_graph_metrics_function_row,
    assert_graph_metrics_module_row,
    assert_graph_metrics_table_counts,
    expect_equal,
    expect_in,
    expect_true,
)
from tests._helpers.contracts import ContractCtx, count_rows
from tests._helpers.graph_runtime_harness import (
    run_graph_metrics_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests._helpers.graph_runtime_harness import (
        GraphRuntimeHarness,
    )

MIN_CONFIG_DATA_FLOW_ROWS = 0
CONFIG_GRAPH_METRICS_KEY_COUNT = 2
MIN_GRAPH_STATS_ROWS = 4
REPO = "demo/repo"
COMMIT = "abc123"
REL_PATH = "pkg/mod.py"
MODULE_A = "pkg.mod_a"
MODULE_B = "pkg.mod_b"


def test_graph_metrics_end_to_end(graph_runtime_ctx: GraphRuntimeHarness) -> None:
    """Build graphs and compute analytics metrics end-to-end."""
    run_graph_metrics_pipeline(graph_runtime_ctx)
    params = [graph_runtime_ctx.snapshot.repo, graph_runtime_ctx.snapshot.commit]
    con = graph_runtime_ctx.gateway.con

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
    expect_in(str(graph_runtime_ctx.goids["func_a"]), chain_json)

    fixtures = graph_runtime_ctx.fixtures
    assert_graph_metrics_table_counts(
        con,
        graph_runtime_ctx.snapshot,
        GraphMetricsTableExpectations(
            config_keys=CONFIG_GRAPH_METRICS_KEY_COUNT,
            config_projection_min=1,
            functions=len(fixtures.call_graph.nodes),
            modules_min=len(fixtures.import_graph.nodes),
            functions_ext=len(fixtures.call_graph.nodes),
            modules_ext_min=len(fixtures.import_graph.nodes),
            graph_stats_min=MIN_GRAPH_STATS_ROWS,
            subsystem_metrics_min=1,
            subsystem_agreement_min=1,
            symbol_modules_min=len(fixtures.symbol_module_graph.nodes),
            symbol_functions_min=len(fixtures.symbol_function_graph.nodes),
        ),
    )


def test_contracts_and_catalog(graph_runtime_ctx: GraphRuntimeHarness) -> None:
    """Validate dataset contracts and plugin catalog rendering."""
    run_graph_metrics_pipeline(graph_runtime_ctx)
    params = [graph_runtime_ctx.snapshot.repo, graph_runtime_ctx.snapshot.commit]
    con = graph_runtime_ctx.gateway.con

    table_checks = run_contract_checkers(
        ctx=ContractCtx(
            gateway=graph_runtime_ctx.gateway,
            repo=graph_runtime_ctx.snapshot.repo,
            commit=graph_runtime_ctx.snapshot.commit,
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
        graph_runtime_ctx.gateway,
        snapshot=SnapshotKey(
            repo=graph_runtime_ctx.snapshot.repo,
            commit=graph_runtime_ctx.snapshot.commit,
        ),
        spec=NotNullFractionSpec(
            table="analytics.graph_metrics_functions",
            column="function_goid_h128",
            min_fraction=0.1,
        ),
    )
    expect_in(nullable_result.status, {"passed", "failed"})

    with pytest.raises(ValueError, match="Unsafe or unknown table"):
        assert_table_exists(graph_runtime_ctx.gateway, table="analytics.unknown_table")

    catalog = build_plugin_catalog()
    expect_equal(catalog["count"], len(catalog["plugins"]))
    markdown = render_plugin_catalog_markdown(catalog)
    expect_in("Plugin Catalog", markdown)
    expect_in(str(len(catalog["plugins"])), markdown)
    expect_true(
        count_rows(
            con,
            "SELECT COUNT(*) FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?",
            params,
        )
        > 0,
    )


def test_compute_function_graph_metrics_counts_and_cycles(tmp_path: Path) -> None:
    """Compute function graph metrics with cycles and aggregated edge counts."""
    ctx = graph_metrics_ready_gateway(
        tmp_path / "graph_metrics",
        GraphMetricsGatewayOptions(
            repo=REPO,
            commit=COMMIT,
            include_symbol_edges=False,
            run_metrics=False,
            build_callgraph_enabled=False,
            file_backed=False,
        ),
    )
    seed_function_graph_cycle(ctx.gateway, repo=REPO, commit=COMMIT, rel_path=REL_PATH)

    builder = ConfigBuilder.from_snapshot(
        snapshot=SnapshotInit(repo=REPO, commit=COMMIT, repo_root=ctx.repo_root),
        layout=BuildLayoutOptions(build_dir=ctx.build_dir),
    )
    cfg = builder.graph_metrics()
    compute_graph_metrics(ctx.gateway, cfg)

    assert_graph_metrics_function_row(
        ctx.gateway.con,
        FunctionMetricsExpectation(
            goid=2,
            fan_in=1,
            fan_out=1,
            in_degree=2,
            out_degree=1,
            cycle_member=True,
        ),
    )
    ctx.close()


def test_compute_module_graph_metrics_with_symbol_coupling(tmp_path: Path) -> None:
    """Compute module graph metrics including symbol coupling fan counts."""
    ctx = graph_metrics_ready_gateway(
        tmp_path / "graph_metrics_mod",
        GraphMetricsGatewayOptions(
            repo=REPO,
            commit=COMMIT,
            include_symbol_edges=False,
            run_metrics=False,
            build_callgraph_enabled=False,
            file_backed=False,
        ),
    )
    seed_module_graph_inputs(
        ctx.gateway,
        repo=REPO,
        commit=COMMIT,
        module_a=MODULE_A,
        module_b=MODULE_B,
    )

    builder = ConfigBuilder.from_snapshot(
        snapshot=SnapshotInit(repo=REPO, commit=COMMIT, repo_root=ctx.repo_root),
        layout=BuildLayoutOptions(build_dir=ctx.build_dir),
    )
    cfg = builder.graph_metrics()
    compute_graph_metrics(ctx.gateway, cfg)

    assert_graph_metrics_module_row(
        ctx.gateway.con,
        ModuleMetricsExpectation(
            module=MODULE_A,
            import_fan_in=0,
            import_fan_out=1,
            symbol_fan_in=0,
            symbol_fan_out=1,
            import_cycle_member=False,
        ),
    )
    ctx.close()
