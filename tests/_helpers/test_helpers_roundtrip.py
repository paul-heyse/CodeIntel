"""Self-tests for builders, provisioning contexts, and assertions."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import assert_columns_not_null, assert_table_has_rows
from tests._helpers.builders import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    ConfigValueRow,
    CoverageFunctionRow,
    GoidRow,
    GraphMetricsModulesExtRow,
    SymbolGraphMetricsModulesRow,
    insert_call_graph_edges,
    insert_call_graph_nodes,
    insert_config_values,
    insert_coverage_functions,
    insert_goids,
    insert_graph_metrics_modules_ext,
    insert_symbol_graph_metrics_modules,
)
from tests._helpers.fixtures import (
    GatewayOptions,
    ProvisionedGateway,
    ProvisioningConfig,
    provision_docs_export_ready,
    provisioned_gateway,
)


def test_provisioned_gateway_round_trip(tmp_path: Path) -> None:
    """Context-managed gateways should ingest and close cleanly."""
    repo_root = tmp_path / "repo"
    with provisioned_gateway(repo_root) as ctx:
        if ctx.gateway is None:
            pytest.fail("Gateway was not provisioned")
        assert_table_has_rows(ctx.gateway, "core.repo_map")


def test_file_backed_gateway_creates_db(tmp_path: Path) -> None:
    """File-backed gateways should materialize a DuckDB file on disk."""
    repo_root = tmp_path / "repo"
    config = ProvisioningConfig(
        gateway_options=GatewayOptions(file_backed=True),
        run_ingestion=False,
    )
    with provisioned_gateway(repo_root, config=config) as ctx:
        if not ctx.db_path.is_file():
            pytest.fail("Expected DuckDB file to exist on disk")


def test_builder_inserts_round_trip(tmp_path: Path) -> None:
    """Builder insert helpers should persist rows for downstream assertions."""
    repo_root = tmp_path / "repo"
    with provisioned_gateway(repo_root, config=ProvisioningConfig(run_ingestion=False)) as ctx:
        gateway = ctx.gateway
        insert_goids(
            gateway,
            [
                GoidRow(
                    goid_h128=1,
                    urn="urn:demo",
                    repo="r",
                    commit="c",
                    rel_path="a.py",
                    kind="function",
                    qualname="a.fn",
                    start_line=1,
                    end_line=2,
                )
            ],
        )
        insert_call_graph_nodes(
            gateway,
            [
                CallGraphNodeRow(
                    1,
                    "python",
                    "function",
                    0,
                    is_public=True,
                    rel_path="a.py",
                )
            ],
        )
        insert_call_graph_edges(
            gateway,
            [
                CallGraphEdgeRow(
                    repo="r",
                    commit="c",
                    caller_goid_h128=1,
                    callee_goid_h128=None,
                    callsite_path="a.py",
                    callsite_line=1,
                    callsite_col=0,
                    language="python",
                    kind="unresolved",
                    resolved_via="unresolved",
                    confidence=0.0,
                )
            ],
        )
        assert_table_has_rows(gateway, "core.goids")
        assert_table_has_rows(gateway, "graph.call_graph_edges")


def test_docs_export_ready_has_non_nulls(tmp_path: Path) -> None:
    """Docs export provisioning should populate required columns without NULLs."""
    ctx = provision_docs_export_ready(tmp_path, file_backed=False)
    try:
        assert_table_has_rows(ctx.gateway, "analytics.coverage_functions")
        assert_columns_not_null(
            ctx.gateway,
            "analytics.coverage_functions",
            ["function_goid_h128", "urn", "repo", "commit", "coverage_ratio"],
        )
        insert_coverage_functions(
            ctx.gateway,
            [
                CoverageFunctionRow(
                    function_goid_h128=2,
                    urn="urn:demo",
                    repo=ctx.repo,
                    commit=ctx.commit,
                    rel_path="foo.py",
                    language="python",
                    kind="function",
                    qualname="pkg.foo:extra",
                    start_line=1,
                    end_line=2,
                    executable_lines=2,
                    covered_lines=2,
                    coverage_ratio=1.0,
                    tested=True,
                    untested_reason=None,
                    created_at=datetime.now(tz=UTC),
                )
            ],
        )
        assert_table_has_rows(ctx.gateway, "analytics.coverage_functions", min_rows=2)
    finally:
        ctx.close()


def test_symbol_and_config_metrics_builder_round_trip(tmp_path: Path) -> None:
    """Symbol/config metrics builders should persist non-null values."""
    with provisioned_gateway(
        tmp_path / "repo", config=ProvisioningConfig(run_ingestion=False)
    ) as ctx:
        gateway = ctx.gateway
        now = datetime.now(tz=UTC)
        insert_symbol_graph_metrics_modules(
            gateway,
            [
                SymbolGraphMetricsModulesRow(
                    repo=ctx.repo,
                    commit=ctx.commit,
                    module="pkg.a",
                    symbol_betweenness=0.0,
                    symbol_closeness=0.0,
                    symbol_eigenvector=0.0,
                    symbol_harmonic=0.0,
                    symbol_k_core=1,
                    symbol_constraint=0.0,
                    symbol_effective_size=0.0,
                    symbol_community_id=1,
                    symbol_component_id=1,
                    symbol_component_size=1,
                    created_at=now,
                )
            ],
        )
        insert_graph_metrics_modules_ext(
            gateway,
            [
                GraphMetricsModulesExtRow(
                    repo=ctx.repo,
                    commit=ctx.commit,
                    module="pkg.a",
                    import_betweenness=0.0,
                    import_closeness=0.0,
                    import_eigenvector=0.0,
                    import_harmonic=0.0,
                    import_k_core=1,
                    import_constraint=0.0,
                    import_effective_size=0.0,
                    import_community_id=1,
                    import_component_id=1,
                    import_component_size=1,
                    import_scc_id=1,
                    import_scc_size=1,
                    created_at=now,
                )
            ],
        )
        insert_config_values(
            gateway,
            [
                ConfigValueRow(
                    repo=ctx.repo,
                    commit=ctx.commit,
                    config_path="cfg/app.yaml",
                    format="yaml",
                    key="feature.flag",
                    reference_paths=[],
                    reference_modules=["pkg.a"],
                    reference_count=1,
                )
            ],
        )
        assert_table_has_rows(gateway, "analytics.symbol_graph_metrics_modules")
        assert_table_has_rows(gateway, "analytics.graph_metrics_modules_ext")
        assert_table_has_rows(gateway, "analytics.config_values")


def test_file_backed_gateway_releases_handles(tmp_path: Path) -> None:
    """File-backed gateways should be deletable after context exit (cleanup)."""
    db_path = tmp_path / "repo" / "build" / "db" / "codeintel.duckdb"
    config = ProvisioningConfig(
        gateway_options=GatewayOptions(file_backed=True),
        run_ingestion=False,
    )
    with provisioned_gateway(tmp_path / "repo", config=config):
        if not db_path.exists():
            pytest.fail("Expected DuckDB file during context")
    db_path.unlink(missing_ok=False)


def test_strict_schema_flag_enforces_views(tmp_path: Path) -> None:
    """strict_schema should force schema+views even when ensure_views is False."""
    config = ProvisioningConfig(
        gateway_options=GatewayOptions(
            apply_schema=True,
            ensure_views=False,
            validate_schema=False,
            strict_schema=True,
            file_backed=False,
        ),
        run_ingestion=False,
    )
    with provisioned_gateway(tmp_path / "repo", config=config) as ctx:
        # docs view should be queryable because strict_schema turns on views
        ctx.gateway.con.execute("SELECT * FROM docs.v_symbol_module_graph LIMIT 0")


def test_loose_gateway_allows_schema_drift(loose_gateway: ProvisionedGateway) -> None:
    """loose_gateway exists solely for schema-drift scenarios."""
    con = loose_gateway.gateway.con
    con.execute("DROP VIEW IF EXISTS docs.v_symbol_module_graph")
    con.execute("CREATE VIEW docs.v_symbol_module_graph AS SELECT 1 AS ok")
    row = con.execute("SELECT ok FROM docs.v_symbol_module_graph").fetchone()
    if row != (1,):
        pytest.fail("Loose gateway drift did not persist custom view")


def test_strict_fixtures_expose_views(
    fresh_gateway: StorageGateway,
    docs_export_gateway: ProvisionedGateway,
    graph_ready_gateway: ProvisionedGateway,
) -> None:
    """Strict fixtures should have canonical docs views available."""
    fresh_gateway.con.execute("SELECT * FROM docs.v_symbol_module_graph LIMIT 0")
    docs_export_gateway.gateway.con.execute("SELECT * FROM docs.v_symbol_module_graph LIMIT 0")
    graph_ready_gateway.gateway.con.execute("SELECT * FROM docs.v_symbol_module_graph LIMIT 0")
