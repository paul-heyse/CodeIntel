"""Self-tests for builders, provisioning contexts, and assertions."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests._helpers.assertions import assert_columns_not_null, assert_table_has_rows
from tests._helpers.builders import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    CoverageFunctionRow,
    GoidRow,
    insert_call_graph_edges,
    insert_call_graph_nodes,
    insert_coverage_functions,
    insert_goids,
)
from tests._helpers.fixtures import (
    GatewayOptions,
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
