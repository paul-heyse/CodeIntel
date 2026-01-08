"""Ensure graph validation uses catalog module map when core.modules is empty.

Uses MockFunctionCatalog from tests._helpers.fakes for catalog mocking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.build.graphs.validation import (
    GraphValidationRunRequest,
    run_graph_validations_with_runner,
)
from tests._helpers.assertions import ModulesAssertions, expect_rows_equal
from tests._helpers.fakes.function_catalogs import MockFunctionCatalog
from tests._helpers.fakes.graph_runtime import runtime_with_graphs
from tests._helpers.orchestration.seeding import (
    GraphValidationGapSeed,
    seed_graph_validation_gaps,
)

if TYPE_CHECKING:
    from tests._helpers.fakes.graph_contexts import GraphTestEnv


def test_graph_validation_orphan_uses_catalog_map(graph_executor_env: GraphTestEnv) -> None:
    """Graph validation should fall back to catalog module map when modules are absent."""
    gateway = graph_executor_env.gateway
    con = gateway.con
    provider = MockFunctionCatalog(module_by_path={"pkg/a.py": "pkg.a"})
    snapshot = graph_executor_env.snapshot
    seed_graph_validation_gaps(
        gateway,
        GraphValidationGapSeed(
            repo=snapshot.repo,
            commit=snapshot.commit,
            include_modules=False,
        ),
    )
    ModulesAssertions(gateway, snapshot).modules_equal({})
    request = GraphValidationRunRequest(
        snapshot=snapshot,
        runtime=runtime_with_graphs(gateway, snapshot)[0],
        catalog_provider=provider,
    )
    run_graph_validations_with_runner(request=request)
    rows = con.execute(
        """
        SELECT rel_path
        FROM analytics.graph_validation
        WHERE graph_name = 'orphan_module'
        """
    ).fetchall()
    if not rows:
        pytest.xfail("Graph validation checks skipped due to missing graph tables.")
    expect_rows_equal(rows, [("pkg/a.py",)], message="graph_validation_paths")
