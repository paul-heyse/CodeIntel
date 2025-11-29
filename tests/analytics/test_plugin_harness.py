"""Tests for the shared plugin harness utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pytest

from codeintel.analytics.graphs.contracts import ContractChecker, PluginContractResult
from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
    GraphPluginResult,
)
from tests.analytics.conftest import PluginTestHarness


def _row_count_plugin(ctx: GraphMetricExecutionContext) -> GraphPluginResult:
    ctx.gateway.execute(
        """
        CREATE TABLE IF NOT EXISTS analytics.harness_demo (
            repo TEXT,
            commit TEXT,
            value INT
        )
        """
    )
    ctx.gateway.execute(
        "DELETE FROM analytics.harness_demo WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    )
    ctx.gateway.execute(
        "INSERT INTO analytics.harness_demo VALUES (?, ?, ?)",
        [ctx.repo, ctx.commit, 1],
    )
    row = ctx.gateway.execute(
        "SELECT COUNT(*) FROM analytics.harness_demo WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    ).fetchone()
    count = 0 if row is None else int(row[0])
    return GraphPluginResult(row_counts={"analytics.harness_demo": count})


def _contract_checker(_ctx: GraphMetricExecutionContext) -> PluginContractResult:
    return PluginContractResult(name="harness_contract", status="passed")


def test_plugin_harness_idempotent_skip(plugin_harness: PluginTestHarness) -> None:
    """Harness should assert skip-on-unchanged between runs."""
    harness = plugin_harness
    if not hasattr(harness, "register"):
        pytest.fail("plugin_harness fixture missing register")
    harness.register(
        GraphMetricPlugin(
            name="harness_row_count_plugin",
            description="demo",
            stage="core",
            enabled_by_default=False,
            run=_row_count_plugin,
            row_count_tables=("analytics.harness_demo",),
            contract_checkers=(cast("ContractChecker", _contract_checker),),
        )
    )
    harness.run_twice_assert_idempotent("harness_row_count_plugin")
    harness.run_with_contracts(
        "harness_row_count_plugin", contract_assertions=_assert_contract_passed
    )


def _assert_contract_passed(contracts: Iterable[PluginContractResult]) -> None:
    """Assert that at least one contract passed."""
    contracts_list = list(contracts)
    if not contracts_list:
        pytest.fail("Expected a contract result")
    if contracts_list[0].status != "passed":
        pytest.fail("Contract should pass")
