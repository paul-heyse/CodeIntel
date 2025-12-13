"""Dataset contract and row schema checks for analytics tables."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, get_type_hints

import pytest

from codeintel.analytics.utilities.datasets import (
    get_analytics_dataset_contract,
    insert_analytics_rows,
)
from codeintel.analytics.utilities.persistence import DeleteScope
from codeintel.config.datasets import (
    BehavioralCoverageRowModel,
    FunctionMetricsRow,
    FunctionTypesRow,
    GraphMetricsFunctionsExtRow,
    GraphMetricsFunctionsRow,
    GraphMetricsModulesExtRow,
    GraphMetricsModulesRow,
    ProfileRowModel,
    get_dataset_contracts_by_table_key,
)
from codeintel.graphs.catalog import FunctionCatalog
from tests._helpers import CORE_PACK, create_test_context
from tests._helpers.analytics_domain import make_graph_metric_function_row
from tests._helpers.catalogs import ensure_catalog_with_goids
from tests._helpers.contracts import ContractCtx, count_rows
from tests._helpers.rows import function_meta

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from pathlib import Path

    from duckdb import DuckDBPyConnection

    from tests._helpers import TestContext


def _function_metrics_row(ctx: ContractCtx) -> FunctionMetricsRow:
    now = datetime.now(UTC)
    return {
        "function_goid_h128": 1,
        "urn": "urn:demo#fn",
        "repo": ctx.repo,
        "commit": ctx.commit,
        "rel_path": "pkg/mod.py",
        "language": "python",
        "kind": "function",
        "qualname": "pkg.mod.fn",
        "start_line": 1,
        "end_line": 2,
        "loc": 2,
        "logical_loc": 2,
        "param_count": 1,
        "positional_params": 1,
        "keyword_only_params": 0,
        "has_varargs": False,
        "has_varkw": False,
        "is_async": False,
        "is_generator": False,
        "return_count": 0,
        "yield_count": 0,
        "raise_count": 0,
        "cyclomatic_complexity": 1,
        "max_nesting_depth": 0,
        "stmt_count": 1,
        "decorator_count": 0,
        "has_docstring": False,
        "complexity_bucket": "low",
        "created_at": now,
    }


def _graph_metrics_functions_row(ctx: ContractCtx) -> GraphMetricsFunctionsRow:
    now = datetime.now(UTC)
    catalog = FunctionCatalog(
        functions=[
            function_meta(
                goid=10,
                rel_path="pkg/mod.py",
                qualname="pkg.mod.fn",
                snapshot=(ctx.repo, ctx.commit),
                line_span=(1, 2),
            )
        ],
        module_by_path={"pkg/mod.py": "pkg.mod"},
    )
    ensure_catalog_with_goids(ctx, catalog)
    return make_graph_metric_function_row(
        repo=ctx.repo,
        commit=ctx.commit,
        function_goid_h128=10,
        overrides={
            "call_fan_in": 1,
            "call_fan_out": 1,
            "call_in_degree": 1,
            "call_out_degree": 1,
            "call_pagerank": 0.1,
            "call_betweenness": 0.2,
            "call_closeness": 0.3,
            "call_cycle_member": False,
            "created_at": now,
        },
    )


def _assert_graph_fk(con: DuckDBPyConnection) -> None:
    """Ensure graph_metrics_functions has matching GOIDs in core.goids."""
    rows = con.execute(
        """
        SELECT s.repo, s.commit, s.function_goid_h128
        FROM analytics.graph_metrics_functions s
        LEFT JOIN core.goids t
          ON (s.repo = t.repo AND s.commit = t.commit AND s.function_goid_h128 = t.goid_h128)
        WHERE t.repo IS NULL OR t.commit IS NULL OR t.goid_h128 IS NULL
        LIMIT 1
        """
    ).fetchall()
    if rows:
        pytest.fail("FK violation from analytics.graph_metrics_functions to core.goids")


COUNT_QUERIES: dict[str, str] = {
    "analytics.function_metrics": """
        SELECT COUNT(*) FROM analytics.function_metrics WHERE repo = ? AND commit = ?
    """,
    "analytics.graph_metrics_functions": """
        SELECT COUNT(*) FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?
    """,
}


def _count_table_rows(ctx: ContractCtx, table_key: str) -> int:
    """Return count for the provided table in the current repo/commit scope.

    Returns
    -------
    int
        Row count for the given table scoped to repo/commit.
    """
    sql = COUNT_QUERIES[table_key]
    return count_rows(ctx.gateway.con, sql, [ctx.repo, ctx.commit])


@pytest.fixture
def contract_ctx(tmp_path: Path) -> Iterator[ContractCtx]:
    """Contract context with an in-memory gateway and standard core schema.

    Yields
    ------
    ContractCtx
        Context containing gateway, repo, and commit.
    """
    ctx: TestContext = create_test_context(tmp_path)
    ctx.require(CORE_PACK)
    try:
        yield ContractCtx(gateway=ctx.gateway, repo=ctx.repo, commit=ctx.commit)
    finally:
        ctx.close()


@pytest.mark.parametrize(
    ("table_key", "row_builder", "extra_assertion"),
    [
        ("analytics.function_metrics", _function_metrics_row, None),
        ("analytics.graph_metrics_functions", _graph_metrics_functions_row, _assert_graph_fk),
    ],
)
def test_dataset_insertion_idempotent(
    contract_ctx: ContractCtx,
    table_key: str,
    row_builder: Callable[[ContractCtx], Mapping[str, object]],
    extra_assertion: Callable[[DuckDBPyConnection], None] | None,
) -> None:
    """Inserting rows twice should be idempotent and FK-clean when applicable."""
    contract = get_analytics_dataset_contract(contract_ctx.gateway, table_key)
    row = row_builder(contract_ctx)
    insert_analytics_rows(
        contract_ctx.gateway,
        contract,
        [row],
        delete_scope=DeleteScope(repo=contract_ctx.repo, commit=contract_ctx.commit),
        scope=f"{contract_ctx.repo}@{contract_ctx.commit}",
    )
    first = _count_table_rows(contract_ctx, table_key)
    insert_analytics_rows(
        contract_ctx.gateway,
        contract,
        [row],
        delete_scope=DeleteScope(repo=contract_ctx.repo, commit=contract_ctx.commit),
        scope=f"{contract_ctx.repo}@{contract_ctx.commit}",
    )
    second = _count_table_rows(contract_ctx, table_key)
    if first != 1 or second != 1:
        pytest.fail(f"Idempotency failure: first={first} second={second}")
    if extra_assertion is not None:
        extra_assertion(contract_ctx.gateway.con)


def _assert_row_matches_table(row_type: type[Mapping[str, object]], table_key: str) -> None:
    """Verify TypedDict annotations align with the DatasetContract schema."""
    contract = get_dataset_contracts_by_table_key().get(table_key)
    if contract is None or contract.schema is None:
        pytest.fail(f"{table_key} has no contract schema")
        return
    expected_cols = [col.name for col in contract.schema.columns]
    annotations = get_type_hints(row_type)
    actual_cols: Sequence[str] = list(annotations.keys())
    if list(actual_cols) != list(expected_cols):
        pytest.fail(f"{table_key} mismatch: {actual_cols} != {expected_cols}")


@pytest.mark.parametrize(
    ("row_type", "table_key"),
    [
        (FunctionMetricsRow, "analytics.function_metrics"),
        (FunctionTypesRow, "analytics.function_types"),
        (ProfileRowModel, "analytics.test_profile"),
        (BehavioralCoverageRowModel, "analytics.behavioral_coverage"),
        (GraphMetricsFunctionsRow, "analytics.graph_metrics_functions"),
        (GraphMetricsModulesRow, "analytics.graph_metrics_modules"),
        (GraphMetricsFunctionsExtRow, "analytics.graph_metrics_functions_ext"),
        (GraphMetricsModulesExtRow, "analytics.graph_metrics_modules_ext"),
    ],
)
def test_row_model_matches_contract(row_type: type[Mapping[str, object]], table_key: str) -> None:
    """Ensure TypedDict row models stay aligned with table schemas."""
    _assert_row_matches_table(row_type, table_key)
