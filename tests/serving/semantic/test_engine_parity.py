"""Polars/DuckDB parity tests for serving queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.semantic.models import SemanticQueryRequest
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.assertions.expectation_assertions import expect_equal
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.anyio
async def test_polars_duckdb_parity_for_simple_query(tmp_path: Path) -> None:
    """Polars and DuckDB engines should return identical rows for simple queries."""
    pytest.importorskip("polars")
    snapshot = ServingSnapshotFactory(tmp_path).demo_snapshot(row_count=4)

    manager = ServingDBManager(
        pointer_path=snapshot.pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
        hot_swap=False,
    )
    await manager.start()
    try:
        polars_kernel = SemanticQueryKernel(
            db=manager,
            settings=ServingSettings(
                serve_dir=snapshot.serve_dir,
                hot_swap=False,
                pool_size=1,
                poll_interval_s=0.01,
                schema_enforcement="strict",
                query_engine="polars",
                result_engine="polars",
            ),
        )
        duckdb_kernel = SemanticQueryKernel(
            db=manager,
            settings=ServingSettings(
                serve_dir=snapshot.serve_dir,
                hot_swap=False,
                pool_size=1,
                poll_interval_s=0.01,
                schema_enforcement="strict",
                query_engine="duckdb",
                result_engine="duckdb",
            ),
        )
        request = SemanticQueryRequest(
            view_id="demo.view",
            order_by=["id"],
            limit=3,
        )
        polars_result = polars_kernel.query(request)
        duckdb_result = duckdb_kernel.query(request)
        expect_equal(polars_result.rows, duckdb_result.rows)
    finally:
        await manager.stop()
